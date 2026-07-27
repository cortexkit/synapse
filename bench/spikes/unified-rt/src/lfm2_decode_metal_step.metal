// LFM2 Metal step kernels.
//
// This library holds the decode-step kernels that are specific to the LFM2
// hybrid backbone. The attention layers reuse the Qwen3 step kernels (RMSNorm,
// QK-norm + RoPE, GQA attention, pack-4 Q8 / f16 matvec, on-GPU argmax) which
// are already proven token-exact by the Qwen3 campaign; those live in
// qwen3_decode_metal_step.metal and are intentionally NOT duplicated or mutated
// here so the Qwen3 byte-identity fixtures stay undisturbed.
//
// The genuinely new kernel is the short-convolution decode step below. It is
// the conv-cache analogue of the KV-cache attention step: each LFM2 conv layer
// keeps a small rolling window (kernel_size rows of `hidden` channels) resident
// on device, and every decoded token advances that window in place and emits
// the gated convolution output for the newest position only.
//
// Exactness contract (mirrors lfm2.rs decode_conv exactly):
//   1. in_proj splits into [x, gate, y]; product = x * y (computed host-side or
//      by a reused matvec kernel; fed here as `product`).
//   2. Advance the rolling cache: row t := row t+1 for t < kernel_size-1, then
//      row kernel_size-1 := product. (decode_conv: state.copy_within + write.)
//   3. Depthwise causal conv at the newest position, serial tap-ascending f32
//      accumulation:
//          conv[c] = sum_{tap=0..kernel_size-1} cache[tap*hidden+c]
//                                         * conv_weight[c*kernel_size+tap]
//      This is the same operand set and reduction order as the CPU reference
//      depthwise_causal_conv1d evaluated at the final sequence position, so the
//      result is bit-identical (per-dot serial f32, no parallel reduction).
//   4. out[c] = gate[c] * conv[c]  (decode_conv: gate *= convolved_state[last]).
//
// Each thread owns one channel column of the cache, so the in-place advance and
// the convolution read/write only that column: there is no cross-thread
// dependency and the kernel is deterministic for a given grid.

#include <metal_stdlib>
using namespace metal;

struct Lfm2ConvStepParams {
    uint hidden;
    uint kernel_size;
};

// cache:       [kernel_size * hidden] f32, rolling window, row 0 = oldest.
// product:     [hidden] f32, the new x*y row to append at row kernel_size-1.
// gate:        [hidden] f32, the c_gate row from in_proj.
// conv_weight: [hidden * kernel_size] f32, depthwise taps, channel-major
//              (index = channel * kernel_size + tap), matching the CPU layout.
// out:         [hidden] f32, gate * conv(newest position), fed to out_proj.
kernel void lfm2_conv_step(
    device float *cache [[buffer(0)]],
    const device float *product [[buffer(1)]],
    const device float *gate [[buffer(2)]],
    const device float *conv_weight [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant Lfm2ConvStepParams &params [[buffer(5)]],
    uint channel [[thread_position_in_grid]]
) {
    const uint hidden = params.hidden;
    const uint kernel_size = params.kernel_size;
    if (channel >= hidden) {
        return;
    }

    // Advance this channel's rolling column in place: shift every row toward the
    // oldest slot, then append the new product row at the newest slot. This is
    // the device form of decode_conv's state.copy_within(hidden.., 0) followed by
    // state[(kernel_size - 1) * hidden..] = product.
    for (uint row = 0; row + 1 < kernel_size; ++row) {
        cache[row * hidden + channel] = cache[(row + 1) * hidden + channel];
    }
    cache[(kernel_size - 1) * hidden + channel] = product[channel];

    // Depthwise causal convolution at the newest position. The cache now holds
    // kernel_size rows of history; the newest-position output taps all of them in
    // ascending tap order with a serial f32 accumulator, exactly matching the CPU
    // reference reduction order so the bits agree.
    float accumulator = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
        accumulator += cache[tap * hidden + channel] * conv_weight[channel * kernel_size + tap];
    }
    out[channel] = gate[channel] * accumulator;
}

struct Lfm2ConvSplitParams {
    uint hidden;
};

// Split the in_proj output into the short-convolution operands, matching
// lfm2.rs::decode_conv exactly. in_proj maps hidden -> 3*hidden and the result
// is laid out as [x | gate | y]; the conv path consumes product = x * y and the
// gate row. The surrounding projection kernels keep activations in f16, while
// the conv step itself runs f32 internally (the stage-A exactness contract), so
// this kernel widens the two f16 slices to f32 as it forms them.
//
//   proj:    [3 * hidden] f16, the in_proj output.
//   product: [hidden] f32, product[c] = proj[c] * proj[2*hidden + c]   (x * y).
//   gate:    [hidden] f32, gate[c]    = proj[hidden + c].
kernel void lfm2_conv_split(
    const device half *proj [[buffer(0)]],
    device float *product [[buffer(1)]],
    device float *gate [[buffer(2)]],
    constant Lfm2ConvSplitParams &params [[buffer(3)]],
    uint channel [[thread_position_in_grid]]
) {
    const uint hidden = params.hidden;
    if (channel >= hidden) {
        return;
    }
    float x = (float)proj[channel];
    float y = (float)proj[2 * hidden + channel];
    product[channel] = x * y;
    gate[channel] = (float)proj[hidden + channel];
}

// Widen the f32 conv-step output back to the f16 activations the reused
// out_proj matvec kernel consumes. The conv step keeps its rolling cache and
// its per-position output in f32 (bit-exact with the CPU reference); only the
// value handed to the next projection is rounded to f16, the same boundary
// rounding the rest of the hybrid engine applies between layers.
//
//   in:  [hidden] f32, gate * conv(newest position) from lfm2_conv_step.
//   out: [hidden] f16, fed to the conv out_proj matvec.
kernel void lfm2_conv_f32_to_f16(
    const device float *in [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant Lfm2ConvSplitParams &params [[buffer(2)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= params.hidden) {
        return;
    }
    out[channel] = (half)in[channel];
}
