// Native Metal harness for the LFM2 decode step kernels.
//
// This file exposes a small C ABI over the lfm2_decode_metal_step.metal kernels
// so the Rust driver (lfm2_decode_metal_step.rs) can drive them. It is additive:
// it shares no mutable state with the Qwen3 step harness and never touches the
// Qwen3 kernels, so the Qwen3 byte-identity fixtures are unaffected.
//
// The harness models the LFM2 conv-cache model directly: each convolution layer
// owns a device-resident rolling cache buffer (kernel_size rows of `hidden`
// channels). A decode step advances that cache in place on the GPU and reads
// back only the gated convolution output for the newest position, exactly like
// the KV-cache attention step reads back one context row. The cache buffers
// persist for the life of the context so a sequence of steps rolls the window
// forward without host round trips; cache_read/cache_write exist for tests and
// to leave room for a future rewind/rollback without changing the ABI.
//
// Memory management follows the rest of this directory: manual retain/release
// (the build does not enable ARC) and MTLResourceStorageModeShared buffers so
// the CPU can feed activations and read results through buffer.contents once a
// command buffer has completed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char lfm2_step_error[1024];

static void set_error(NSString *message) {
    if (message == nil) {
        message = @"unknown LFM2 Metal step error";
    }
    const char *utf8 = [message UTF8String];
    if (utf8 == NULL) {
        lfm2_step_error[0] = '\0';
        return;
    }
    strncpy(lfm2_step_error, utf8, sizeof(lfm2_step_error) - 1);
    lfm2_step_error[sizeof(lfm2_step_error) - 1] = '\0';
}

const char *synapse_lfm2_metal_step_last_error(void) {
    return lfm2_step_error;
}

// Mirrors Lfm2ConvStepParams in the Metal source. Two uint32 fields keep the
// struct layout identical on both sides of the FFI boundary (8 bytes, 4-byte
// aligned), which is what setBytes:index: binds into the kernel's constant
// address space.
typedef struct Lfm2ConvStepParams {
    uint32_t hidden;
    uint32_t kernel_size;
} Lfm2ConvStepParams;

typedef struct Lfm2ConvLayerBuffers {
    id<MTLBuffer> cache;       // [kernel_size * hidden] f32, zeroed, rolling window.
    id<MTLBuffer> conv_weight; // [hidden * kernel_size] f32, static depthwise taps.
} Lfm2ConvLayerBuffers;

typedef struct Lfm2MetalStepContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> conv_step;
    uint64_t hidden;
    uint64_t kernel_size;
    Lfm2ConvLayerBuffers *layers;
    uint64_t layer_count;
    // Reusable per-step scratch sized to `hidden` so a step performs no
    // allocation: the host feeds product/gate into these and reads `out`.
    id<MTLBuffer> product;
    id<MTLBuffer> gate;
    id<MTLBuffer> out;
} Lfm2MetalStepContext;

static id<MTLBuffer> shared_buffer(id<MTLDevice> device, const void *bytes, NSUInteger length) {
    if (bytes == NULL) {
        return [device newBufferWithLength:length
                                   options:MTLResourceStorageModeShared];
    }
    return [device newBufferWithBytes:bytes
                               length:length
                              options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (result == nil) {
        set_error(error.localizedDescription
                      ?: [NSString stringWithFormat:@"failed to compile Metal kernel %@", name]);
    }
    return result;
}

void *synapse_lfm2_metal_step_context_new(
    uint64_t hidden,
    uint64_t kernel_size,
    const char *metallib_path
) {
    @autoreleasepool {
        if (hidden == 0 || kernel_size == 0 || hidden > UINT32_MAX || kernel_size > UINT32_MAX ||
            metallib_path == NULL) {
            set_error(@"invalid LFM2 Metal step dimensions or metallib path");
            return NULL;
        }
        Lfm2MetalStepContext *context = calloc(1, sizeof(*context));
        if (context == NULL) {
            set_error(@"failed to allocate LFM2 Metal step context");
            return NULL;
        }
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            set_error(@"no Metal device for LFM2 Metal step");
            free(context);
            return NULL;
        }
        context->queue = [context->device newCommandQueue];
        NSError *error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        context->library = [context->device newLibraryWithURL:library_url error:&error];
        if (context->queue == nil || context->library == nil) {
            set_error(error.localizedDescription ?: @"failed to load LFM2 Metal step metallib");
            [context->queue release];
            [context->library release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->conv_step = pipeline(context->device, context->library, @"lfm2_conv_step");
        if (context->conv_step == nil) {
            [context->library release];
            [context->queue release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->hidden = hidden;
        context->kernel_size = kernel_size;
        return context;
    }
}

int32_t synapse_lfm2_metal_step_prepare(
    void *raw,
    uint64_t conv_layer_count,
    const float *const *conv_weights
) {
    @autoreleasepool {
        @try {
            Lfm2MetalStepContext *context = raw;
            if (context == NULL || conv_layer_count == 0 || conv_weights == NULL) {
                set_error(@"invalid LFM2 Metal step preparation arguments");
                return -1;
            }
            if (context->layers != NULL) {
                set_error(@"LFM2 Metal step context already prepared");
                return -2;
            }
            const NSUInteger hidden = (NSUInteger)context->hidden;
            const NSUInteger kernel_size = (NSUInteger)context->kernel_size;
            const NSUInteger cache_bytes = (NSUInteger)(kernel_size * hidden * sizeof(float));
            const NSUInteger weight_bytes = (NSUInteger)(hidden * kernel_size * sizeof(float));

            context->layers = calloc((size_t)conv_layer_count, sizeof(*context->layers));
            if (context->layers == NULL) {
                set_error(@"failed to allocate LFM2 conv layer table");
                return -3;
            }
            context->layer_count = conv_layer_count;
            for (uint64_t layer = 0; layer < conv_layer_count; ++layer) {
                if (conv_weights[layer] == NULL) {
                    set_error([NSString stringWithFormat:@"LFM2 conv layer %llu weight is null", layer]);
                    return -4;
                }
                // Zero-initialised rolling cache: matches empty_decode_cache, which
                // starts every conv state at all zeros.
                context->layers[layer].cache = shared_buffer(context->device, NULL, cache_bytes);
                context->layers[layer].conv_weight =
                    shared_buffer(context->device, conv_weights[layer], weight_bytes);
                if (context->layers[layer].cache == nil || context->layers[layer].conv_weight == nil) {
                    set_error(@"failed to allocate LFM2 conv layer buffers");
                    return -5;
                }
            }
            const NSUInteger scratch_bytes = (NSUInteger)(hidden * sizeof(float));
            context->product = shared_buffer(context->device, NULL, scratch_bytes);
            context->gate = shared_buffer(context->device, NULL, scratch_bytes);
            context->out = shared_buffer(context->device, NULL, scratch_bytes);
            if (context->product == nil || context->gate == nil || context->out == nil) {
                set_error(@"failed to allocate LFM2 conv step scratch buffers");
                return -6;
            }
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 Metal step preparation raised");
            return -7;
        }
    }
}

// Run one conv decode step for `layer`: feed product/gate, advance the layer's
// device-resident cache in place, and read back out[hidden] = gate * conv(newest
// position). The cache stays on device between calls, so a sequence of steps
// rolls the window forward exactly as the CPU decode_conv advances its Vec state.
int32_t synapse_lfm2_conv_step(
    void *raw,
    uint64_t layer,
    const float *product,
    const float *gate,
    float *out
) {
    @autoreleasepool {
        @try {
            Lfm2MetalStepContext *context = raw;
            if (context == NULL || context->layers == NULL || layer >= context->layer_count ||
                product == NULL || gate == NULL || out == NULL) {
                set_error(@"invalid LFM2 conv step arguments");
                return -1;
            }
            const NSUInteger scratch_bytes = (NSUInteger)(context->hidden * sizeof(float));
            memcpy(context->product.contents, product, scratch_bytes);
            memcpy(context->gate.contents, gate, scratch_bytes);

            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:context->conv_step];
            [encoder setBuffer:context->layers[layer].cache offset:0 atIndex:0];
            [encoder setBuffer:context->product offset:0 atIndex:1];
            [encoder setBuffer:context->gate offset:0 atIndex:2];
            [encoder setBuffer:context->layers[layer].conv_weight offset:0 atIndex:3];
            [encoder setBuffer:context->out offset:0 atIndex:4];
            Lfm2ConvStepParams params = {
                (uint32_t)context->hidden,
                (uint32_t)context->kernel_size,
            };
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            NSUInteger threads_per_group = context->conv_step.maxTotalThreadsPerThreadgroup;
            if (threads_per_group > context->hidden) {
                threads_per_group = (NSUInteger)context->hidden;
            }
            if (threads_per_group == 0) {
                threads_per_group = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(context->hidden, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                set_error(command_buffer.error.localizedDescription ?: @"LFM2 conv step command failed");
                return -2;
            }
            memcpy(out, context->out.contents, scratch_bytes);
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 conv step raised");
            return -3;
        }
    }
}

// Read the layer's current device-resident cache back to the host. Used by the
// exactness gate to confirm the rolling window matches the CPU state, and a
// building block for any future rewind/rollback.
int32_t synapse_lfm2_conv_cache_read(void *raw, uint64_t layer, float *host) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL || context->layers == NULL || layer >= context->layer_count || host == NULL) {
            set_error(@"invalid LFM2 conv cache read arguments");
            return -1;
        }
        const NSUInteger cache_bytes =
            (NSUInteger)(context->kernel_size * context->hidden * sizeof(float));
        memcpy(host, context->layers[layer].cache.contents, cache_bytes);
        return 0;
    }
}

// Overwrite the layer's device-resident cache from the host. Lets tests seed a
// starting window and leaves a clean hook for future rewind without an ABI break.
int32_t synapse_lfm2_conv_cache_write(void *raw, uint64_t layer, const float *host) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL || context->layers == NULL || layer >= context->layer_count || host == NULL) {
            set_error(@"invalid LFM2 conv cache write arguments");
            return -1;
        }
        const NSUInteger cache_bytes =
            (NSUInteger)(context->kernel_size * context->hidden * sizeof(float));
        memcpy(context->layers[layer].cache.contents, host, cache_bytes);
        return 0;
    }
}

void synapse_lfm2_metal_step_context_free(void *raw) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL) {
            return;
        }
        if (context->layers != NULL) {
            for (uint64_t layer = 0; layer < context->layer_count; ++layer) {
                [context->layers[layer].cache release];
                [context->layers[layer].conv_weight release];
            }
            free(context->layers);
        }
        [context->product release];
        [context->gate release];
        [context->out release];
        [context->conv_step release];
        [context->library release];
        [context->queue release];
        [context->device release];
        free(context);
    }
}


// ===========================================================================
// Hybrid decode-step context (stage C).
//
// This is the end-to-end LFM2 step engine. It walks the hybrid backbone -- ten
// short-convolution layers and six full-attention layers
// (Config::full_attn_idxs = [2,5,8,10,12,14]) -- entirely on device, reusing the
// proven Qwen3 step kernels for everything the two families share (RMSNorm,
// QKV matvec, QK-norm + RoPE, GQA attention, matvec + residual, SwiGLU, LM head,
// on-GPU argmax, embedding gather) and adding the LFM2-specific conv path
// (lfm2_conv_split -> lfm2_conv_step -> lfm2_conv_f32_to_f16) for the conv
// layers. Per-layer dispatch follows lfm2.rs::decode_embedding exactly:
//
//   operator_norm -> mixer -> +residual -> ffn_norm -> SwiGLU FFN -> +residual
//
// with the mixer being either the conv path or the attention path. final_norm,
// the tied LM head, and argmax form the shared tail.
//
// This context is deliberately separate from the conv-only context above (which
// the stage-A exactness gates drive) and from the Qwen3 context (which stays
// untouched): it owns its own device-resident KV caches (attention layers) and
// rolling conv caches (conv layers), its own f16 weights, and its own scratch.
// Activations are f16 between layers (the Qwen3 kernel discipline); the conv
// step keeps its cache and per-position output in f32, widening at the split
// and narrowing again before out_proj.
//
// Memory management matches this directory: manual retain/release (no ARC),
// weights and caches uploaded to private storage via a blit, host-visible
// (shared) buffers only where the CPU feeds or reads (logits, the chained token
// id scratch).
// ===========================================================================

typedef struct Lfm2HybridLayerParams {
    const void *operator_norm;       // f16 [hidden]
    const void *ffn_norm;            // f16 [hidden]
    const void *gate_weight;         // f16 [intermediate * hidden]   (w1)
    const void *gate_weight_q8;      // Q8_0 blocks for gate_weight (or NULL)
    const void *up_weight;           // f16 [intermediate * hidden]   (w3)
    const void *up_weight_q8;        // Q8_0 blocks for up_weight (or NULL)
    const void *down_weight;         // f16 [hidden * intermediate]   (w2)
    const void *down_weight_q8;      // Q8_0 blocks for down_weight (or NULL)
    const void *in_proj_weight;      // conv: f16 [3*hidden * hidden]
    const void *in_proj_weight_q8;   // Q8_0 blocks for in_proj_weight (or NULL)
    const void *conv_weight;         // conv: f32 [hidden * kernel_size] (never quantized)
    const void *out_proj_weight;     // conv: f16 [hidden * hidden]
    const void *out_proj_weight_q8;  // Q8_0 blocks for out_proj_weight (or NULL)
    const void *q_weight;            // attn: f16 [query_width * hidden]
    const void *q_weight_q8;         // Q8_0 blocks for q_weight (or NULL)
    const void *k_weight;            // attn: f16 [kv_width * hidden]
    const void *k_weight_q8;         // Q8_0 blocks for k_weight (or NULL)
    const void *v_weight;            // attn: f16 [kv_width * hidden]
    const void *v_weight_q8;         // Q8_0 blocks for v_weight (or NULL)
    const void *o_weight;            // attn: f16 [hidden * query_width]
    const void *o_weight_q8;         // Q8_0 blocks for o_weight (or NULL)
    const void *q_norm;              // attn: f16 [head_dim]
    const void *k_norm;              // attn: f16 [head_dim]
    uint64_t is_attention;           // 1 = full attention, 0 = short conv
} Lfm2HybridLayerParams;

// One matmul weight slot. The reused Qwen3 kernels read either the f16 buffer
// (quantized == 0) or the Q8_0 block buffer (quantized != 0); exactly one of the
// two is resident for a given engine, matching the Qwen3 step context's
// StepWeight. The f16 and Q8 buffers are never both allocated: an f16 engine
// leaves q8 nil, a Q8 engine leaves fp16 nil.
typedef struct Lfm2StepWeight {
    id<MTLBuffer> fp16;
    id<MTLBuffer> q8;
} Lfm2StepWeight;

typedef struct Lfm2HybridLayerBuffers {
    id<MTLBuffer> operator_norm;
    id<MTLBuffer> ffn_norm;
    Lfm2StepWeight gate_weight;
    Lfm2StepWeight up_weight;
    Lfm2StepWeight down_weight;
    // Conv layers.
    Lfm2StepWeight in_proj_weight;
    id<MTLBuffer> conv_weight;
    Lfm2StepWeight out_proj_weight;
    id<MTLBuffer> conv_cache;     // f32 [kernel_size * hidden], rolling window.
    // Attention layers.
    Lfm2StepWeight q_weight;
    Lfm2StepWeight k_weight;
    Lfm2StepWeight v_weight;
    Lfm2StepWeight o_weight;
    id<MTLBuffer> q_norm;
    id<MTLBuffer> k_norm;
    id<MTLBuffer> key_cache;      // f16 [kv_heads * bucket * head_dim].
    id<MTLBuffer> value_cache;    // f16 [kv_heads * bucket * head_dim].
    BOOL is_attention;
} Lfm2HybridLayerBuffers;

typedef struct Lfm2HybridStepContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    // Reused Qwen3 step pipelines.
    id<MTLComputePipelineState> rmsnorm;
    id<MTLComputePipelineState> qkv_matvec;
    id<MTLComputePipelineState> qk_norm_rope;
    id<MTLComputePipelineState> attention;
    id<MTLComputePipelineState> matvec_residual;
    id<MTLComputePipelineState> residual_rmsnorm;
    id<MTLComputePipelineState> gate_up_swiglu;
    id<MTLComputePipelineState> lm_head;
    id<MTLComputePipelineState> argmax_partial;
    id<MTLComputePipelineState> argmax_final;
    id<MTLComputePipelineState> embedding_gather;
    // LFM2-specific conv pipelines.
    id<MTLComputePipelineState> conv_step;
    id<MTLComputePipelineState> conv_split;
    id<MTLComputePipelineState> conv_f32_to_f16;
    Lfm2HybridLayerBuffers *layers;
    uint64_t layer_count;
    uint64_t bucket;
    uint64_t hidden;
    uint64_t query_heads;
    uint64_t kv_heads;
    uint64_t head_dim;
    uint64_t intermediate;
    uint64_t vocab;
    uint64_t kernel_size;
    float epsilon;
    // Quantization mode for the whole engine: 0 = f16 matmuls, 1 = Q8_0 matmuls.
    // Set once at prepare; every matmul encode helper threads it into its kernel
    // config so the reused kernels select the matching weight path and dispatch.
    uint32_t quantized;
    // Per-step activation scratch (f16 unless noted), sized so a step allocates
    // nothing. current/next ping-pong the running residual across layers.
    id<MTLBuffer> x_a;
    id<MTLBuffer> x_b;
    id<MTLBuffer> normalized;
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> attention_context;
    id<MTLBuffer> attention_scores;   // f32 [query_heads * bucket]
    id<MTLBuffer> mlp;
    id<MTLBuffer> final_norm;
    id<MTLBuffer> logits;             // f32 [vocab], shared for host readback.
    // Conv-path scratch.
    id<MTLBuffer> conv_proj;          // f16 [3 * hidden]
    id<MTLBuffer> conv_product;       // f32 [hidden]
    id<MTLBuffer> conv_gate;          // f32 [hidden]
    id<MTLBuffer> conv_out;           // f32 [hidden]
    id<MTLBuffer> conv_out_f16;       // f16 [hidden]
    // Shared tail / chained-decode residents.
    id<MTLBuffer> final_norm_weight;
    Lfm2StepWeight lm_head_weight;    // tied head: f16 or Q8_0.
    id<MTLBuffer> embeddings;         // f16 [vocab * hidden], tied table.
    uint64_t argmax_partials;
    id<MTLBuffer> argmax_partial_keys;
    id<MTLBuffer> argmax_partial_ids;
    id<MTLBuffer> chain_token_ids;    // shared, host seeds step 0.
    id<MTLBuffer> chain_input;
    id<MTLBuffer> zero_buffer;        // shared zeros for cache reset blits.
} Lfm2HybridStepContext;

static MTLSize hybrid_grid(NSUInteger count) {
    return MTLSizeMake(count, 1, 1);
}

static MTLSize hybrid_group(NSUInteger count) {
    return MTLSizeMake(MIN((NSUInteger)256, MAX((NSUInteger)1, count)), 1, 1);
}

// Upload `bytes` (length bytes) into a fresh private buffer via the blit encoder.
static id<MTLBuffer> hybrid_private(id<MTLDevice> device, id<MTLBlitCommandEncoder> blit, const void *bytes, NSUInteger length) {
    if (bytes == NULL || length == 0 || blit == nil) return nil;
    id<MTLBuffer> source = [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    id<MTLBuffer> destination = [device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    if (source == nil || destination == nil) {
        [source release];
        [destination release];
        return nil;
    }
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:length];
    [source release];
    return destination;
}

// Upload `elements` f16 values as a private weight buffer.
static id<MTLBuffer> hybrid_weight_f16(id<MTLDevice> device, id<MTLBlitCommandEncoder> blit, const void *fp16, NSUInteger elements) {
    if (fp16 == NULL) return nil;
    return hybrid_private(device, blit, fp16, elements * sizeof(uint16_t));
}

// Upload `elements` f32 values as a private buffer (the conv depthwise taps).
static id<MTLBuffer> hybrid_weight_f32(id<MTLDevice> device, id<MTLBlitCommandEncoder> blit, const void *fp32, NSUInteger elements) {
    if (fp32 == NULL) return nil;
    return hybrid_private(device, blit, fp32, elements * sizeof(float));
}

// Upload one matmul weight as EITHER f16 or Q8_0, mirroring the Qwen3 step
// context's new_weight. When `q8` is non-NULL the weight is stored as
// `elements / 32 * 34` Q8_0 block bytes (each 32-element row block is the GGUF
// 34-byte layout: f16 scale + 32 i8 quants) and the f16 slot stays nil;
// otherwise the f16 buffer is uploaded and the Q8 slot stays nil. `elements` is
// the matrix element count (rows * cols); it is a multiple of 32 because every
// LFM2 matmul column dimension is a multiple of 32.
static Lfm2StepWeight hybrid_weight(id<MTLDevice> device, id<MTLBlitCommandEncoder> blit, const void *fp16, const void *q8, NSUInteger elements) {
    Lfm2StepWeight weight = { nil, nil };
    if (q8 != NULL) {
        weight.q8 = hybrid_private(device, blit, q8, elements / 32 * 34);
    } else {
        weight.fp16 = hybrid_weight_f16(device, blit, fp16, elements);
    }
    return weight;
}

static id<MTLBuffer> hybrid_zero(id<MTLDevice> device, NSUInteger length, MTLResourceOptions options) {
    if (length == 0) return nil;
    return [device newBufferWithLength:length options:options];
}

void *synapse_lfm2_hybrid_step_context_new(
    uint64_t bucket,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t vocab,
    uint64_t kernel_size,
    float epsilon,
    const char *metallib_path
) {
    @autoreleasepool {
        if (bucket == 0 || hidden == 0 || query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
            intermediate == 0 || vocab == 0 || kernel_size == 0 || query_heads % kv_heads != 0 ||
            head_dim % 2 != 0 || metallib_path == NULL) {
            set_error(@"invalid LFM2 hybrid step dimensions or metallib path");
            return NULL;
        }
        Lfm2HybridStepContext *context = calloc(1, sizeof(*context));
        if (context == NULL) {
            set_error(@"failed to allocate LFM2 hybrid step context");
            return NULL;
        }
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            set_error(@"no Metal device for LFM2 hybrid step");
            free(context);
            return NULL;
        }
        context->queue = [context->device newCommandQueue];
        NSError *error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        context->library = [context->device newLibraryWithURL:library_url error:&error];
        if (context->queue == nil || context->library == nil) {
            set_error(error.localizedDescription ?: @"failed to load LFM2 hybrid step metallib");
            [context->queue release];
            [context->library release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->rmsnorm = pipeline(context->device, context->library, @"metal_step_rmsnorm");
        context->qkv_matvec = pipeline(context->device, context->library, @"metal_step_qkv_matvec");
        context->qk_norm_rope = pipeline(context->device, context->library, @"metal_step_qk_norm_rope");
        context->attention = pipeline(context->device, context->library, @"metal_step_attention");
        context->matvec_residual = pipeline(context->device, context->library, @"metal_step_matvec_residual");
        context->residual_rmsnorm = pipeline(context->device, context->library, @"metal_step_residual_rmsnorm");
        context->gate_up_swiglu = pipeline(context->device, context->library, @"metal_step_gate_up_swiglu");
        context->lm_head = pipeline(context->device, context->library, @"metal_step_lm_head");
        context->argmax_partial = pipeline(context->device, context->library, @"metal_step_argmax_partial");
        context->argmax_final = pipeline(context->device, context->library, @"metal_step_argmax_final");
        context->embedding_gather = pipeline(context->device, context->library, @"metal_step_embedding_gather");
        context->conv_step = pipeline(context->device, context->library, @"lfm2_conv_step");
        context->conv_split = pipeline(context->device, context->library, @"lfm2_conv_split");
        context->conv_f32_to_f16 = pipeline(context->device, context->library, @"lfm2_conv_f32_to_f16");
        if (context->rmsnorm == nil || context->qkv_matvec == nil || context->qk_norm_rope == nil ||
            context->attention == nil || context->matvec_residual == nil || context->residual_rmsnorm == nil ||
            context->gate_up_swiglu == nil || context->lm_head == nil || context->argmax_partial == nil ||
            context->argmax_final == nil || context->embedding_gather == nil || context->conv_step == nil ||
            context->conv_split == nil || context->conv_f32_to_f16 == nil) {
            set_error(@"failed to compile an LFM2 hybrid step kernel");
            [context->library release];
            [context->queue release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->bucket = bucket;
        context->hidden = hidden;
        context->query_heads = query_heads;
        context->kv_heads = kv_heads;
        context->head_dim = head_dim;
        context->intermediate = intermediate;
        context->vocab = vocab;
        context->kernel_size = kernel_size;
        context->epsilon = epsilon;
        return context;
    }
}

int32_t synapse_lfm2_hybrid_step_prepare(
    void *raw,
    uint64_t layer_count,
    uint32_t quantized,
    const Lfm2HybridLayerParams *params,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *lm_head_q8,
    const void *embeddings
) {
    @autoreleasepool {
        @try {
            Lfm2HybridStepContext *context = raw;
            // The tied LM head is f16 in an f16 engine and Q8_0 in a quantized
            // engine; require the slot matching the mode (the other is null).
            if (context == NULL || layer_count == 0 || params == NULL || final_norm_weight == NULL ||
                (quantized ? lm_head_q8 == NULL : lm_head_weight == NULL) || embeddings == NULL) {
                set_error(@"invalid LFM2 hybrid step preparation arguments");
                return -1;
            }
            if (context->layers != NULL) {
                set_error(@"LFM2 hybrid step context already prepared");
                return -2;
            }
            const uint64_t query_width = context->query_heads * context->head_dim;
            const uint64_t kv_width = context->kv_heads * context->head_dim;
            const NSUInteger kv_cache_elements = (NSUInteger)(context->kv_heads * context->bucket * context->head_dim);
            const NSUInteger conv_cache_elements = (NSUInteger)(context->kernel_size * context->hidden);

            context->layers = calloc((size_t)layer_count, sizeof(*context->layers));
            if (context->layers == NULL) {
                set_error(@"failed to allocate LFM2 hybrid step layer table");
                return -3;
            }
            context->layer_count = layer_count;
            context->quantized = quantized;

            id<MTLCommandBuffer> upload_command = [context->queue commandBuffer];
            id<MTLBlitCommandEncoder> upload_blit = [upload_command blitCommandEncoder];
            if (upload_command == nil || upload_blit == nil) {
                set_error(@"failed to create LFM2 hybrid weight upload command");
                return -4;
            }
            BOOL alloc_ok = YES;
            for (uint64_t i = 0; i < layer_count; ++i) {
                const Lfm2HybridLayerParams *source = &params[i];
                Lfm2HybridLayerBuffers *target = &context->layers[i];
                target->is_attention = source->is_attention != 0;
                // A quantized engine must be handed a Q8_0 buffer for every matmul
                // it runs; a missing block buffer would make the kernel dereference
                // a nil weight slot. Conv and attention layers quantize different
                // projections, so the required set depends on the layer type.
                if (quantized && (source->gate_weight_q8 == NULL || source->up_weight_q8 == NULL ||
                                  source->down_weight_q8 == NULL ||
                                  (target->is_attention
                                       ? (source->q_weight_q8 == NULL || source->k_weight_q8 == NULL ||
                                          source->v_weight_q8 == NULL || source->o_weight_q8 == NULL)
                                       : (source->in_proj_weight_q8 == NULL || source->out_proj_weight_q8 == NULL)))) {
                    set_error(@"quantized LFM2 hybrid step is missing a Q8_0 weight buffer");
                    alloc_ok = NO;
                    break;
                }
                // Weights shared by every layer: the two norms (always f16) and the
                // SwiGLU FFN projections (f16 or Q8_0).
                target->operator_norm = hybrid_weight_f16(context->device, upload_blit, source->operator_norm, context->hidden);
                target->ffn_norm = hybrid_weight_f16(context->device, upload_blit, source->ffn_norm, context->hidden);
                target->gate_weight = hybrid_weight(context->device, upload_blit, source->gate_weight, source->gate_weight_q8, context->intermediate * context->hidden);
                target->up_weight = hybrid_weight(context->device, upload_blit, source->up_weight, source->up_weight_q8, context->intermediate * context->hidden);
                target->down_weight = hybrid_weight(context->device, upload_blit, source->down_weight, source->down_weight_q8, context->hidden * context->intermediate);
                if (target->operator_norm == nil || target->ffn_norm == nil ||
                    (quantized ? (target->gate_weight.q8 == nil || target->up_weight.q8 == nil || target->down_weight.q8 == nil)
                               : (target->gate_weight.fp16 == nil || target->up_weight.fp16 == nil || target->down_weight.fp16 == nil))) {
                    alloc_ok = NO;
                    break;
                }
                if (target->is_attention) {
                    target->q_weight = hybrid_weight(context->device, upload_blit, source->q_weight, source->q_weight_q8, query_width * context->hidden);
                    target->k_weight = hybrid_weight(context->device, upload_blit, source->k_weight, source->k_weight_q8, kv_width * context->hidden);
                    target->v_weight = hybrid_weight(context->device, upload_blit, source->v_weight, source->v_weight_q8, kv_width * context->hidden);
                    target->o_weight = hybrid_weight(context->device, upload_blit, source->o_weight, source->o_weight_q8, context->hidden * query_width);
                    target->q_norm = hybrid_weight_f16(context->device, upload_blit, source->q_norm, context->head_dim);
                    target->k_norm = hybrid_weight_f16(context->device, upload_blit, source->k_norm, context->head_dim);
                    target->key_cache = hybrid_zero(context->device, kv_cache_elements * sizeof(uint16_t), MTLResourceStorageModePrivate);
                    target->value_cache = hybrid_zero(context->device, kv_cache_elements * sizeof(uint16_t), MTLResourceStorageModePrivate);
                    if ((quantized ? (target->q_weight.q8 == nil || target->k_weight.q8 == nil ||
                                      target->v_weight.q8 == nil || target->o_weight.q8 == nil)
                                   : (target->q_weight.fp16 == nil || target->k_weight.fp16 == nil ||
                                      target->v_weight.fp16 == nil || target->o_weight.fp16 == nil)) ||
                        target->q_norm == nil || target->k_norm == nil ||
                        target->key_cache == nil || target->value_cache == nil) {
                        alloc_ok = NO;
                        break;
                    }
                } else {
                    target->in_proj_weight = hybrid_weight(context->device, upload_blit, source->in_proj_weight, source->in_proj_weight_q8, 3 * context->hidden * context->hidden);
                    target->conv_weight = hybrid_weight_f32(context->device, upload_blit, source->conv_weight, context->hidden * context->kernel_size);
                    target->out_proj_weight = hybrid_weight(context->device, upload_blit, source->out_proj_weight, source->out_proj_weight_q8, context->hidden * context->hidden);
                    // Rolling conv cache: shared so the reset path can zero it from
                    // the host, matching empty_decode_cache's all-zero conv state.
                    target->conv_cache = hybrid_zero(context->device, conv_cache_elements * sizeof(float), MTLResourceStorageModeShared);
                    if ((quantized ? (target->in_proj_weight.q8 == nil || target->out_proj_weight.q8 == nil)
                                   : (target->in_proj_weight.fp16 == nil || target->out_proj_weight.fp16 == nil)) ||
                        target->conv_weight == nil || target->conv_cache == nil) {
                        alloc_ok = NO;
                        break;
                    }
                }
            }
            if (!alloc_ok) {
                [upload_blit endEncoding];
                [upload_command commit];
                set_error(@"failed to allocate LFM2 hybrid step layer buffers");
                return -5;
            }
            context->final_norm_weight = hybrid_weight_f16(context->device, upload_blit, final_norm_weight, context->hidden);
            context->lm_head_weight = hybrid_weight(context->device, upload_blit, lm_head_weight, lm_head_q8, context->vocab * context->hidden);
            context->embeddings = hybrid_weight_f16(context->device, upload_blit, embeddings, context->vocab * context->hidden);


            const NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
            const NSUInteger query_bytes = (NSUInteger)query_width * sizeof(uint16_t);
            const NSUInteger kv_bytes = (NSUInteger)kv_width * sizeof(uint16_t);
            const NSUInteger intermediate_bytes = (NSUInteger)context->intermediate * sizeof(uint16_t);
            context->x_a = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->x_b = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->normalized = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->query = hybrid_zero(context->device, query_bytes, MTLResourceStorageModePrivate);
            context->key = hybrid_zero(context->device, kv_bytes, MTLResourceStorageModePrivate);
            context->attention_context = hybrid_zero(context->device, query_bytes, MTLResourceStorageModePrivate);
            context->attention_scores = hybrid_zero(
                context->device,
                (NSUInteger)context->query_heads * (NSUInteger)context->bucket * sizeof(float),
                MTLResourceStorageModePrivate
            );
            context->mlp = hybrid_zero(context->device, intermediate_bytes, MTLResourceStorageModePrivate);
            context->final_norm = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->logits = hybrid_zero(context->device, (NSUInteger)context->vocab * sizeof(float), MTLResourceStorageModeShared);
            context->conv_proj = hybrid_zero(context->device, (NSUInteger)(3 * context->hidden) * sizeof(uint16_t), MTLResourceStorageModePrivate);
            context->conv_product = hybrid_zero(context->device, (NSUInteger)context->hidden * sizeof(float), MTLResourceStorageModePrivate);
            context->conv_gate = hybrid_zero(context->device, (NSUInteger)context->hidden * sizeof(float), MTLResourceStorageModePrivate);
            context->conv_out = hybrid_zero(context->device, (NSUInteger)context->hidden * sizeof(float), MTLResourceStorageModePrivate);
            context->conv_out_f16 = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->argmax_partials = (context->vocab + 4095) / 4096;
            if (context->argmax_partials == 0) context->argmax_partials = 1;
            context->argmax_partial_keys = hybrid_zero(context->device, (NSUInteger)context->argmax_partials * sizeof(int32_t), MTLResourceStorageModePrivate);
            context->argmax_partial_ids = hybrid_zero(context->device, (NSUInteger)context->argmax_partials * sizeof(uint32_t), MTLResourceStorageModePrivate);
            context->chain_token_ids = hybrid_zero(context->device, sizeof(uint32_t), MTLResourceStorageModeShared);
            context->chain_input = hybrid_zero(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            // Zero buffer big enough to blit-reset the largest cache (a KV cache).
            context->zero_buffer = hybrid_zero(context->device, kv_cache_elements * sizeof(uint16_t), MTLResourceStorageModeShared);

            [upload_blit endEncoding];
            [upload_command commit];
            [upload_command waitUntilCompleted];
            if (upload_command.status == MTLCommandBufferStatusError) {
                set_error(upload_command.error.localizedDescription ?: @"LFM2 hybrid weight upload failed");
                return -6;
            }
            if (context->final_norm_weight == nil ||
                (quantized ? context->lm_head_weight.q8 == nil : context->lm_head_weight.fp16 == nil) ||
                context->embeddings == nil ||
                context->x_a == nil || context->x_b == nil || context->normalized == nil || context->query == nil ||
                context->key == nil || context->attention_context == nil || context->attention_scores == nil ||
                context->mlp == nil || context->final_norm == nil || context->logits == nil ||
                context->conv_proj == nil || context->conv_product == nil || context->conv_gate == nil ||
                context->conv_out == nil || context->conv_out_f16 == nil || context->argmax_partial_keys == nil ||
                context->argmax_partial_ids == nil || context->chain_token_ids == nil || context->chain_input == nil ||
                context->zero_buffer == nil) {
                set_error(@"failed to allocate LFM2 hybrid step activation buffers");
                return -7;
            }
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 hybrid step preparation raised");
            return -100;
        }
    }
}

// Bind a matmul weight's f16 buffer at its fp16 index and its Q8_0 buffer at
// its q8 index. The reused kernels read only the slot selected by their
// `quantized` config flag, so the unused slot may be nil and is never
// dereferenced. A given engine populates exactly one slot per weight (see
// hybrid_weight), so one of the two bindings is always nil.
static void hybrid_set_weight(id<MTLComputeCommandEncoder> encoder, Lfm2StepWeight *weight, NSUInteger fp16_index, NSUInteger q8_index) {
    [encoder setBuffer:weight->fp16 offset:0 atIndex:fp16_index];
    [encoder setBuffer:weight->q8 offset:0 atIndex:q8_index];
}

static void hybrid_encode_rmsnorm(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    id<MTLBuffer> weight,
    uint32_t width,
    float epsilon
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->rmsnorm];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    struct { uint32_t width; float epsilon; } config = { width, epsilon };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:hybrid_grid(32) threadsPerThreadgroup:hybrid_group(32)];
    [encoder endEncoding];
}

static void hybrid_encode_qkv(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    Lfm2HybridLayerBuffers *layer,
    id<MTLBuffer> input,
    uint32_t position
) {
    uint32_t query_width = (uint32_t)(context->query_heads * context->head_dim);
    uint32_t kv_width = (uint32_t)(context->kv_heads * context->head_dim);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qkv_matvec];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->query offset:0 atIndex:1];
    [encoder setBuffer:context->key offset:0 atIndex:2];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:3];
    hybrid_set_weight(encoder, &layer->q_weight, 4, 5);
    hybrid_set_weight(encoder, &layer->k_weight, 6, 7);
    hybrid_set_weight(encoder, &layer->v_weight, 8, 9);
    struct { uint32_t input_width; uint32_t query_width; uint32_t kv_width; uint32_t head_dim; uint32_t capacity; uint32_t position; uint32_t quantized; } config = {
        (uint32_t)context->hidden, query_width, kv_width, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position, context->quantized
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:10];
    // F16 dispatches one thread per output row (the row dot is serial). Q8 packs
    // the query rows plus both KV projections into the grid and gives each row a
    // full 32-lane simdgroup, mirroring the Qwen3 step engine's Q8 QKV dispatch.
    NSUInteger qkv_rows = context->quantized
        ? (NSUInteger)query_width + (NSUInteger)kv_width * 2
        : (NSUInteger)MAX(query_width, kv_width);
    NSUInteger qkv_threads = context->quantized ? qkv_rows * 32 : qkv_rows;
    [encoder dispatchThreads:hybrid_grid(qkv_threads) threadsPerThreadgroup:hybrid_group(context->quantized ? 256 : qkv_rows)];
    [encoder endEncoding];
}

static void hybrid_encode_qk_norm_rope(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    Lfm2HybridLayerBuffers *layer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    NSUInteger rope_offset,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qk_norm_rope];
    [encoder setBuffer:context->query offset:0 atIndex:0];
    [encoder setBuffer:context->key offset:0 atIndex:1];
    [encoder setBuffer:layer->q_norm offset:0 atIndex:2];
    [encoder setBuffer:layer->k_norm offset:0 atIndex:3];
    [encoder setBuffer:rope_cos offset:rope_offset atIndex:4];
    [encoder setBuffer:rope_sin offset:rope_offset atIndex:5];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:6];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; float epsilon; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim, context->epsilon,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:7];
    [encoder dispatchThreads:hybrid_grid(MAX(context->query_heads, context->kv_heads) * 32) threadsPerThreadgroup:hybrid_group(256)];
    [encoder endEncoding];
}

static void hybrid_encode_attention(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    Lfm2HybridLayerBuffers *layer,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->attention];
    [encoder setBuffer:context->query offset:0 atIndex:0];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:1];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:2];
    [encoder setBuffer:context->attention_context offset:0 atIndex:3];
    [encoder setBuffer:context->attention_scores offset:0 atIndex:4];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    [encoder dispatchThreads:hybrid_grid(context->query_heads * 32) threadsPerThreadgroup:hybrid_group(256)];
    [encoder endEncoding];
}

static void hybrid_encode_matvec_residual(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> residual,
    id<MTLBuffer> output,
    Lfm2StepWeight *weight,
    uint32_t input_width,
    uint32_t output_width,
    BOOL add_residual
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->matvec_residual];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    hybrid_set_weight(encoder, weight, 3, 4);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        input_width, output_width, context->quantized, (uint32_t)add_residual
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    // F16 uses one lane per independent output row; the row dot itself stays
    // serial. Q8 packs four independent rows into each 32-lane simdgroup, eight
    // sub-lanes per row, each row still reduced by its own simd_sum.
    NSUInteger matvec_threads =
        context->quantized ? (NSUInteger)((output_width + 3) / 4) * 32 : output_width;
    [encoder dispatchThreads:hybrid_grid(matvec_threads) threadsPerThreadgroup:hybrid_group(context->quantized ? 256 : output_width)];
    [encoder endEncoding];
}

static void hybrid_encode_residual_rmsnorm(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> projection,
    id<MTLBuffer> residual,
    id<MTLBuffer> normalized,
    id<MTLBuffer> weight
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->residual_rmsnorm];
    [encoder setBuffer:projection offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:normalized offset:0 atIndex:2];
    [encoder setBuffer:weight offset:0 atIndex:3];
    struct { uint32_t width; float epsilon; } config = { (uint32_t)context->hidden, context->epsilon };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    [encoder dispatchThreads:hybrid_grid(32) threadsPerThreadgroup:hybrid_group(32)];
    [encoder endEncoding];
}

static void hybrid_encode_gate_up(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    Lfm2HybridLayerBuffers *layer,
    id<MTLBuffer> input
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->gate_up_swiglu];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->mlp offset:0 atIndex:1];
    hybrid_set_weight(encoder, &layer->gate_weight, 2, 3);
    hybrid_set_weight(encoder, &layer->up_weight, 4, 5);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        (uint32_t)context->hidden, (uint32_t)context->intermediate, context->quantized, 0
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:6];
    NSUInteger gate_threads =
        context->quantized ? (NSUInteger)((context->intermediate + 3) / 4) * 32 : (NSUInteger)context->intermediate;
    [encoder dispatchThreads:hybrid_grid(gate_threads) threadsPerThreadgroup:hybrid_group(context->quantized ? 256 : context->intermediate)];
    [encoder endEncoding];
}

static void hybrid_encode_lm_head(Lfm2HybridStepContext *context, id<MTLCommandBuffer> command_buffer) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->lm_head];
    [encoder setBuffer:context->final_norm offset:0 atIndex:0];
    [encoder setBuffer:context->logits offset:0 atIndex:1];
    hybrid_set_weight(encoder, &context->lm_head_weight, 2, 3);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        (uint32_t)context->hidden, (uint32_t)context->vocab, context->quantized, 0
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    // The large vocabulary projection is row-parallel in f16 (one lane per full
    // serial dot); Q8 packs four vocabulary rows per simdgroup, eight sub-lanes
    // each, mirroring the Qwen3 step engine's Q8 LM-head dispatch.
    NSUInteger lm_head_threads =
        context->quantized ? (NSUInteger)((context->vocab + 3) / 4) * 32 : (NSUInteger)context->vocab;
    [encoder dispatchThreads:hybrid_grid(lm_head_threads) threadsPerThreadgroup:hybrid_group(context->quantized ? 256 : context->vocab)];
    [encoder endEncoding];
}

static void hybrid_encode_argmax_offset(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> token_out,
    NSUInteger out_offset
) {
    struct { uint32_t vocab; uint32_t partials; } config = {
        (uint32_t)context->vocab, (uint32_t)context->argmax_partials
    };
    id<MTLComputeCommandEncoder> partial = [command_buffer computeCommandEncoder];
    [partial setComputePipelineState:context->argmax_partial];
    [partial setBuffer:context->logits offset:0 atIndex:0];
    [partial setBuffer:context->argmax_partial_keys offset:0 atIndex:1];
    [partial setBuffer:context->argmax_partial_ids offset:0 atIndex:2];
    [partial setBytes:&config length:sizeof(config) atIndex:3];
    [partial dispatchThreadgroups:MTLSizeMake((NSUInteger)context->argmax_partials, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [partial endEncoding];

    id<MTLComputeCommandEncoder> final = [command_buffer computeCommandEncoder];
    [final setComputePipelineState:context->argmax_final];
    [final setBuffer:context->argmax_partial_keys offset:0 atIndex:0];
    [final setBuffer:context->argmax_partial_ids offset:0 atIndex:1];
    [final setBuffer:token_out offset:out_offset atIndex:2];
    [final setBytes:&config length:sizeof(config) atIndex:3];
    [final dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [final endEncoding];
}

static void hybrid_encode_embedding_gather_offset(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> token_in,
    NSUInteger in_offset,
    id<MTLBuffer> input_out
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->embedding_gather];
    [encoder setBuffer:context->embeddings offset:0 atIndex:0];
    [encoder setBuffer:token_in offset:in_offset atIndex:1];
    [encoder setBuffer:input_out offset:0 atIndex:2];
    struct { uint32_t hidden; } config = { (uint32_t)context->hidden };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:hybrid_grid(context->hidden) threadsPerThreadgroup:hybrid_group(context->hidden)];
    [encoder endEncoding];
}

// Conv-path encoders. The split widens the f16 in_proj output into the f32
// product/gate the conv step consumes; the conv step (proven bit-exact, stage A)
// advances the layer's rolling cache and emits the f32 gated output; the
// f32->f16 kernel narrows it back for the reused out_proj matvec.
static void hybrid_encode_conv_split(Lfm2HybridStepContext *context, id<MTLCommandBuffer> command_buffer) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->conv_split];
    [encoder setBuffer:context->conv_proj offset:0 atIndex:0];
    [encoder setBuffer:context->conv_product offset:0 atIndex:1];
    [encoder setBuffer:context->conv_gate offset:0 atIndex:2];
    struct { uint32_t hidden; } config = { (uint32_t)context->hidden };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:hybrid_grid(context->hidden) threadsPerThreadgroup:hybrid_group(context->hidden)];
    [encoder endEncoding];
}

static void hybrid_encode_conv_step(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    Lfm2HybridLayerBuffers *layer
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->conv_step];
    [encoder setBuffer:layer->conv_cache offset:0 atIndex:0];
    [encoder setBuffer:context->conv_product offset:0 atIndex:1];
    [encoder setBuffer:context->conv_gate offset:0 atIndex:2];
    [encoder setBuffer:layer->conv_weight offset:0 atIndex:3];
    [encoder setBuffer:context->conv_out offset:0 atIndex:4];
    Lfm2ConvStepParams params = {
        (uint32_t)context->hidden,
        (uint32_t)context->kernel_size,
    };
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    NSUInteger threads_per_group = context->conv_step.maxTotalThreadsPerThreadgroup;
    if (threads_per_group > context->hidden) {
        threads_per_group = (NSUInteger)context->hidden;
    }
    if (threads_per_group == 0) {
        threads_per_group = 1;
    }
    [encoder dispatchThreads:MTLSizeMake(context->hidden, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    [encoder endEncoding];
}

static void hybrid_encode_conv_f32_to_f16(Lfm2HybridStepContext *context, id<MTLCommandBuffer> command_buffer) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->conv_f32_to_f16];
    [encoder setBuffer:context->conv_out offset:0 atIndex:0];
    [encoder setBuffer:context->conv_out_f16 offset:0 atIndex:1];
    struct { uint32_t hidden; } config = { (uint32_t)context->hidden };
    [encoder setBytes:&config length:sizeof(config) atIndex:2];
    [encoder dispatchThreads:hybrid_grid(context->hidden) threadsPerThreadgroup:hybrid_group(context->hidden)];
    [encoder endEncoding];
}

// One full hybrid forward pass for a single token at `position`, reading `input`
// (an f16 hidden-wide embedding row) and advancing every layer's cache. Mirrors
// lfm2.rs::decode_embedding layer by layer: operator_norm -> mixer -> +residual
// -> ffn_norm -> SwiGLU FFN -> +residual, then final_norm and the tied LM head.
// `current`/`next` ping-pong the running residual exactly as the Qwen3 engine
// does; the conv and attention mixers slot into the same residual discipline.
static void hybrid_encode_forward(
    Lfm2HybridStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    NSUInteger rope_offset,
    uint32_t position,
    float epsilon
) {
    id<MTLBuffer> current = input;
    id<MTLBuffer> next = context->x_a;
    for (uint64_t index = 0; index < context->layer_count; ++index) {
        Lfm2HybridLayerBuffers *layer = &context->layers[index];
        hybrid_encode_rmsnorm(context, command_buffer, current, context->normalized, layer->operator_norm,
                              (uint32_t)context->hidden, epsilon);
        if (layer->is_attention) {
            hybrid_encode_qkv(context, command_buffer, layer, context->normalized, position);
            hybrid_encode_qk_norm_rope(context, command_buffer, layer, rope_cos, rope_sin, rope_offset, position);
            hybrid_encode_attention(context, command_buffer, layer, position);
            hybrid_encode_matvec_residual(context, command_buffer, context->attention_context, current, next,
                                          &layer->o_weight, (uint32_t)(context->query_heads * context->head_dim),
                                          (uint32_t)context->hidden, NO);
        } else {
            // in_proj: hidden -> 3*hidden, no residual. The residual slot is bound
            // (not read) because add_residual is false.
            hybrid_encode_matvec_residual(context, command_buffer, context->normalized, current, context->conv_proj,
                                          &layer->in_proj_weight, (uint32_t)context->hidden,
                                          (uint32_t)(3 * context->hidden), NO);
            hybrid_encode_conv_split(context, command_buffer);
            hybrid_encode_conv_step(context, command_buffer, layer);
            hybrid_encode_conv_f32_to_f16(context, command_buffer);
            // out_proj: hidden -> hidden, no residual (the residual add happens in
            // the residual_rmsnorm below, matching decode_conv + add_residual).
            hybrid_encode_matvec_residual(context, command_buffer, context->conv_out_f16, current, next,
                                          &layer->out_proj_weight, (uint32_t)context->hidden,
                                          (uint32_t)context->hidden, NO);
        }
        hybrid_encode_residual_rmsnorm(context, command_buffer, next, current, context->normalized, layer->ffn_norm);
        hybrid_encode_gate_up(context, command_buffer, layer, context->normalized);
        hybrid_encode_matvec_residual(context, command_buffer, context->mlp, next, current,
                                      &layer->down_weight, (uint32_t)context->intermediate,
                                      (uint32_t)context->hidden, YES);
    }
    hybrid_encode_rmsnorm(context, command_buffer, current, context->final_norm, context->final_norm_weight,
                          (uint32_t)context->hidden, epsilon);
    hybrid_encode_lm_head(context, command_buffer);
}

// Chained greedy decode: encode `steps` full forward passes plus an on-GPU
// argmax into one command buffer, gathering each step's input from the previous
// step's device-side argmax. Position advances per step; rope tables cover the
// whole span (one head_dim block per step). The first step's token is seeded by
// the host in token_in_first. After completion the host reads back `steps`
// token ids at once. Mirrors the Qwen3 chain; correct because the embedding
// gather and argmax are byte-exact with the per-token host path.
int32_t synapse_lfm2_hybrid_step_chain(
    void *raw,
    uint64_t position,
    uint32_t steps,
    uint32_t token_in_first,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    uint32_t *token_ids_out,
    float epsilon
) {
    @autoreleasepool {
        @try {
            lfm2_step_error[0] = '\0';
            Lfm2HybridStepContext *context = raw;
            if (context == NULL || rope_cos == NULL || rope_sin == NULL || token_ids_out == NULL ||
                context->layers == NULL || steps == 0 || position + steps > context->bucket) {
                set_error(@"invalid LFM2 hybrid step chain arguments");
                return -1;
            }
            NSUInteger rope_span = (NSUInteger)steps * (NSUInteger)context->head_dim;
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> ids_buffer = [context->device newBufferWithLength:(NSUInteger)steps * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (cosine_buffer == nil || sine_buffer == nil || ids_buffer == nil || command_buffer == nil) {
                [cosine_buffer release];
                [sine_buffer release];
                [ids_buffer release];
                set_error(@"failed to allocate LFM2 hybrid step chain buffers");
                return -2;
            }
            *(uint32_t *)context->chain_token_ids.contents = token_in_first;
            NSUInteger head_dim_bytes = (NSUInteger)context->head_dim * sizeof(uint16_t);
            for (uint32_t step = 0; step < steps; ++step) {
                id<MTLBuffer> step_ids = (step == 0) ? context->chain_token_ids : ids_buffer;
                NSUInteger id_offset = (step == 0) ? 0 : (NSUInteger)(step - 1) * sizeof(uint32_t);
                hybrid_encode_embedding_gather_offset(context, command_buffer, step_ids, id_offset, context->chain_input);
                hybrid_encode_forward(context, command_buffer, context->chain_input, cosine_buffer, sine_buffer,
                                      (NSUInteger)step * head_dim_bytes, (uint32_t)position + step, epsilon);
                hybrid_encode_argmax_offset(context, command_buffer, ids_buffer, (NSUInteger)step * sizeof(uint32_t));
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"LFM2 hybrid step chain command buffer failed");
            } else {
                memcpy(token_ids_out, ids_buffer.contents, (NSUInteger)steps * sizeof(uint32_t));
            }
            [cosine_buffer release];
            [sine_buffer release];
            [ids_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 hybrid step chain raised");
            return -100;
        }
    }
}

// Explicit-token forward: run `steps` forward passes feeding host-supplied token
// ids (one per step), emitting the greedy argmax after each. Used to prefill a
// prompt token-by-token (each prompt token advances the conv and KV caches just
// as lfm2.rs::decode_token does), with the last argmax being the first generated
// token. Mirrors the Qwen3 verify path.
int32_t synapse_lfm2_hybrid_step_verify(
    void *raw,
    uint64_t position,
    const uint32_t *token_ids,
    uint32_t steps,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    uint32_t *argmaxes_out,
    float epsilon
) {
    @autoreleasepool {
        @try {
            lfm2_step_error[0] = '\0';
            Lfm2HybridStepContext *context = raw;
            if (context == NULL || token_ids == NULL || rope_cos == NULL || rope_sin == NULL ||
                argmaxes_out == NULL || context->layers == NULL || steps == 0 ||
                position + steps > context->bucket) {
                set_error(@"invalid LFM2 hybrid step verify arguments");
                return -1;
            }
            NSUInteger rope_span = (NSUInteger)steps * (NSUInteger)context->head_dim;
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> proposal_buffer = [context->device newBufferWithBytes:token_ids
                length:(NSUInteger)steps * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> argmax_buffer = [context->device newBufferWithLength:(NSUInteger)steps * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (cosine_buffer == nil || sine_buffer == nil || proposal_buffer == nil ||
                argmax_buffer == nil || command_buffer == nil) {
                [cosine_buffer release];
                [sine_buffer release];
                [proposal_buffer release];
                [argmax_buffer release];
                set_error(@"failed to allocate LFM2 hybrid step verify buffers");
                return -2;
            }
            NSUInteger head_dim_bytes = (NSUInteger)context->head_dim * sizeof(uint16_t);
            for (uint32_t step = 0; step < steps; ++step) {
                hybrid_encode_embedding_gather_offset(context, command_buffer, proposal_buffer,
                                                      (NSUInteger)step * sizeof(uint32_t), context->chain_input);
                hybrid_encode_forward(context, command_buffer, context->chain_input, cosine_buffer, sine_buffer,
                                      (NSUInteger)step * head_dim_bytes, (uint32_t)position + step, epsilon);
                hybrid_encode_argmax_offset(context, command_buffer, argmax_buffer, (NSUInteger)step * sizeof(uint32_t));
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"LFM2 hybrid step verify command buffer failed");
            } else {
                memcpy(argmaxes_out, argmax_buffer.contents, (NSUInteger)steps * sizeof(uint32_t));
            }
            [cosine_buffer release];
            [sine_buffer release];
            [proposal_buffer release];
            [argmax_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 hybrid step verify raised");
            return -100;
        }
    }
}

// Single forward pass for one host-fed f16 embedding row, reading back the full
// f32 logits. Used by the layer-parity probes (Metal logits vs the lfm2.rs CPU
// reference per position) and as a building block for host-driven decoding.
int32_t synapse_lfm2_hybrid_step(
    void *raw,
    uint64_t position,
    const uint16_t *input,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    float *logits,
    float epsilon
) {
    @autoreleasepool {
        @try {
            lfm2_step_error[0] = '\0';
            Lfm2HybridStepContext *context = raw;
            if (context == NULL || input == NULL || rope_cos == NULL || rope_sin == NULL || logits == NULL ||
                position >= context->bucket || context->layers == NULL) {
                set_error(@"invalid LFM2 hybrid step arguments");
                return -1;
            }
            NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input length:hidden_bytes
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:(NSUInteger)context->head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:(NSUInteger)context->head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (input_buffer == nil || cosine_buffer == nil || sine_buffer == nil || command_buffer == nil) {
                [input_buffer release];
                [cosine_buffer release];
                [sine_buffer release];
                set_error(@"failed to allocate LFM2 hybrid step input buffers");
                return -2;
            }
            hybrid_encode_forward(context, command_buffer, input_buffer, cosine_buffer, sine_buffer, 0,
                                  (uint32_t)position, epsilon);
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"LFM2 hybrid step command buffer failed");
            } else {
                memcpy(logits, context->logits.contents, (NSUInteger)context->vocab * sizeof(float));
            }
            [input_buffer release];
            [cosine_buffer release];
            [sine_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 hybrid step raised");
            return -100;
        }
    }
}

// Reset every layer's cache to the empty-decode state: conv rolling windows
// zeroed (they persist across steps, so a new sequence must start clean) and KV
// caches zeroed. KV slots at positions <= the current one are always overwritten
// before attention reads them when a sequence starts at position 0, so the KV
// zeroing is defensive rather than required; the conv zeroing is required.
int32_t synapse_lfm2_hybrid_step_reset(void *raw) {
    @autoreleasepool {
        Lfm2HybridStepContext *context = raw;
        if (context == NULL || context->layers == NULL) {
            set_error(@"invalid LFM2 hybrid step reset arguments");
            return -1;
        }
        const NSUInteger conv_cache_bytes = (NSUInteger)(context->kernel_size * context->hidden * sizeof(float));
        const NSUInteger kv_cache_bytes = (NSUInteger)(context->kv_heads * context->bucket * context->head_dim * sizeof(uint16_t));
        // Conv caches are shared: zero them directly from the host.
        for (uint64_t layer = 0; layer < context->layer_count; ++layer) {
            if (!context->layers[layer].is_attention) {
                memset(context->layers[layer].conv_cache.contents, 0, conv_cache_bytes);
            }
        }
        // KV caches are private: blit zeros from the shared zero buffer.
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || blit == nil) {
            set_error(@"failed to create LFM2 hybrid step reset command");
            return -2;
        }
        for (uint64_t layer = 0; layer < context->layer_count; ++layer) {
            if (context->layers[layer].is_attention) {
                [blit copyFromBuffer:context->zero_buffer sourceOffset:0
                            toBuffer:context->layers[layer].key_cache destinationOffset:0 size:kv_cache_bytes];
                [blit copyFromBuffer:context->zero_buffer sourceOffset:0
                            toBuffer:context->layers[layer].value_cache destinationOffset:0 size:kv_cache_bytes];
            }
        }
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            set_error(command_buffer.error.localizedDescription ?: @"LFM2 hybrid step reset failed");
            return -3;
        }
        return 0;
    }
}

static void hybrid_release_weight(Lfm2StepWeight *weight) {
    [weight->fp16 release];
    [weight->q8 release];
    weight->fp16 = nil;
    weight->q8 = nil;
}

static void hybrid_release_layer(Lfm2HybridLayerBuffers *layer) {
    [layer->operator_norm release];
    [layer->ffn_norm release];
    hybrid_release_weight(&layer->gate_weight);
    hybrid_release_weight(&layer->up_weight);
    hybrid_release_weight(&layer->down_weight);
    hybrid_release_weight(&layer->in_proj_weight);
    [layer->conv_weight release];
    hybrid_release_weight(&layer->out_proj_weight);
    [layer->conv_cache release];
    hybrid_release_weight(&layer->q_weight);
    hybrid_release_weight(&layer->k_weight);
    hybrid_release_weight(&layer->v_weight);
    hybrid_release_weight(&layer->o_weight);
    [layer->q_norm release];
    [layer->k_norm release];
    [layer->key_cache release];
    [layer->value_cache release];
    memset(layer, 0, sizeof(*layer));
}

void synapse_lfm2_hybrid_step_context_free(void *raw) {
    if (raw == NULL) return;
    Lfm2HybridStepContext *context = raw;
    if (context->layers != NULL) {
        for (uint64_t index = 0; index < context->layer_count; ++index) hybrid_release_layer(&context->layers[index]);
        free(context->layers);
    }
    [context->x_a release];
    [context->x_b release];
    [context->normalized release];
    [context->query release];
    [context->key release];
    [context->attention_context release];
    [context->attention_scores release];
    [context->mlp release];
    [context->final_norm release];
    [context->logits release];
    [context->conv_proj release];
    [context->conv_product release];
    [context->conv_gate release];
    [context->conv_out release];
    [context->conv_out_f16 release];
    [context->final_norm_weight release];
    hybrid_release_weight(&context->lm_head_weight);
    [context->embeddings release];
    [context->argmax_partial_keys release];
    [context->argmax_partial_ids release];
    [context->chain_token_ids release];
    [context->chain_input release];
    [context->zero_buffer release];
    [context->conv_f32_to_f16 release];
    [context->conv_split release];
    [context->conv_step release];
    [context->embedding_gather release];
    [context->argmax_final release];
    [context->argmax_partial release];
    [context->lm_head release];
    [context->gate_up_swiglu release];
    [context->residual_rmsnorm release];
    [context->matvec_residual release];
    [context->attention release];
    [context->qk_norm_rope release];
    [context->qkv_matvec release];
    [context->rmsnorm release];
    [context->library release];
    [context->queue release];
    [context->device release];
    free(context);
}
