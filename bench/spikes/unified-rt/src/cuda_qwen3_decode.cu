#include "cuda_family_common.cuh"

#include <cfloat>
#include <cstdio>

using namespace synapse_cuda_family;

extern "C" {
typedef struct Qwen3DecodeLayerParams {
    const float *input_norm;
    const float *post_attention_norm;
    SynapseCudaWeight q_weight;
    const float *q_norm;
    SynapseCudaWeight k_weight;
    const float *k_norm;
    SynapseCudaWeight v_weight;
    SynapseCudaWeight o_weight;
    SynapseCudaWeight gate_weight;
    SynapseCudaWeight up_weight;
    SynapseCudaWeight down_weight;
} Qwen3DecodeLayerParams;
}

namespace {

struct DecodeLayer {
    DeviceAllocation<float> input_norm, post_attention_norm, q_norm, k_norm;
    DeviceMatrix q_weight, k_weight, v_weight, o_weight;
    DeviceMatrix gate_weight, up_weight, down_weight;
    DeviceAllocation<float> key_cache, value_cache;
};

__global__ void decode_fp32_matvec(
    const float *weights,
    const float *input,
    float *output,
    int rows,
    int columns
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        partial += weights[static_cast<size_t>(row) * columns + column] * input[column];
    }
    partial = warp_sum(partial);
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        int warps = blockDim.x >> 5;
        float sum = lane < warps ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) output[row] = sum;
    }
}

__global__ void decode_rms_norm(
    const float *input,
    const float *weight,
    float *output,
    int width,
    float epsilon
) {
    float square = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float value = input[column];
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < warps ? reductions[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) reductions[0] = rsqrtf(sum / width + epsilon);
    }
    __syncthreads();
    float inverse = reductions[0];
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        output[column] = input[column] * inverse * weight[column];
    }
}

__global__ void decode_head_norm_rope(
    const float *input,
    const float *weight,
    float *output,
    int heads,
    int head_dim,
    int position,
    float epsilon,
    float theta
) {
    int head = blockIdx.x;
    if (head >= heads) return;
    int base = head * head_dim;
    float square = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = input[base + dimension];
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < warps ? reductions[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) reductions[0] = rsqrtf(sum / head_dim + epsilon);
    }
    __syncthreads();
    float inverse = reductions[0];
    int half_dim = head_dim / 2;
    for (int dimension = threadIdx.x; dimension < half_dim; dimension += blockDim.x) {
        float frequency = powf(theta, -2.0f * dimension / head_dim);
        float angle = position * frequency;
        float cosine = cosf(angle);
        float sine = sinf(angle);
        float first = input[base + dimension] * inverse * weight[dimension];
        float second = input[base + half_dim + dimension] * inverse * weight[half_dim + dimension];
        output[base + dimension] = first * cosine - second * sine;
        output[base + half_dim + dimension] = second * cosine + first * sine;
    }
}

__global__ void decode_attention(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    float *output,
    int sequence,
    int query_heads,
    int kv_heads,
    int head_dim
) {
    int query_head = blockIdx.x;
    if (query_head >= query_heads) return;
    int groups = query_heads / kv_heads;
    int kv_head = query_head / groups;
    extern __shared__ float scores[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int query_base = query_head * head_dim;
    float inverse_scale = rsqrtf(static_cast<float>(head_dim));
    float maximum = -FLT_MAX;
    for (int key = warp; key < sequence; key += warps) {
        float partial = 0.0f;
        int key_base = (key * kv_heads + kv_head) * head_dim;
        for (int dimension = lane; dimension < head_dim; dimension += 32) {
            partial += query[query_base + dimension] * key_cache[key_base + dimension];
        }
        float score = warp_sum(partial) * inverse_scale;
        if (lane == 0) {
            scores[key] = score;
            maximum = fmaxf(maximum, score);
        }
    }
    maximum = warp_max(maximum);
    __shared__ float reductions[32];
    if (lane == 0) reductions[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? reductions[lane] : -FLT_MAX;
        value = warp_max(value);
        if (lane == 0) reductions[0] = value;
    }
    __syncthreads();
    maximum = reductions[0];
    float sum = 0.0f;
    for (int key = threadIdx.x; key < sequence; key += blockDim.x) {
        float probability = expf(scores[key] - maximum);
        scores[key] = probability;
        sum += probability;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? reductions[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) reductions[0] = value;
    }
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1.0e-20f);
    for (int key = threadIdx.x; key < sequence; key += blockDim.x) scores[key] *= inverse;
    __syncthreads();
    int output_base = query_head * head_dim;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = 0.0f;
        for (int key = 0; key < sequence; ++key) {
            int value_base = (key * kv_heads + kv_head) * head_dim;
            value += scores[key] * value_cache[value_base + dimension];
        }
        output[output_base + dimension] = value;
    }
}

__global__ void decode_add_residual(
    const float *update,
    const float *residual,
    float *output,
    int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = update[index] + residual[index];
}

__global__ void decode_add_residual_rms_norm(
    const float *update,
    const float *residual,
    const float *weight,
    float *output,
    float *norm_output,
    int width,
    float epsilon
) {
    float square = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float value = update[column] + residual[column];
        output[column] = value;
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < warps ? reductions[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) reductions[0] = rsqrtf(sum / width + epsilon);
    }
    __syncthreads();
    float inverse = reductions[0];
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        norm_output[column] = output[column] * inverse * weight[column];
    }
}

__global__ void decode_swiglu(const float *gate, const float *up, float *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
}

__global__ void decode_fused_gate_up_swiglu_q8_0(
    const uint8_t *gate_weights,
    const uint8_t *up_weights,
    const float *input,
    float *output,
    int rows,
    int columns
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int blocks_per_row = columns / Q8_0_BLOCK_ELEMENTS;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (int column_block = warp; column_block < blocks_per_row; column_block += warps) {
        size_t offset = (static_cast<size_t>(row) * blocks_per_row + column_block) * Q8_0_BLOCK_BYTES;
        const uint8_t *gate_block = gate_weights + offset;
        const uint8_t *up_block = up_weights + offset;
        half gate_scale = *reinterpret_cast<const half *>(gate_block);
        half up_scale = *reinterpret_cast<const half *>(up_block);
        int8_t gate_q = reinterpret_cast<const int8_t *>(gate_block + sizeof(half))[lane];
        int8_t up_q = reinterpret_cast<const int8_t *>(up_block + sizeof(half))[lane];
        float in = input[column_block * Q8_0_BLOCK_ELEMENTS + lane];
        gate_partial += __half2float(gate_scale) * static_cast<float>(gate_q) * in;
        up_partial += __half2float(up_scale) * static_cast<float>(up_q) * in;
    }
    gate_partial = warp_sum(gate_partial);
    up_partial = warp_sum(up_partial);
    __shared__ float gate_sums[32];
    __shared__ float up_sums[32];
    if (lane == 0) { gate_sums[warp] = gate_partial; up_sums[warp] = up_partial; }
    __syncthreads();
    if (warp == 0) {
        float g = lane < warps ? gate_sums[lane] : 0.0f;
        float u = lane < warps ? up_sums[lane] : 0.0f;
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane == 0) output[row] = (g / (1.0f + expf(-g))) * u;
    }
}

struct Qwen3DecodeContext {
    cudaStream_t stream = nullptr;
    bool weights_loaded = false;
    int capacity;
    int position = 0;
    int hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    int intermediate = 0, layer_count = 0, vocab = 0;
    float epsilon = 0.0f, theta = 0.0f;
    std::vector<DecodeLayer> layers;
    DeviceAllocation<float> final_norm;
    DeviceMatrix lm_head;
    DeviceAllocation<float> x0, x1, normed, q_raw, k_raw, v_raw, q, k;
    DeviceAllocation<float> attention, projected, gate, up, activated, logits;

    explicit Qwen3DecodeContext(int cache_capacity) : capacity(cache_capacity) {
        if (capacity <= 0) throw std::runtime_error("Qwen3 CUDA decode capacity must be positive");
        FAMILY_CUDA_CHECK(cudaFree(nullptr));
        FAMILY_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    ~Qwen3DecodeContext() {
        if (stream) cudaStreamDestroy(stream);
    }

    void load_model(
        int h,
        int qh,
        int kvh,
        int hd,
        int inter,
        int count,
        int vocabulary,
        float eps,
        float rope_theta,
        const Qwen3DecodeLayerParams *params,
        const float *host_final_norm,
        const SynapseCudaWeight *host_lm_head
    ) {
        if (weights_loaded) return;
        if (!params || !host_final_norm || !host_lm_head) {
            throw std::runtime_error("Qwen3 CUDA decode received a null model pointer");
        }
        if (h <= 0 || qh <= 0 || kvh <= 0 || qh % kvh || hd <= 0
            || inter <= 0 || count <= 0 || vocabulary <= 0) {
            throw std::runtime_error("Qwen3 CUDA decode received invalid model dimensions");
        }
        hidden = h;
        query_heads = qh;
        kv_heads = kvh;
        head_dim = hd;
        intermediate = inter;
        layer_count = count;
        vocab = vocabulary;
        epsilon = eps;
        theta = rope_theta;
        int query_width = query_heads * head_dim;
        int kv_width = kv_heads * head_dim;
        layers.resize(layer_count);
        for (int index = 0; index < layer_count; ++index) {
            DecodeLayer &target = layers[index];
            const Qwen3DecodeLayerParams &source = params[index];
            copy_float(target.input_norm, source.input_norm, hidden);
            copy_float(target.post_attention_norm, source.post_attention_norm, hidden);
            copy_float(target.q_norm, source.q_norm, head_dim);
            copy_float(target.k_norm, source.k_norm, head_dim);
            copy_matrix(target.q_weight, source.q_weight, static_cast<size_t>(query_width) * hidden);
            copy_matrix(target.k_weight, source.k_weight, static_cast<size_t>(kv_width) * hidden);
            copy_matrix(target.v_weight, source.v_weight, static_cast<size_t>(kv_width) * hidden);
            copy_matrix(target.o_weight, source.o_weight, static_cast<size_t>(hidden) * query_width);
            copy_matrix(target.gate_weight, source.gate_weight, static_cast<size_t>(intermediate) * hidden);
            copy_matrix(target.up_weight, source.up_weight, static_cast<size_t>(intermediate) * hidden);
            copy_matrix(target.down_weight, source.down_weight, static_cast<size_t>(hidden) * intermediate);
            target.key_cache.allocate(static_cast<size_t>(capacity) * kv_width);
            target.value_cache.allocate(static_cast<size_t>(capacity) * kv_width);
            FAMILY_CUDA_CHECK(cudaMemset(target.key_cache.pointer, 0, target.key_cache.count * sizeof(float)));
            FAMILY_CUDA_CHECK(cudaMemset(target.value_cache.pointer, 0, target.value_cache.count * sizeof(float)));
        }
        copy_float(final_norm, host_final_norm, hidden);
        copy_matrix(lm_head, *host_lm_head, static_cast<size_t>(vocab) * hidden);
        x0.allocate(hidden);
        x1.allocate(hidden);
        normed.allocate(hidden);
        q_raw.allocate(query_width);
        k_raw.allocate(kv_width);
        v_raw.allocate(kv_width);
        q.allocate(query_width);
        k.allocate(kv_width);
        attention.allocate(query_width);
        projected.allocate(hidden);
        gate.allocate(intermediate);
        up.allocate(intermediate);
        activated.allocate(intermediate);
        logits.allocate(vocab);
        FAMILY_CUDA_CHECK(cudaDeviceSynchronize());
        weights_loaded = true;
        std::fprintf(
            stderr,
            "CUDA Qwen3 decode persistent weights: layers=%d dtype=%s activations=fp32 accum=fp32 cache=fp32\n",
            layer_count,
            lm_head.quantized ? "q8_0" : "fp32"
        );
    }

    void matvec(const DeviceMatrix &weight, const float *input, float *output, int rows, int columns) {
        if (weight.quantized) {
            launch_decode_matvec(weight, input, output, rows, columns, stream);
        } else {
            decode_fp32_matvec<<<rows, 256, 0, stream>>>(
                weight.fp32.pointer,
                input,
                output,
                rows,
                columns
            );
        }
    }

    void run_token(const float *host_embedding, int token_position, bool produce_logits) {
        if (token_position < 0 || token_position >= capacity) {
            throw std::runtime_error("Qwen3 CUDA decode position exceeds cache capacity");
        }
        int query_width = query_heads * head_dim;
        int kv_width = kv_heads * head_dim;
        int threads = 256;
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(
            x0.pointer,
            host_embedding,
            static_cast<size_t>(hidden) * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        ));
        decode_rms_norm<<<1, threads, 0, stream>>>(
            x0.pointer, layers.front().input_norm.pointer, normed.pointer, hidden, epsilon
        );
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            DecodeLayer &layer = layers[layer_index];
            matvec(layer.q_weight, normed.pointer, q_raw.pointer, query_width, hidden);
            matvec(layer.k_weight, normed.pointer, k_raw.pointer, kv_width, hidden);
            matvec(layer.v_weight, normed.pointer, layer.value_cache.pointer + static_cast<size_t>(token_position) * kv_width, kv_width, hidden);
            decode_head_norm_rope<<<query_heads, threads, 0, stream>>>(
                q_raw.pointer, layer.q_norm.pointer, q.pointer,
                query_heads, head_dim, token_position, epsilon, theta
            );
            decode_head_norm_rope<<<kv_heads, threads, 0, stream>>>(
                k_raw.pointer, layer.k_norm.pointer, layer.key_cache.pointer + static_cast<size_t>(token_position) * kv_width,
                kv_heads, head_dim, token_position, epsilon, theta
            );
            decode_attention<<<query_heads, threads, static_cast<size_t>(token_position + 1) * sizeof(float), stream>>>(
                q.pointer,
                layer.key_cache.pointer,
                layer.value_cache.pointer,
                attention.pointer,
                token_position + 1,
                query_heads,
                kv_heads,
                head_dim
            );
            matvec(layer.o_weight, attention.pointer, projected.pointer, hidden, query_width);
            decode_add_residual_rms_norm<<<1, threads, 0, stream>>>(
                projected.pointer,
                x0.pointer,
                layer.post_attention_norm.pointer,
                x1.pointer,
                normed.pointer,
                hidden,
                epsilon
            );
            if (layer.gate_weight.quantized && layer.up_weight.quantized) {
                decode_fused_gate_up_swiglu_q8_0<<<intermediate, threads, 0, stream>>>(
                    layer.gate_weight.q8_0.pointer,
                    layer.up_weight.q8_0.pointer,
                    normed.pointer,
                    activated.pointer,
                    intermediate,
                    hidden
                );
            } else {
                matvec(layer.gate_weight, normed.pointer, gate.pointer, intermediate, hidden);
                matvec(layer.up_weight, normed.pointer, up.pointer, intermediate, hidden);
                decode_swiglu<<<(intermediate + threads - 1) / threads, threads, 0, stream>>>(
                    gate.pointer, up.pointer, activated.pointer, intermediate
                );
            }
            matvec(layer.down_weight, activated.pointer, projected.pointer, hidden, intermediate);
            if (layer_index + 1 < layers.size()) {
                decode_add_residual_rms_norm<<<1, threads, 0, stream>>>(
                    projected.pointer,
                    x1.pointer,
                    layers[layer_index + 1].input_norm.pointer,
                    x0.pointer,
                    normed.pointer,
                    hidden,
                    epsilon
                );
            } else if (produce_logits) {
                decode_add_residual_rms_norm<<<1, threads, 0, stream>>>(
                    projected.pointer,
                    x1.pointer,
                    final_norm.pointer,
                    x0.pointer,
                    x1.pointer,
                    hidden,
                    epsilon
                );
            } else {
                decode_add_residual<<<(hidden + threads - 1) / threads, threads, 0, stream>>>(
                    projected.pointer, x1.pointer, x0.pointer, hidden
                );
            }
        }
        if (produce_logits) {
            matvec(lm_head, x1.pointer, logits.pointer, vocab, hidden);
        }
        FAMILY_CUDA_CHECK(cudaGetLastError());
    }

    void prefill(const float *host_embeddings, int sequence, float *host_logits) {
        if (!weights_loaded || !host_embeddings || !host_logits || sequence <= 0 || sequence > capacity) {
            throw std::runtime_error("Qwen3 CUDA prefill received invalid inputs");
        }
        position = 0;
        for (int index = 0; index < sequence; ++index) {
            run_token(
                host_embeddings + static_cast<size_t>(index) * hidden,
                index,
                index + 1 == sequence
            );
        }
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(
            host_logits,
            logits.pointer,
            static_cast<size_t>(vocab) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ));
        FAMILY_CUDA_CHECK(cudaStreamSynchronize(stream));
        position = sequence;
    }

    void step(const float *host_embedding, int requested_position, float *host_logits) {
        if (!weights_loaded || !host_embedding || !host_logits || requested_position != position) {
            throw std::runtime_error("Qwen3 CUDA decode cache position mismatch");
        }
        run_token(host_embedding, requested_position, true);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(
            host_logits,
            logits.pointer,
            static_cast<size_t>(vocab) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ));
        FAMILY_CUDA_CHECK(cudaStreamSynchronize(stream));
        ++position;
    }
};

}  // namespace

extern "C" {

void *synapse_cuda_qwen3_decode_context_new(uint64_t capacity) {
    try {
        return new Qwen3DecodeContext(static_cast<int>(capacity));
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return nullptr;
    }
}

void synapse_cuda_qwen3_decode_context_free(void *raw_context) {
    delete static_cast<Qwen3DecodeContext *>(raw_context);
}

int32_t synapse_cuda_qwen3_decode_prepare(
    void *raw_context,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon,
    float rope_theta,
    const Qwen3DecodeLayerParams *layers,
    const float *final_norm,
    const SynapseCudaWeight *lm_head
) {
    try {
        if (!raw_context) throw std::runtime_error("Qwen3 CUDA decode context is null");
        static_cast<Qwen3DecodeContext *>(raw_context)->load_model(
            hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, vocab,
            epsilon, rope_theta, layers, final_norm, lm_head
        );
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

int32_t synapse_cuda_qwen3_decode_prefill(
    void *raw_context,
    uint64_t sequence,
    const float *embeddings,
    float *logits
) {
    try {
        if (!raw_context) throw std::runtime_error("Qwen3 CUDA decode context is null");
        static_cast<Qwen3DecodeContext *>(raw_context)->prefill(
            embeddings, static_cast<int>(sequence), logits
        );
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

int32_t synapse_cuda_qwen3_decode_step(
    void *raw_context,
    uint64_t position,
    const float *embedding,
    float *logits
) {
    try {
        if (!raw_context) throw std::runtime_error("Qwen3 CUDA decode context is null");
        static_cast<Qwen3DecodeContext *>(raw_context)->step(
            embedding, static_cast<int>(position), logits
        );
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

int32_t synapse_cuda_qwen3_decode_cache_copy(
    void *raw_context,
    uint64_t layer,
    float *output,
    uint64_t elements
) {
    try {
        if (!raw_context || !output) throw std::runtime_error("Qwen3 CUDA cache copy received a null pointer");
        Qwen3DecodeContext *context = static_cast<Qwen3DecodeContext *>(raw_context);
        if (layer >= context->layers.size()) throw std::runtime_error("Qwen3 CUDA cache layer is out of range");
        size_t per_cache = static_cast<size_t>(context->capacity) * context->kv_heads * context->head_dim;
        if (elements != 2 * per_cache) throw std::runtime_error("Qwen3 CUDA cache copy size mismatch");
        DecodeLayer &source = context->layers[layer];
        FAMILY_CUDA_CHECK(cudaMemcpy(
            output,
            source.key_cache.pointer,
            per_cache * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        FAMILY_CUDA_CHECK(cudaMemcpy(
            output + per_cache,
            source.value_cache.pointer,
            per_cache * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

}  // extern "C"
