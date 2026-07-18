#include "cuda_family_common.cuh"

#include <cfloat>
#include <cstdio>

using namespace synapse_cuda_family;

extern "C" {
typedef struct Lfm2LayerParams {
    int32_t mixer_type;
    const float *operator_norm;
    const float *ffn_norm;
    SynapseCudaWeight conv_in_weight;
    const float *conv_weight;
    SynapseCudaWeight conv_out_weight;
    SynapseCudaWeight q_weight;
    const float *q_norm;
    SynapseCudaWeight k_weight;
    const float *k_norm;
    SynapseCudaWeight v_weight;
    SynapseCudaWeight attention_out_weight;
    SynapseCudaWeight w1_weight;
    SynapseCudaWeight w2_weight;
    SynapseCudaWeight w3_weight;
} Lfm2LayerParams;
}

namespace {

struct DeviceLayer {
    int mixer_type = 0;
    DeviceAllocation<float> operator_norm, ffn_norm;
    DeviceMatrix conv_in_weight, conv_out_weight;
    DeviceAllocation<float> conv_weight;
    DeviceMatrix q_weight, k_weight, v_weight, attention_out_weight;
    DeviceAllocation<float> q_norm, k_norm;
    DeviceMatrix w1_weight, w2_weight, w3_weight;
    DeviceAllocation<float> conv_state, key_cache, value_cache;
};

__global__ void rms_norm(
    const float *input,
    const float *weight,
    float *output,
    int rows,
    int width,
    float epsilon
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int base = row * width;
    float square = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float value = input[base + column];
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    square = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) square = warp_sum(square);
    if (threadIdx.x == 0) reductions[0] = rsqrtf(square / width + epsilon);
    __syncthreads();
    float inverse = reductions[0];
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = input[base + column] * inverse * weight[column];
    }
}

__global__ void add_residual(const float *update, const float *residual, float *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = update[index] + residual[index];
}

__global__ void swiglu(const float *gate, const float *up, float *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
}

__global__ void full_conv_mix(
    const float *projected,
    const float *weights,
    float *output,
    int batch,
    int seq,
    int hidden,
    int kernel
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = batch * seq * hidden;
    if (index >= count) return;
    int channel = index % hidden;
    int row = index / hidden;
    int position = row % seq;
    int batch_index = row / seq;
    float convolved = 0.0f;
    for (int tap = 0; tap < kernel; ++tap) {
        int source_position = position + tap + 1 - kernel;
        if (source_position < 0) continue;
        int source = (batch_index * seq + source_position) * 3 * hidden;
        convolved += projected[source + channel] * projected[source + 2 * hidden + channel]
            * weights[channel * kernel + tap];
    }
    int current = row * 3 * hidden;
    output[index] = projected[current + hidden + channel] * convolved;
}

__global__ void save_conv_state(
    const float *projected,
    float *state,
    int seq,
    int hidden,
    int kernel
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = kernel * hidden;
    if (index >= count) return;
    int tap = index / hidden;
    int channel = index % hidden;
    int source_position = seq - kernel + tap;
    if (source_position < 0) {
        state[index] = 0.0f;
    } else {
        int source = source_position * 3 * hidden;
        state[index] = projected[source + channel] * projected[source + 2 * hidden + channel];
    }
}

__global__ void decode_conv_mix(
    const float *projected,
    const float *weights,
    float *state,
    float *output,
    int hidden,
    int kernel
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;
    for (int tap = 0; tap + 1 < kernel; ++tap) {
        state[tap * hidden + channel] = state[(tap + 1) * hidden + channel];
    }
    state[(kernel - 1) * hidden + channel] =
        projected[channel] * projected[2 * hidden + channel];
    float convolved = 0.0f;
    for (int tap = 0; tap < kernel; ++tap) {
        convolved += state[tap * hidden + channel] * weights[channel * kernel + tap];
    }
    output[channel] = projected[hidden + channel] * convolved;
}

__global__ void head_norm_rope(
    const float *input,
    const float *weight,
    float *output,
    int rows,
    int heads,
    int head_dim,
    int position_offset,
    float epsilon,
    float theta
) {
    int row_head = blockIdx.x;
    int row = row_head / heads;
    int head = row_head % heads;
    if (row >= rows) return;
    int base = (row * heads + head) * head_dim;
    float square = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = input[base + dimension];
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    square = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) square = warp_sum(square);
    if (threadIdx.x == 0) reductions[0] = rsqrtf(square / head_dim + epsilon);
    __syncthreads();
    float inverse = reductions[0];
    int half_dim = head_dim / 2;
    int position = position_offset + row;
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

__global__ void causal_attention(
    const float *q,
    const float *k,
    const float *v,
    const uint8_t *mask,
    float *output,
    int seq,
    int query_heads,
    int kv_heads,
    int head_dim
) {
    int row_head = blockIdx.x;
    int query = row_head / query_heads;
    int query_head = row_head % query_heads;
    if (query >= seq) return;
    int groups = query_heads / kv_heads;
    int kv_head = query_head / groups;
    extern __shared__ float scores[];
    float maximum = -FLT_MAX;
    for (int key_position = threadIdx.x; key_position < seq; key_position += blockDim.x) {
        float score = -FLT_MAX;
        if (key_position <= query && mask[key_position]) {
            score = 0.0f;
            int q_base = (query * query_heads + query_head) * head_dim;
            int k_base = (key_position * kv_heads + kv_head) * head_dim;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                score += q[q_base + dimension] * k[k_base + dimension];
            }
            score /= sqrtf(static_cast<float>(head_dim));
        }
        scores[key_position] = score;
        maximum = fmaxf(maximum, score);
    }
    maximum = warp_max(maximum);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = maximum;
    __syncthreads();
    maximum = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : -FLT_MAX;
    if (warp == 0) maximum = warp_max(maximum);
    if (threadIdx.x == 0) reductions[0] = maximum;
    __syncthreads();
    maximum = reductions[0];
    float sum = 0.0f;
    for (int key_position = threadIdx.x; key_position < seq; key_position += blockDim.x) {
        float probability = expf(scores[key_position] - maximum);
        scores[key_position] = probability;
        sum += probability;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) reductions[0] = sum;
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1.0e-20f);
    for (int key_position = threadIdx.x; key_position < seq; key_position += blockDim.x) {
        scores[key_position] *= inverse;
    }
    __syncthreads();
    int out_base = (query * query_heads + query_head) * head_dim;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = 0.0f;
        for (int key_position = 0; key_position <= query; ++key_position) {
            int v_base = (key_position * kv_heads + kv_head) * head_dim;
            value += scores[key_position] * v[v_base + dimension];
        }
        output[out_base + dimension] = value;
    }
}

__global__ void decode_attention(
    const float *q,
    const float *key_cache,
    const float *value_cache,
    float *output,
    int seq,
    int query_heads,
    int kv_heads,
    int head_dim
) {
    int query_head = blockIdx.x;
    if (query_head >= query_heads) return;
    int groups = query_heads / kv_heads;
    int kv_head = query_head / groups;
    extern __shared__ float scores[];
    float maximum = -FLT_MAX;
    for (int position = threadIdx.x; position < seq; position += blockDim.x) {
        float score = 0.0f;
        int q_base = query_head * head_dim;
        int k_base = (position * kv_heads + kv_head) * head_dim;
        for (int dimension = 0; dimension < head_dim; ++dimension) {
            score += q[q_base + dimension] * key_cache[k_base + dimension];
        }
        score /= sqrtf(static_cast<float>(head_dim));
        scores[position] = score;
        maximum = fmaxf(maximum, score);
    }
    maximum = warp_max(maximum);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = maximum;
    __syncthreads();
    maximum = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : -FLT_MAX;
    if (warp == 0) maximum = warp_max(maximum);
    if (threadIdx.x == 0) reductions[0] = maximum;
    __syncthreads();
    maximum = reductions[0];
    float sum = 0.0f;
    for (int position = threadIdx.x; position < seq; position += blockDim.x) {
        float probability = expf(scores[position] - maximum);
        scores[position] = probability;
        sum += probability;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < (blockDim.x + 31) / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) reductions[0] = sum;
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1.0e-20f);
    for (int position = threadIdx.x; position < seq; position += blockDim.x) scores[position] *= inverse;
    __syncthreads();
    int out_base = query_head * head_dim;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = 0.0f;
        for (int position = 0; position < seq; ++position) {
            int v_base = (position * kv_heads + kv_head) * head_dim;
            value += scores[position] * value_cache[v_base + dimension];
        }
        output[out_base + dimension] = value;
    }
}

struct Workspace {
    int seq = 0;
    DeviceAllocation<float> x0, x1, normed, projected;
    DeviceAllocation<float> mixer_raw, q, k, v, attention;
    DeviceAllocation<float> gate, up, activated, logits;
    DeviceAllocation<uint8_t> mask;

    void ensure(int requested_seq, int hidden, int kv_width, int intermediate, int vocab) {
        if (seq >= requested_seq) return;
        seq = requested_seq;
        size_t rows = requested_seq;
        x0.allocate(rows * hidden);
        x1.allocate(rows * hidden);
        normed.allocate(rows * hidden);
        projected.allocate(rows * hidden);
        mixer_raw.allocate(rows * 3 * hidden);
        q.allocate(rows * hidden);
        k.allocate(rows * kv_width);
        v.allocate(rows * kv_width);
        attention.allocate(rows * hidden);
        gate.allocate(rows * intermediate);
        up.allocate(rows * intermediate);
        activated.allocate(rows * intermediate);
        logits.allocate(vocab);
        mask.allocate(rows);
    }
};

struct Lfm2Context {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    bool graphs_enabled = false;
    bool weights_loaded = false;
    int hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    int intermediate = 0, layer_count = 0, kernel = 0, vocab = 0, capacity = 0;
    int decode_position = 0;
    float epsilon = 0.0f, theta = 0.0f;
    std::vector<DeviceLayer> layers;
    DeviceAllocation<float> final_norm;
    DeviceMatrix lm_head;
    DeviceAllocation<unsigned char> workspace;
    Workspace buffers;
    std::unordered_map<std::string, std::unique_ptr<MatmulPlan>> matmuls;

    explicit Lfm2Context(bool graphs) : graphs_enabled(graphs) {
        FAMILY_CUDA_CHECK(cudaFree(nullptr));
        FAMILY_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        FAMILY_CUBLAS_CHECK(cublasLtCreate(&lt));
        workspace.allocate(64 * 1024 * 1024);
    }

    ~Lfm2Context() {
        matmuls.clear();
        if (lt) cublasLtDestroy(lt);
        if (stream) cudaStreamDestroy(stream);
    }

    const MatmulPlan &matmul(int m, int n, int k) {
        std::string key = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
        auto found = matmuls.find(key);
        if (found == matmuls.end()) {
            auto plan = std::make_unique<MatmulPlan>(select_matmul(lt, key.c_str(), CUDA_R_32F, m, n, k, true));
            found = matmuls.emplace(key, std::move(plan)).first;
        }
        return *found->second;
    }

    void gemm(int m, int n, int k, const float *a, const float *b, float *c) {
        launch_matmul(lt, matmul(m, n, k), a, b, c, workspace.pointer, stream);
    }

    void decode_matvec(int rows, int columns, const float *input, const DeviceMatrix &weight, float *output) {
        if (weight.quantized) {
            launch_decode_matvec(weight, input, output, rows, columns, stream);
        } else {
            gemm(1, rows, columns, input, weight.fp32.pointer, output);
        }
    }

    void load_model(
        int h,
        int qh,
        int kvh,
        int hd,
        int inter,
        int layers_count,
        int kernel_size,
        int vocabulary,
        float eps,
        float rope_theta,
        const Lfm2LayerParams *params,
        const float *host_final_norm,
        const SynapseCudaWeight *host_lm_head
    ) {
        if (weights_loaded) {
            if (hidden != h || query_heads != qh || kv_heads != kvh || head_dim != hd ||
                intermediate != inter || layer_count != layers_count || kernel != kernel_size || vocab != vocabulary) {
                throw std::runtime_error("LFM2 CUDA model dimensions changed");
            }
            return;
        }
        hidden = h;
        query_heads = qh;
        kv_heads = kvh;
        head_dim = hd;
        intermediate = inter;
        layer_count = layers_count;
        kernel = kernel_size;
        vocab = vocabulary;
        epsilon = eps;
        theta = rope_theta;
        int kv_width = kv_heads * head_dim;
        layers.resize(layer_count);
        for (int index = 0; index < layer_count; ++index) {
            DeviceLayer &target = layers[index];
            const Lfm2LayerParams &source = params[index];
            target.mixer_type = source.mixer_type;
            copy_float(target.operator_norm, source.operator_norm, hidden);
            copy_float(target.ffn_norm, source.ffn_norm, hidden);
            if (target.mixer_type == 0) {
                copy_matrix(target.conv_in_weight, source.conv_in_weight, static_cast<size_t>(3) * hidden * hidden);
                copy_float(target.conv_weight, source.conv_weight, static_cast<size_t>(hidden) * kernel);
                copy_matrix(target.conv_out_weight, source.conv_out_weight, static_cast<size_t>(hidden) * hidden);
            } else {
                copy_matrix(target.q_weight, source.q_weight, static_cast<size_t>(hidden) * hidden);
                copy_float(target.q_norm, source.q_norm, head_dim);
                copy_matrix(target.k_weight, source.k_weight, static_cast<size_t>(kv_width) * hidden);
                copy_float(target.k_norm, source.k_norm, head_dim);
                copy_matrix(target.v_weight, source.v_weight, static_cast<size_t>(kv_width) * hidden);
                copy_matrix(target.attention_out_weight, source.attention_out_weight, static_cast<size_t>(hidden) * hidden);
            }
            copy_matrix(target.w1_weight, source.w1_weight, static_cast<size_t>(intermediate) * hidden);
            copy_matrix(target.w2_weight, source.w2_weight, static_cast<size_t>(hidden) * intermediate);
            copy_matrix(target.w3_weight, source.w3_weight, static_cast<size_t>(intermediate) * hidden);
        }
        copy_float(final_norm, host_final_norm, hidden);
        if (!host_lm_head) throw std::runtime_error("LFM2 CUDA received a null LM head");
        copy_matrix(lm_head, *host_lm_head, static_cast<size_t>(vocab) * hidden);
        // Weight uploads use the default stream; synchronize the device before the
        // nonblocking inference stream can consume the final upload.
        FAMILY_CUDA_CHECK(cudaDeviceSynchronize());
        weights_loaded = true;
        int attention_layers = 0;
        for (const DeviceLayer &layer : layers) attention_layers += layer.mixer_type == 1;
        std::fprintf(
            stderr,
            "CUDA LFM2 persistent weights: layers=%d conv=%d attention=%d dtype=%s accum=fp32 graphs=%s\n",
            layer_count,
            layer_count - attention_layers,
            attention_layers,
            lm_head.quantized ? "q8_0" : "fp32",
            graphs_enabled ? "requested-decode-uncaptured" : "disabled"
        );
    }

    void ensure_capacity(int requested_capacity) {
        if (capacity == requested_capacity) return;
        int kv_width = kv_heads * head_dim;
        for (DeviceLayer &layer : layers) {
            if (layer.mixer_type == 0) {
                layer.conv_state.allocate(static_cast<size_t>(kernel) * hidden);
            } else {
                layer.key_cache.allocate(static_cast<size_t>(requested_capacity) * kv_width);
                layer.value_cache.allocate(static_cast<size_t>(requested_capacity) * kv_width);
            }
        }
        capacity = requested_capacity;
        decode_position = 0;
    }

    void full(
        const float *host_input,
        const uint8_t *host_mask,
        int seq,
        int cache_capacity,
        bool initialize_cache,
        float *host_hidden,
        float *host_logits
    ) {
        int kv_width = kv_heads * head_dim;
        int threads = 256;
        int rows = seq;
        int hidden_values = rows * hidden;
        int hidden_blocks = (hidden_values + threads - 1) / threads;
        int intermediate_values = rows * intermediate;
        int intermediate_blocks = (intermediate_values + threads - 1) / threads;
        buffers.ensure(seq, hidden, kv_width, intermediate, vocab);
        if (initialize_cache) ensure_capacity(cache_capacity);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(buffers.x0.pointer, host_input, static_cast<size_t>(hidden_values) * sizeof(float), cudaMemcpyHostToDevice, stream));
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(buffers.mask.pointer, host_mask, seq, cudaMemcpyHostToDevice, stream));

        for (DeviceLayer &layer : layers) {
            rms_norm<<<rows, threads, 0, stream>>>(buffers.x0.pointer, layer.operator_norm.pointer, buffers.normed.pointer, rows, hidden, epsilon);
            if (layer.mixer_type == 0) {
                gemm(rows, 3 * hidden, hidden, buffers.normed.pointer, layer.conv_in_weight.fp32.pointer, buffers.mixer_raw.pointer);
                full_conv_mix<<<(hidden_values + threads - 1) / threads, threads, 0, stream>>>(
                    buffers.mixer_raw.pointer,
                    layer.conv_weight.pointer,
                    buffers.attention.pointer,
                    1,
                    seq,
                    hidden,
                    kernel
                );
                if (initialize_cache) {
                    int state_count = kernel * hidden;
                    save_conv_state<<<(state_count + threads - 1) / threads, threads, 0, stream>>>(
                        buffers.mixer_raw.pointer,
                        layer.conv_state.pointer,
                        seq,
                        hidden,
                        kernel
                    );
                }
                gemm(rows, hidden, hidden, buffers.attention.pointer, layer.conv_out_weight.fp32.pointer, buffers.projected.pointer);
            } else {
                gemm(rows, hidden, hidden, buffers.normed.pointer, layer.q_weight.fp32.pointer, buffers.mixer_raw.pointer);
                gemm(rows, kv_width, hidden, buffers.normed.pointer, layer.k_weight.fp32.pointer, buffers.k.pointer);
                gemm(rows, kv_width, hidden, buffers.normed.pointer, layer.v_weight.fp32.pointer, buffers.v.pointer);
                head_norm_rope<<<rows * query_heads, threads, 0, stream>>>(
                    buffers.mixer_raw.pointer, layer.q_norm.pointer, buffers.q.pointer,
                    rows, query_heads, head_dim, 0, epsilon, theta
                );
                head_norm_rope<<<rows * kv_heads, threads, 0, stream>>>(
                    buffers.k.pointer, layer.k_norm.pointer, buffers.projected.pointer,
                    rows, kv_heads, head_dim, 0, epsilon, theta
                );
                if (initialize_cache) {
                    FAMILY_CUDA_CHECK(cudaMemcpyAsync(layer.key_cache.pointer, buffers.projected.pointer, static_cast<size_t>(rows) * kv_width * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    FAMILY_CUDA_CHECK(cudaMemcpyAsync(layer.value_cache.pointer, buffers.v.pointer, static_cast<size_t>(rows) * kv_width * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                }
                causal_attention<<<rows * query_heads, threads, static_cast<size_t>(seq) * sizeof(float), stream>>>(
                    buffers.q.pointer,
                    buffers.projected.pointer,
                    buffers.v.pointer,
                    buffers.mask.pointer,
                    buffers.attention.pointer,
                    seq,
                    query_heads,
                    kv_heads,
                    head_dim
                );
                gemm(rows, hidden, hidden, buffers.attention.pointer, layer.attention_out_weight.fp32.pointer, buffers.projected.pointer);
            }
            add_residual<<<hidden_blocks, threads, 0, stream>>>(buffers.projected.pointer, buffers.x0.pointer, buffers.x1.pointer, hidden_values);
            rms_norm<<<rows, threads, 0, stream>>>(buffers.x1.pointer, layer.ffn_norm.pointer, buffers.normed.pointer, rows, hidden, epsilon);
            gemm(rows, intermediate, hidden, buffers.normed.pointer, layer.w1_weight.fp32.pointer, buffers.gate.pointer);
            gemm(rows, intermediate, hidden, buffers.normed.pointer, layer.w3_weight.fp32.pointer, buffers.up.pointer);
            swiglu<<<intermediate_blocks, threads, 0, stream>>>(buffers.gate.pointer, buffers.up.pointer, buffers.activated.pointer, intermediate_values);
            gemm(rows, hidden, intermediate, buffers.activated.pointer, layer.w2_weight.fp32.pointer, buffers.projected.pointer);
            add_residual<<<hidden_blocks, threads, 0, stream>>>(buffers.projected.pointer, buffers.x1.pointer, buffers.x0.pointer, hidden_values);
        }
        rms_norm<<<rows, threads, 0, stream>>>(buffers.x0.pointer, final_norm.pointer, buffers.x1.pointer, rows, hidden, epsilon);
        if (host_logits) {
            decode_matvec(vocab, hidden, buffers.x1.pointer + static_cast<size_t>(seq - 1) * hidden, lm_head, buffers.logits.pointer);
            FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_logits, buffers.logits.pointer, static_cast<size_t>(vocab) * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (host_hidden) {
            FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_hidden, buffers.x1.pointer, static_cast<size_t>(hidden_values) * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        FAMILY_CUDA_CHECK(cudaStreamSynchronize(stream));
        FAMILY_CUDA_CHECK(cudaGetLastError());
        if (initialize_cache) decode_position = seq;
    }

    void decode(const float *host_embedding, int position, int cache_capacity, float *host_hidden, float *host_logits) {
        if (position == 0) {
            ensure_capacity(cache_capacity);
            decode_position = 0;
        }
        if (capacity != cache_capacity || position != decode_position || position >= capacity) {
            throw std::runtime_error("LFM2 CUDA decode cache position/capacity mismatch");
        }
        int kv_width = kv_heads * head_dim;
        int threads = 256;
        int hidden_blocks = (hidden + threads - 1) / threads;
        int intermediate_blocks = (intermediate + threads - 1) / threads;
        buffers.ensure(1, hidden, kv_width, intermediate, vocab);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(buffers.x0.pointer, host_embedding, static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyHostToDevice, stream));
        for (DeviceLayer &layer : layers) {
            rms_norm<<<1, threads, 0, stream>>>(buffers.x0.pointer, layer.operator_norm.pointer, buffers.normed.pointer, 1, hidden, epsilon);
            if (layer.mixer_type == 0) {
                decode_matvec(3 * hidden, hidden, buffers.normed.pointer, layer.conv_in_weight, buffers.mixer_raw.pointer);
                decode_conv_mix<<<hidden_blocks, threads, 0, stream>>>(
                    buffers.mixer_raw.pointer,
                    layer.conv_weight.pointer,
                    layer.conv_state.pointer,
                    buffers.attention.pointer,
                    hidden,
                    kernel
                );
                decode_matvec(hidden, hidden, buffers.attention.pointer, layer.conv_out_weight, buffers.projected.pointer);
            } else {
                decode_matvec(hidden, hidden, buffers.normed.pointer, layer.q_weight, buffers.mixer_raw.pointer);
                decode_matvec(kv_width, hidden, buffers.normed.pointer, layer.k_weight, buffers.k.pointer);
                decode_matvec(kv_width, hidden, buffers.normed.pointer, layer.v_weight, buffers.v.pointer);
                head_norm_rope<<<query_heads, threads, 0, stream>>>(
                    buffers.mixer_raw.pointer, layer.q_norm.pointer, buffers.q.pointer,
                    1, query_heads, head_dim, position, epsilon, theta
                );
                head_norm_rope<<<kv_heads, threads, 0, stream>>>(
                    buffers.k.pointer, layer.k_norm.pointer, buffers.projected.pointer,
                    1, kv_heads, head_dim, position, epsilon, theta
                );
                FAMILY_CUDA_CHECK(cudaMemcpyAsync(layer.key_cache.pointer + static_cast<size_t>(position) * kv_width, buffers.projected.pointer, static_cast<size_t>(kv_width) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                FAMILY_CUDA_CHECK(cudaMemcpyAsync(layer.value_cache.pointer + static_cast<size_t>(position) * kv_width, buffers.v.pointer, static_cast<size_t>(kv_width) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                decode_attention<<<query_heads, threads, static_cast<size_t>(position + 1) * sizeof(float), stream>>>(
                    buffers.q.pointer,
                    layer.key_cache.pointer,
                    layer.value_cache.pointer,
                    buffers.attention.pointer,
                    position + 1,
                    query_heads,
                    kv_heads,
                    head_dim
                );
                decode_matvec(hidden, hidden, buffers.attention.pointer, layer.attention_out_weight, buffers.projected.pointer);
            }
            add_residual<<<hidden_blocks, threads, 0, stream>>>(buffers.projected.pointer, buffers.x0.pointer, buffers.x1.pointer, hidden);
            rms_norm<<<1, threads, 0, stream>>>(buffers.x1.pointer, layer.ffn_norm.pointer, buffers.normed.pointer, 1, hidden, epsilon);
            decode_matvec(intermediate, hidden, buffers.normed.pointer, layer.w1_weight, buffers.gate.pointer);
            decode_matvec(intermediate, hidden, buffers.normed.pointer, layer.w3_weight, buffers.up.pointer);
            swiglu<<<intermediate_blocks, threads, 0, stream>>>(buffers.gate.pointer, buffers.up.pointer, buffers.activated.pointer, intermediate);
            decode_matvec(hidden, intermediate, buffers.activated.pointer, layer.w2_weight, buffers.projected.pointer);
            add_residual<<<hidden_blocks, threads, 0, stream>>>(buffers.projected.pointer, buffers.x1.pointer, buffers.x0.pointer, hidden);
        }
        rms_norm<<<1, threads, 0, stream>>>(buffers.x0.pointer, final_norm.pointer, buffers.x1.pointer, 1, hidden, epsilon);
        decode_matvec(vocab, hidden, buffers.x1.pointer, lm_head, buffers.logits.pointer);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_hidden, buffers.x1.pointer, static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyDeviceToHost, stream));
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_logits, buffers.logits.pointer, static_cast<size_t>(vocab) * sizeof(float), cudaMemcpyDeviceToHost, stream));
        FAMILY_CUDA_CHECK(cudaStreamSynchronize(stream));
        FAMILY_CUDA_CHECK(cudaGetLastError());
        ++decode_position;
    }
};

void validate_common(
    void *raw_context,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t kernel,
    uint64_t vocab,
    const Lfm2LayerParams *layers,
    const float *final_norm,
    const SynapseCudaWeight *lm_head
) {
    if (!raw_context || !layers || !final_norm || !lm_head) throw std::runtime_error("LFM2 CUDA received a null model pointer");
    if (!hidden || !query_heads || !kv_heads || query_heads % kv_heads || !head_dim ||
        query_heads * head_dim != hidden || !intermediate || !layer_count || !kernel || !vocab) {
        throw std::runtime_error("LFM2 CUDA received invalid model dimensions");
    }
}

}  // namespace

extern "C" {

void *synapse_cuda_lfm2_context_new(int32_t graphs_enabled, int32_t precision) {
    try {
        if (precision != 0) throw std::runtime_error("LFM2 CUDA currently requires fp32");
        return new Lfm2Context(graphs_enabled != 0);
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return nullptr;
    }
}

void synapse_cuda_lfm2_context_free(void *raw_context) {
    delete static_cast<Lfm2Context *>(raw_context);
}

int32_t synapse_cuda_lfm2_full_forward(
    void *raw_context,
    uint64_t seq,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t kernel,
    uint64_t vocab,
    float epsilon,
    float rope_theta,
    const float *input,
    const uint8_t *mask,
    const Lfm2LayerParams *layers,
    const float *final_norm,
    const SynapseCudaWeight *lm_head,
    float *output
) {
    try {
        validate_common(raw_context, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, layers, final_norm, lm_head);
        if (!seq || !input || !mask || !output) throw std::runtime_error("LFM2 CUDA full forward received invalid inputs");
        Lfm2Context *context = static_cast<Lfm2Context *>(raw_context);
        context->load_model(hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, epsilon, rope_theta, layers, final_norm, lm_head);
        context->full(input, mask, seq, 0, false, output, nullptr);
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

int32_t synapse_cuda_lfm2_prefill(
    void *raw_context,
    uint64_t seq,
    uint64_t capacity,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t kernel,
    uint64_t vocab,
    float epsilon,
    float rope_theta,
    const float *input,
    const Lfm2LayerParams *layers,
    const float *final_norm,
    const SynapseCudaWeight *lm_head,
    float *logits
) {
    try {
        validate_common(raw_context, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, layers, final_norm, lm_head);
        if (!seq || seq > capacity || !input || !logits) throw std::runtime_error("LFM2 CUDA prefill received invalid inputs");
        Lfm2Context *context = static_cast<Lfm2Context *>(raw_context);
        context->load_model(hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, epsilon, rope_theta, layers, final_norm, lm_head);
        std::vector<uint8_t> mask(seq, 1);
        context->full(input, mask.data(), seq, capacity, true, nullptr, logits);
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

int32_t synapse_cuda_lfm2_decode(
    void *raw_context,
    uint64_t position,
    uint64_t capacity,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t kernel,
    uint64_t vocab,
    float epsilon,
    float rope_theta,
    const float *embedding,
    const Lfm2LayerParams *layers,
    const float *final_norm,
    const SynapseCudaWeight *lm_head,
    float *output_hidden,
    float *logits
) {
    try {
        validate_common(raw_context, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, layers, final_norm, lm_head);
        if (!capacity || !embedding || !output_hidden || !logits) throw std::runtime_error("LFM2 CUDA decode received invalid inputs");
        Lfm2Context *context = static_cast<Lfm2Context *>(raw_context);
        context->load_model(hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, kernel, vocab, epsilon, rope_theta, layers, final_norm, lm_head);
        context->decode(embedding, position, capacity, output_hidden, logits);
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

}  // extern "C"
