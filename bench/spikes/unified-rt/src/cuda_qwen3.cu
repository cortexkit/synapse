#include "cuda_family_common.cuh"

#include <cfloat>
#include <cstdio>

using namespace synapse_cuda_family;

extern "C" {
typedef struct Qwen3LayerParams {
    const float *input_norm;
    const float *post_attention_norm;
    const float *q_weight;
    const float *q_norm;
    const float *k_weight;
    const float *k_norm;
    const float *v_weight;
    const float *o_weight;
    const float *gate_weight;
    const float *up_weight;
    const float *down_weight;
} Qwen3LayerParams;
}

namespace {

struct DeviceLayer {
    DeviceAllocation<float> input_norm, post_attention_norm, q_norm, k_norm;
    DeviceAllocation<unsigned char> q_weight, k_weight, v_weight, o_weight;
    DeviceAllocation<unsigned char> gate_weight, up_weight, down_weight;
};

__global__ void rms_norm(const half *input, const float *weight, half *output, int width, float epsilon) {
    int row = blockIdx.x;
    int base = row * width;
    float square = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float value = __half2float(input[base + column]);
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    square = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) square = warp_sum(square);
    if (threadIdx.x == 0) reductions[0] = rsqrtf(square / width + epsilon);
    __syncthreads();
    float inverse = reductions[0];
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = __float2half(__half2float(input[base + column]) * inverse * weight[column]);
    }
}

__global__ void head_norm_rope_transpose(
    const half *input,
    const float *weight,
    half *output,
    const float *cosine,
    const float *sine,
    int batch,
    int seq,
    int heads,
    int head_dim,
    float epsilon
) {
    int row_head = blockIdx.x;
    int head = row_head % heads;
    int row = row_head / heads;
    int position = row % seq;
    int batch_index = row / seq;
    int source = row * heads * head_dim + head * head_dim;
    float square = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        float value = __half2float(input[source + dimension]);
        square += value * value;
    }
    square = warp_sum(square);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = square;
    __syncthreads();
    square = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) square = warp_sum(square);
    if (threadIdx.x == 0) reductions[0] = rsqrtf(square / head_dim + epsilon);
    __syncthreads();
    float inverse = reductions[0];
    int groups = heads == 16 ? 2 : 1;
    int kv_heads = heads / groups;
    int group = head % groups;
    int kv_head = head / groups;
    int target = ((group * batch * kv_heads + batch_index * kv_heads + kv_head) * seq + position) * head_dim;
    int half_dim = head_dim / 2;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
        int pair = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
        float value = __half2float(input[source + dimension]) * inverse * weight[dimension];
        float pair_value = __half2float(input[source + pair]) * inverse * weight[pair];
        float sign = dimension < half_dim ? -1.0f : 1.0f;
        output[target + dimension] = __float2half(
            value * cosine[position * head_dim + dimension] +
            sign * pair_value * sine[position * head_dim + dimension]
        );
    }
}

__global__ void transpose_value(const half *input, half *output, int batch, int seq, int heads, int head_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * heads * head_dim;
    if (index >= total) return;
    int dimension = index % head_dim;
    int remainder = index / head_dim;
    int head = remainder % heads;
    remainder /= heads;
    int position = remainder % seq;
    int batch_index = remainder / seq;
    int target = ((batch_index * heads + head) * seq + position) * head_dim + dimension;
    output[target] = input[index];
}

__global__ void causal_softmax(half *scores, const uint8_t *mask, int batch, int kv_heads, int seq, float scale) {
    int row = blockIdx.x;
    int matrix = row / seq;
    int batch_index = (matrix / kv_heads) % batch;
    int query = row % seq;
    int base = row * seq;
    float maximum = -FLT_MAX;
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        float value = (key > query || !mask[batch_index * seq + key]) ? -10000.0f : __half2float(scores[base + key]) * scale;
        maximum = fmaxf(maximum, value);
    }
    maximum = warp_max(maximum);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = maximum;
    __syncthreads();
    maximum = threadIdx.x < blockDim.x / 32 ? reductions[lane] : -FLT_MAX;
    if (warp == 0) maximum = warp_max(maximum);
    if (threadIdx.x == 0) reductions[0] = maximum;
    __syncthreads();
    maximum = reductions[0];
    float sum = 0.0f;
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        float value = (key > query || !mask[batch_index * seq + key]) ? -10000.0f : __half2float(scores[base + key]) * scale;
        float exponential = expf(value - maximum);
        scores[base + key] = __float2half(exponential);
        sum += exponential;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) reductions[0] = sum;
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1e-20f);
    for (int key = threadIdx.x; key < seq; key += blockDim.x) scores[base + key] = __float2half(__half2float(scores[base + key]) * inverse);
}

__global__ void transpose_context(const half *source, half *target, int batch, int seq, int query_heads, int kv_heads, int head_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * query_heads * head_dim;
    if (index >= total) return;
    int dimension = index % head_dim;
    int remainder = index / head_dim;
    int head = remainder % query_heads;
    remainder /= query_heads;
    int position = remainder % seq;
    int batch_index = remainder / seq;
    int groups = query_heads / kv_heads;
    int group = head % groups;
    int kv_head = head / groups;
    int source_index = ((group * batch * kv_heads + batch_index * kv_heads + kv_head) * seq + position) * head_dim + dimension;
    target[index] = source[source_index];
}

__global__ void add_residual(const half *projected, const half *residual, half *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half(__half2float(projected[index]) + __half2float(residual[index]));
}

__global__ void swiglu(const half *gate, const half *up, half *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = __half2float(gate[index]);
    value = value / (1.0f + expf(-value));
    output[index] = __float2half(value * __half2float(up[index]));
}

__global__ void to_float(const half *input, float *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __half2float(input[index]);
}

struct QwenContext;

struct ShapePlan {
    QwenContext *context;
    int batch, seq, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count;
    float epsilon;
    size_t arena_bytes = 0;
    DeviceAllocation<unsigned char> arena, workspace;
    DeviceAllocation<uint8_t> mask;
    DeviceAllocation<float> cosine, sine, output;
    half *x0 = nullptr, *x1 = nullptr, *normed = nullptr;
    half *q_raw = nullptr, *k_raw = nullptr, *v_raw = nullptr;
    half *q = nullptr, *k = nullptr, *v = nullptr, *scores = nullptr, *attention = nullptr;
    half *context_rows = nullptr, *projected = nullptr, *gate = nullptr, *up = nullptr, *activated = nullptr;
    MatmulPlan hq, hkv, qh, hi, ih, qk, pv;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;


    ShapePlan(QwenContext *owner, int b, int s, int h, int qh_count, int kvh, int hd, int inter, int layers, float eps, float theta);
    ~ShapePlan();
    void compute(StageProfile *profile = nullptr);
    void initialize_and_verify(const uint16_t *input, const uint8_t *host_mask);
    void run(const uint16_t *input, const uint8_t *host_mask, float *host_output);
};

struct QwenContext {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    bool graphs_enabled;
    bool weights_loaded = false;
    int hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, layer_count = 0;
    std::vector<DeviceLayer> layers;
    DeviceAllocation<float> final_norm;
    std::unordered_map<std::string, std::unique_ptr<ShapePlan>> plans;

    explicit QwenContext(bool graphs) : graphs_enabled(graphs) {
        FAMILY_CUDA_CHECK(cudaFree(nullptr));
        FAMILY_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        FAMILY_CUBLAS_CHECK(cublasLtCreate(&lt));
    }
    ~QwenContext() {
        plans.clear();
        if (lt) cublasLtDestroy(lt);
        if (stream) cudaStreamDestroy(stream);
    }

    void load_weights(int h, int qh_count, int kvh, int hd, int inter, int count, const Qwen3LayerParams *params, const float *host_final_norm) {
        if (weights_loaded) {
            if (hidden != h || query_heads != qh_count || kv_heads != kvh || head_dim != hd || intermediate != inter || layer_count != count) throw std::runtime_error("Qwen3 CUDA model dimensions changed");
            return;
        }
        hidden = h; query_heads = qh_count; kv_heads = kvh; head_dim = hd; intermediate = inter; layer_count = count;
        int q_width = qh_count * hd;
        int kv_width = kvh * hd;
        layers.resize(count);
        for (int index = 0; index < count; ++index) {
            DeviceLayer &target = layers[index];
            const Qwen3LayerParams &source = params[index];
            copy_float(target.input_norm, source.input_norm, h);
            copy_float(target.post_attention_norm, source.post_attention_norm, h);
            copy_float(target.q_norm, source.q_norm, hd);
            copy_float(target.k_norm, source.k_norm, hd);
            copy_weight<half>(target.q_weight, source.q_weight, static_cast<size_t>(q_width) * h);
            copy_weight<half>(target.k_weight, source.k_weight, static_cast<size_t>(kv_width) * h);
            copy_weight<half>(target.v_weight, source.v_weight, static_cast<size_t>(kv_width) * h);
            copy_weight<half>(target.o_weight, source.o_weight, static_cast<size_t>(h) * q_width);
            copy_weight<half>(target.gate_weight, source.gate_weight, static_cast<size_t>(inter) * h);
            copy_weight<half>(target.up_weight, source.up_weight, static_cast<size_t>(inter) * h);
            copy_weight<half>(target.down_weight, source.down_weight, static_cast<size_t>(h) * inter);
        }
        copy_float(final_norm, host_final_norm, h);
        FAMILY_CUDA_CHECK(cudaDeviceSynchronize());
        weights_loaded = true;
        std::fprintf(stderr, "CUDA Qwen3 persistent weights: layers=%d dtype=f16 accum=fp32 norm_params=fp32\n", count);
    }
};

ShapePlan::ShapePlan(QwenContext *owner, int b, int s, int h, int qh_count, int kvh, int hd, int inter, int layers_count, float eps, float theta)
    : context(owner), batch(b), seq(s), hidden(h), query_heads(qh_count), kv_heads(kvh), head_dim(hd), intermediate(inter), layer_count(layers_count), epsilon(eps) {
    size_t rows = static_cast<size_t>(batch) * seq;
    size_t hidden_values = rows * hidden;
    size_t q_values = rows * query_heads * head_dim;
    size_t kv_values = rows * kv_heads * head_dim;
    size_t scores_count = static_cast<size_t>(batch) * query_heads * seq * seq;
    size_t intermediate_values = rows * intermediate;
    size_t total = hidden_values * 7 + q_values * 3 + kv_values * 4 + scores_count + intermediate_values * 3;
    arena_bytes = total * sizeof(half) + 20 * 256;
    arena.allocate(arena_bytes);
    mask.allocate(rows);
    output.allocate(hidden_values);
    unsigned char *cursor = arena.pointer;
    auto take = [&](size_t count) {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(cursor) + 255) & ~uintptr_t(255);
        cursor = reinterpret_cast<unsigned char *>(aligned);
        half *result = reinterpret_cast<half *>(cursor);
        cursor += count * sizeof(half);
        return result;
    };
    x0 = take(hidden_values); x1 = take(hidden_values); normed = take(hidden_values);
    q_raw = take(q_values); k_raw = take(kv_values); v_raw = take(kv_values);
    q = take(q_values); k = take(kv_values); v = take(kv_values); scores = take(scores_count);
    attention = take(q_values); context_rows = take(q_values); projected = take(hidden_values);
    gate = take(intermediate_values); up = take(intermediate_values); activated = take(intermediate_values);

    std::vector<float> host_cos(seq * head_dim), host_sin(seq * head_dim);
    for (int position = 0; position < seq; ++position) for (int index = 0; index < head_dim / 2; ++index) {
        float frequency = std::pow(theta, -2.0f * index / head_dim);
        float angle = position * frequency;
        float c = std::cos(angle), sn = std::sin(angle);
        host_cos[position * head_dim + index] = host_cos[position * head_dim + head_dim / 2 + index] = c;
        host_sin[position * head_dim + index] = host_sin[position * head_dim + head_dim / 2 + index] = sn;
    }
    copy_float(cosine, host_cos.data(), host_cos.size());
    copy_float(sine, host_sin.data(), host_sin.size());

    hq = select_matmul(owner->lt, "hq", CUDA_R_16F, rows, query_heads * head_dim, hidden, true);
    hkv = select_matmul(owner->lt, "hkv", CUDA_R_16F, rows, kv_heads * head_dim, hidden, true);
    qh = select_matmul(owner->lt, "qh", CUDA_R_16F, rows, hidden, query_heads * head_dim, true);
    hi = select_matmul(owner->lt, "hi", CUDA_R_16F, rows, intermediate, hidden, true);
    ih = select_matmul(owner->lt, "ih", CUDA_R_16F, rows, hidden, intermediate, true);
    int group_batches = batch * kv_heads;
    qk = select_matmul(owner->lt, "gqa_qk", CUDA_R_16F, seq, seq, head_dim, true, group_batches, seq * head_dim, seq * head_dim, seq * seq);
    pv = select_matmul(owner->lt, "gqa_pv", CUDA_R_16F, seq, head_dim, seq, false, group_batches, seq * seq, seq * head_dim, seq * head_dim);
    size_t max_workspace = std::max({hq.workspace_bytes, hkv.workspace_bytes, qh.workspace_bytes, hi.workspace_bytes, ih.workspace_bytes, qk.workspace_bytes, pv.workspace_bytes});
    workspace.allocate(max_workspace);
}

ShapePlan::~ShapePlan() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
}

void ShapePlan::compute(StageProfile *profile) {
    int rows = batch * seq;
    int threads = 256;
    int hidden_count = rows * hidden;
    int hidden_blocks = (hidden_count + threads - 1) / threads;
    int query_count = rows * query_heads * head_dim;
    int query_blocks = (query_count + threads - 1) / threads;
    int kv_count = rows * kv_heads * head_dim;
    int kv_blocks = (kv_count + threads - 1) / threads;
    int intermediate_count = rows * intermediate;
    int intermediate_blocks = (intermediate_count + threads - 1) / threads;
    int groups = query_heads / kv_heads;
    size_t q_group_values = static_cast<size_t>(batch) * kv_heads * seq * head_dim;
    size_t score_group_values = static_cast<size_t>(batch) * kv_heads * seq * seq;
    auto begin = [&](const char *name) { if (profile) profile->begin(name, context->stream); };
    auto end = [&] { if (profile) profile->end(context->stream); };
    for (int index = 0; index < layer_count; ++index) {
        DeviceLayer &layer = context->layers[index];
        begin("pointwise_layout");
        rms_norm<<<rows, threads, 0, context->stream>>>(x0, layer.input_norm.pointer, normed, hidden, epsilon);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, hq, normed, layer.q_weight.pointer, q_raw, workspace.pointer, context->stream);
        launch_matmul(context->lt, hkv, normed, layer.k_weight.pointer, k_raw, workspace.pointer, context->stream);
        launch_matmul(context->lt, hkv, normed, layer.v_weight.pointer, v_raw, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        head_norm_rope_transpose<<<rows * query_heads, threads, 0, context->stream>>>(q_raw, layer.q_norm.pointer, q, cosine.pointer, sine.pointer, batch, seq, query_heads, head_dim, epsilon);
        head_norm_rope_transpose<<<rows * kv_heads, threads, 0, context->stream>>>(k_raw, layer.k_norm.pointer, k, cosine.pointer, sine.pointer, batch, seq, kv_heads, head_dim, epsilon);
        transpose_value<<<kv_blocks, threads, 0, context->stream>>>(v_raw, v, batch, seq, kv_heads, head_dim);
        end();
        begin("attention_gemm");
        for (int group = 0; group < groups; ++group) {
            launch_matmul(context->lt, qk, q + group * q_group_values, k, scores + group * score_group_values, workspace.pointer, context->stream);
        }
        end();
        begin("score_softmax");
        causal_softmax<<<batch * query_heads * seq, threads, 0, context->stream>>>(scores, mask.pointer, batch, kv_heads, seq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        end();
        begin("attention_gemm");
        for (int group = 0; group < groups; ++group) {
            launch_matmul(context->lt, pv, scores + group * score_group_values, v, attention + group * q_group_values, workspace.pointer, context->stream);
        }
        end();
        begin("pointwise_layout");
        transpose_context<<<query_blocks, threads, 0, context->stream>>>(attention, context_rows, batch, seq, query_heads, kv_heads, head_dim);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, qh, context_rows, layer.o_weight.pointer, projected, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        add_residual<<<hidden_blocks, threads, 0, context->stream>>>(projected, x0, x1, hidden_count);
        rms_norm<<<rows, threads, 0, context->stream>>>(x1, layer.post_attention_norm.pointer, normed, hidden, epsilon);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, hi, normed, layer.gate_weight.pointer, gate, workspace.pointer, context->stream);
        launch_matmul(context->lt, hi, normed, layer.up_weight.pointer, up, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        swiglu<<<intermediate_blocks, threads, 0, context->stream>>>(gate, up, activated, intermediate_count);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, ih, activated, layer.down_weight.pointer, projected, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        add_residual<<<hidden_blocks, threads, 0, context->stream>>>(projected, x1, x0, hidden_count);
        end();
    }
    begin("final_norm_output");
    rms_norm<<<rows, threads, 0, context->stream>>>(x0, context->final_norm.pointer, x1, hidden, epsilon);
    to_float<<<hidden_blocks, threads, 0, context->stream>>>(x1, output.pointer, hidden_count);
    end();
    FAMILY_CUDA_CHECK(cudaGetLastError());
}

void ShapePlan::initialize_and_verify(const uint16_t *input, const uint8_t *host_mask) {
    size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * sizeof(half);
    size_t mask_bytes = static_cast<size_t>(batch) * seq;
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    StageProfile profile;
    compute(&profile);
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
    std::unordered_map<std::string, double> stage_ms = profile.collect();
    std::vector<float> uncaptured(output.count);
    FAMILY_CUDA_CHECK(cudaMemcpy(uncaptured.data(), output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost));
    FAMILY_CUDA_CHECK(cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeThreadLocal));
    compute();
    FAMILY_CUDA_CHECK(cudaStreamEndCapture(context->stream, &graph));
    FAMILY_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
    std::vector<float> captured(output.count);
    FAMILY_CUDA_CHECK(cudaMemcpy(captured.data(), output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(uncaptured.data(), captured.data(), output.count * sizeof(float)) != 0) throw std::runtime_error("Qwen3 captured output differs from uncaptured output");
    std::fprintf(stderr, "CUDA Qwen3 shape %dx%d: arena=%zu workspace=%zu captured_exact=true launches=%d gqa=two-group-strided kv_repeat_bytes=0 stage_projection_mlp_gemm=%.3fms stage_attention_gemm=%.3fms stage_score_softmax=%.3fms stage_pointwise_layout=%.3fms stage_final_norm_output=%.3fms\n", batch, seq, arena_bytes, workspace.count, layer_count * (15 + 2 * (query_heads / kv_heads)) + 2, stage_ms["projection_mlp_gemm"], stage_ms["attention_gemm"], stage_ms["score_softmax"], stage_ms["pointwise_layout"], stage_ms["final_norm_output"]);
}

void ShapePlan::run(const uint16_t *input, const uint8_t *host_mask, float *host_output) {
    size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * sizeof(half);
    size_t mask_bytes = static_cast<size_t>(batch) * seq;
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    if (context->graphs_enabled) FAMILY_CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    else compute();
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_output, output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost, context->stream));
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
}

}  // namespace

extern "C" {

void *synapse_cuda_qwen3_context_new(int32_t graphs_enabled) {
    try {
        return new QwenContext(graphs_enabled != 0);
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return nullptr;
    }
}

void synapse_cuda_qwen3_context_free(void *raw_context) {
    delete static_cast<QwenContext *>(raw_context);
}

int32_t synapse_cuda_qwen3_forward(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    float rope_theta,
    const uint16_t *input,
    const uint8_t *attention_mask,
    const Qwen3LayerParams *layers,
    const float *final_norm,
    float *output
) {
    try {
        if (!raw_context || !input || !attention_mask || !layers || !final_norm || !output) throw std::runtime_error("Qwen3 CUDA received a null pointer");
        if (!batch || !seq || !hidden || !query_heads || !kv_heads || query_heads % kv_heads || !head_dim || !layer_count) throw std::runtime_error("Qwen3 CUDA received invalid dimensions");
        QwenContext *context = static_cast<QwenContext *>(raw_context);
        context->load_weights(hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, layers, final_norm);
        std::string key = shape_key(batch, seq);
        auto found = context->plans.find(key);
        if (found == context->plans.end()) {
            auto plan = std::make_unique<ShapePlan>(context, batch, seq, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, epsilon, rope_theta);
            plan->initialize_and_verify(input, attention_mask);
            found = context->plans.emplace(key, std::move(plan)).first;
        }
        found->second->run(input, attention_mask, output);
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

}  // extern "C"
