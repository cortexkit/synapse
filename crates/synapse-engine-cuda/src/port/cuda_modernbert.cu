#include "cuda_family_common.cuh"

#include <cfloat>
#include <cstdio>

using namespace synapse_cuda_family;

extern "C" {
typedef struct ModernBertLayerParams {
    const float *qkv_weight;
    const float *attention_output_weight;
    const float *attention_norm_weight;
    const float *mlp_input_weight;
    const float *mlp_output_weight;
    const float *mlp_norm_weight;
    int32_t attention_type;
} ModernBertLayerParams;
}

namespace {

struct DeviceLayer {
    DeviceAllocation<unsigned char> qkv_weight;
    DeviceAllocation<unsigned char> attention_output_weight;
    DeviceAllocation<float> attention_norm_weight;
    DeviceAllocation<unsigned char> mlp_input_weight;
    DeviceAllocation<unsigned char> mlp_output_weight;
    DeviceAllocation<float> mlp_norm_weight;
    int attention_type = 0;
};

template <typename T>
__global__ void layer_norm_kernel(const T *input, const float *weight, T *output, int hidden, float epsilon) {
    int row = blockIdx.x;
    int base = row * hidden;
    float sum = 0.0f;
    float square = 0.0f;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = load_value(input, base + column);
        sum += value;
        square += value * value;
    }
    sum = warp_sum(sum);
    square = warp_sum(square);
    __shared__ float sums[32];
    __shared__ float squares[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sums[warp] = sum;
        squares[warp] = square;
    }
    __syncthreads();
    sum = threadIdx.x < blockDim.x / 32 ? sums[lane] : 0.0f;
    square = threadIdx.x < blockDim.x / 32 ? squares[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_sum(sum);
        square = warp_sum(square);
    }
    if (threadIdx.x == 0) {
        sums[0] = sum / hidden;
        squares[0] = square / hidden;
    }
    __syncthreads();
    float mean = sums[0];
    float variance = fmaxf(squares[0] - mean * mean, 0.0f);
    float inverse = rsqrtf(variance + epsilon);
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = (load_value(input, base + column) - mean) * inverse;
        store_value(output, base + column, value * weight[column]);
    }
}

template <typename T>
__global__ void copy_kernel(const T *input, T *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index];
}

template <typename T>
__global__ void add_kernel(const T *projected, const T *residual, T *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) store_value(output, index, load_value(projected, index) + load_value(residual, index));
}

template <typename T>
__global__ void qkv_rope_transpose(
    const T *raw,
    T *q,
    T *k,
    T *v,
    const float *cosine,
    const float *sine,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * head_dim;
    if (index >= total) return;
    int dimension = index % head_dim;
    int remainder = index / head_dim;
    int position = remainder % seq;
    remainder /= seq;
    int head = remainder % heads;
    int batch_index = remainder / heads;
    int hidden = heads * head_dim;
    int half = head_dim / 2;
    int pair_dimension = dimension < half ? dimension + half : dimension - half;
    int hidden_index = head * head_dim + dimension;
    int pair_hidden_index = head * head_dim + pair_dimension;
    int row = batch_index * seq + position;
    int q_offset = row * 3 * hidden;
    float q_value = load_value(raw, q_offset + hidden_index);
    float k_value = load_value(raw, q_offset + hidden + hidden_index);
    float q_pair = load_value(raw, q_offset + pair_hidden_index);
    float k_pair = load_value(raw, q_offset + hidden + pair_hidden_index);
    float cos_value = cosine[position * head_dim + dimension];
    float sin_value = sine[position * head_dim + dimension];
    float sign = dimension < half ? -1.0f : 1.0f;
    store_value(q, index, q_value * cos_value + sign * q_pair * sin_value);
    store_value(k, index, k_value * cos_value + sign * k_pair * sin_value);
    store_value(v, index, load_value(raw, q_offset + 2 * hidden + hidden_index));
}

template <typename T>
__global__ void mask_softmax(
    T *scores,
    const uint8_t *padding_mask,
    const float *band_mask,
    int heads,
    int seq,
    bool sliding,
    float scale
) {
    int row = blockIdx.x;
    int batch_index = row / (heads * seq);
    int query = row % seq;
    int base = row * seq;
    float maximum = -FLT_MAX;
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        float mask = padding_mask[batch_index * seq + key] ? 0.0f : -10000.0f;
        if (sliding) mask = fminf(mask, band_mask[query * seq + key]);
        maximum = fmaxf(maximum, load_value(scores, base + key) * scale + mask);
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
        float mask = padding_mask[batch_index * seq + key] ? 0.0f : -10000.0f;
        if (sliding) mask = fminf(mask, band_mask[query * seq + key]);
        float value = expf(load_value(scores, base + key) * scale + mask - maximum);
        store_value(scores, base + key, value);
        sum += value;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) reductions[0] = sum;
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1e-20f);
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        store_value(scores, base + key, load_value(scores, base + key) * inverse);
    }
}

template <typename T>
__global__ void transpose_context(const T *source, T *target, int batch, int seq, int heads, int head_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * heads * head_dim;
    if (index >= total) return;
    int dimension = index % head_dim;
    int remainder = index / head_dim;
    int head = remainder % heads;
    remainder /= heads;
    int position = remainder % seq;
    int batch_index = remainder / seq;
    int source_index = ((batch_index * heads + head) * seq + position) * head_dim + dimension;
    target[index] = source[source_index];
}

template <typename T>
__global__ void geglu_kernel(T *projected, T *activated, int rows, int intermediate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * intermediate;
    if (index >= total) return;
    int row = index / intermediate;
    int column = index % intermediate;
    int base = row * intermediate * 2;
    float gate = load_value(projected, base + column);
    float value = 0.5f * gate * (1.0f + erff(gate * 0.7071067811865475f));
    value *= load_value(projected, base + intermediate + column);
    store_value(activated, index, value);
}

template <typename T>
__global__ void to_float_kernel(const T *input, float *output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = load_value(input, index);
}

struct ModernContext;

struct ShapePlan {
    ModernContext *context;
    int batch, seq, hidden, heads, head_dim, intermediate, layer_count;
    float epsilon;
    size_t arena_bytes = 0;
    DeviceAllocation<unsigned char> arena;
    DeviceAllocation<unsigned char> workspace;
    DeviceAllocation<uint8_t> mask;
    DeviceAllocation<float> band_mask, global_cos, global_sin, local_cos, local_sin, output;
    void *x0 = nullptr, *x1 = nullptr, *normed = nullptr, *qkv = nullptr;
    void *q = nullptr, *k = nullptr, *v = nullptr, *scores = nullptr, *attention = nullptr;
    void *context_rows = nullptr, *projected = nullptr, *mlp_projected = nullptr, *activated = nullptr;
    MatmulPlan h3h, hh, h2i, ih, qk_plan, pv_plan;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    ShapePlan(ModernContext *owner, int b, int s, int h, int nh, int inter, int layers, float eps, float global_theta, float local_theta, int half_window);
    ~ShapePlan();
    template <typename T> void compute(StageProfile *profile = nullptr);
    void initialize_and_verify(const void *input, const uint8_t *host_mask);
    void run(const void *input, const uint8_t *host_mask, float *host_output);
};

struct ModernContext {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    bool graphs_enabled;
    bool f16;
    bool weights_loaded = false;
    int hidden = 0, intermediate = 0, layer_count = 0;
    std::vector<DeviceLayer> layers;
    DeviceAllocation<float> final_norm;
    std::unordered_map<std::string, std::unique_ptr<ShapePlan>> plans;

    ModernContext(bool graphs, bool use_f16) : graphs_enabled(graphs), f16(use_f16) {
        FAMILY_CUDA_CHECK(cudaFree(nullptr));
        FAMILY_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        FAMILY_CUBLAS_CHECK(cublasLtCreate(&lt));
    }
    ~ModernContext() {
        plans.clear();
        if (lt) cublasLtDestroy(lt);
        if (stream) cudaStreamDestroy(stream);
    }

    template <typename T>
    void load_weights(int h, int inter, int count, const ModernBertLayerParams *params, const float *host_final_norm) {
        if (weights_loaded) {
            if (hidden != h || intermediate != inter || layer_count != count) throw std::runtime_error("ModernBERT CUDA model dimensions changed");
            return;
        }
        hidden = h;
        intermediate = inter;
        layer_count = count;
        layers.resize(count);
        for (int index = 0; index < count; ++index) {
            DeviceLayer &target = layers[index];
            const ModernBertLayerParams &source = params[index];
            copy_weight<T>(target.qkv_weight, source.qkv_weight, 3ULL * h * h);
            copy_weight<T>(target.attention_output_weight, source.attention_output_weight, 1ULL * h * h);
            if (source.attention_norm_weight) copy_float(target.attention_norm_weight, source.attention_norm_weight, h);
            copy_weight<T>(target.mlp_input_weight, source.mlp_input_weight, 2ULL * inter * h);
            copy_weight<T>(target.mlp_output_weight, source.mlp_output_weight, 1ULL * h * inter);
            copy_float(target.mlp_norm_weight, source.mlp_norm_weight, h);
            target.attention_type = source.attention_type;
        }
        copy_float(final_norm, host_final_norm, h);
        FAMILY_CUDA_CHECK(cudaDeviceSynchronize());
        weights_loaded = true;
        std::fprintf(stderr, "CUDA ModernBERT persistent weights: layers=%d dtype=%s norm_params=fp32\n", count, f16 ? "f16" : "f32");
    }
};

ShapePlan::ShapePlan(ModernContext *owner, int b, int s, int h, int nh, int inter, int layers_count, float eps, float global_theta, float local_theta, int half_window)
    : context(owner), batch(b), seq(s), hidden(h), heads(nh), head_dim(h / nh), intermediate(inter), layer_count(layers_count), epsilon(eps) {
    size_t rows = static_cast<size_t>(batch) * seq;
    size_t hidden_values = rows * hidden;
    size_t scores_count = static_cast<size_t>(batch) * heads * seq * seq;
    size_t mlp_values = rows * intermediate * 2;
    size_t activated_values = rows * intermediate;
    size_t element_bytes = context->f16 ? sizeof(half) : sizeof(float);
    size_t total_elements = hidden_values * 12 + scores_count + mlp_values + activated_values;
    arena_bytes = total_elements * element_bytes + 16 * 256;
    arena.allocate(arena_bytes);
    mask.allocate(rows);
    output.allocate(hidden_values);
    unsigned char *cursor = arena.pointer;
    auto take = [&](size_t count) -> void * {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(cursor) + 255) & ~uintptr_t(255);
        cursor = reinterpret_cast<unsigned char *>(aligned);
        void *result = cursor;
        cursor += count * element_bytes;
        return result;
    };
    x0 = take(hidden_values); x1 = take(hidden_values); normed = take(hidden_values); qkv = take(hidden_values * 3);
    q = take(hidden_values); k = take(hidden_values); v = take(hidden_values); scores = take(scores_count);
    attention = take(hidden_values); context_rows = take(hidden_values); projected = take(hidden_values);
    mlp_projected = take(mlp_values); activated = take(activated_values);

    std::vector<float> band(seq * seq, 0.0f);
    for (int query = 0; query < seq; ++query) for (int key = 0; key < seq; ++key) if (std::abs(query - key) > half_window) band[query * seq + key] = -10000.0f;
    copy_float(band_mask, band.data(), band.size());
    auto make_rope = [&](float theta, DeviceAllocation<float> &cos_target, DeviceAllocation<float> &sin_target) {
        std::vector<float> cosine(seq * head_dim), sine(seq * head_dim);
        for (int position = 0; position < seq; ++position) for (int index = 0; index < head_dim / 2; ++index) {
            float frequency = std::pow(theta, -2.0f * index / head_dim);
            float angle = position * frequency;
            float c = std::cos(angle), sn = std::sin(angle);
            cosine[position * head_dim + index] = cosine[position * head_dim + head_dim / 2 + index] = c;
            sine[position * head_dim + index] = sine[position * head_dim + head_dim / 2 + index] = sn;
        }
        copy_float(cos_target, cosine.data(), cosine.size());
        copy_float(sin_target, sine.data(), sine.size());
    };
    make_rope(global_theta, global_cos, global_sin);
    make_rope(local_theta, local_cos, local_sin);

    cudaDataType_t dtype = context->f16 ? CUDA_R_16F : CUDA_R_32F;
    h3h = select_matmul(owner->lt, "h3h", dtype, rows, 3LL * hidden, hidden, true);
    hh = select_matmul(owner->lt, "hh", dtype, rows, hidden, hidden, true);
    h2i = select_matmul(owner->lt, "h2i", dtype, rows, 2LL * intermediate, hidden, true);
    ih = select_matmul(owner->lt, "ih", dtype, rows, hidden, intermediate, true);
    int batch_heads = batch * heads;
    qk_plan = select_matmul(owner->lt, "qk", dtype, seq, seq, head_dim, true, batch_heads, seq * head_dim, seq * head_dim, seq * seq);
    pv_plan = select_matmul(owner->lt, "pv", dtype, seq, head_dim, seq, false, batch_heads, seq * seq, seq * head_dim, seq * head_dim);
    size_t max_workspace = std::max({h3h.workspace_bytes, hh.workspace_bytes, h2i.workspace_bytes, ih.workspace_bytes, qk_plan.workspace_bytes, pv_plan.workspace_bytes});
    workspace.allocate(max_workspace);
}

ShapePlan::~ShapePlan() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
}

template <typename T>
void ShapePlan::compute(StageProfile *profile) {
    int rows = batch * seq;
    int threads = 256;
    int hidden_count = rows * hidden;
    int blocks = (hidden_count + threads - 1) / threads;
    int head_blocks = (batch * heads * seq * head_dim + threads - 1) / threads;
    int activated_blocks = (rows * intermediate + threads - 1) / threads;
    auto begin = [&](const char *name) { if (profile) profile->begin(name, context->stream); };
    auto end = [&] { if (profile) profile->end(context->stream); };
    for (int index = 0; index < layer_count; ++index) {
        DeviceLayer &layer = context->layers[index];
        begin("pointwise_layout");
        if (layer.attention_norm_weight.pointer) {
            layer_norm_kernel<<<rows, threads, 0, context->stream>>>(static_cast<T *>(x0), layer.attention_norm_weight.pointer, static_cast<T *>(normed), hidden, epsilon);
        } else {
            copy_kernel<<<blocks, threads, 0, context->stream>>>(static_cast<T *>(x0), static_cast<T *>(normed), hidden_count);
        }
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, h3h, normed, layer.qkv_weight.pointer, qkv, workspace.pointer, context->stream);
        end();
        const float *cosine = layer.attention_type ? local_cos.pointer : global_cos.pointer;
        const float *sine = layer.attention_type ? local_sin.pointer : global_sin.pointer;
        begin("pointwise_layout");
        qkv_rope_transpose<<<head_blocks, threads, 0, context->stream>>>(static_cast<T *>(qkv), static_cast<T *>(q), static_cast<T *>(k), static_cast<T *>(v), cosine, sine, batch, seq, heads, head_dim);
        end();
        begin("attention_gemm");
        launch_matmul(context->lt, qk_plan, q, k, scores, workspace.pointer, context->stream);
        end();
        begin("score_softmax");
        mask_softmax<<<batch * heads * seq, threads, 0, context->stream>>>(static_cast<T *>(scores), mask.pointer, band_mask.pointer, heads, seq, layer.attention_type != 0, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        end();
        begin("attention_gemm");
        launch_matmul(context->lt, pv_plan, scores, v, attention, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        transpose_context<<<blocks, threads, 0, context->stream>>>(static_cast<T *>(attention), static_cast<T *>(context_rows), batch, seq, heads, head_dim);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, hh, context_rows, layer.attention_output_weight.pointer, projected, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        add_kernel<<<blocks, threads, 0, context->stream>>>(static_cast<T *>(projected), static_cast<T *>(x0), static_cast<T *>(x1), hidden_count);
        layer_norm_kernel<<<rows, threads, 0, context->stream>>>(static_cast<T *>(x1), layer.mlp_norm_weight.pointer, static_cast<T *>(normed), hidden, epsilon);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, h2i, normed, layer.mlp_input_weight.pointer, mlp_projected, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        geglu_kernel<<<activated_blocks, threads, 0, context->stream>>>(static_cast<T *>(mlp_projected), static_cast<T *>(activated), rows, intermediate);
        end();
        begin("projection_mlp_gemm");
        launch_matmul(context->lt, ih, activated, layer.mlp_output_weight.pointer, projected, workspace.pointer, context->stream);
        end();
        begin("pointwise_layout");
        add_kernel<<<blocks, threads, 0, context->stream>>>(static_cast<T *>(projected), static_cast<T *>(x1), static_cast<T *>(x0), hidden_count);
        end();
    }
    begin("final_norm_output");
    layer_norm_kernel<<<rows, threads, 0, context->stream>>>(static_cast<T *>(x0), context->final_norm.pointer, static_cast<T *>(x1), hidden, epsilon);
    to_float_kernel<<<blocks, threads, 0, context->stream>>>(static_cast<T *>(x1), output.pointer, hidden_count);
    end();
    FAMILY_CUDA_CHECK(cudaGetLastError());
}

void ShapePlan::initialize_and_verify(const void *input, const uint8_t *host_mask) {
    size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * (context->f16 ? sizeof(half) : sizeof(float));
    size_t mask_bytes = static_cast<size_t>(batch) * seq;
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    StageProfile profile;
    if (context->f16) compute<half>(&profile); else compute<float>(&profile);
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
    std::unordered_map<std::string, double> stage_ms = profile.collect();
    std::vector<float> uncaptured(output.count);
    FAMILY_CUDA_CHECK(cudaMemcpy(uncaptured.data(), output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost));
    FAMILY_CUDA_CHECK(cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeThreadLocal));
    if (context->f16) compute<half>(); else compute<float>();
    FAMILY_CUDA_CHECK(cudaStreamEndCapture(context->stream, &graph));
    FAMILY_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
    std::vector<float> captured(output.count);
    FAMILY_CUDA_CHECK(cudaMemcpy(captured.data(), output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(uncaptured.data(), captured.data(), output.count * sizeof(float)) != 0) {
        float maximum = 0.0f;
        size_t maximum_index = 0;
        for (size_t index = 0; index < output.count; ++index) {
            float difference = std::abs(uncaptured[index] - captured[index]);
            if (difference > maximum) {
                maximum = difference;
                maximum_index = index;
            }
        }
        std::ostringstream message;
        message << "ModernBERT captured output differs from uncaptured output; max_abs=" << maximum << " index=" << maximum_index;
        throw std::runtime_error(message.str());
    }
    std::fprintf(stderr, "CUDA ModernBERT shape %dx%d: arena=%zu workspace=%zu captured_exact=true launches=%d band_mask=precomputed gqa=none stage_projection_mlp_gemm=%.3fms stage_attention_gemm=%.3fms stage_score_softmax=%.3fms stage_pointwise_layout=%.3fms stage_final_norm_output=%.3fms\n", batch, seq, arena_bytes, workspace.count, layer_count * 15 + 2, stage_ms["projection_mlp_gemm"], stage_ms["attention_gemm"], stage_ms["score_softmax"], stage_ms["pointwise_layout"], stage_ms["final_norm_output"]);
    const MatmulPlan *plans[] = {&h3h, &hh, &h2i, &ih, &qk_plan, &pv_plan};
    for (const MatmulPlan *plan : plans) std::fprintf(stderr, "CUDA ModernBERT algo %dx%d %s: id=%d split_k=%d workspace=%zu\n", batch, seq, plan->name.c_str(), plan->algorithm_id, plan->split_k, plan->workspace_bytes);
}

void ShapePlan::run(const void *input, const uint8_t *host_mask, float *host_output) {
    size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * (context->f16 ? sizeof(half) : sizeof(float));
    size_t mask_bytes = static_cast<size_t>(batch) * seq;
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    if (context->graphs_enabled) FAMILY_CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    else if (context->f16) compute<half>();
    else compute<float>();
    FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_output, output.pointer, output.count * sizeof(float), cudaMemcpyDeviceToHost, context->stream));
    FAMILY_CUDA_CHECK(cudaStreamSynchronize(context->stream));
}

}  // namespace

extern "C" {

void *synapse_cuda_modernbert_context_new(int32_t graphs_enabled, int32_t precision) {
    try {
        return new ModernContext(graphs_enabled != 0, precision != 0);
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return nullptr;
    }
}

void synapse_cuda_modernbert_context_free(void *raw_context) {
    delete static_cast<ModernContext *>(raw_context);
}

int32_t synapse_cuda_modernbert_forward(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    float global_theta,
    float local_theta,
    uint64_t local_half_window,
    const void *input,
    const uint8_t *attention_mask,
    const ModernBertLayerParams *layers,
    const float *final_norm,
    float *output
) {
    try {
        if (!raw_context || !input || !attention_mask || !layers || !final_norm || !output) throw std::runtime_error("ModernBERT CUDA received a null pointer");
        if (!batch || !seq || !hidden || !heads || hidden % heads || !layer_count) throw std::runtime_error("ModernBERT CUDA received invalid dimensions");
        ModernContext *context = static_cast<ModernContext *>(raw_context);
        if (context->f16) context->load_weights<half>(hidden, intermediate, layer_count, layers, final_norm);
        else context->load_weights<float>(hidden, intermediate, layer_count, layers, final_norm);
        std::string key = shape_key(batch, seq);
        auto found = context->plans.find(key);
        if (found == context->plans.end()) {
            auto plan = std::make_unique<ShapePlan>(context, batch, seq, hidden, heads, intermediate, layer_count, epsilon, global_theta, local_theta, local_half_window);
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
