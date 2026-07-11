#include "cuda_minilm.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) cublas_check((call), #call, __FILE__, __LINE__)

void cuda_check(cudaError_t status, const char *call, const char *file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << call << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void cublas_check(cublasStatus_t status, const char *call, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << call << " failed at " << file << ':' << line << " with cuBLAS status " << status;
        throw std::runtime_error(message.str());
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

template <typename T>
struct DeviceAllocation {
    T *pointer = nullptr;
    size_t count = 0;

    DeviceAllocation() = default;
    explicit DeviceAllocation(size_t elements) { allocate(elements); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept : pointer(other.pointer), count(other.count) {
        other.pointer = nullptr;
        other.count = 0;
    }
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this != &other) {
            reset();
            pointer = other.pointer;
            count = other.count;
            other.pointer = nullptr;
            other.count = 0;
        }
        return *this;
    }
    ~DeviceAllocation() { reset(); }

    void allocate(size_t elements) {
        reset();
        count = elements;
        if (count != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)));
    }
    void reset() {
        if (pointer != nullptr) cudaFree(pointer);
        pointer = nullptr;
        count = 0;
    }
};

struct DeviceLayer {
    DeviceAllocation<half> query_weight, query_bias;
    DeviceAllocation<half> key_weight, key_bias;
    DeviceAllocation<half> value_weight, value_bias;
    DeviceAllocation<half> attention_output_weight, attention_output_bias;
    DeviceAllocation<float> attention_ln_weight, attention_ln_bias;
    DeviceAllocation<half> intermediate_weight, intermediate_bias;
    DeviceAllocation<half> output_weight, output_bias;
    DeviceAllocation<float> output_ln_weight, output_ln_bias;
};

std::vector<half> to_half(const float *values, size_t count) {
    std::vector<half> converted(count);
    for (size_t index = 0; index < count; ++index) converted[index] = __float2half(values[index]);
    return converted;
}

void copy_half(DeviceAllocation<half> &target, const float *source, size_t count) {
    target.allocate(count);
    std::vector<half> converted = to_half(source, count);
    CUDA_CHECK(cudaMemcpy(target.pointer, converted.data(), count * sizeof(half), cudaMemcpyHostToDevice));
}

void copy_float(DeviceAllocation<float> &target, const float *source, size_t count) {
    target.allocate(count);
    CUDA_CHECK(cudaMemcpy(target.pointer, source, count * sizeof(float), cudaMemcpyHostToDevice));
}

struct MatmulPlan {
    std::string name;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;
    int algorithm_id = -1;

    MatmulPlan() = default;
    MatmulPlan(const MatmulPlan &) = delete;
    MatmulPlan &operator=(const MatmulPlan &) = delete;
    MatmulPlan(MatmulPlan &&other) noexcept { *this = std::move(other); }
    MatmulPlan &operator=(MatmulPlan &&other) noexcept {
        if (this != &other) {
            release();
            name = std::move(other.name);
            operation = other.operation;
            a_layout = other.a_layout;
            b_layout = other.b_layout;
            c_layout = other.c_layout;
            algorithm = other.algorithm;
            workspace_bytes = other.workspace_bytes;
            algorithm_id = other.algorithm_id;
            other.operation = nullptr;
            other.a_layout = nullptr;
            other.b_layout = nullptr;
            other.c_layout = nullptr;
        }
        return *this;
    }
    ~MatmulPlan() { release(); }

    void release() {
        if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
        if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
        if (operation) cublasLtMatmulDescDestroy(operation);
        operation = nullptr;
        a_layout = b_layout = c_layout = nullptr;
    }
};

MatmulPlan select_matmul(
    cublasLtHandle_t handle,
    const char *name,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transpose_b,
    int batch_count,
    int64_t a_stride,
    int64_t b_stride,
    int64_t c_stride,
    size_t workspace_limit
) {
    MatmulPlan plan;
    plan.name = name;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op_a = CUBLAS_OP_N;
    cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    const int64_t b_rows = transpose_b ? n : k;
    const int64_t b_cols = transpose_b ? k : n;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a_layout, CUDA_R_16F, m, k, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b_layout, CUDA_R_16F, b_rows, b_cols, b_cols));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c_layout, CUDA_R_16F, m, n, n));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    if (batch_count > 1) {
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)));
    }

    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit,
        sizeof(workspace_limit)
    ));
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        plan.operation,
        plan.a_layout,
        plan.b_layout,
        plan.c_layout,
        plan.c_layout,
        preference,
        1,
        &result,
        &returned
    );
    cublasLtMatmulPreferenceDestroy(preference);
    CUBLAS_CHECK(status);
    if (returned != 1 || result.state != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("no cuBLASLt algorithm for ") + name);
    }
    plan.algorithm = result.algo;
    plan.workspace_bytes = result.workspaceSize;
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
        &plan.algorithm,
        CUBLASLT_ALGO_CONFIG_ID,
        &plan.algorithm_id,
        sizeof(plan.algorithm_id),
        nullptr
    ));
    return plan;
}

void launch_matmul(
    cublasLtHandle_t handle,
    const MatmulPlan &plan,
    const half *a,
    const half *b,
    half *c,
    void *workspace,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        handle,
        plan.operation,
        &alpha,
        a,
        plan.a_layout,
        b,
        plan.b_layout,
        &beta,
        c,
        plan.c_layout,
        c,
        plan.c_layout,
        &plan.algorithm,
        workspace,
        plan.workspace_bytes,
        stream
    ));
}

__global__ void qkv_bias_transpose(
    const half *q_raw,
    const half *k_raw,
    const half *v_raw,
    const half *q_bias,
    const half *k_bias,
    const half *v_bias,
    half *q,
    half *k,
    half *v,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * head_dim;
    if (index >= total) return;
    int d = index % head_dim;
    int remainder = index / head_dim;
    int s = remainder % seq;
    remainder /= seq;
    int head = remainder % heads;
    int b = remainder / heads;
    int hidden_index = head * head_dim + d;
    int raw_index = (b * seq + s) * (heads * head_dim) + hidden_index;
    q[index] = __float2half(__half2float(q_raw[raw_index]) + __half2float(q_bias[hidden_index]));
    k[index] = __float2half(__half2float(k_raw[raw_index]) + __half2float(k_bias[hidden_index]));
    v[index] = __float2half(__half2float(v_raw[raw_index]) + __half2float(v_bias[hidden_index]));
}

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__inline__ __device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

__global__ void scale_mask_softmax(
    half *scores,
    const uint8_t *mask,
    int batch,
    int heads,
    int seq,
    float scale
) {
    int row = blockIdx.x;
    int query_batch = row / (heads * seq);
    int base = row * seq;
    float local_max = -FLT_MAX;
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        float value = mask[query_batch * seq + key] ? __half2float(scores[base + key]) * scale : -10000.0f;
        local_max = fmaxf(local_max, value);
    }
    local_max = warp_max(local_max);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = local_max;
    __syncthreads();
    float maximum = threadIdx.x < blockDim.x / 32 ? reductions[lane] : -FLT_MAX;
    if (warp == 0) maximum = warp_max(maximum);
    if (threadIdx.x == 0) reductions[0] = maximum;
    __syncthreads();
    maximum = reductions[0];

    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        float value = mask[query_batch * seq + key] ? __half2float(scores[base + key]) * scale : -10000.0f;
        float exponential = expf(value - maximum);
        scores[base + key] = __float2half(exponential);
        local_sum += exponential;
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) reductions[warp] = local_sum;
    __syncthreads();
    float sum = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) reductions[0] = sum;
    __syncthreads();
    float inverse = 1.0f / fmaxf(reductions[0], 1e-20f);
    for (int key = threadIdx.x; key < seq; key += blockDim.x) {
        scores[base + key] = __float2half(__half2float(scores[base + key]) * inverse);
    }
}

__global__ void transpose_context(
    const half *source,
    half *target,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * heads * head_dim;
    if (index >= total) return;
    int d = index % head_dim;
    int remainder = index / head_dim;
    int head = remainder % heads;
    remainder /= heads;
    int s = remainder % seq;
    int b = remainder / seq;
    int source_index = ((b * heads + head) * seq + s) * head_dim + d;
    target[index] = source[source_index];
}

__global__ void bias_gelu(half *values, const half *bias, int rows, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (index >= total) return;
    float value = __half2float(values[index]) + __half2float(bias[index % width]);
    value = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    values[index] = __float2half(value);
}

__global__ void residual_layer_norm(
    const half *projected,
    const half *bias,
    const half *residual,
    const float *weight,
    const float *norm_bias,
    half *output,
    int hidden,
    float epsilon
) {
    int row = blockIdx.x;
    float local_sum = 0.0f;
    float local_square = 0.0f;
    int base = row * hidden;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = __half2float(projected[base + column]) + __half2float(bias[column]) + __half2float(residual[base + column]);
        local_sum += value;
        local_square += value * value;
    }
    local_sum = warp_sum(local_sum);
    local_square = warp_sum(local_square);
    __shared__ float sums[32];
    __shared__ float squares[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sums[warp] = local_sum;
        squares[warp] = local_square;
    }
    __syncthreads();
    float sum = threadIdx.x < blockDim.x / 32 ? sums[lane] : 0.0f;
    float square = threadIdx.x < blockDim.x / 32 ? squares[lane] : 0.0f;
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
        float value = __half2float(projected[base + column]) + __half2float(bias[column]) + __half2float(residual[base + column]);
        output[base + column] = __float2half((value - mean) * inverse * weight[column] + norm_bias[column]);
    }
}

__global__ void mean_pool_l2(
    const half *hidden_states,
    const uint8_t *mask,
    float *pooled,
    int seq,
    int hidden
) {
    int batch = blockIdx.x;
    float local_norm = 0.0f;
    int count = 0;
    for (int token = 0; token < seq; ++token) count += mask[batch * seq + token] != 0;
    count = max(count, 1);
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float sum = 0.0f;
        for (int token = 0; token < seq; ++token) {
            if (mask[batch * seq + token]) {
                sum += __half2float(hidden_states[(batch * seq + token) * hidden + column]);
            }
        }
        float value = sum / count;
        pooled[batch * hidden + column] = value;
        local_norm += value * value;
    }
    local_norm = warp_sum(local_norm);
    __shared__ float reductions[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) reductions[warp] = local_norm;
    __syncthreads();
    float norm = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0f;
    if (warp == 0) norm = warp_sum(norm);
    if (threadIdx.x == 0) reductions[0] = rsqrtf(fmaxf(norm, 1e-24f));
    __syncthreads();
    float inverse = reductions[0];
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        pooled[batch * hidden + column] *= inverse;
    }
}

struct StageEvent {
    const char *name;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

struct StageProfile {
    std::vector<StageEvent> events;
    void begin(const char *name, cudaStream_t stream) {
        StageEvent event{name};
        CUDA_CHECK(cudaEventCreate(&event.start));
        CUDA_CHECK(cudaEventCreate(&event.stop));
        CUDA_CHECK(cudaEventRecord(event.start, stream));
        events.push_back(event);
    }
    void end(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(events.back().stop, stream)); }
    std::unordered_map<std::string, double> collect() {
        std::unordered_map<std::string, double> totals;
        for (StageEvent &event : events) {
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, event.start, event.stop));
            totals[event.name] += milliseconds;
            cudaEventDestroy(event.start);
            cudaEventDestroy(event.stop);
        }
        events.clear();
        return totals;
    }
};

struct CudaContext;

struct ShapePlan {
    CudaContext *context;
    int batch, seq, hidden, heads, head_dim, intermediate, layer_count;
    float epsilon;
    size_t arena_bytes = 0;
    DeviceAllocation<unsigned char> arena;
    DeviceAllocation<unsigned char> workspace;
    DeviceAllocation<uint8_t> mask;
    DeviceAllocation<float> pooled;
    half *x0 = nullptr;
    half *x1 = nullptr;
    half *q_raw = nullptr;
    half *k_raw = nullptr;
    half *v_raw = nullptr;
    half *q = nullptr;
    half *k = nullptr;
    half *v = nullptr;
    half *scores = nullptr;
    half *attention = nullptr;
    half *context_rows = nullptr;
    half *projected = nullptr;
    half *intermediate_values = nullptr;
    half *ffn_output = nullptr;
    MatmulPlan hidden_hidden;
    MatmulPlan hidden_intermediate;
    MatmulPlan intermediate_hidden;
    MatmulPlan qk;
    MatmulPlan pv;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    double algo_selection_ms = 0.0;
    double warmup_ms = 0.0;
    double capture_ms = 0.0;
    double instantiate_ms = 0.0;
    double first_launch_ms = 0.0;
    std::unordered_map<std::string, double> stage_ms;

    ShapePlan(CudaContext *owner, int b, int s, int h, int nh, int i, int layers, float eps);
    ~ShapePlan();
    void compute(StageProfile *profile);
    void initialize_and_verify(const uint16_t *input, const uint8_t *host_mask);
    void run(const uint16_t *input, const uint8_t *host_mask, float *output, bool graph_enabled);
};

struct CudaContext {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    bool graphs_enabled = true;
    bool weights_loaded = false;
    int hidden = 0;
    int intermediate = 0;
    int layer_count = 0;
    std::vector<DeviceLayer> layers;
    std::unordered_map<std::string, std::unique_ptr<ShapePlan>> plans;
    double context_init_ms = 0.0;
    double handle_init_ms = 0.0;
    double weight_upload_ms = 0.0;

    explicit CudaContext(bool graphs) : graphs_enabled(graphs) {
        auto started = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaFree(nullptr));
        context_init_ms = elapsed_ms(started);
        started = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasLtCreate(&lt));
        handle_init_ms = elapsed_ms(started);
        std::fprintf(stderr, "CUDA init: context=%.3fms cublaslt_handle=%.3fms cublaslt_version=%zu\n", context_init_ms, handle_init_ms, cublasLtGetVersion());
    }
    ~CudaContext() {
        plans.clear();
        if (lt) cublasLtDestroy(lt);
        if (stream) cudaStreamDestroy(stream);
    }

    void load_weights(int h, int inter, int count, const SynapseCudaEncoderLayerParams *host_layers) {
        if (weights_loaded) {
            if (hidden != h || intermediate != inter || layer_count != count) throw std::runtime_error("CUDA context model dimensions changed");
            return;
        }
        auto started = std::chrono::steady_clock::now();
        hidden = h;
        intermediate = inter;
        layer_count = count;
        layers.resize(count);
        for (int index = 0; index < count; ++index) {
            DeviceLayer &target = layers[index];
            const SynapseCudaEncoderLayerParams &source = host_layers[index];
            copy_half(target.query_weight, source.query_weight, h * h);
            copy_half(target.query_bias, source.query_bias, h);
            copy_half(target.key_weight, source.key_weight, h * h);
            copy_half(target.key_bias, source.key_bias, h);
            copy_half(target.value_weight, source.value_weight, h * h);
            copy_half(target.value_bias, source.value_bias, h);
            copy_half(target.attention_output_weight, source.attention_output_weight, h * h);
            copy_half(target.attention_output_bias, source.attention_output_bias, h);
            copy_float(target.attention_ln_weight, source.attention_ln_weight, h);
            copy_float(target.attention_ln_bias, source.attention_ln_bias, h);
            copy_half(target.intermediate_weight, source.intermediate_weight, inter * h);
            copy_half(target.intermediate_bias, source.intermediate_bias, inter);
            copy_half(target.output_weight, source.output_weight, h * inter);
            copy_half(target.output_bias, source.output_bias, h);
            copy_float(target.output_ln_weight, source.output_ln_weight, h);
            copy_float(target.output_ln_bias, source.output_ln_bias, h);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        weight_upload_ms = elapsed_ms(started);
        weights_loaded = true;
        std::fprintf(stderr, "CUDA persistent weights: upload=%.3fms layers=%d hidden=%d intermediate=%d norm_params=fp32\n", weight_upload_ms, count, h, inter);
    }
};

ShapePlan::ShapePlan(CudaContext *owner, int b, int s, int h, int nh, int i, int layers_count, float eps)
    : context(owner), batch(b), seq(s), hidden(h), heads(nh), head_dim(h / nh), intermediate(i), layer_count(layers_count), epsilon(eps) {
    const size_t rows = static_cast<size_t>(batch) * seq;
    const size_t hidden_values = rows * hidden;
    const size_t head_values = hidden_values;
    const size_t score_values = static_cast<size_t>(batch) * heads * seq * seq;
    const size_t intermediate_count = rows * intermediate;
    const size_t half_count = hidden_values * 10 + head_values * 4 + score_values + intermediate_count;
    arena_bytes = half_count * sizeof(half) + 16 * 256;
    arena.allocate(arena_bytes);
    mask.allocate(rows);
    pooled.allocate(static_cast<size_t>(batch) * hidden);
    unsigned char *cursor = arena.pointer;
    auto take = [&cursor](size_t count) {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(cursor) + 255) & ~uintptr_t(255);
        cursor = reinterpret_cast<unsigned char *>(aligned);
        half *result = reinterpret_cast<half *>(cursor);
        cursor += count * sizeof(half);
        return result;
    };
    x0 = take(hidden_values);
    x1 = take(hidden_values);
    q_raw = take(hidden_values);
    k_raw = take(hidden_values);
    v_raw = take(hidden_values);
    q = take(head_values);
    k = take(head_values);
    v = take(head_values);
    scores = take(score_values);
    attention = take(head_values);
    context_rows = take(hidden_values);
    projected = take(hidden_values);
    intermediate_values = take(intermediate_count);
    ffn_output = take(hidden_values);

    auto started = std::chrono::steady_clock::now();
    constexpr size_t workspace_limit = 64 * 1024 * 1024;
    hidden_hidden = select_matmul(owner->lt, "hidden_hidden", rows, hidden, hidden, true, 1, 0, 0, 0, workspace_limit);
    hidden_intermediate = select_matmul(owner->lt, "hidden_intermediate", rows, intermediate, hidden, true, 1, 0, 0, 0, workspace_limit);
    intermediate_hidden = select_matmul(owner->lt, "intermediate_hidden", rows, hidden, intermediate, true, 1, 0, 0, 0, workspace_limit);
    int batch_heads = batch * heads;
    qk = select_matmul(owner->lt, "qk", seq, seq, head_dim, true, batch_heads, seq * head_dim, seq * head_dim, seq * seq, workspace_limit);
    pv = select_matmul(owner->lt, "pv", seq, head_dim, seq, false, batch_heads, seq * seq, seq * head_dim, seq * head_dim, workspace_limit);
    algo_selection_ms = elapsed_ms(started);
    size_t max_workspace = std::max({hidden_hidden.workspace_bytes, hidden_intermediate.workspace_bytes, intermediate_hidden.workspace_bytes, qk.workspace_bytes, pv.workspace_bytes});
    workspace.allocate(max_workspace);
}

ShapePlan::~ShapePlan() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
}

void ShapePlan::compute(StageProfile *profile) {
    const int rows = batch * seq;
    const int threads = 256;
    const int hidden_blocks = (rows * hidden + threads - 1) / threads;
    const int intermediate_blocks = (rows * intermediate + threads - 1) / threads;
    const int head_blocks = (batch * heads * seq * head_dim + threads - 1) / threads;
    for (int index = 0; index < layer_count; ++index) {
        DeviceLayer &layer = context->layers[index];
        if (profile) profile->begin("gemm", context->stream);
        launch_matmul(context->lt, hidden_hidden, x0, layer.query_weight.pointer, q_raw, workspace.pointer, context->stream);
        launch_matmul(context->lt, hidden_hidden, x0, layer.key_weight.pointer, k_raw, workspace.pointer, context->stream);
        launch_matmul(context->lt, hidden_hidden, x0, layer.value_weight.pointer, v_raw, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("pointwise", context->stream);
        qkv_bias_transpose<<<head_blocks, threads, 0, context->stream>>>(q_raw, k_raw, v_raw, layer.query_bias.pointer, layer.key_bias.pointer, layer.value_bias.pointer, q, k, v, batch, seq, heads, head_dim);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("attention_gemm", context->stream);
        launch_matmul(context->lt, qk, q, k, scores, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("score_softmax", context->stream);
        scale_mask_softmax<<<batch * heads * seq, threads, 0, context->stream>>>(scores, mask.pointer, batch, heads, seq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("attention_gemm", context->stream);
        launch_matmul(context->lt, pv, scores, v, attention, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("pointwise", context->stream);
        transpose_context<<<hidden_blocks, threads, 0, context->stream>>>(attention, context_rows, batch, seq, heads, head_dim);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("gemm", context->stream);
        launch_matmul(context->lt, hidden_hidden, context_rows, layer.attention_output_weight.pointer, projected, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("pointwise", context->stream);
        residual_layer_norm<<<rows, threads, 0, context->stream>>>(projected, layer.attention_output_bias.pointer, x0, layer.attention_ln_weight.pointer, layer.attention_ln_bias.pointer, x1, hidden, epsilon);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("gemm", context->stream);
        launch_matmul(context->lt, hidden_intermediate, x1, layer.intermediate_weight.pointer, intermediate_values, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("pointwise", context->stream);
        bias_gelu<<<intermediate_blocks, threads, 0, context->stream>>>(intermediate_values, layer.intermediate_bias.pointer, rows, intermediate);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("gemm", context->stream);
        launch_matmul(context->lt, intermediate_hidden, intermediate_values, layer.output_weight.pointer, ffn_output, workspace.pointer, context->stream);
        if (profile) profile->end(context->stream);
        if (profile) profile->begin("pointwise", context->stream);
        residual_layer_norm<<<rows, threads, 0, context->stream>>>(ffn_output, layer.output_bias.pointer, x1, layer.output_ln_weight.pointer, layer.output_ln_bias.pointer, x0, hidden, epsilon);
        if (profile) profile->end(context->stream);
    }
    if (profile) profile->begin("pool_l2", context->stream);
    mean_pool_l2<<<batch, 256, 0, context->stream>>>(x0, mask.pointer, pooled.pointer, seq, hidden);
    if (profile) profile->end(context->stream);
    CUDA_CHECK(cudaGetLastError());
}

void ShapePlan::initialize_and_verify(const uint16_t *input, const uint8_t *host_mask) {
    const size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * sizeof(half);
    const size_t mask_bytes = static_cast<size_t>(batch) * seq;
    CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    StageProfile profile;
    auto started = std::chrono::steady_clock::now();
    compute(&profile);
    CUDA_CHECK(cudaStreamSynchronize(context->stream));
    warmup_ms = elapsed_ms(started);
    stage_ms = profile.collect();
    std::vector<float> uncaptured(static_cast<size_t>(batch) * hidden);
    CUDA_CHECK(cudaMemcpy(uncaptured.data(), pooled.pointer, uncaptured.size() * sizeof(float), cudaMemcpyDeviceToHost));

    started = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeThreadLocal));
    compute(nullptr);
    CUDA_CHECK(cudaStreamEndCapture(context->stream, &graph));
    capture_ms = elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    instantiate_ms = elapsed_ms(started);
    CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    started = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    CUDA_CHECK(cudaStreamSynchronize(context->stream));
    first_launch_ms = elapsed_ms(started);
    std::vector<float> captured(uncaptured.size());
    CUDA_CHECK(cudaMemcpy(captured.data(), pooled.pointer, captured.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(uncaptured.data(), captured.data(), captured.size() * sizeof(float)) != 0) {
        float maximum = 0.0f;
        for (size_t index = 0; index < captured.size(); ++index) maximum = std::max(maximum, std::abs(captured[index] - uncaptured[index]));
        std::ostringstream message;
        message << "captured output differs from uncaptured output; max_abs=" << maximum;
        throw std::runtime_error(message.str());
    }

    const size_t max_workspace = workspace.count;
    std::fprintf(
        stderr,
        "CUDA shape %dx%d: arena=%zu workspace=%zu algo_select=%.3fms warmup=%.3fms capture=%.3fms instantiate=%.3fms first_launch=%.3fms captured_exact=true launches=%d stage_gemm=%.3fms stage_attention_gemm=%.3fms stage_score_softmax=%.3fms stage_pointwise=%.3fms stage_pool_l2=%.3fms\n",
        batch, seq, arena_bytes, max_workspace, algo_selection_ms, warmup_ms, capture_ms, instantiate_ms, first_launch_ms,
        layer_count * 14 + 1, stage_ms["gemm"], stage_ms["attention_gemm"], stage_ms["score_softmax"], stage_ms["pointwise"], stage_ms["pool_l2"]
    );
    const MatmulPlan *plans[] = {&hidden_hidden, &hidden_intermediate, &intermediate_hidden, &qk, &pv};
    for (const MatmulPlan *plan : plans) {
        std::fprintf(stderr, "CUDA algo %dx%d %s: id=%d workspace=%zu compute=CUBLAS_COMPUTE_32F operands=CUDA_R_16F\n", batch, seq, plan->name.c_str(), plan->algorithm_id, plan->workspace_bytes);
    }
}

void ShapePlan::run(const uint16_t *input, const uint8_t *host_mask, float *output, bool graph_enabled) {
    const size_t input_bytes = static_cast<size_t>(batch) * seq * hidden * sizeof(half);
    const size_t mask_bytes = static_cast<size_t>(batch) * seq;
    CUDA_CHECK(cudaMemcpyAsync(x0, input, input_bytes, cudaMemcpyHostToDevice, context->stream));
    CUDA_CHECK(cudaMemcpyAsync(mask.pointer, host_mask, mask_bytes, cudaMemcpyHostToDevice, context->stream));
    if (graph_enabled) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, context->stream));
    } else {
        compute(nullptr);
    }
    CUDA_CHECK(cudaMemcpyAsync(output, pooled.pointer, static_cast<size_t>(batch) * hidden * sizeof(float), cudaMemcpyDeviceToHost, context->stream));
    CUDA_CHECK(cudaStreamSynchronize(context->stream));
}

std::string shape_key(int batch, int seq) {
    return std::to_string(batch) + "x" + std::to_string(seq);
}

}  // namespace

extern "C" {

void *synapse_cuda_context_new(int32_t graphs_enabled) {
    try {
        last_error.clear();
        return new CudaContext(graphs_enabled != 0);
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    }
}

void synapse_cuda_context_free(void *raw_context) {
    delete static_cast<CudaContext *>(raw_context);
}

int32_t synapse_cuda_encoder_forward(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps,
    const uint16_t *input,
    const uint8_t *attention_mask,
    float *output,
    const SynapseCudaEncoderLayerParams *layers
) {
    try {
        last_error.clear();
        if (!raw_context || !input || !attention_mask || !output || !layers) throw std::runtime_error("CUDA encoder received a null pointer");
        if (hidden % heads != 0 || batch == 0 || seq == 0 || hidden == 0 || layer_count == 0) throw std::runtime_error("CUDA encoder received invalid dimensions");
        CudaContext *context = static_cast<CudaContext *>(raw_context);
        context->load_weights(static_cast<int>(hidden), static_cast<int>(intermediate), static_cast<int>(layer_count), layers);
        std::string key = shape_key(static_cast<int>(batch), static_cast<int>(seq));
        auto found = context->plans.find(key);
        if (found == context->plans.end()) {
            auto plan = std::make_unique<ShapePlan>(context, static_cast<int>(batch), static_cast<int>(seq), static_cast<int>(hidden), static_cast<int>(heads), static_cast<int>(intermediate), static_cast<int>(layer_count), layer_norm_eps);
            plan->initialize_and_verify(input, attention_mask);
            found = context->plans.emplace(key, std::move(plan)).first;
        }
        found->second->run(input, attention_mask, output, context->graphs_enabled);
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}

const char *synapse_cuda_last_error(void) {
    return last_error.c_str();
}

uint64_t synapse_cuda_cublaslt_version(void) {
    return static_cast<uint64_t>(cublasLtGetVersion());
}

}  // extern "C"
