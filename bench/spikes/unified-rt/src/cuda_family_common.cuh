#ifndef SYNAPSE_CUDA_FAMILY_COMMON_CUH
#define SYNAPSE_CUDA_FAMILY_COMMON_CUH

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
typedef struct SynapseCudaWeight {
    const float *fp32;
    const uint8_t *q8_0;
} SynapseCudaWeight;
}

namespace synapse_cuda_family {

inline void cuda_check(cudaError_t status, const char *call, const char *file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << call << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

inline void cublas_check(cublasStatus_t status, const char *call, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << call << " failed at " << file << ':' << line << " with cuBLAS status " << status;
        throw std::runtime_error(message.str());
    }
}

#define FAMILY_CUDA_CHECK(call) ::synapse_cuda_family::cuda_check((call), #call, __FILE__, __LINE__)
#define FAMILY_CUBLAS_CHECK(call) ::synapse_cuda_family::cublas_check((call), #call, __FILE__, __LINE__)

template <typename T>
struct DeviceAllocation {
    T *pointer = nullptr;
    size_t count = 0;

    DeviceAllocation() = default;
    explicit DeviceAllocation(size_t elements) { allocate(elements); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept { *this = std::move(other); }
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
        if (count != 0) FAMILY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)));
    }
    void reset() {
        if (pointer != nullptr) cudaFree(pointer);
        pointer = nullptr;
        count = 0;
    }
};

inline __device__ float warp_sum(float value);

constexpr int Q8_0_BLOCK_ELEMENTS = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;

struct DeviceMatrix {
    DeviceAllocation<float> fp32;
    DeviceAllocation<uint8_t> q8_0;
    bool quantized = false;
};

static __global__ void dequantize_q8_0(
    const uint8_t *weights,
    float *output,
    size_t elements
) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    size_t block_index = index / Q8_0_BLOCK_ELEMENTS;
    int lane = index % Q8_0_BLOCK_ELEMENTS;
    const uint8_t *block = weights + block_index * Q8_0_BLOCK_BYTES;
    half scale = *reinterpret_cast<const half *>(block);
    int8_t quantized = reinterpret_cast<const int8_t *>(block + sizeof(half))[lane];
    output[index] = __half2float(scale) * static_cast<float>(quantized);
}

static __global__ void q8_0_matvec(
    const uint8_t *weights,
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
    float partial = 0.0f;
    for (int column_block = warp; column_block < blocks_per_row; column_block += warps) {
        const uint8_t *block = weights
            + (static_cast<size_t>(row) * blocks_per_row + column_block) * Q8_0_BLOCK_BYTES;
        half scale = *reinterpret_cast<const half *>(block);
        int8_t quantized = reinterpret_cast<const int8_t *>(block + sizeof(half))[lane];
        partial += __half2float(scale) * static_cast<float>(quantized)
            * input[column_block * Q8_0_BLOCK_ELEMENTS + lane];
    }
    partial = warp_sum(partial);
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < warps ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) output[row] = sum;
    }
}

static __global__ void q8_0_matvec_2row(
    const uint8_t *weights,
    const float *input,
    float *output,
    int rows,
    int columns
) {
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int blocks_per_row = columns / Q8_0_BLOCK_ELEMENTS;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (int column_block = warp; column_block < blocks_per_row; column_block += warps) {
        float in = input[column_block * Q8_0_BLOCK_ELEMENTS + lane];
        const uint8_t *block0 = weights
            + (static_cast<size_t>(row0) * blocks_per_row + column_block) * Q8_0_BLOCK_BYTES;
        half scale0 = *reinterpret_cast<const half *>(block0);
        int8_t quantized0 = reinterpret_cast<const int8_t *>(block0 + sizeof(half))[lane];
        partial0 += __half2float(scale0) * static_cast<float>(quantized0) * in;
        const uint8_t *block1 = weights
            + (static_cast<size_t>(row1) * blocks_per_row + column_block) * Q8_0_BLOCK_BYTES;
        half scale1 = *reinterpret_cast<const half *>(block1);
        int8_t quantized1 = reinterpret_cast<const int8_t *>(block1 + sizeof(half))[lane];
        partial1 += __half2float(scale1) * static_cast<float>(quantized1) * in;
    }
    partial0 = warp_sum(partial0);
    partial1 = warp_sum(partial1);
    __shared__ float warp_sums0[32];
    __shared__ float warp_sums1[32];
    if (lane == 0) {
        warp_sums0[warp] = partial0;
        warp_sums1[warp] = partial1;
    }
    __syncthreads();
    if (warp == 0) {
        float sum0 = lane < warps ? warp_sums0[lane] : 0.0f;
        float sum1 = lane < warps ? warp_sums1[lane] : 0.0f;
        sum0 = warp_sum(sum0);
        sum1 = warp_sum(sum1);
        if (lane == 0) {
            output[row0] = sum0;
            output[row1] = sum1;
        }
    }
}

inline void copy_matrix(DeviceMatrix &target, const SynapseCudaWeight &source, size_t elements) {
    if (!source.fp32) throw std::runtime_error("CUDA matrix is missing its fp32 source");
    target.quantized = source.q8_0 != nullptr;
    target.fp32.allocate(elements);
    if (target.quantized) {
        if (elements % Q8_0_BLOCK_ELEMENTS != 0) {
            throw std::runtime_error("Q8_0 CUDA matrix element count is not block aligned");
        }
        size_t bytes = elements / Q8_0_BLOCK_ELEMENTS * Q8_0_BLOCK_BYTES;
        target.q8_0.allocate(bytes);
        FAMILY_CUDA_CHECK(cudaMemcpy(
            target.q8_0.pointer,
            source.q8_0,
            bytes,
            cudaMemcpyHostToDevice
        ));
        int threads = 256;
        dequantize_q8_0<<<(elements + threads - 1) / threads, threads>>>(
            target.q8_0.pointer,
            target.fp32.pointer,
            elements
        );
        FAMILY_CUDA_CHECK(cudaGetLastError());
    } else {
        FAMILY_CUDA_CHECK(cudaMemcpy(
            target.fp32.pointer,
            source.fp32,
            elements * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
}

inline void launch_decode_matvec(
    const DeviceMatrix &weight,
    const float *input,
    float *output,
    int rows,
    int columns,
    cudaStream_t stream
) {
    if (!weight.quantized) {
        throw std::runtime_error("fused decode matvec was requested for an fp32 matrix");
    }
    if (rows >= 2 && (rows & 1) == 0) {
        int blocks = rows / 2;
        q8_0_matvec_2row<<<blocks, 256, 0, stream>>>(weight.q8_0.pointer, input, output, rows, columns);
    } else {
        q8_0_matvec<<<rows, 256, 0, stream>>>(weight.q8_0.pointer, input, output, rows, columns);
    }
}

inline std::vector<half> to_half(const float *values, size_t count) {
    std::vector<half> converted(count);
    for (size_t index = 0; index < count; ++index) converted[index] = __float2half(values[index]);
    return converted;
}

template <typename T>
void copy_weight(DeviceAllocation<unsigned char> &target, const float *source, size_t count);

template <>
inline void copy_weight<float>(DeviceAllocation<unsigned char> &target, const float *source, size_t count) {
    target.allocate(count * sizeof(float));
    FAMILY_CUDA_CHECK(cudaMemcpy(target.pointer, source, count * sizeof(float), cudaMemcpyHostToDevice));
}

template <>
inline void copy_weight<half>(DeviceAllocation<unsigned char> &target, const float *source, size_t count) {
    target.allocate(count * sizeof(half));
    std::vector<half> converted = to_half(source, count);
    FAMILY_CUDA_CHECK(cudaMemcpy(target.pointer, converted.data(), count * sizeof(half), cudaMemcpyHostToDevice));
}

inline void copy_float(DeviceAllocation<float> &target, const float *source, size_t count) {
    target.allocate(count);
    FAMILY_CUDA_CHECK(cudaMemcpy(target.pointer, source, count * sizeof(float), cudaMemcpyHostToDevice));
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
    int split_k = 1;

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
            split_k = other.split_k;
            other.operation = nullptr;
            other.a_layout = other.b_layout = other.c_layout = nullptr;
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

inline MatmulPlan select_matmul(
    cublasLtHandle_t handle,
    const char *name,
    cudaDataType_t dtype,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transpose_b,
    int batch_count = 1,
    int64_t a_stride = 0,
    int64_t b_stride = 0,
    int64_t c_stride = 0
) {
    MatmulPlan plan;
    plan.name = name;
    FAMILY_CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op_a = CUBLAS_OP_N;
    cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    FAMILY_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    FAMILY_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    const int64_t b_rows = transpose_b ? n : k;
    const int64_t b_cols = transpose_b ? k : n;
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a_layout, dtype, m, k, k));
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b_layout, dtype, b_rows, b_cols, b_cols));
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c_layout, dtype, m, n, n));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    if (batch_count > 1) {
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride)));
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride)));
        FAMILY_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan.c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)));
    }
    constexpr size_t workspace_limit = 64 * 1024 * 1024;
    cublasLtMatmulPreference_t preference = nullptr;
    FAMILY_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    FAMILY_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, plan.operation, plan.a_layout, plan.b_layout, plan.c_layout, plan.c_layout, preference, 1, &result, &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    FAMILY_CUBLAS_CHECK(status);
    if (returned != 1 || result.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("no cuBLASLt algorithm for ") + name);
    plan.algorithm = result.algo;
    plan.workspace_bytes = result.workspaceSize;
    FAMILY_CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&plan.algorithm, CUBLASLT_ALGO_CONFIG_ID, &plan.algorithm_id, sizeof(plan.algorithm_id), nullptr));
    FAMILY_CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&plan.algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &plan.split_k, sizeof(plan.split_k), nullptr));
    return plan;
}

inline void launch_matmul(cublasLtHandle_t handle, const MatmulPlan &plan, const void *a, const void *b, void *c, void *workspace, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    FAMILY_CUBLAS_CHECK(cublasLtMatmul(handle, plan.operation, &alpha, a, plan.a_layout, b, plan.b_layout, &beta, c, plan.c_layout, c, plan.c_layout, &plan.algorithm, workspace, plan.workspace_bytes, stream));
}

inline __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

inline __device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

template <typename T> inline __device__ float load_value(const T *values, int index);
template <> inline __device__ float load_value<half>(const half *values, int index) { return __half2float(values[index]); }
template <> inline __device__ float load_value<float>(const float *values, int index) { return values[index]; }
template <typename T> inline __device__ void store_value(T *values, int index, float value);
template <> inline __device__ void store_value<half>(half *values, int index, float value) { values[index] = __float2half(value); }
template <> inline __device__ void store_value<float>(float *values, int index, float value) { values[index] = value; }

struct StageEvent {
    std::string name;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

struct StageProfile {
    std::vector<StageEvent> events;

    void begin(const char *name, cudaStream_t stream) {
        StageEvent event;
        event.name = name;
        FAMILY_CUDA_CHECK(cudaEventCreate(&event.start));
        FAMILY_CUDA_CHECK(cudaEventCreate(&event.stop));
        FAMILY_CUDA_CHECK(cudaEventRecord(event.start, stream));
        events.push_back(std::move(event));
    }

    void end(cudaStream_t stream) {
        FAMILY_CUDA_CHECK(cudaEventRecord(events.back().stop, stream));
    }

    std::unordered_map<std::string, double> collect() {
        std::unordered_map<std::string, double> totals;
        for (StageEvent &event : events) {
            float milliseconds = 0.0f;
            FAMILY_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, event.start, event.stop));
            totals[event.name] += milliseconds;
            cudaEventDestroy(event.start);
            cudaEventDestroy(event.stop);
        }
        events.clear();
        return totals;
    }
};

inline std::string shape_key(int batch, int seq) {
    return std::to_string(batch) + "x" + std::to_string(seq);
}

}  // namespace synapse_cuda_family

extern "C" void synapse_cuda_set_last_error(const char *message);

#endif
