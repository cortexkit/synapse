#include "cuda_family_common.cuh"

using namespace synapse_cuda_family;

namespace {

struct StaticWeight {
    DeviceAllocation<float> device;
};

struct OpsContext {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    DeviceAllocation<float> a, dynamic_b, c;
    DeviceAllocation<unsigned char> workspace;
    std::unordered_map<std::string, std::unique_ptr<MatmulPlan>> plans;
    std::unordered_map<std::string, StaticWeight> static_weights;

    OpsContext() {
        FAMILY_CUDA_CHECK(cudaFree(nullptr));
        FAMILY_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        FAMILY_CUBLAS_CHECK(cublasLtCreate(&lt));
        workspace.allocate(64 * 1024 * 1024);
    }

    ~OpsContext() {
        static_weights.clear();
        plans.clear();
        if (lt) cublasLtDestroy(lt);
        if (stream) cudaStreamDestroy(stream);
    }

    void ensure(DeviceAllocation<float> &allocation, size_t count) {
        if (allocation.count < count) allocation.allocate(count);
    }

    const MatmulPlan &plan(int m, int n, int k, bool transpose_b) {
        std::string key = std::to_string(m) + "x" + std::to_string(n) + "x" +
            std::to_string(k) + (transpose_b ? "t" : "n");
        auto found = plans.find(key);
        if (found == plans.end()) {
            auto value = std::make_unique<MatmulPlan>(
                select_matmul(lt, key.c_str(), CUDA_R_32F, m, n, k, transpose_b)
            );
            found = plans.emplace(key, std::move(value)).first;
        }
        return *found->second;
    }

    const float *static_weight(const float *host, size_t count) {
        std::string key = std::to_string(reinterpret_cast<uintptr_t>(host)) + ":" + std::to_string(count);
        auto found = static_weights.find(key);
        if (found == static_weights.end()) {
            StaticWeight weight;
            weight.device.allocate(count);
            FAMILY_CUDA_CHECK(cudaMemcpyAsync(
                weight.device.pointer,
                host,
                count * sizeof(float),
                cudaMemcpyHostToDevice,
                stream
            ));
            found = static_weights.emplace(key, std::move(weight)).first;
        }
        return found->second.device.pointer;
    }

    void matmul(
        int m,
        int n,
        int k,
        const float *host_a,
        const float *host_b,
        bool transpose_b,
        bool static_rhs,
        float *host_c
    ) {
        size_t a_count = static_cast<size_t>(m) * k;
        size_t b_count = static_cast<size_t>(n) * k;
        size_t c_count = static_cast<size_t>(m) * n;
        ensure(a, a_count);
        ensure(c, c_count);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(a.pointer, host_a, a_count * sizeof(float), cudaMemcpyHostToDevice, stream));
        const float *device_b = nullptr;
        if (static_rhs) {
            device_b = static_weight(host_b, b_count);
        } else {
            ensure(dynamic_b, b_count);
            FAMILY_CUDA_CHECK(cudaMemcpyAsync(dynamic_b.pointer, host_b, b_count * sizeof(float), cudaMemcpyHostToDevice, stream));
            device_b = dynamic_b.pointer;
        }
        launch_matmul(lt, plan(m, n, k, transpose_b), a.pointer, device_b, c.pointer, workspace.pointer, stream);
        FAMILY_CUDA_CHECK(cudaMemcpyAsync(host_c, c.pointer, c_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        FAMILY_CUDA_CHECK(cudaStreamSynchronize(stream));
        FAMILY_CUDA_CHECK(cudaGetLastError());
    }
};

}  // namespace

extern "C" {

void *synapse_cuda_ops_context_new() {
    try {
        return new OpsContext();
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return nullptr;
    }
}

void synapse_cuda_ops_context_free(void *raw_context) {
    delete static_cast<OpsContext *>(raw_context);
}

int32_t synapse_cuda_ops_matmul(
    void *raw_context,
    uint64_t m,
    uint64_t n,
    uint64_t k,
    const float *a,
    const float *b,
    int32_t transpose_b,
    int32_t static_rhs,
    float *c
) {
    try {
        if (!raw_context || !m || !n || !k || !a || !b || !c) {
            throw std::runtime_error("CUDA matmul received invalid dimensions or a null pointer");
        }
        static_cast<OpsContext *>(raw_context)->matmul(m, n, k, a, b, transpose_b != 0, static_rhs != 0, c);
        return 0;
    } catch (const std::exception &error) {
        synapse_cuda_set_last_error(error.what());
        return -1;
    }
}

}  // extern "C"
