#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

using SynapseProgressCallback = void (*)(void*, int);

struct SynapseProgressObserver {
    SynapseProgressCallback callback = nullptr;
    void* cookie = nullptr;
    bool inference = false;
    void report(bool setup) {
        if (!setup) inference = true;
        if (callback) callback(cookie, setup && !inference ? 1 : 0);
    }
};

// Captured graphs record real layer-completion events too. Waiting on these
// from the calling thread does not stall GPU work or call socket code during
// capture; an enqueued kernel alone is never counted as completed progress.
struct SynapseLayerProgress {
    std::vector<cudaEvent_t> events;
    explicit SynapseLayerProgress(int layers) { init(layers); }
    static void check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void init(int layers) {
        events.resize(layers, nullptr);
        try {
            for (auto& event : events) check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        } catch (...) {
            for (auto event : events) if (event) cudaEventDestroy(event);
            events.clear();
            throw;
        }
    }
    void record(int layer, cudaStream_t stream) {
        // Default capture events can be graph-internal dependency markers.
        // External record nodes overwrite the host-visible event on every
        // serialized replay, so a warmup record cannot count as new progress.
        check(cudaEventRecordWithFlags(events.at(layer), stream, cudaEventRecordExternal));
    }
    void wait(SynapseProgressObserver& observer, bool setup) {
        if (!observer.callback) return;
        for (auto event : events) {
            check(cudaEventSynchronize(event));
            observer.report(setup);
        }
    }
    ~SynapseLayerProgress() {
        for (auto event : events) if (event) cudaEventDestroy(event);
    }
};
