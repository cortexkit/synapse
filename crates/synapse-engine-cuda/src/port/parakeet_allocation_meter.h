#ifndef SYNAPSE_PARAKEET_ALLOCATION_METER_H
#define SYNAPSE_PARAKEET_ALLOCATION_METER_H
#include <algorithm>
#include <cstddef>

// Counts explicit engine-owned allocations, not device-wide memory usage.
struct ParakeetAllocationMeter {
    std::size_t live = 0;
    std::size_t peak = 0;
    void allocate(std::size_t bytes) { live += bytes; peak = std::max(peak, live); }
    void release(std::size_t bytes) { live -= bytes; }
    void begin_request() { peak = live; }
};
#endif
