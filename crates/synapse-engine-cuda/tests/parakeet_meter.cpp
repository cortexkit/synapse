#include "../src/port/parakeet_allocation_meter.h"
#include <cassert>
#include <iostream>

int main() {
    ParakeetAllocationMeter meter;
    meter.allocate(100); // Persistent weights survive request resets.
    meter.begin_request();
    meter.allocate(80);
    meter.allocate(40);
    meter.release(80);
    assert(meter.live == 140 && meter.peak == 220);
    meter.begin_request();
    assert(meter.peak == 140);
    meter.release(40);
    meter.allocate(10);
    assert(meter.live == 110 && meter.peak == 140);
    meter.release(110);
    meter.begin_request();
    assert(meter.live == 0 && meter.peak == 0);
    std::cout << "parakeet_allocation_meter: 1 test passed\n";
}
