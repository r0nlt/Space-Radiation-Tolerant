/**
 * @file memory_scrubber_test.cpp
 * @brief Tests for the unified rad_ml::memory::MemoryScrubber
 *
 * Covers both region kinds of the unified scrubber:
 *  - callback-repaired regions (caller-provided repair routine, e.g. TMR
 *    repair() over an array), driven by the background thread
 *  - CRC-verified regions (per-64-byte-block checksums stored at
 *    registration, mismatches counted on scrub)
 */

#include <rad_ml/core/redundancy/tmr.hpp>
#include <rad_ml/memory/memory_scrubber.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using rad_ml::core::redundancy::TMR;
using rad_ml::memory::MemoryScrubber;

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition) assert(condition)

// Callback-repaired region: background scrubbing repairs a corrupted TMR copy
TEST(test_callback_scrubbing)
{
    TMR<int> tmr_values[10];
    for (int i = 0; i < 10; ++i) {
        tmr_values[i] = i;
    }

    MemoryScrubber scrubber(100);  // 100ms interval

    size_t handle = scrubber.registerMemoryRegion<TMR<int>>(
        tmr_values, sizeof(tmr_values), [](TMR<int>* ptr, size_t size) {
            size_t count = size / sizeof(TMR<int>);
            for (size_t i = 0; i < count; ++i) {
                ptr[i].repair();
            }
        });

    // Simulate a bit flip by corrupting one of the TMR replicas
    int* raw_values = reinterpret_cast<int*>(&tmr_values[5]);
    raw_values[0] = 99;

    // The value should still be correct due to TMR majority voting
    ASSERT(tmr_values[5].get() == 5);

    scrubber.start();
    ASSERT(scrubber.isRunning());

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    scrubber.stop();
    ASSERT(!scrubber.isRunning());

    // Verify that the corrupted replica was repaired
    ASSERT(raw_values[0] == 5);
    ASSERT(raw_values[1] == 5);
    ASSERT(raw_values[2] == 5);

    ASSERT(scrubber.unregisterMemoryRegion(handle));
    ASSERT(!scrubber.unregisterMemoryRegion(handle));
}

// CRC-verified region: corruption in any 64-byte block is detected on scrub
TEST(test_crc_region_detection)
{
    std::vector<uint8_t> buffer(256, 0xA5);

    MemoryScrubber scrubber;
    size_t handle = scrubber.registerMemoryRegion(buffer.data(), buffer.size());

    ASSERT(scrubber.getRegionCount() == 1);
    ASSERT(scrubber.getTotalMemorySize() == buffer.size());

    // Clean scrub: no errors
    ASSERT(scrubber.scrubMemory() == 0);

    // Corrupt two separate 64-byte blocks
    buffer[10] ^= 0x01;
    buffer[130] ^= 0x80;

    ASSERT(scrubber.scrubMemory() == 2);

    // The scrub re-armed the checksums, so the same contents are now clean
    ASSERT(scrubber.scrubMemory() == 0);

    const auto stats = scrubber.getStatistics();
    ASSERT(stats.errors_detected == 2);
    ASSERT(stats.scrub_cycles == 3);

    ASSERT(scrubber.unregisterMemoryRegion(handle));
    ASSERT(scrubber.getRegionCount() == 0);
}

int main()
{
    std::cout << "Running memory scrubber tests..." << std::endl;

    test_callback_scrubbing();
    test_crc_region_detection();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
