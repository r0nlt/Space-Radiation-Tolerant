/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#include <rad_ml/core/memory/memory_scrubber.hpp>
#include <rad_ml/core/redundancy/tmr.hpp>
#include <cassert>
#include <iostream>
#include <thread>

using namespace rad_ml::core::memory;
using namespace rad_ml::core::redundancy;

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition) assert(condition)

// Test basic scrubbing functionality
TEST(test_memory_scrubbing) {
    // Create some TMR variables
    TMR<int> tmr_values[10];
    for (int i = 0; i < 10; ++i) {
        tmr_values[i] = i;
    }
    
    // Initialize scrubber
    MemoryScrubber scrubber(100); // 100ms interval
    
    // Register memory region
    size_t handle = scrubber.registerMemoryRegion<TMR<int>>(
        tmr_values,
        sizeof(tmr_values),
        [](TMR<int>* ptr, size_t size) {
            size_t count = size / sizeof(TMR<int>);
            for (size_t i = 0; i < count; ++i) {
                ptr[i].repair();
            }
        }
    );
    
    // Simulate a bit flip by corrupting one of the TMR replicas
    // In a real test, we would use a test-specific subclass or friend function
    // This is a simplified demonstration
    int* raw_values = reinterpret_cast<int*>(&tmr_values[5]);
    raw_values[0] = 99; // Corrupt the first replica
    
    // The value should still be correct due to TMR
    ASSERT(tmr_values[5].get() == 5);
    
    // Start scrubbing
    scrubber.start();
    
    // Wait for scrubbing to happen
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    
    // Stop scrubbing
    scrubber.stop();
    
    // Verify that the value was repaired
    ASSERT(raw_values[0] == 5);
    ASSERT(raw_values[1] == 5);
    ASSERT(raw_values[2] == 5);
    
    // Unregister memory region
    bool result = scrubber.unregisterMemoryRegion(handle);
    ASSERT(result);
}

int main() {
    std::cout << "Running memory scrubber tests..." << std::endl;
    
    test_memory_scrubbing();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
} 
