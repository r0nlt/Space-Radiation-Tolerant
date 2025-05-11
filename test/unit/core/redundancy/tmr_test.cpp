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
#include <rad_ml/core/redundancy/tmr.hpp>
#include <cassert>
#include <iostream>

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition) assert(condition)

using namespace rad_ml::core::redundancy;

// Test initialization
TEST(test_initialization) {
    TMR<int> tmr(42);
    ASSERT(tmr.get() == 42);
}

// Test assignment
TEST(test_assignment) {
    TMR<int> tmr;
    tmr = 123;
    ASSERT(tmr.get() == 123);
}

// Test majority voting
TEST(test_majority_voting) {
    // Create a TMR with a known initial value
    TMR<int> tmr(10);
    
    // Access the private members directly for testing
    // In a real test, we would use a test-specific subclass or friend function
    // This is a simplified demonstration
    
    // Simulate a bit flip in one copy
    // WARNING: This is a hack for demonstration purposes only
    // In a real test, we would use a proper testing interface
    int* raw_values = reinterpret_cast<int*>(&tmr);
    raw_values[0] = 99;  // Corrupt the first copy
    
    // Majority voting should still return the correct value
    ASSERT(tmr.get() == 10);
    
    // Repair should fix the corrupted value
    tmr.repair();
    
    // All values should now be correct
    ASSERT(raw_values[0] == 10);
    ASSERT(raw_values[1] == 10);
    ASSERT(raw_values[2] == 10);
}

int main() {
    std::cout << "Running TMR tests..." << std::endl;
    
    test_initialization();
    test_assignment();
    test_majority_voting();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
} 
