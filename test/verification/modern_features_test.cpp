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
/**
 * @file modern_features_test.cpp
 * @brief Tests for modern C++ enhancements to the radiation-tolerant framework
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <tuple>
#include <string>
#include <thread>

#include "../../include/rad_ml/core/redundancy/enhanced_voting.hpp"
#include "../../include/rad_ml/core/memory/protected_value.hpp"
#include "../../include/rad_ml/core/memory/aligned_memory.hpp"
#include "../../include/rad_ml/core/runtime/error_tracker.hpp"
#include "../../include/rad_ml/core/adaptive/adaptive_framework.hpp"

using namespace rad_ml::core;

// Helper function to print binary representation of a value
template<typename T>
void printBinary(const std::string& label, T value) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    std::cout << label << ": ";
    
    for (int i = sizeof(T) * 8 - 1; i >= 0; i--) {
        std::cout << ((bits >> i) & 1);
        if (i % 8 == 0) std::cout << " ";
    }
    std::cout << " (decimal: " << value << ")" << std::endl;
}

// Helper function to inject bit error
template<typename T>
T injectBitError(T value, int bitPosition) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    bits ^= (UintType(1) << bitPosition);
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Test the enhanced voting mechanisms
void testEnhancedVoting() {
    std::cout << "\n=== Testing Enhanced Voting Mechanisms ===" << std::endl;
    
    float original = 3.14159f;
    float corrupted1 = injectBitError(original, 5);
    float corrupted2 = injectBitError(original, 15);
    
    printBinary("Original", original);
    printBinary("Corrupted1", corrupted1);
    printBinary("Corrupted2", corrupted2);
    
    // Test standard voting vs. weighted voting
    {
        std::cout << "\n--- Standard vs. Weighted Voting ---" << std::endl;
        
        auto start1 = std::chrono::high_resolution_clock::now();
        float standard_result = redundancy::EnhancedVoting::standardVote(corrupted1, corrupted2, original);
        auto end1 = std::chrono::high_resolution_clock::now();
        
        auto start2 = std::chrono::high_resolution_clock::now();
        float weighted_result = redundancy::EnhancedVoting::weightedVote(corrupted1, corrupted2, original, 0.3f, 0.3f, 1.0f);
        auto end2 = std::chrono::high_resolution_clock::now();
        
        auto standard_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
        auto weighted_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
        
        printBinary("Standard Vote Result", standard_result);
        printBinary("Weighted Vote Result", weighted_result);
        
        std::cout << "Standard voting time: " << standard_time.count() << " ns" << std::endl;
        std::cout << "Weighted voting time: " << weighted_time.count() << " ns" << std::endl;
        
        std::cout << "Standard vote correct: " << (standard_result == original ? "Yes" : "No") << std::endl;
        std::cout << "Weighted vote correct: " << (weighted_result == original ? "Yes" : "No") << std::endl;
    }
    
    // Test optimized bit-level voting
    {
        std::cout << "\n--- Standard vs. Fast Bit Correction ---" << std::endl;
        
        auto start1 = std::chrono::high_resolution_clock::now();
        float bit_result = redundancy::EnhancedVoting::bitLevelVote(corrupted1, original, original);
        auto end1 = std::chrono::high_resolution_clock::now();
        
        auto start2 = std::chrono::high_resolution_clock::now();
        float fast_result = redundancy::EnhancedVoting::fastBitCorrection(corrupted1, original, original);
        auto end2 = std::chrono::high_resolution_clock::now();
        
        auto bit_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
        auto fast_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
        
        printBinary("Bit-level Vote Result", bit_result);
        printBinary("Fast Bit Correction Result", fast_result);
        
        std::cout << "Bit-level voting time: " << bit_time.count() << " ns" << std::endl;
        std::cout << "Fast bit correction time: " << fast_time.count() << " ns" << std::endl;
        std::cout << "Speedup: " << (float)bit_time.count() / fast_time.count() << "x" << std::endl;
        
        std::cout << "Bit-level vote correct: " << (bit_result == original ? "Yes" : "No") << std::endl;
        std::cout << "Fast bit correction correct: " << (fast_result == original ? "Yes" : "No") << std::endl;
    }
    
    // Test pattern detection with confidence
    {
        std::cout << "\n--- Enhanced Pattern Detection with Confidence ---" << std::endl;
        
        auto [pattern1, confidence1] = redundancy::EnhancedVoting::detectFaultPatternWithConfidence(
            corrupted1, original, original);
        
        auto [pattern2, confidence2] = redundancy::EnhancedVoting::detectFaultPatternWithConfidence(
            corrupted1, corrupted2, original);
        
        std::cout << "Single corruption - Pattern: " << static_cast<int>(pattern1) 
                  << ", Confidence: " << confidence1 << std::endl;
        
        std::cout << "Multiple corruption - Pattern: " << static_cast<int>(pattern2) 
                  << ", Confidence: " << confidence2 << std::endl;
    }
}

// Test the protected value class
void testProtectedValue() {
    std::cout << "\n=== Testing Protected Value ===" << std::endl;
    
    // Create a protected value
    memory::ProtectedValue<float> protected_pi(3.14159f);
    
    // Test normal operation
    {
        std::cout << "\n--- Normal Operation ---" << std::endl;
        
        auto result = protected_pi.get();
        if (std::holds_alternative<memory::ValueCorruptionError>(result)) {
            std::cout << "Error: " << std::get<memory::ValueCorruptionError>(result).message << std::endl;
        } else {
            std::cout << "Value: " << std::get<float>(result) << std::endl;
        }
    }
    
    // Test with corruption
    {
        std::cout << "\n--- With Corruption ---" << std::endl;
        
        // Corrupt internal value using compiler-specific hack (for testing only)
        float* raw_access = reinterpret_cast<float*>(&protected_pi);
        *raw_access = injectBitError(*raw_access, 10);
        
        auto result = protected_pi.get();
        if (std::holds_alternative<memory::ValueCorruptionError>(result)) {
            auto error = std::get<memory::ValueCorruptionError>(result);
            std::cout << "Error detected: " << error.message << std::endl;
            std::cout << "Pattern: " << static_cast<int>(error.pattern) 
                      << ", Confidence: " << error.confidence << std::endl;
        } else {
            std::cout << "Value successfully corrected: " << std::get<float>(result) << std::endl;
        }
    }
    
    // Test scrubbing
    {
        std::cout << "\n--- Memory Scrubbing ---" << std::endl;
        
        protected_pi.set(3.14159f);
        
        // Corrupt two copies
        float* raw_access = reinterpret_cast<float*>(&protected_pi);
        *(raw_access + 0) = injectBitError(3.14159f, 2);
        *(raw_access + 1) = injectBitError(3.14159f, 7);
        
        std::cout << "Before scrubbing:" << std::endl;
        auto before = protected_pi.get();
        if (std::holds_alternative<float>(before)) {
            std::cout << "Value: " << std::get<float>(before) << std::endl;
        }
        
        bool scrubbed = protected_pi.scrub();
        std::cout << "Scrubbing performed: " << (scrubbed ? "Yes" : "No") << std::endl;
        
        std::cout << "After scrubbing:" << std::endl;
        auto after = protected_pi.get();
        if (std::holds_alternative<float>(after)) {
            std::cout << "Value: " << std::get<float>(after) << std::endl;
        }
    }
    
    // Test monadic operations
    {
        std::cout << "\n--- Monadic Operations ---" << std::endl;
        
        protected_pi.set(3.14159f);
        
        // Transform with doubling function
        auto doubled = protected_pi.transform([](float f) { return f * 2; });
        
        auto result = doubled.get();
        if (std::holds_alternative<float>(result)) {
            std::cout << "Transformed value: " << std::get<float>(result) << std::endl;
        }
        
        // Test with error propagation
        float* raw_access = reinterpret_cast<float*>(&protected_pi);
        for (int i = 0; i < 3; i++) {
            *(raw_access + i) = injectBitError(*(raw_access + i), i + 5);
        }
        
        auto corrupted_transform = protected_pi.transform([](float f) { return f * 10; });
        auto corrupted_result = corrupted_transform.get();
        
        if (std::holds_alternative<memory::ValueCorruptionError>(corrupted_result)) {
            std::cout << "Error propagated correctly through transform" << std::endl;
        } else {
            std::cout << "Value: " << std::get<float>(corrupted_result) << std::endl;
        }
    }
}

// Test memory protection strategies
void testMemoryProtection() {
    std::cout << "\n=== Testing Memory Protection Strategies ===" << std::endl;
    
    // Test aligned memory
    {
        std::cout << "\n--- Aligned Protected Memory ---" << std::endl;
        
        memory::AlignedProtectedMemory<float, 64> aligned_value(3.14159f);
        std::cout << "Original value: " << aligned_value.get() << std::endl;
        
        // Corrupt one copy
        aligned_value.corruptCopy(0, injectBitError(3.14159f, 12));
        
        std::cout << "After corruption:" << std::endl;
        std::cout << "Raw copy 0: " << aligned_value.getRawCopy(0) << std::endl;
        std::cout << "Raw copy 1: " << aligned_value.getRawCopy(1) << std::endl;
        std::cout << "Raw copy 2: " << aligned_value.getRawCopy(2) << std::endl;
        std::cout << "Corrected value: " << aligned_value.get() << std::endl;
        
        // Test auto-scrubbing
        aligned_value.enableScrubbing(true);
        aligned_value.get(); // This should auto-scrub
        
        std::cout << "After auto-scrubbing:" << std::endl;
        std::cout << "Raw copy 0: " << aligned_value.getRawCopy(0) << std::endl;
        std::cout << "All copies equal: " << ((aligned_value.getRawCopy(0) == aligned_value.getRawCopy(1) && 
                                             aligned_value.getRawCopy(1) == aligned_value.getRawCopy(2)) ? "Yes" : "No") << std::endl;
    }
    
    // Test interleaved memory (only for integral types)
    {
        std::cout << "\n--- Interleaved Bit Memory ---" << std::endl;
        
        memory::InterleavedBitMemory<int32_t> interleaved_value(42);
        std::cout << "Original value: " << interleaved_value.get() << std::endl;
        
        // Corrupt interleaved data using a bit mask (this would typically be caused by radiation)
        // We'll use a hack to directly modify the internal storage
        uint64_t* raw_access = reinterpret_cast<uint64_t*>(&interleaved_value);
        *raw_access ^= (1ULL << 10) | (1ULL << 11); // Flip two adjacent bits
        
        std::cout << "After corruption:" << std::endl;
        std::cout << "Corrected value: " << interleaved_value.get() << std::endl;
        
        // Test if error is corrected after scrubbing
        interleaved_value.scrub();
        std::cout << "After scrubbing:" << std::endl;
        std::cout << "Value: " << interleaved_value.get() << std::endl;
    }
}

// Test error tracking
void testErrorTracking() {
    std::cout << "\n=== Testing Error Tracking ===" << std::endl;
    
    runtime::RadiationErrorTracker tracker;
    
    // Record various errors
    tracker.recordError(redundancy::FaultPattern::SINGLE_BIT, "Memory address 0x1000");
    tracker.recordError(redundancy::FaultPattern::SINGLE_BIT, "Memory address 0x2000");
    tracker.recordError(redundancy::FaultPattern::ADJACENT_BITS, "Memory address 0x3000");
    tracker.recordError(redundancy::FaultPattern::BYTE_ERROR, "Memory address 0x4000");
    
    // Check statistics
    std::cout << "Total errors: " << tracker.getTotalErrorCount() << std::endl;
    std::cout << "Error rate: " << tracker.getErrorRate() << " errors/second" << std::endl;
    
    std::cout << "Error distribution:" << std::endl;
    auto distribution = tracker.getPatternDistribution();
    std::cout << "SINGLE_BIT: " << (distribution[0] * 100) << "%" << std::endl;
    std::cout << "ADJACENT_BITS: " << (distribution[1] * 100) << "%" << std::endl;
    std::cout << "BYTE_ERROR: " << (distribution[2] * 100) << "%" << std::endl;
    
    // Test history tracking
    auto history = tracker.getRecentErrors(10);
    std::cout << "Error history size: " << history.size() << std::endl;
    
    if (!history.empty()) {
        auto& latest = history.back();
        std::cout << "Latest error: Pattern=" << static_cast<int>(latest.pattern)
                  << ", Data=" << latest.data << std::endl;
    }
    
    // Reset statistics
    tracker.reset();
    std::cout << "After reset, total errors: " << tracker.getTotalErrorCount() << std::endl;
}

// Test adaptive framework
void testAdaptiveFramework() {
    std::cout << "\n=== Testing Adaptive Framework ===" << std::endl;
    
    adaptive::AdaptiveFramework framework;
    
    // Test initial state
    std::cout << "Initial environment: " << static_cast<int>(framework.getEnvironment()) << std::endl;
    
    auto& settings = framework.getCurrentSettings();
    std::cout << "Initial settings:" << std::endl;
    std::cout << "  Scrubbing interval: " << settings.scrubbing_interval_ms << " ms" << std::endl;
    std::cout << "  Error threshold: " << settings.error_threshold << std::endl;
    std::cout << "  Redundancy level: " << settings.redundancy_level << std::endl;
    
    // Test environment changes
    framework.setEnvironment(adaptive::AdaptiveFramework::EnvironmentType::JUPITER);
    std::cout << "New environment: " << static_cast<int>(framework.getEnvironment()) << std::endl;
    
    auto& new_settings = framework.getCurrentSettings();
    std::cout << "Updated settings:" << std::endl;
    std::cout << "  Scrubbing interval: " << new_settings.scrubbing_interval_ms << " ms" << std::endl;
    std::cout << "  Error threshold: " << new_settings.error_threshold << std::endl;
    std::cout << "  Redundancy level: " << new_settings.redundancy_level << std::endl;
    
    // Test environment callback
    int callback_called = 0;
    framework.registerEnvironmentChangeCallback([&callback_called](adaptive::AdaptiveFramework::EnvironmentType env) {
        callback_called++;
        std::cout << "Environment changed to: " << static_cast<int>(env) << std::endl;
    });
    
    framework.setEnvironment(adaptive::AdaptiveFramework::EnvironmentType::SOLAR_FLARE);
    std::cout << "Callback called: " << (callback_called > 0 ? "Yes" : "No") << std::endl;
    
    // Test auto-detection
    for (int i = 0; i < 20; i++) {
        framework.logError(redundancy::FaultPattern::WORD_ERROR, "Simulated severe error");
    }
    
    framework.autoDetectEnvironment();
    std::cout << "Auto-detected environment: " << static_cast<int>(framework.getEnvironment()) << std::endl;
}

// Test performance with batch processing
void testBatchProcessing() {
#if __cplusplus >= 202002L
    std::cout << "\n=== Testing Batch Processing with Ranges ===" << std::endl;
    
    // Create a batch of triplets with some errors
    std::vector<std::tuple<float, float, float>> batch;
    for (int i = 0; i < 1000; i++) {
        float base = i * 0.1f;
        
        // Every 5th value has a corrupted copy
        if (i % 5 == 0) {
            batch.emplace_back(base, base, injectBitError(base, i % 16));
        } else if (i % 7 == 0) {
            batch.emplace_back(injectBitError(base, i % 8), base, base);
        } else {
            batch.emplace_back(base, base, base);
        }
    }
    
    // Process using batch mode
    auto start = std::chrono::high_resolution_clock::now();
    
    auto results = redundancy::EnhancedVoting::batchProcess(batch);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Batch processing time: " << duration.count() << " µs" << std::endl;
    std::cout << "Processed " << results.size() << " triplets" << std::endl;
    
    // Compare with serial processing
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> serial_results;
    serial_results.reserve(batch.size());
    
    for (const auto& [a, b, c] : batch) {
        serial_results.push_back(
            redundancy::EnhancedVoting::adaptiveVote(
                a, b, c, redundancy::EnhancedVoting::detectFaultPattern(a, b, c)
            )
        );
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto serial_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Serial processing time: " << serial_duration.count() << " µs" << std::endl;
    std::cout << "Speedup: " << (float)serial_duration.count() / duration.count() << "x" << std::endl;
#else
    std::cout << "\n=== Batch Processing requires C++20 ===" << std::endl;
#endif
}

int main() {
    std::cout << "Modern Radiation-Tolerant ML Framework Features Test" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    testEnhancedVoting();
    testProtectedValue();
    testMemoryProtection();
    testErrorTracking();
    testAdaptiveFramework();
    testBatchProcessing();
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
} 
