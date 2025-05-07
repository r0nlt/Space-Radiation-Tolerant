/**
 * @file enhanced_tmr_test.cpp
 * @brief Tests for the enhanced TMR voting mechanisms
 */

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <bitset>
#include <cmath>
#include <cassert>

#include "../../include/rad_ml/core/redundancy/enhanced_voting.hpp"

using namespace rad_ml::core::redundancy;

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

// Function to inject single bit errors
template<typename T>
T injectSingleBitError(T value, int bitPosition) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    // Flip the bit at the specified position
    bits ^= (UintType(1) << bitPosition);
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Function to inject adjacent bit errors (MCU simulation)
template<typename T>
T injectAdjacentBitErrors(T value, int startBit, int numBits = 2) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    // Flip consecutive bits
    for (int i = 0; i < numBits; i++) {
        int bitPos = (startBit + i) % (sizeof(T) * 8);
        bits ^= (UintType(1) << bitPos);
    }
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Function to inject byte errors
template<typename T>
T injectByteError(T value, int byteIndex) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    // Corrupt an entire byte
    UintType mask = UintType(0xFF) << (byteIndex * 8);
    bits ^= mask;
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Function to inject word errors (32-bit)
template<typename T>
T injectWordError(T value) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    // For 32-bit or smaller types, corrupt the entire value
    if (sizeof(T) <= 4) {
        bits = ~bits; // Invert all bits for a worst-case test
    } else {
        // For 64-bit, corrupt only the lower or upper 32 bits
        if (rand() % 2 == 0) {
            bits ^= 0xFFFFFFFF; // Corrupt lower 32 bits
        } else {
            bits ^= (0xFFFFFFFFULL << 32); // Corrupt upper 32 bits
        }
    }
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Function to inject burst errors
template<typename T>
T injectBurstError(T value, int startBit, int burstLength) {
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    
    // Create a pattern with clustered errors
    UintType mask = 0;
    for (int i = 0; i < burstLength; i++) {
        int bitPos = (startBit + i) % (sizeof(T) * 8);
        mask |= (UintType(1) << bitPos);
    }
    
    bits ^= mask;
    
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Function to test the voting mechanisms with different error patterns
template<typename T>
void testVotingMechanisms(T originalValue) {
    std::cout << "\n=== Testing with type " << typeid(T).name() << " ===" << std::endl;
    printBinary("Original", originalValue);
    
    std::cout << "\n--- Single Bit Error Test ---" << std::endl;
    {
        // Create copies with different single bit errors
        T copy1 = injectSingleBitError(originalValue, 3);
        T copy2 = originalValue;
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        assert(standardResult == originalValue);
        
        // Test bit-level voting
        T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
        printBinary("Bit-Level Vote Result", bitResult);
        assert(bitResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Adjacent Bits Error Test ---" << std::endl;
    {
        // Create copies with adjacent bit errors
        T copy1 = injectAdjacentBitErrors(originalValue, 10, 2);
        T copy2 = originalValue;
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        assert(standardResult == originalValue);
        
        // Test bit-level voting
        T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
        printBinary("Bit-Level Vote Result", bitResult);
        assert(bitResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Byte Error Test ---" << std::endl;
    {
        // Create copies with byte errors
        T copy1 = injectByteError(originalValue, 0);
        T copy2 = originalValue;
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting (may fail with byte errors)
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        
        // Test bit-level voting
        T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
        printBinary("Bit-Level Vote Result", bitResult);
        assert(bitResult == originalValue);
        
        // Test burst error voting
        T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
        printBinary("Burst Error Vote Result", burstResult);
        assert(burstResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Word Error Test ---" << std::endl;
    {
        // Create copies with word errors
        T copy1 = injectWordError(originalValue);
        T copy2 = originalValue;
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        
        // Test word error voting
        T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
        printBinary("Word Error Vote Result", wordResult);
        assert(wordResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Burst Error Test ---" << std::endl;
    {
        // Create copies with burst errors
        T copy1 = injectBurstError(originalValue, 12, 5);
        T copy2 = originalValue;
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        
        // Test burst error voting
        T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
        printBinary("Burst Error Vote Result", burstResult);
        assert(burstResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Multiple Copy Corruption Test ---" << std::endl;
    {
        // Create copies with different error patterns
        T copy1 = injectSingleBitError(originalValue, 5);
        T copy2 = injectSingleBitError(originalValue, 15);
        T copy3 = originalValue;
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2 (corrupted)", copy2);
        printBinary("Copy3", copy3);
        
        // Test standard voting (may fail with multiple corruptions)
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        
        // Test bit-level voting
        T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
        printBinary("Bit-Level Vote Result", bitResult);
        assert(bitResult == originalValue);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        assert(adaptiveResult == originalValue);
    }
    
    std::cout << "\n--- Extreme Case: All Copies Corrupted ---" << std::endl;
    {
        // Create copies with different error patterns, all corrupted
        T copy1 = injectSingleBitError(originalValue, 7);
        T copy2 = injectSingleBitError(originalValue, 8);
        T copy3 = injectSingleBitError(originalValue, 9);
        
        printBinary("Copy1 (corrupted)", copy1);
        printBinary("Copy2 (corrupted)", copy2);
        printBinary("Copy3 (corrupted)", copy3);
        
        // Test all voting mechanisms
        T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
        printBinary("Standard Vote Result", standardResult);
        
        T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
        printBinary("Bit-Level Vote Result", bitResult);
        
        T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
        printBinary("Word Error Vote Result", wordResult);
        
        T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
        printBinary("Burst Error Vote Result", burstResult);
        
        // Test adaptive voting
        FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
        std::cout << "Detected pattern: " << static_cast<int>(detectedPattern) << std::endl;
        
        T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
        printBinary("Adaptive Vote Result", adaptiveResult);
        
        // Calculate error rates
        int standard_errors = (standardResult != originalValue) ? 1 : 0;
        int bit_errors = (bitResult != originalValue) ? 1 : 0;
        int word_errors = (wordResult != originalValue) ? 1 : 0;
        int burst_errors = (burstResult != originalValue) ? 1 : 0;
        int adaptive_errors = (adaptiveResult != originalValue) ? 1 : 0;
        
        std::cout << "Error rates:" << std::endl;
        std::cout << "Standard voting: " << standard_errors * 100 << "%" << std::endl;
        std::cout << "Bit-level voting: " << bit_errors * 100 << "%" << std::endl;
        std::cout << "Word error voting: " << word_errors * 100 << "%" << std::endl;
        std::cout << "Burst error voting: " << burst_errors * 100 << "%" << std::endl;
        std::cout << "Adaptive voting: " << adaptive_errors * 100 << "%" << std::endl;
    }
}

// Function to run statistical tests to measure correction effectiveness
template<typename T>
void runStatisticalTests(int numTests = 1000) {
    std::cout << "\n=== Statistical Tests with type " << typeid(T).name() 
              << " (" << numTests << " trials) ===" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1000.0, 1000.0);
    
    // Track success rates for different voting methods
    struct Results {
        int standardSuccess = 0;
        int bitLevelSuccess = 0;
        int wordErrorSuccess = 0;
        int burstErrorSuccess = 0;
        int adaptiveSuccess = 0;
    };
    
    // Results for different error patterns
    Results singleBitResults;
    Results adjacentBitsResults;
    Results byteErrorResults;
    Results wordErrorResults;
    Results burstErrorResults;
    
    // Run tests
    for (int i = 0; i < numTests; i++) {
        T originalValue = dis(gen);
        
        // 1. Single bit error tests
        {
            T copy1 = injectSingleBitError(originalValue, i % (sizeof(T) * 8));
            T copy2 = originalValue;
            T copy3 = originalValue;
            
            T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
            T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
            T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
            T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
            
            FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
            T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
            
            if (standardResult == originalValue) singleBitResults.standardSuccess++;
            if (bitResult == originalValue) singleBitResults.bitLevelSuccess++;
            if (wordResult == originalValue) singleBitResults.wordErrorSuccess++;
            if (burstResult == originalValue) singleBitResults.burstErrorSuccess++;
            if (adaptiveResult == originalValue) singleBitResults.adaptiveSuccess++;
        }
        
        // 2. Adjacent bits error tests
        {
            T copy1 = injectAdjacentBitErrors(originalValue, i % (sizeof(T) * 8 - 2), 2 + (i % 2));
            T copy2 = originalValue;
            T copy3 = originalValue;
            
            T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
            T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
            T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
            T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
            
            FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
            T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
            
            if (standardResult == originalValue) adjacentBitsResults.standardSuccess++;
            if (bitResult == originalValue) adjacentBitsResults.bitLevelSuccess++;
            if (wordResult == originalValue) adjacentBitsResults.wordErrorSuccess++;
            if (burstResult == originalValue) adjacentBitsResults.burstErrorSuccess++;
            if (adaptiveResult == originalValue) adjacentBitsResults.adaptiveSuccess++;
        }
        
        // 3. Byte error tests
        {
            T copy1 = injectByteError(originalValue, i % sizeof(T));
            T copy2 = originalValue;
            T copy3 = originalValue;
            
            T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
            T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
            T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
            T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
            
            FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
            T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
            
            if (standardResult == originalValue) byteErrorResults.standardSuccess++;
            if (bitResult == originalValue) byteErrorResults.bitLevelSuccess++;
            if (wordResult == originalValue) byteErrorResults.wordErrorSuccess++;
            if (burstResult == originalValue) byteErrorResults.burstErrorSuccess++;
            if (adaptiveResult == originalValue) byteErrorResults.adaptiveSuccess++;
        }
        
        // 4. Word error tests
        {
            T copy1 = injectWordError(originalValue);
            T copy2 = originalValue;
            T copy3 = originalValue;
            
            T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
            T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
            T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
            T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
            
            FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
            T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
            
            if (standardResult == originalValue) wordErrorResults.standardSuccess++;
            if (bitResult == originalValue) wordErrorResults.bitLevelSuccess++;
            if (wordResult == originalValue) wordErrorResults.wordErrorSuccess++;
            if (burstResult == originalValue) wordErrorResults.burstErrorSuccess++;
            if (adaptiveResult == originalValue) wordErrorResults.adaptiveSuccess++;
        }
        
        // 5. Burst error tests
        {
            T copy1 = injectBurstError(originalValue, i % (sizeof(T) * 8 - 5), 3 + (i % 3));
            T copy2 = originalValue;
            T copy3 = originalValue;
            
            T standardResult = EnhancedVoting::standardVote(copy1, copy2, copy3);
            T bitResult = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
            T wordResult = EnhancedVoting::wordErrorVote(copy1, copy2, copy3);
            T burstResult = EnhancedVoting::burstErrorVote(copy1, copy2, copy3);
            
            FaultPattern detectedPattern = EnhancedVoting::detectFaultPattern(copy1, copy2, copy3);
            T adaptiveResult = EnhancedVoting::adaptiveVote(copy1, copy2, copy3, detectedPattern);
            
            if (standardResult == originalValue) burstErrorResults.standardSuccess++;
            if (bitResult == originalValue) burstErrorResults.bitLevelSuccess++;
            if (wordResult == originalValue) burstErrorResults.wordErrorSuccess++;
            if (burstResult == originalValue) burstErrorResults.burstErrorSuccess++;
            if (adaptiveResult == originalValue) burstErrorResults.adaptiveSuccess++;
        }
    }
    
    // Print results
    auto printResults = [numTests](const std::string& errorType, const Results& results) {
        std::cout << "=== " << errorType << " Error Results ===" << std::endl;
        std::cout << "Standard voting:    " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << (results.standardSuccess * 100.0 / numTests) << "%" << std::endl;
        std::cout << "Bit-level voting:   " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << (results.bitLevelSuccess * 100.0 / numTests) << "%" << std::endl;
        std::cout << "Word error voting:  " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << (results.wordErrorSuccess * 100.0 / numTests) << "%" << std::endl;
        std::cout << "Burst error voting: " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << (results.burstErrorSuccess * 100.0 / numTests) << "%" << std::endl;
        std::cout << "Adaptive voting:    " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << (results.adaptiveSuccess * 100.0 / numTests) << "%" << std::endl;
        std::cout << std::endl;
    };
    
    printResults("Single Bit", singleBitResults);
    printResults("Adjacent Bits", adjacentBitsResults);
    printResults("Byte Error", byteErrorResults);
    printResults("Word Error", wordErrorResults);
    printResults("Burst Error", burstErrorResults);
}

int main() {
    std::cout << "Enhanced TMR Voting Mechanism Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Test with different data types
    testVotingMechanisms<float>(3.14159f);
    testVotingMechanisms<double>(2.71828);
    testVotingMechanisms<int32_t>(42);
    testVotingMechanisms<int64_t>(9223372036854775807);
    
    // Run statistical tests
    runStatisticalTests<float>(1000);
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
} 