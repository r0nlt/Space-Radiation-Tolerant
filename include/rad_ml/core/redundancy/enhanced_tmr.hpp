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
 * @file enhanced_tmr.hpp
 * @brief Enhanced Triple Modular Redundancy implementation with specialized voting mechanisms
 * 
 * This file defines advanced voting mechanisms that go beyond simple majority voting
 * to handle complex fault patterns like WORD_ERROR and BURST_ERROR more effectively.
 */

#ifndef RAD_ML_ENHANCED_TMR_HPP
#define RAD_ML_ENHANCED_TMR_HPP

#include <rad_ml/core/redundancy/tmr.hpp>
#include <array>
#include <atomic>
#include <functional>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <cassert>
#include <iostream>
#include <bitset>
#include <cstring>
#include <algorithm>
#include <vector>

namespace rad_ml {
namespace core {
namespace redundancy {

/**
 * @brief CRC calculation utility
 * 
 * Provides functions for calculating CRC checksums of arbitrary data.
 * This is used to verify the integrity of values in radiation environments.
 */
class CRC {
public:
    /**
     * @brief Calculate CRC32 checksum for arbitrary data
     * 
     * @param data Pointer to the data
     * @param size Size of the data in bytes
     * @return CRC32 checksum
     */
    static uint32_t calculate(const void* data, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFF;
        
        for (size_t i = 0; i < size; ++i) {
            crc ^= bytes[i];
            for (int j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        
        return ~crc;
    }
    
    /**
     * @brief Calculate CRC32 checksum for a value of any type
     * 
     * @tparam T Type of the value
     * @param value The value to calculate CRC for
     * @return CRC32 checksum
     */
    template <typename T>
    static uint32_t calculateForValue(const T& value) {
        return calculate(&value, sizeof(T));
    }
};

/**
 * Fault pattern categories to optimize voting strategy
 */
enum class FaultPattern {
    SINGLE_BIT,    // Single-Event Upset (SEU)
    ADJACENT_BITS, // Multiple-Cell Upset (MCU)
    BYTE_ERROR,    // Byte-level corruption
    WORD_ERROR,    // 32-bit word corruption
    BURST_ERROR,   // Clustered errors
    UNKNOWN        // Unknown pattern (default to conservative strategy)
};

/**
 * @class EnhancedTMR
 * @brief Provides enhanced voting strategies for different radiation fault patterns
 */
template <typename T>
class EnhancedTMR {
public:
    /// Statistics about detected errors
    struct ErrorStats {
        uint64_t detected_errors;       ///< Total number of detected errors
        uint64_t corrected_errors;      ///< Number of errors that were corrected
        uint64_t uncorrectable_errors;  ///< Number of errors that couldn't be corrected
    };
    
    /**
     * @brief Default constructor
     */
    EnhancedTMR() noexcept : values_{T{}, T{}, T{}} {
        recalculateChecksums();
    }
    
    /**
     * @brief Constructor with initial value
     * 
     * @param value Initial value for all copies
     */
    explicit EnhancedTMR(const T& value) noexcept : values_{value, value, value} {
        recalculateChecksums();
    }
    
    /**
     * @brief Copy constructor
     *
     * @param other EnhancedTMR to copy from
     */
    EnhancedTMR(const EnhancedTMR& other) noexcept {
        values_ = other.values_;
        checksums_ = other.checksums_;
        
        // Don't copy error stats
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }
    
    /**
     * @brief Assignment operator
     *
     * @param other EnhancedTMR to copy from
     * @return Reference to this object
     */
    EnhancedTMR& operator=(const EnhancedTMR& other) noexcept {
        if (this != &other) {
            values_ = other.values_;
            checksums_ = other.checksums_;
            
            // Don't copy error stats
            error_stats_.detected_errors = 0;
            error_stats_.corrected_errors = 0;
            error_stats_.uncorrectable_errors = 0;
        }
        return *this;
    }
    
    /**
     * @brief Get the correct value using majority voting
     * 
     * @return The correct value, or best guess if uncorrectable
     */
    T get() const {
        // First check checksums to see if any copy is definitely corrupted
        bool checksum_valid[3] = {
            verifyChecksum(0),
            verifyChecksum(1),
            verifyChecksum(2)
        };
        
        // Count valid copies
        int valid_count = 0;
        for (int i = 0; i < 3; ++i) {
            if (checksum_valid[i]) valid_count++;
        }
        
        // If all copies valid, do normal TMR voting
        if (valid_count == 3) {
            return performMajorityVoting();
        }
        
        // If only some copies valid, use only those for voting
        if (valid_count > 0) {
            // If only one valid, return it
            if (valid_count == 1) {
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        incrementErrorStats(true);
                        return values_[i];
                    }
                }
            }
            
            // If two valid, compare them
            if (valid_count == 2) {
                if (checksum_valid[0] && checksum_valid[1] && values_[0] == values_[1]) {
                    incrementErrorStats(true);
                    return values_[0];
                }
                if (checksum_valid[0] && checksum_valid[2] && values_[0] == values_[2]) {
                    incrementErrorStats(true);
                    return values_[0];
                }
                if (checksum_valid[1] && checksum_valid[2] && values_[1] == values_[2]) {
                    incrementErrorStats(true);
                    return values_[1];
                }
                
                // Two valid but different values - uncertain which is correct
                std::cerr << "WARNING: EnhancedTMR has two valid checksums but different values" << std::endl;
                incrementErrorStats(false);
                
                // Return first valid as a best guess
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        return values_[i];
                    }
                }
            }
        }
        
        // At this point all checksums are invalid
        // Resort to normal TMR voting and hope for the best
        incrementErrorStats(false);
        std::cerr << "WARNING: EnhancedTMR detected corrupted checksums for all copies" << std::endl;
        return performMajorityVoting();
    }
    
    /**
     * @brief Set the value in all three copies
     * 
     * @param value The value to set
     */
    void set(const T& value) noexcept {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
        recalculateChecksums();
    }
    
    /**
     * @brief Repair any corrupted values and checksums
     *
     * @return true if repair was successful
     */
    bool repair() {
        T corrected_value = get(); // This will do majority voting
        
        // Forcibly correct all copies
        values_[0] = corrected_value;
        values_[1] = corrected_value;
        values_[2] = corrected_value;
        
        // Recalculate checksums
        recalculateChecksums();
        
        return true;
    }
    
    /**
     * @brief Verify the integrity of all values
     * 
     * @return true if all values and checksums are consistent
     */
    bool verify() const {
        // Check all checksums
        bool all_valid = verifyChecksum(0) && verifyChecksum(1) && verifyChecksum(2);
        
        // If checksums valid, check values match
        if (all_valid) {
            return (values_[0] == values_[1] && values_[1] == values_[2]);
        }
        
        return false;
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Current error statistics
     */
    ErrorStats getErrorStats() const {
        return error_stats_;
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }
    
    /**
     * @brief Assignment operator for value
     * 
     * @param value The value to assign
     * @return Reference to this object
     */
    EnhancedTMR& operator=(const T& value) {
        set(value);
        return *this;
    }
    
    /**
     * @brief Value conversion operator
     * 
     * @return The protected value
     */
    explicit operator T() const {
        return get();
    }
    
    /**
     * Standard majority voting strategy
     */
    template<typename U>
    static U standardVote(const U& a, const U& b, const U& c) {
        if (a == b) return a;
        if (a == c) return a;
        if (b == c) return b;
        
        // No majority found, fall back to bit-level voting
        return bitLevelVote(a, b, c);
    }
    
    /**
     * Bit-level voting strategy for integer types
     */
    template<typename U>
    static U bitLevelVote(const U& a, const U& b, const U& c) {
        static_assert(std::is_arithmetic<U>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<
            sizeof(U) == 8, 
            uint64_t, 
            typename std::conditional<
                sizeof(U) == 4,
                uint32_t,
                typename std::conditional<
                    sizeof(U) == 2,
                    uint16_t,
                    uint8_t
                >::type
            >::type
        >::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(U));
        std::memcpy(&b_bits, &b, sizeof(U));
        std::memcpy(&c_bits, &c, sizeof(U));
        
        UintType result = 0;
        for (size_t i = 0; i < sizeof(U) * 8; i++) {
            UintType bit_a = (a_bits >> i) & 1;
            UintType bit_b = (b_bits >> i) & 1;
            UintType bit_c = (c_bits >> i) & 1;
            
            // Majority vote for this bit
            UintType majority_bit = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c);
            result |= (majority_bit << i);
        }
        
        U final_result;
        std::memcpy(&final_result, &result, sizeof(U));
        return final_result;
    }
    
    /**
     * Word-level voting strategy for handling word corruptions
     */
    template<typename U>
    static U wordErrorVote(const U& a, const U& b, const U& c) {
        static_assert(std::is_arithmetic<U>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(U) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(U));
        std::memcpy(&b_bits, &b, sizeof(U));
        std::memcpy(&c_bits, &c, sizeof(U));
        
        // Calculate Hamming distances between all values
        int dist_ab = hammingDistance(a_bits, b_bits);
        int dist_ac = hammingDistance(a_bits, c_bits);
        int dist_bc = hammingDistance(b_bits, c_bits);
        
        // Find the closest pair and use them to reconstruct
        if (dist_ab <= dist_ac && dist_ab <= dist_bc) {
            // a and b are closest
            return reconstructFromClosestPair(a, b, c);
        } else if (dist_ac <= dist_ab && dist_ac <= dist_bc) {
            // a and c are closest
            return reconstructFromClosestPair(a, c, b);
        } else {
            // b and c are closest
            return reconstructFromClosestPair(b, c, a);
        }
    }
    
    /**
     * Burst error voting strategy for handling clustered bit errors
     */
    template<typename U>
    static U burstErrorVote(const U& a, const U& b, const U& c) {
        static_assert(std::is_arithmetic<U>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(U) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(U));
        std::memcpy(&b_bits, &b, sizeof(U));
        std::memcpy(&c_bits, &c, sizeof(U));
        
        // Perform segment-based voting
        UintType result = 0;
        constexpr int SEGMENT_SIZE = 8; // 8-bit segments
        constexpr int NUM_SEGMENTS = sizeof(U) * 8 / SEGMENT_SIZE;
        
        for (int segment = 0; segment < NUM_SEGMENTS; segment++) {
            int shift = segment * SEGMENT_SIZE;
            UintType mask = ((1ULL << SEGMENT_SIZE) - 1) << shift;
            
            UintType segment_a = (a_bits & mask) >> shift;
            UintType segment_b = (b_bits & mask) >> shift;
            UintType segment_c = (c_bits & mask) >> shift;
            
            // If any two segments match, use that value
            UintType segment_result;
            if (segment_a == segment_b) {
                segment_result = segment_a;
            } else if (segment_a == segment_c) {
                segment_result = segment_a;
            } else if (segment_b == segment_c) {
                segment_result = segment_b;
            } else {
                // No match, use bit-level voting for this segment
                segment_result = segmentBitVote(segment_a, segment_b, segment_c, SEGMENT_SIZE);
            }
            
            // Place the segment in the result
            result |= (segment_result << shift);
        }
        
        U final_result;
        std::memcpy(&final_result, &result, sizeof(U));
        return final_result;
    }
    
    /**
     * Adaptive voting strategy that selects the appropriate algorithm
     * based on the detected fault pattern
     */
    template<typename U>
    static U adaptiveVote(const U& a, const U& b, const U& c, FaultPattern pattern) {
        // Fast path for exact matches
        if (a == b) return a;
        if (a == c) return a;
        if (b == c) return b;
        
        // Apply specialized voting based on pattern
        switch (pattern) {
            case FaultPattern::SINGLE_BIT:
                return bitLevelVote(a, b, c);
                
            case FaultPattern::WORD_ERROR:
                return wordErrorVote(a, b, c);
                
            case FaultPattern::BURST_ERROR:
                return burstErrorVote(a, b, c);
                
            case FaultPattern::ADJACENT_BITS:
                // Use bit-level for adjacent bits
                return bitLevelVote(a, b, c);
                
            case FaultPattern::BYTE_ERROR:
                // Use segment-based voting for byte errors
                return burstErrorVote(a, b, c);
                
            case FaultPattern::UNKNOWN:
            default:
                // For unknown patterns, use most conservative approach
                // Try all strategies and select best result
                U bit_result = bitLevelVote(a, b, c);
                U word_result = wordErrorVote(a, b, c);
                U burst_result = burstErrorVote(a, b, c);
                
                // Select result with highest confidence
                return selectMostConfidentResult(a, b, c, bit_result, word_result, burst_result);
        }
    }
    
    /**
     * Detect the likely fault pattern by analyzing the differences
     * between the three values
     */
    template<typename U>
    static FaultPattern detectFaultPattern(const U& a, const U& b, const U& c) {
        static_assert(std::is_arithmetic<U>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(U) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(U));
        std::memcpy(&b_bits, &b, sizeof(U));
        std::memcpy(&c_bits, &c, sizeof(U));
        
        // Calculate XOR to see the differing bits
        UintType diff_ab = a_bits ^ b_bits;
        UintType diff_ac = a_bits ^ c_bits;
        UintType diff_bc = b_bits ^ c_bits;
        
        // Count number of bit differences
        int bits_diff_ab = countBits(diff_ab);
        int bits_diff_ac = countBits(diff_ac);
        int bits_diff_bc = countBits(diff_bc);
        
        // Get the maximum number of differing bits
        int max_diff = std::max({bits_diff_ab, bits_diff_ac, bits_diff_bc});
        
        // Analyze bit patterns to determine fault type
        if (max_diff == 1) {
            return FaultPattern::SINGLE_BIT;
        } else if (max_diff <= 3 && areAdjacentBits(diff_ab) && areAdjacentBits(diff_ac) && areAdjacentBits(diff_bc)) {
            return FaultPattern::ADJACENT_BITS;
        } else if (max_diff <= 8 && areByteBoundary(diff_ab) && areByteBoundary(diff_ac) && areByteBoundary(diff_bc)) {
            return FaultPattern::BYTE_ERROR;
        } else if (max_diff <= 32 && areWordBoundary(diff_ab) && areWordBoundary(diff_ac) && areWordBoundary(diff_bc)) {
            return FaultPattern::WORD_ERROR;
        } else if (areBurstPattern(diff_ab) || areBurstPattern(diff_ac) || areBurstPattern(diff_bc)) {
            return FaultPattern::BURST_ERROR;
        }
        
        return FaultPattern::UNKNOWN;
    }
    
public:
#ifdef ENABLE_TESTING
    // Test hooks for direct access to internal state
    void setForTesting(int index, const T& value) {
        if (index >= 0 && index < 3) {
            values_[index] = value;
        }
    }
    
    void recalculateChecksumsForTesting() {
        recalculateChecksums();
    }
    
    T getForTesting(int index) const {
        if (index >= 0 && index < 3) {
            return values_[index];
        }
        return T{};
    }
#endif // ENABLE_TESTING

private:
    /// The redundant copies of the value
    std::array<T, 3> values_;
    
    /// Checksums for each value
    mutable std::array<uint32_t, 3> checksums_;
    
    /// Error counters (no longer using atomic for simplicity)
    mutable ErrorStats error_stats_ = {0, 0, 0};
    
    /**
     * @brief Calculate checksums for all values
     */
    void recalculateChecksums() {
        checksums_[0] = CRC::calculateForValue(values_[0]);
        checksums_[1] = CRC::calculateForValue(values_[1]);
        checksums_[2] = CRC::calculateForValue(values_[2]);
    }
    
    /**
     * @brief Verify checksum for a specific value
     * 
     * @param index Index of the value (0-2)
     * @return true if checksum is valid
     */
    bool verifyChecksum(int index) const {
        uint32_t computed = CRC::calculateForValue(values_[index]);
        return (computed == checksums_[index]);
    }
    
    /**
     * @brief Perform majority voting on values
     * 
     * @return Majority value or best guess
     */
    T performMajorityVoting() const {
        // Majority voting
        if (values_[0] == values_[1]) {
            return values_[0];
        } else if (values_[0] == values_[2]) {
            return values_[0];
        } else if (values_[1] == values_[2]) {
            return values_[1];
        }
        
        // No majority - try to detect which one is wrong using checksums
        if (verifyChecksum(0) && !verifyChecksum(1) && !verifyChecksum(2)) {
            return values_[0];
        }
        if (!verifyChecksum(0) && verifyChecksum(1) && !verifyChecksum(2)) {
            return values_[1];
        }
        if (!verifyChecksum(0) && !verifyChecksum(1) && verifyChecksum(2)) {
            return values_[2];
        }
        
        // Still no clear answer - return first value as a last resort
        // In a real system, this would trigger a more robust recovery mechanism
        return values_[0];
    }
    
    /**
     * @brief Increment error statistics
     * 
     * @param corrected Whether the error was corrected
     */
    void incrementErrorStats(bool corrected) const {
        error_stats_.detected_errors++;
        if (corrected) {
            error_stats_.corrected_errors++;
        } else {
            error_stats_.uncorrectable_errors++;
        }
    }
    
    /**
     * Count the number of set bits in a value
     */
    template<typename UintType>
    static int countBits(UintType value) {
        int count = 0;
        while (value) {
            count += (value & 1);
            value >>= 1;
        }
        return count;
    }
    
    /**
     * Check if differing bits are adjacent (for MCU detection)
     */
    template<typename UintType>
    static bool areAdjacentBits(UintType diff) {
        if (countBits(diff) <= 1) return true;
        
        // Check if all set bits are adjacent
        UintType shifted = diff >> 1;
        return ((diff & shifted) != 0) && (countBits(diff | shifted) <= countBits(diff) + 1);
    }
    
    /**
     * Check if differing bits align with byte boundaries
     */
    template<typename UintType>
    static bool areByteBoundary(UintType diff) {
        // Check if all differing bits are within a single byte
        for (int i = 0; i < sizeof(UintType); i++) {
            UintType mask = 0xFF << (i * 8);
            if ((diff & mask) == diff) return true;
        }
        return false;
    }
    
    /**
     * Check if differing bits align with word boundaries
     */
    template<typename UintType>
    static bool areWordBoundary(UintType diff) {
        // For 32-bit or smaller types, any difference is within a word
        if (sizeof(UintType) <= 4) return true;
        
        // For 64-bit, check if all differ bits are in the same 32-bit word
        UintType lower_mask = 0xFFFFFFFF;
        UintType upper_mask = ~lower_mask;
        
        return (diff & lower_mask) == diff || (diff & upper_mask) == diff;
    }
    
    /**
     * Check if differing bits follow a burst pattern
     */
    template<typename UintType>
    static bool areBurstPattern(UintType diff) {
        // Burst errors typically have clusters of errors
        // Count the number of transitions from 0->1 and 1->0
        int transitions = 0;
        UintType mask = 1;
        bool prev_bit = false;
        
        for (size_t i = 0; i < sizeof(UintType) * 8; i++) {
            bool current_bit = (diff & mask) != 0;
            if (current_bit != prev_bit) {
                transitions++;
            }
            prev_bit = current_bit;
            mask <<= 1;
        }
        
        // Burst errors have few transitions relative to bit count
        return (transitions <= 4) && (countBits(diff) > 3);
    }
    
    /**
     * Calculate Hamming distance between two values
     */
    template<typename UintType>
    static int hammingDistance(UintType a, UintType b) {
        return countBits(a ^ b);
    }
    
    /**
     * Reconstruct a value from the two closest copies
     */
    template<typename U>
    static U reconstructFromClosestPair(const U& a, const U& b, const U& outlier) {
        using UintType = typename std::conditional<sizeof(U) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, outlier_bits;
        std::memcpy(&a_bits, &a, sizeof(U));
        std::memcpy(&b_bits, &b, sizeof(U));
        std::memcpy(&outlier_bits, &outlier, sizeof(U));
        
        // For bits that agree between a and b, use that value
        // For bits that disagree, use bit-level voting
        UintType agreement_mask = ~(a_bits ^ b_bits);
        UintType result = (a_bits & agreement_mask); // Take bits where a and b agree
        
        // For disagreeing bits, do bit-by-bit comparison with outlier
        UintType disagreement_mask = (a_bits ^ b_bits);
        for (size_t i = 0; i < sizeof(U) * 8; i++) {
            UintType mask = 1ULL << i;
            if (disagreement_mask & mask) {
                // This is a disagreeing bit - check with outlier
                UintType bit_a = (a_bits & mask);
                UintType bit_b = (b_bits & mask);
                UintType bit_outlier = (outlier_bits & mask);
                
                // If outlier agrees with either a or b, use that value
                if (bit_a == bit_outlier) {
                    result |= bit_a;
                } else if (bit_b == bit_outlier) {
                    result |= bit_b;
                } else {
                    // No agreement, use a as default (arbitrary choice)
                    result |= bit_a;
                }
            }
        }
        
        U final_result;
        std::memcpy(&final_result, &result, sizeof(U));
        return final_result;
    }
    
    /**
     * Bit-level voting for a specific segment (used in burst error voting)
     */
    template<typename UintType>
    static UintType segmentBitVote(UintType a, UintType b, UintType c, int segment_size) {
        UintType result = 0;
        for (int i = 0; i < segment_size; i++) {
            UintType bit_a = (a >> i) & 1;
            UintType bit_b = (b >> i) & 1;
            UintType bit_c = (c >> i) & 1;
            
            // Majority vote for this bit
            UintType majority_bit = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c);
            result |= (majority_bit << i);
        }
        return result;
    }
    
    /**
     * Select the most confident result from different voting strategies
     */
    template<typename U>
    static U selectMostConfidentResult(const U& a, const U& b, const U& c,
                                     const U& bit_result, const U& word_result, const U& burst_result) {
        // Count how many of the original values agree with each result
        int bit_confidence = 0;
        int word_confidence = 0;
        int burst_confidence = 0;
        
        if (bit_result == a) bit_confidence++;
        if (bit_result == b) bit_confidence++;
        if (bit_result == c) bit_confidence++;
        
        if (word_result == a) word_confidence++;
        if (word_result == b) word_confidence++;
        if (word_result == c) word_confidence++;
        
        if (burst_result == a) burst_confidence++;
        if (burst_result == b) burst_confidence++;
        if (burst_result == c) burst_confidence++;
        
        // Return the result with highest confidence
        if (bit_confidence >= word_confidence && bit_confidence >= burst_confidence) {
            return bit_result;
        } else if (word_confidence >= bit_confidence && word_confidence >= burst_confidence) {
            return word_result;
        } else {
            return burst_result;
        }
    }
};

/// Template alias for convenience
template <typename T>
using ETMR = EnhancedTMR<T>;

} // namespace redundancy
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ENHANCED_TMR_HPP 
