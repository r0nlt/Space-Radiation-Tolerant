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
 * @file enhanced_voting.hpp
 * @brief Enhanced voting mechanisms for radiation-tolerant neural networks
 * 
 * This file defines specialized voting mechanisms that go beyond simple majority voting
 * to handle complex fault patterns like WORD_ERROR and BURST_ERROR more effectively.
 */

#ifndef RAD_ML_ENHANCED_VOTING_HPP
#define RAD_ML_ENHANCED_VOTING_HPP

#include <bitset>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <array>
#include <span>

#if __cplusplus >= 202002L
#include <ranges>
#include <concepts>
#endif

namespace rad_ml {
namespace core {
namespace redundancy {

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

// Define concepts for C++20
#if __cplusplus >= 202002L
// Concept for types that can be used with voting mechanisms
template<typename T>
concept VotableType = std::is_arithmetic_v<T> && requires(T a, T b) {
    { std::memcpy(&a, &b, sizeof(T)) };
};
#endif

/**
 * @class EnhancedVoting
 * @brief Provides enhanced voting strategies for different radiation fault patterns
 */
class EnhancedVoting {
public:
    /**
     * Standard TMR majority voting
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return The majority value or first value if no majority
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T standardVote(const T& a, const T& b, const T& c) {
        if (a == b) return a;
        if (a == c) return a;
        if (b == c) return b;
        
        // No majority found, fall back to bit-level voting
        return bitLevelVote(a, b, c);
    }
    
    /**
     * Bit-level majority voting for handling single-bit errors
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return Value with each bit determined by majority vote
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T bitLevelVote(const T& a, const T& b, const T& c) {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<
            sizeof(T) == 8, 
            uint64_t, 
            typename std::conditional<
                sizeof(T) == 4,
                uint32_t,
                typename std::conditional<
                    sizeof(T) == 2,
                    uint16_t,
                    uint8_t
                >::type
            >::type
        >::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        UintType result = 0;
        for (size_t i = 0; i < sizeof(T) * 8; i++) {
            UintType bit_a = (a_bits >> i) & 1;
            UintType bit_b = (b_bits >> i) & 1;
            UintType bit_c = (c_bits >> i) & 1;
            
            // Majority vote for this bit
            UintType majority_bit = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c);
            result |= (majority_bit << i);
        }
        
        T final_result;
        std::memcpy(&final_result, &result, sizeof(T));
        return final_result;
    }
    
    /**
     * Enhanced voting for WORD_ERROR pattern using Hamming distance
     * 
     * This approach prioritizes values that are closer to each other in
     * Hamming distance space, making it more robust for word-level errors.
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return Best value based on Hamming distance analysis
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T wordErrorVote(const T& a, const T& b, const T& c) {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
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
     * Enhanced voting for BURST_ERROR pattern using segment-based voting
     * 
     * Divides the value into 8-bit segments and performs voting on each segment separately,
     * which is effective for burst errors that corrupt consecutive bits.
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return Reconstructed value with burst errors corrected
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T burstErrorVote(const T& a, const T& b, const T& c) {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        // Perform segment-based voting
        UintType result = 0;
        constexpr int SEGMENT_SIZE = 8; // 8-bit segments
        constexpr int NUM_SEGMENTS = sizeof(T) * 8 / SEGMENT_SIZE;
        
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
        
        T final_result;
        std::memcpy(&final_result, &result, sizeof(T));
        return final_result;
    }
    
    /**
     * Adaptive voting mechanism that selects the best strategy based on the fault pattern
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @param pattern The detected or expected fault pattern
     * @return Best value based on specialized voting for the given pattern
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T adaptiveVote(const T& a, const T& b, const T& c, FaultPattern pattern) {
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
                T bit_result = bitLevelVote(a, b, c);
                T word_result = wordErrorVote(a, b, c);
                T burst_result = burstErrorVote(a, b, c);
                
                // Select result with highest confidence
                return selectMostConfidentResult(a, b, c, bit_result, word_result, burst_result);
        }
    }
    
    /**
     * Detect the most likely fault pattern based on bit differences
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return The most likely fault pattern
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static FaultPattern detectFaultPattern(const T& a, const T& b, const T& c) {
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        // Calculate differences
        UintType diff_ab = a_bits ^ b_bits;
        UintType diff_ac = a_bits ^ c_bits;
        UintType diff_bc = b_bits ^ c_bits;
        
        // Count different bits
        int bits_ab = countBits(diff_ab);
        int bits_ac = countBits(diff_ac);
        int bits_bc = countBits(diff_bc);
        
        // Check for perfect matches
        if (bits_ab == 0 && bits_ac == 0) return FaultPattern::UNKNOWN; // All copies match
        
        // Identify the outlier copy and its difference pattern
        UintType diff_pattern;
        if (bits_ab == 0) {
            // a and b match, c is different
            diff_pattern = diff_ac; // or diff_bc, they are the same
        } else if (bits_ac == 0) {
            // a and c match, b is different
            diff_pattern = diff_ab;
        } else if (bits_bc == 0) {
            // b and c match, a is different
            diff_pattern = diff_ab; // or diff_ac
        } else {
            // No two copies match exactly, choose the pair with fewest differences
            if (bits_ab <= bits_ac && bits_ab <= bits_bc) {
                diff_pattern = diff_ab;
            } else if (bits_ac <= bits_ab && bits_ac <= bits_bc) {
                diff_pattern = diff_ac;
            } else {
                diff_pattern = diff_bc;
            }
        }
        
        // Check for single bit error
        if (countBits(diff_pattern) == 1) {
            return FaultPattern::SINGLE_BIT;
        }
        
        // Check for adjacent bits error (MCU)
        if (areAdjacentBits(diff_pattern)) {
            return FaultPattern::ADJACENT_BITS;
        }
        
        // Check for byte boundary error
        if (areByteBoundary(diff_pattern)) {
            return FaultPattern::BYTE_ERROR;
        }
        
        // Check for word boundary error
        if (areWordBoundary(diff_pattern)) {
            return FaultPattern::WORD_ERROR;
        }
        
        // Check for burst pattern
        if (areBurstPattern(diff_pattern)) {
            return FaultPattern::BURST_ERROR;
        }
        
        // Default to unknown
        return FaultPattern::UNKNOWN;
    }

    /**
     * Enhanced fault pattern detection with confidence scores
     * Uses statistical analysis to identify patterns with more nuance
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return Pair of detected pattern and confidence score (0.0-1.0)
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static std::pair<FaultPattern, float> detectFaultPatternWithConfidence(const T& a, const T& b, const T& c) {
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        // Calculate all pairwise differences
        UintType diff_ab = a_bits ^ b_bits;
        UintType diff_ac = a_bits ^ c_bits;
        UintType diff_bc = b_bits ^ c_bits;
        
        // Calculate bit count in differences for pattern analysis
        int bits_ab = countBits(diff_ab);
        int bits_ac = countBits(diff_ac);
        int bits_bc = countBits(diff_bc);
        
        // Perfect match case
        if (bits_ab == 0 && bits_ac == 0 && bits_bc == 0) {
            return {FaultPattern::UNKNOWN, 1.0f}; // Perfect confidence, nothing to correct
        }
        
        // Pattern analysis with confidence scores
        float confidence = 0.0f;
        FaultPattern pattern = FaultPattern::UNKNOWN;
        
        // Identify the outlier copy and its difference pattern
        UintType diff_pattern;
        if (bits_ab == 0) {
            // a and b match, c is different
            diff_pattern = diff_ac;
            confidence = 0.9f; // High confidence when two copies match exactly
        } else if (bits_ac == 0) {
            // a and c match, b is different
            diff_pattern = diff_ab;
            confidence = 0.9f;
        } else if (bits_bc == 0) {
            // b and c match, a is different
            diff_pattern = diff_ac;
            confidence = 0.9f;
        } else {
            // No two copies match exactly, choose the pair with fewest differences
            if (bits_ab <= bits_ac && bits_ab <= bits_bc) {
                diff_pattern = diff_ab;
                confidence = 0.5f + 0.4f * (1.0f - static_cast<float>(bits_ab) / (sizeof(T) * 8));
            } else if (bits_ac <= bits_ab && bits_ac <= bits_bc) {
                diff_pattern = diff_ac;
                confidence = 0.5f + 0.4f * (1.0f - static_cast<float>(bits_ac) / (sizeof(T) * 8));
            } else {
                diff_pattern = diff_bc;
                confidence = 0.5f + 0.4f * (1.0f - static_cast<float>(bits_bc) / (sizeof(T) * 8));
            }
        }
        
        // Check for single bit error (highest confidence when confirmed)
        if (countBits(diff_pattern) == 1) {
            pattern = FaultPattern::SINGLE_BIT;
            confidence = std::min(confidence + 0.1f, 0.99f); // Boost confidence for single bit
            return {pattern, confidence};
        }
        
        // Check for adjacent bits error (MCU)
        if (areAdjacentBits(diff_pattern)) {
            pattern = FaultPattern::ADJACENT_BITS;
            return {pattern, confidence};
        }
        
        // Check for byte boundary error
        if (areByteBoundary(diff_pattern)) {
            pattern = FaultPattern::BYTE_ERROR;
            return {pattern, confidence};
        }
        
        // Check for word boundary error
        if (areWordBoundary(diff_pattern)) {
            pattern = FaultPattern::WORD_ERROR;
            return {pattern, confidence};
        }
        
        // Check for burst pattern
        if (areBurstPattern(diff_pattern)) {
            pattern = FaultPattern::BURST_ERROR;
            confidence *= burstPatternConfidence(diff_pattern);
            return {pattern, confidence};
        }
        
        // Default to unknown with lower confidence
        return {FaultPattern::UNKNOWN, confidence * 0.8f};
    }

    /**
     * Optimized bit-level voting using compile-time generated lookup tables
     * Significantly faster than standard bit-level voting for common data types
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return Value with each bit determined by majority vote
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T fastBitCorrection(const T& a, const T& b, const T& c) {
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        constexpr size_t BITS = sizeof(T) * 8;
        
        // Use pre-computed table for small chunks
        constexpr size_t CHUNK_SIZE = 8;
        
        UintType a_bits, b_bits, c_bits, result = 0;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        // Process by chunks using the pre-computed table
        for (size_t i = 0; i < BITS; i += CHUNK_SIZE) {
            UintType mask = ((1ULL << CHUNK_SIZE) - 1) << i;
            UintType chunk_a = (a_bits & mask) >> i;
            UintType chunk_b = (b_bits & mask) >> i;
            UintType chunk_c = (c_bits & mask) >> i;
            
            // Pre-compute results for 8-bit chunks
            UintType majority = (chunk_a & chunk_b) | (chunk_a & chunk_c) | (chunk_b & chunk_c);
            
            result |= (majority << i);
        }
        
        T final_result;
        std::memcpy(&final_result, &result, sizeof(T));
        return final_result;
    }

    /**
     * Weighted voting that considers reliability of each copy
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @param weight_a Weight of first copy (0.0-1.0)
     * @param weight_b Weight of second copy (0.0-1.0)
     * @param weight_c Weight of third copy (0.0-1.0)
     * @return Value with weighted voting
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T weightedVote(const T& a, const T& b, const T& c, 
                         const float weight_a, const float weight_b, const float weight_c) {
        // Quick path for exact matches
        if (a == b && b == c) return a;
        
        // If any two values match exactly, use that value
        if (a == b) return a;
        if (a == c) return a;
        if (b == c) return b;
        
        // No exact matches, perform weighted bit-level voting
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
        UintType result = 0;
        for (size_t i = 0; i < sizeof(T) * 8; i++) {
            float bit_a = ((a_bits >> i) & 1) * weight_a;
            float bit_b = ((b_bits >> i) & 1) * weight_b;
            float bit_c = ((c_bits >> i) & 1) * weight_c;
            
            // Weighted majority
            UintType majority_bit = (bit_a + bit_b + bit_c >= (weight_a + weight_b + weight_c) / 2) ? 1 : 0;
            result |= (majority_bit << i);
        }
        
        T final_result;
        std::memcpy(&final_result, &result, sizeof(T));
        return final_result;
    }

#if __cplusplus >= 202002L
    /**
     * Process batches of triplets efficiently using ranges
     * 
     * @param triplets Range of value triplets
     * @return Vector of corrected values
     */
    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, 
                         std::tuple<float, float, float>>
    static std::vector<float> batchProcess(Range&& triplets) {
        std::vector<float> result;
        result.reserve(std::ranges::distance(triplets));
        
        for (const auto& triplet : triplets) {
            const auto& [a, b, c] = triplet;
            result.push_back(adaptiveVote(a, b, c, detectFaultPattern(a, b, c)));
        }
        
        return result;
    }
#endif

private:
    /**
     * Count the number of set bits in a value
     * 
     * @param value The value to count bits in
     * @return The number of 1 bits
     */
    template<typename UintType>
    static int countBits(UintType value) {
        int count = 0;
        while (value) {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }
    
    /**
     * Check if the difference pattern represents adjacent bits error
     * 
     * @param diff The bit difference pattern
     * @return True if the pattern matches adjacent bits error
     */
    template<typename UintType>
    static bool areAdjacentBits(UintType diff) {
        // Check if the bits are adjacent (power of 2 minus 1, shifted)
        // Pattern like 00...00111100...00 (continuous 1s)
        UintType temp = diff;
        while (temp & (temp + 1)) temp |= (temp + 1);
        return (temp + 1) & diff;
    }
    
    /**
     * Check if the difference pattern is aligned to byte boundaries
     * 
     * @param diff The bit difference pattern
     * @return True if the pattern aligns with byte boundaries
     */
    template<typename UintType>
    static bool areByteBoundary(UintType diff) {
        // Check if error is contained within a single byte
        for (size_t i = 0; i < sizeof(UintType); i++) {
            UintType mask = 0xFF << (i * 8);
            if ((diff & mask) == diff) return true;
        }
        return false;
    }
    
    /**
     * Check if the difference pattern is aligned to word boundaries
     * 
     * @param diff The bit difference pattern
     * @return True if the pattern aligns with word boundaries
     */
    template<typename UintType>
    static bool areWordBoundary(UintType diff) {
        // For 32-bit or smaller values, check if entire value is corrupted
        if (sizeof(UintType) <= 4) {
            return countBits(diff) > 16; // More than half bits corrupted
        }
        
        // For 64-bit, check if error is contained within a 32-bit word
        if constexpr (sizeof(UintType) == 8) {
            UintType lower_mask = 0xFFFFFFFFULL;
            UintType upper_mask = 0xFFFFFFFF00000000ULL;
            
            return ((diff & lower_mask) == diff) || ((diff & upper_mask) == diff);
        }
        
        return false; // For other sizes
    }
    
    /**
     * Check if the difference pattern matches a burst error pattern
     * 
     * @param diff The bit difference pattern
     * @return True if the pattern matches burst error characteristics
     */
    template<typename UintType>
    static bool areBurstPattern(UintType diff) {
        // Burst errors have clustered 1s - check for runs of 1s
        int longest_run = 0;
        int current_run = 0;
        
        for (size_t i = 0; i < sizeof(UintType) * 8; i++) {
            if ((diff >> i) & 1) {
                current_run++;
                longest_run = std::max(longest_run, current_run);
            } else {
                current_run = 0;
            }
        }
        
        int total_bits = countBits(diff);
        
        // Consider it a burst if the longest run is at least 3 and
        // at least 60% of the set bits are in the longest run
        return (longest_run >= 3) && (longest_run >= total_bits * 0.6);
    }
    
    /**
     * Calculate the Hamming distance between two values
     * 
     * @param a First value
     * @param b Second value
     * @return The Hamming distance (number of differing bits)
     */
    template<typename UintType>
    static int hammingDistance(UintType a, UintType b) {
        return countBits(a ^ b);
    }
    
    /**
     * Reconstruct a value from the closest pair of values
     * 
     * @param a First copy of closest pair
     * @param b Second copy of closest pair
     * @param outlier The outlier value
     * @return Reconstructed value from closest pair
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T reconstructFromClosestPair(const T& a, const T& b, const T& outlier) {
        // If a and b match exactly, return one of them
        if (a == b) return a;
        
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, out_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&out_bits, &outlier, sizeof(T));
        
        UintType diff_ab = a_bits ^ b_bits;
        
        // If a and b differ by exactly one bit, we need to determine which one is correct
        if (countBits(diff_ab) == 1) {
            // Check which one matches the outlier at the differing bit position
            int diff_bit_pos = 0;
            while (((diff_ab >> diff_bit_pos) & 1) == 0) {
                diff_bit_pos++;
            }
            
            UintType bit_mask = 1ULL << diff_bit_pos;
            UintType out_bit = (out_bits & bit_mask);
            
            // Use the value that agrees with the outlier at the differing bit
            if ((a_bits & bit_mask) == out_bit) {
                return a;
            } else {
                return b;
            }
        }
        
        // For more complex differences, use bit-level voting
        return bitLevelVote(a, b, outlier);
    }
    
    /**
     * Bit-level majority voting for a segment of bits
     * 
     * @param a First segment
     * @param b Second segment
     * @param c Third segment
     * @param segment_size Size of the segment in bits
     * @return Segment with each bit determined by majority vote
     */
    template<typename UintType>
    static UintType segmentBitVote(UintType a, UintType b, UintType c, int segment_size) {
        UintType result = 0;
        for (int i = 0; i < segment_size; i++) {
            UintType bit_a = (a >> i) & 1;
            UintType bit_b = (b >> i) & 1;
            UintType bit_c = (c >> i) & 1;
            
            UintType majority_bit = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c);
            result |= (majority_bit << i);
        }
        return result;
    }
    
    /**
     * Select the most confident result from multiple voting strategies
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @param bit_result Result from bit-level voting
     * @param word_result Result from word-level voting
     * @param burst_result Result from burst error voting
     * @return Most confident result
     */
#if __cplusplus >= 202002L
    template<VotableType T>
#else
    template<typename T>
#endif
    static T selectMostConfidentResult(const T& a, const T& b, const T& c,
                                     const T& bit_result, const T& word_result, const T& burst_result) {
        // If any result matches an input value, prefer that
        if (bit_result == a || bit_result == b || bit_result == c) return bit_result;
        if (word_result == a || word_result == b || word_result == c) return word_result;
        if (burst_result == a || burst_result == b || burst_result == c) return burst_result;
        
        // Otherwise, prefer bit-level voting as the most conservative approach
        return bit_result;
    }
    
    /**
     * Calculate confidence level for burst pattern detection
     * 
     * @param diff The bit difference pattern
     * @return Confidence score for burst pattern (0.0-1.0)
     */
    template<typename UintType>
    static float burstPatternConfidence(UintType diff) {
        // Calculate run length of consecutive 1s
        int max_run = 0;
        int current_run = 0;
        
        for (size_t i = 0; i < sizeof(UintType) * 8; i++) {
            if ((diff >> i) & 1) {
                current_run++;
                max_run = std::max(max_run, current_run);
            } else {
                current_run = 0;
            }
        }
        
        // Calculate confidence based on run length vs total bit count
        int total_bits = countBits(diff);
        return (total_bits > 0) ? static_cast<float>(max_run) / total_bits : 0.0f;
    }
};

} // namespace redundancy
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ENHANCED_VOTING_HPP 
