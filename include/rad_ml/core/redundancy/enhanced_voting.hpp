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
    template<typename T>
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
    template<typename T>
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
    template<typename T>
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
    template<typename T>
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
    template<typename T>
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
     * Detect the most likely fault pattern based on the values
     * 
     * @param a First copy
     * @param b Second copy
     * @param c Third copy
     * @return The detected fault pattern
     */
    template<typename T>
    static FaultPattern detectFaultPattern(const T& a, const T& b, const T& c) {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
        
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, c_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&c_bits, &c, sizeof(T));
        
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
    
private:
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
     * Reconstruct value from the closest pair (used in word error voting)
     */
    template<typename T>
    static T reconstructFromClosestPair(const T& a, const T& b, const T& outlier) {
        using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
        
        UintType a_bits, b_bits, outlier_bits;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        std::memcpy(&outlier_bits, &outlier, sizeof(T));
        
        // For bits that agree between a and b, use that value
        // For bits that disagree, use bit-level voting
        UintType agreement_mask = ~(a_bits ^ b_bits);
        UintType result = (a_bits & agreement_mask); // Take bits where a and b agree
        
        // For disagreeing bits, do bit-by-bit comparison with outlier
        UintType disagreement_mask = (a_bits ^ b_bits);
        for (size_t i = 0; i < sizeof(T) * 8; i++) {
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
        
        T final_result;
        std::memcpy(&final_result, &result, sizeof(T));
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
     * Select the most confident result among multiple voting algorithms
     */
    template<typename T>
    static T selectMostConfidentResult(const T& a, const T& b, const T& c,
                                     const T& bit_result, const T& word_result, const T& burst_result) {
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

} // namespace redundancy
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ENHANCED_VOTING_HPP 