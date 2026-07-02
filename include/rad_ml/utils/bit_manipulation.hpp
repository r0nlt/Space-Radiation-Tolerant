/**
 * @file bit_manipulation.hpp
 * @brief Bit manipulation utilities for radiation simulation
 * 
 * This file provides utilities for manipulating bits in different data types,
 * particularly useful for simulating bit flips due to radiation effects.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

namespace rad_ml {
namespace utils {

/**
 * @brief Utilities for bit-level manipulation
 */
class BitManipulation {
public:
    /**
     * @brief Flip a specific bit in a float value
     * 
     * @param value The float value to modify
     * @param bit_position The position of the bit to flip (0-31 for 32-bit float)
     * @return The float value with the specified bit flipped
     */
    static float flipBit(float value, int bit_position) {
        if (bit_position < 0 || bit_position >= 32) {
            return value; // Invalid bit position
        }
        
        // Copy bits into an integer via memcpy (union punning is UB in C++)
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(value));
        
        // Flip the specified bit using XOR
        bits ^= (1u << bit_position);
        
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    
    /**
     * @brief Flip a specific bit in a double value
     * 
     * @param value The double value to modify
     * @param bit_position The position of the bit to flip (0-63 for 64-bit double)
     * @return The double value with the specified bit flipped
     */
    static double flipBit(double value, int bit_position) {
        if (bit_position < 0 || bit_position >= 64) {
            return value; // Invalid bit position
        }
        
        // Copy bits into an integer via memcpy (union punning is UB in C++)
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(value));
        
        // Flip the specified bit using XOR
        bits ^= (1ULL << bit_position);
        
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    
    /**
     * @brief Flip a specific bit in an integer type
     * 
     * @tparam T Integer type
     * @param value The integer value to modify
     * @param bit_position The position of the bit to flip
     * @return The integer value with the specified bit flipped
     */
    template<typename T>
    static typename std::enable_if<std::is_integral<T>::value, T>::type
    flipBit(T value, int bit_position) {
        if (bit_position < 0 || bit_position >= static_cast<int>(sizeof(T) * 8)) {
            return value; // Invalid bit position
        }
        
        // Flip the specified bit using XOR
        return value ^ (static_cast<T>(1) << bit_position);
    }
    
    /**
     * @brief Count the number of bit flips between two values
     * 
     * @tparam T Data type
     * @param a First value
     * @param b Second value
     * @return Number of bits that differ between a and b
     */
    template<typename T>
    static int countBitDifferences(T a, T b) {
        // Convert to unsigned integer representation for bit counting
        using UIntType = typename std::conditional<
            sizeof(T) == 8, uint64_t,
            typename std::conditional<
                sizeof(T) == 4, uint32_t,
                typename std::conditional<
                    sizeof(T) == 2, uint16_t,
                    uint8_t
                >::type
            >::type
        >::type;
        
        static_assert(sizeof(UIntType) == sizeof(T),
                      "Only 1, 2, 4, and 8 byte types are supported");
        
        UIntType a_bits = 0;
        UIntType b_bits = 0;
        std::memcpy(&a_bits, &a, sizeof(T));
        std::memcpy(&b_bits, &b, sizeof(T));
        
        // XOR the values to get bits that differ
        UIntType diff = a_bits ^ b_bits;
        
        // Count the bits set in the difference
        int count = 0;
        while (diff) {
            count += diff & 1;
            diff >>= 1;
        }
        
        return count;
    }
    
    /**
     * @brief Check if a specific bit is set in a value
     * 
     * @tparam T Data type
     * @param value The value to check
     * @param bit_position The position of the bit to check
     * @return true if the bit is set, false otherwise
     */
    template<typename T>
    static bool isBitSet(T value, int bit_position) {
        if (bit_position < 0 || bit_position >= static_cast<int>(sizeof(T) * 8)) {
            return false; // Invalid bit position
        }
        
        // For floating-point types, convert to integer representation
        if constexpr (std::is_floating_point<T>::value) {
            using UIntType = typename std::conditional<
                sizeof(T) == 8, uint64_t, uint32_t
            >::type;
            static_assert(sizeof(UIntType) == sizeof(T),
                          "Unsupported floating-point size");
            
            UIntType bits;
            std::memcpy(&bits, &value, sizeof(T));
            return (bits & (static_cast<UIntType>(1) << bit_position)) != 0;
        }
        else {
            return (value & (static_cast<T>(1) << bit_position)) != 0;
        }
    }
};

} // namespace utils
} // namespace rad_ml 