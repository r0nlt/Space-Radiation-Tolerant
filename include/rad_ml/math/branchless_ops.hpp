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
#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace rad_ml {
namespace math {

/**
 * @brief Branchless operations for predictable code execution
 * 
 * This class provides branchless implementations of common operations to avoid
 * potential issues with branch prediction units in radiation environments.
 * Branch mispredictions can be more susceptible to radiation-induced errors.
 */
class BranchlessOps {
public:
    /**
     * @brief Branchless min operation
     * 
     * @param a First value
     * @param b Second value 
     * @return The minimum of a and b
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    static T min(T a, T b) {
        // If a <= b, mask will be all 1s, otherwise all 0s
        T mask = -(a <= b);
        return (mask & a) | (~mask & b);
    }
    
    /**
     * @brief Branchless max operation
     * 
     * @param a First value
     * @param b Second value
     * @return The maximum of a and b
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    static T max(T a, T b) {
        // If a >= b, mask will be all 1s, otherwise all 0s
        T mask = -(a >= b);
        return (mask & a) | (~mask & b);
    }
    
    /**
     * @brief Branchless absolute value
     * 
     * @param x Input value
     * @return The absolute value of x
     */
    template <typename T, typename = std::enable_if_t<std::is_signed_v<T> && std::is_integral_v<T>>>
    static T abs(T x) {
        // Get the sign bit (0 if positive, all 1s if negative)
        T mask = x >> (sizeof(T) * 8 - 1);
        // XOR with mask and subtract mask to negate if negative
        return (x ^ mask) - mask;
    }
    
    /**
     * @brief Branchless sign function
     * 
     * @param x Input value
     * @return -1 if x < 0, 0 if x == 0, 1 if x > 0
     */
    template <typename T, typename = std::enable_if_t<std::is_signed_v<T> && std::is_integral_v<T>>>
    static T sign(T x) {
        // Compute (x > 0) - (x < 0) without branches
        return (T(0) < x) - (x < T(0));
    }
    
    /**
     * @brief Branchless selection operation (like a ternary operator)
     * 
     * @param condition Condition value (0 for false, non-zero for true)
     * @param if_true Value to return if condition is true
     * @param if_false Value to return if condition is false
     * @return if_true if condition is non-zero, if_false otherwise
     */
    template <typename T, typename C, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    static T select(C condition, T if_true, T if_false) {
        // Convert condition to a mask (all 1s if true, all 0s if false)
        T mask = -static_cast<T>(condition != 0);
        return (mask & if_true) | (~mask & if_false);
    }
    
    /**
     * @brief Branchless implementation of a clamp operation
     * 
     * @param x Value to clamp
     * @param low Lower bound
     * @param high Upper bound
     * @return x clamped between low and high
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    static T clamp(T x, T low, T high) {
        // First clamp to the upper bound
        T mask1 = -(x <= high);
        T result = (mask1 & x) | (~mask1 & high);
        
        // Then clamp to the lower bound
        T mask2 = -(result >= low);
        return (mask2 & result) | (~mask2 & low);
    }
};

} // namespace math
} // namespace rad_ml 
