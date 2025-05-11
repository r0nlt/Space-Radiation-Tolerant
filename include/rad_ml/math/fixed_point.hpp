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
 * @brief Fixed-point arithmetic implementation
 * 
 * This class implements fixed-point arithmetic for deterministic calculations
 * in radiation environments. It provides precise control over the number of
 * bits used for the integer and fractional parts, allowing for predictable
 * numeric behavior.
 * 
 * @tparam IntBits Number of bits for the integer part
 * @tparam FracBits Number of bits for the fractional part
 * @tparam T Underlying integer type
 */
template <unsigned IntBits, unsigned FracBits, typename T = std::int32_t>
class FixedPoint {
    static_assert(std::is_integral_v<T>, "Base type must be integral");
    static_assert(IntBits + FracBits <= sizeof(T) * 8, "Total bits must fit in the underlying type");
    
public:
    using value_type = T;
    
    static constexpr unsigned integer_bits = IntBits;
    static constexpr unsigned fractional_bits = FracBits;
    static constexpr T scale = static_cast<T>(1) << FracBits;
    
    // Constructors
    constexpr FixedPoint() noexcept : value_(0) {}
    
    constexpr explicit FixedPoint(float value) noexcept
        : value_(static_cast<T>(value * scale)) {}
    
    constexpr explicit FixedPoint(double value) noexcept
        : value_(static_cast<T>(value * scale)) {}
    
    constexpr explicit FixedPoint(int value) noexcept
        : value_(static_cast<T>(value * scale)) {}
    
    // Conversion to floating-point
    [[nodiscard]] constexpr float to_float() const noexcept {
        return static_cast<float>(value_) / scale;
    }
    
    [[nodiscard]] constexpr double to_double() const noexcept {
        return static_cast<double>(value_) / scale;
    }
    
    // Arithmetic operations
    constexpr FixedPoint operator+(const FixedPoint& other) const noexcept {
        FixedPoint result;
        result.value_ = value_ + other.value_;
        return result;
    }
    
    constexpr FixedPoint operator-(const FixedPoint& other) const noexcept {
        FixedPoint result;
        result.value_ = value_ - other.value_;
        return result;
    }
    
    constexpr FixedPoint operator*(const FixedPoint& other) const noexcept {
        // For multiplication, we need to be careful about overflow
        using wider_t = typename std::conditional<
            sizeof(T) <= 4,
            std::int64_t,
            std::int64_t  // Would use int128_t if available
        >::type;
        
        wider_t wide_result = static_cast<wider_t>(value_) * other.value_;
        
        // Scale back down
        FixedPoint result;
        result.value_ = static_cast<T>(wide_result >> FracBits);
        return result;
    }
    
    constexpr FixedPoint operator/(const FixedPoint& other) const noexcept {
        // For division, scale up first to maintain precision
        using wider_t = typename std::conditional<
            sizeof(T) <= 4,
            std::int64_t,
            std::int64_t  // Would use int128_t if available
        >::type;
        
        wider_t scaled_dividend = static_cast<wider_t>(value_) << FracBits;
        
        // Compute the division
        FixedPoint result;
        result.value_ = static_cast<T>(scaled_dividend / other.value_);
        return result;
    }
    
    // Assignment operators
    constexpr FixedPoint& operator+=(const FixedPoint& other) noexcept {
        value_ += other.value_;
        return *this;
    }
    
    constexpr FixedPoint& operator-=(const FixedPoint& other) noexcept {
        value_ -= other.value_;
        return *this;
    }
    
    constexpr FixedPoint& operator*=(const FixedPoint& other) noexcept {
        *this = *this * other;
        return *this;
    }
    
    constexpr FixedPoint& operator/=(const FixedPoint& other) noexcept {
        *this = *this / other;
        return *this;
    }
    
    // Comparison operators
    constexpr bool operator==(const FixedPoint& other) const noexcept {
        return value_ == other.value_;
    }
    
    constexpr bool operator!=(const FixedPoint& other) const noexcept {
        return value_ != other.value_;
    }
    
    constexpr bool operator<(const FixedPoint& other) const noexcept {
        return value_ < other.value_;
    }
    
    constexpr bool operator<=(const FixedPoint& other) const noexcept {
        return value_ <= other.value_;
    }
    
    constexpr bool operator>(const FixedPoint& other) const noexcept {
        return value_ > other.value_;
    }
    
    constexpr bool operator>=(const FixedPoint& other) const noexcept {
        return value_ >= other.value_;
    }
    
    // Access to raw value
    [[nodiscard]] constexpr T raw_value() const noexcept {
        return value_;
    }
    
private:
    T value_;  // The fixed-point value
};

// Common type aliases
using Fixed16_16 = FixedPoint<16, 16, std::int32_t>;  // 16.16 fixed-point (1 bit for sign)
using Fixed8_8 = FixedPoint<8, 8, std::int16_t>;      // 8.8 fixed-point (1 bit for sign)
using Fixed8_24 = FixedPoint<8, 24, std::int32_t>;    // 8.24 fixed-point (1 bit for sign)

} // namespace math
} // namespace rad_ml 
