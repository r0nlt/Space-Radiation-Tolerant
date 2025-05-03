#pragma once

#include "rad_ml/core/redundancy/enhanced_tmr.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <limits>
#include <cmath>

namespace rad_ml {
namespace tmr {

/**
 * @brief Approximation Type Enum
 * 
 * Specifies the type of approximation to use for each module
 */
enum class ApproximationType {
    EXACT,              // No approximation
    REDUCED_PRECISION,  // Reduced bit precision
    RANGE_LIMITED,      // Limit to valid range, clamp outliers
    CUSTOM              // Custom approximation function
};

/**
 * @brief Approximate Triple Modular Redundancy
 * 
 * Implements ATMR using configurable approximation strategies
 * for the redundant modules to reduce resource usage while
 * maintaining fault tolerance.
 * 
 * @tparam T The type of the value to protect with ATMR
 */
template <typename T>
class ApproximateTMR {
public:
    /**
     * @brief Constructor with initial value and approximation types
     * 
     * @param initial_value Initial value for all copies
     * @param approx_types Array of approximation types for each copy
     * @param custom_approx Custom approximation function (if needed)
     */
    ApproximateTMR(
        const T& initial_value,
        const std::array<ApproximationType, 3>& approx_types = {
            ApproximationType::EXACT,
            ApproximationType::REDUCED_PRECISION,
            ApproximationType::RANGE_LIMITED
        },
        std::function<T(const T&)> custom_approx = nullptr
    ) : approximation_types_(approx_types),
        custom_approximation_(custom_approx) {
        
        // Set initial value with appropriate approximations
        set(initial_value);
    }
    
    /**
     * @brief Get the current value using majority voting
     * 
     * @return The correct value based on majority voting
     */
    T get() const {
        // Convert approximate representations back to exact for comparison
        T values[3];
        for (size_t i = 0; i < 3; ++i) {
            values[i] = getExactValue(i);
        }
        
        // Majority voting
        if (values[0] == values[1]) {
            return values[0];
        } else if (values[0] == values[2]) {
            return values[0];
        } else if (values[1] == values[2]) {
            return values[1];
        }
        
        // No majority - return the exact copy
        for (size_t i = 0; i < 3; ++i) {
            if (approximation_types_[i] == ApproximationType::EXACT) {
                return values[i];
            }
        }
        
        // Fallback to first copy if no exact copy exists
        return values[0];
    }
    
    /**
     * @brief Set the value in all three copies with appropriate approximations
     * 
     * @param value The exact value to set
     */
    void set(const T& value) {
        for (size_t i = 0; i < 3; ++i) {
            copies_[i] = applyApproximation(value, approximation_types_[i]);
        }
        
        // Recalculate checksums
        for (size_t i = 0; i < 3; ++i) {
            checksums_[i] = calculateChecksum(copies_[i]);
        }
    }
    
    /**
     * @brief Get the approximate representation for a specific copy
     * 
     * @param index Index of the copy (0-2)
     * @return The approximate value
     */
    T getApproximateValue(size_t index) const {
        if (index >= 3) return T{};
        return copies_[index];
    }
    
    /**
     * @brief Verify integrity of all copies
     * 
     * @return true if all copies' checksums are valid
     */
    bool verify() const {
        for (size_t i = 0; i < 3; ++i) {
            if (!verifyChecksum(i)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Repair any corrupted values
     */
    void repair() {
        // Get correct exact value via voting
        T exact_value = get();
        
        // Reset all copies with proper approximations
        set(exact_value);
    }
    
private:
    // The redundant copies (with approximations)
    std::array<T, 3> copies_;
    
    // Checksums for each value
    std::array<uint32_t, 3> checksums_;
    
    // Approximation type for each copy
    std::array<ApproximationType, 3> approximation_types_;
    
    // Custom approximation function
    std::function<T(const T&)> custom_approximation_;
    
    /**
     * @brief Apply approximation to a value
     * 
     * @param value The exact value
     * @param approx_type Type of approximation to apply
     * @return The approximate value
     */
    T applyApproximation(const T& value, ApproximationType approx_type) const {
        switch (approx_type) {
            case ApproximationType::EXACT:
                return value;
                
            case ApproximationType::REDUCED_PRECISION:
                return reducePrecision(value);
                
            case ApproximationType::RANGE_LIMITED:
                return limitRange(value);
                
            case ApproximationType::CUSTOM:
                if (custom_approximation_) {
                    return custom_approximation_(value);
                }
                return value;
                
            default:
                return value;
        }
    }
    
    /**
     * @brief Get exact value from an approximate representation
     * 
     * @param index Index of the copy (0-2)
     * @return The exact value
     */
    T getExactValue(size_t index) const {
        // For most approximations, we just return the value as-is
        // because the approximation is designed to preserve voting.
        // For custom approximations, we would need more sophisticated
        // conversion if needed.
        return copies_[index];
    }
    
    /**
     * @brief Reduce precision of a value
     * 
     * @param value The exact value
     * @return The value with reduced precision
     */
    T reducePrecision(const T& value) const {
        if constexpr (std::is_floating_point_v<T>) {
            // For floating point, reduce precision by truncating
            constexpr int shift = 5; // Reduce mantissa bits
            T result = value;
            
            // Type punning to manipulate bits safely
            union {
                T f;
                uint64_t i;
            } u;
            
            u.f = result;
            // Clear lower bits of mantissa
            u.i &= ~((1ULL << shift) - 1);
            result = u.f;
            
            return result;
        } 
        else if constexpr (std::is_integral_v<T>) {
            // For integers, mask off lower bits
            constexpr int shift = sizeof(T) <= 2 ? 2 : 3;
            return (value >> shift) << shift;
        }
        else {
            // For other types, just return as is
            return value;
        }
    }
    
    /**
     * @brief Limit value to a predefined range
     * 
     * @param value The exact value
     * @return The value clamped to valid range
     */
    T limitRange(const T& value) const {
        // For this example, we'll use a simple approximation
        // of clamping to a smaller range
        if constexpr (std::is_floating_point_v<T>) {
            // For floating point, limit to +/- 1e6
            constexpr T limit = 1e6;
            return std::max(std::min(value, limit), -limit);
        } 
        else if constexpr (std::is_integral_v<T>) {
            // For integers, limit to half the representable range
            constexpr T limit = std::numeric_limits<T>::max() / 2;
            return std::max(std::min(value, limit), static_cast<T>(-limit));
        }
        else {
            // For other types, return as is
            return value;
        }
    }
    
    /**
     * @brief Calculate checksum for a value
     * 
     * @param value Value to calculate checksum for
     * @return Checksum value
     */
    uint32_t calculateChecksum(const T& value) const {
        // CRC-32 implementation (simplified for example)
        const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
        uint32_t crc = 0xFFFFFFFF;
        
        for (size_t i = 0; i < sizeof(T); ++i) {
            uint8_t byte = data[i];
            crc ^= byte;
            for (size_t j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        
        return ~crc;
    }
    
    /**
     * @brief Verify checksum for a specific value
     * 
     * @param index Index of the value (0-2)
     * @return true if checksum is valid
     */
    bool verifyChecksum(size_t index) const {
        if (index >= 3) return false;
        return checksums_[index] == calculateChecksum(copies_[index]);
    }
};

} // namespace tmr
} // namespace rad_ml 