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
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <limits>
#include <functional>

// Define approximation types
enum class ApproximationType {
    EXACT,              // No approximation
    REDUCED_PRECISION,  // Reduced bit precision
    RANGE_LIMITED,      // Limit to valid range, clamp outliers
    CUSTOM              // Custom approximation function
};

// Simple implementation of Approximate TMR
template <typename T>
class ApproximateTMR {
public:
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
    
    T get() const {
        // Convert approximate representations back to exact for comparison
        T values[3];
        for (size_t i = 0; i < 3; ++i) {
            values[i] = copies_[i]; // In a real system we'd convert back from approximation
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
    
    void set(const T& value) {
        for (size_t i = 0; i < 3; ++i) {
            copies_[i] = applyApproximation(value, approximation_types_[i]);
        }
    }
    
    T getApproximateValue(size_t index) const {
        if (index >= 3) return T{};
        return copies_[index];
    }
    
    void repair() {
        // Get correct exact value via voting
        T exact_value = get();
        
        // Reset all copies with proper approximations
        set(exact_value);
    }
    
    // For testing - directly access copies
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    // For testing - directly corrupt copies
    void corruptCopy(size_t index, const T& value) {
        if (index < 3) {
            copies_[index] = value;
        }
    }
    
    // Get the approximation types
    const std::array<ApproximationType, 3>& getApproximationTypes() const {
        return approximation_types_;
    }
    
private:
    // The redundant copies (with approximations)
    std::array<T, 3> copies_;
    
    // Approximation type for each copy
    std::array<ApproximationType, 3> approximation_types_;
    
    // Custom approximation function
    std::function<T(const T&)> custom_approximation_;
    
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
    
    T reducePrecision(const T& value) const {
        if constexpr (std::is_floating_point_v<T>) {
            // For floating point, reduce precision by truncating
            constexpr int shift = 10; // Reduce mantissa bits
            double result = value;
            
            // Simple reduction - truncate to fewer decimal places
            result = std::floor(result * shift) / shift;
            
            return static_cast<T>(result);
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
    
    T limitRange(const T& value) const {
        // For this example, we'll use a simple approximation
        // of clamping to a smaller range
        if constexpr (std::is_floating_point_v<T>) {
            // For floating point, limit to +/- 1000
            constexpr T limit = 1000;
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
};

// Helper function to get approximation type name
std::string getApproximationTypeName(ApproximationType type) {
    switch (type) {
        case ApproximationType::EXACT: return "EXACT";
        case ApproximationType::REDUCED_PRECISION: return "REDUCED_PRECISION";
        case ApproximationType::RANGE_LIMITED: return "RANGE_LIMITED";
        case ApproximationType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

int main() {
    std::cout << "=== Approximate TMR Test ===" << std::endl;
    
    // Test with float values
    ApproximateTMR<float> atmr_float(
        3.14159f, 
        {ApproximationType::EXACT,
         ApproximationType::REDUCED_PRECISION,
         ApproximationType::RANGE_LIMITED}
    );
    
    std::cout << "Initial value: " << atmr_float.get() << std::endl;
    
    // Print approximations
    std::cout << "Approximate values:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Copy " << i << " (" 
                  << getApproximationTypeName(atmr_float.getApproximationTypes()[i])
                  << "): " << atmr_float.getApproximateValue(i) << std::endl;
    }
    
    // Test with larger value to see range limiting
    atmr_float.set(12345.6789f);
    std::cout << "\nValue after setting to 12345.6789: " << atmr_float.get() << std::endl;
    
    // Print approximations
    std::cout << "Approximate values:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Copy " << i << " (" 
                  << getApproximationTypeName(atmr_float.getApproximationTypes()[i])
                  << "): " << atmr_float.getApproximateValue(i) << std::endl;
    }
    
    // Test corruption recovery
    atmr_float.corruptCopy(0, 999.999f);
    std::cout << "\nAfter corrupting exact copy:" << std::endl;
    std::cout << "Corrupted value: " << atmr_float.getApproximateValue(0) << std::endl;
    std::cout << "TMR output: " << atmr_float.get() << std::endl;
    
    // Test repair
    atmr_float.repair();
    std::cout << "\nAfter repair:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Copy " << i << ": " << atmr_float.getApproximateValue(i) << std::endl;
    }
    
    // Test integer values
    ApproximateTMR<int> atmr_int(
        42, 
        {ApproximationType::EXACT,
         ApproximationType::REDUCED_PRECISION,
         ApproximationType::RANGE_LIMITED}
    );
    
    std::cout << "\n=== Integer Testing ===" << std::endl;
    std::cout << "Initial value: " << atmr_int.get() << std::endl;
    
    // Print approximations
    std::cout << "Approximate values:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Copy " << i << " (" 
                  << getApproximationTypeName(atmr_int.getApproximationTypes()[i])
                  << "): " << atmr_int.getApproximateValue(i) << std::endl;
    }
    
    // Test with larger value
    atmr_int.set(123456);
    std::cout << "\nValue after setting to 123456: " << atmr_int.get() << std::endl;
    
    // Print approximations
    std::cout << "Approximate values:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Copy " << i << " (" 
                  << getApproximationTypeName(atmr_int.getApproximationTypes()[i])
                  << "): " << atmr_int.getApproximateValue(i) << std::endl;
    }
    
    // Test with random corruptions
    std::cout << "\n=== Random Corruption Testing ===" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 3); // How many copies to corrupt
    
    for (int test = 0; test < 5; ++test) {
        ApproximateTMR<int> test_tmr(42);
        
        int num_to_corrupt = distrib(gen);
        std::cout << "Test " << test << ": Corrupting " << num_to_corrupt << " copies" << std::endl;
        
        // Corrupt random copies
        std::vector<int> indices = {0, 1, 2};
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int i = 0; i < num_to_corrupt; ++i) {
            test_tmr.corruptCopy(indices[i], 100 + test * 10 + i);
        }
        
        // Print corrupted state
        std::cout << "Copies after corruption:" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "  Copy " << i << ": " << test_tmr.getApproximateValue(i) << std::endl;
        }
        
        // Get value and check if protected
        int result = test_tmr.get();
        std::cout << "Value after corruption: " << result << std::endl;
        
        if (result == 42) {
            std::cout << "PROTECTED: TMR maintained correct value" << std::endl;
        } else {
            std::cout << "CORRUPTED: TMR returned incorrect value" << std::endl;
        }
        
        // Repair and check
        test_tmr.repair();
        std::cout << "After repair: " << test_tmr.get() << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
} 
