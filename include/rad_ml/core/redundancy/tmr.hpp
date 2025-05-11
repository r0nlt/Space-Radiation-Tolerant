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

#include <array>
#include <type_traits>
#include <functional>

namespace rad_ml {
namespace core {
namespace redundancy {

/**
 * @brief Triple Modular Redundancy implementation
 * 
 * This class implements Triple Modular Redundancy (TMR) for fault tolerance.
 * It stores three copies of a value and uses majority voting to determine
 * the correct value. This provides protection against Single Event Upsets (SEUs)
 * that might corrupt memory in radiation environments.
 * 
 * @tparam T The type of the value to protect with TMR
 */
template <typename T>
class TripleModularRedundancy {
public:
    // Default constructor
    constexpr TripleModularRedundancy() noexcept {
        values_[0] = T{};
        values_[1] = T{};
        values_[2] = T{};
    }
    
    // Constructor with initial value
    constexpr explicit TripleModularRedundancy(const T& value) noexcept {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
    }
    
    // Get the value using majority voting
    [[nodiscard]] T get() const {
        // Majority voting
        if (values_[0] == values_[1]) {
            return values_[0];
        } else if (values_[0] == values_[2]) {
            return values_[0];
        } else if (values_[1] == values_[2]) {
            return values_[1];
        }
        
        // If no majority, use the first value
        // In a real implementation, this would log an error or trigger a fallback
        return values_[0];
    }
    
    // Set the value in all three copies
    void set(const T& value) noexcept {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
    }
    
    // Repair any corrupted values
    void repair() {
        const T correct_value = get();
        values_[0] = correct_value;
        values_[1] = correct_value;
        values_[2] = correct_value;
    }
    
    // Conversion operator
    explicit operator T() const {
        return get();
    }
    
    // Assignment operator
    TripleModularRedundancy& operator=(const T& value) {
        set(value);
        return *this;
    }
    
private:
    std::array<T, 3> values_;
};

// Template alias for convenience
template <typename T>
using TMR = TripleModularRedundancy<T>;

} // namespace redundancy
} // namespace core
} // namespace rad_ml 
