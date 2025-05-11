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

#include <functional>
#include <algorithm>

namespace rad_ml {
namespace tmr {

/**
 * @brief Triple Modular Redundancy implementation
 * 
 * Basic implementation of Triple Modular Redundancy for fault tolerance.
 * Uses three copies of the data and majority voting to correct single errors.
 * 
 * @tparam T Type of the data to be protected
 */
template <typename T>
class TMR {
public:
    /**
     * @brief Constructor
     * 
     * @param initial_value Initial value
     * @param on_error_callback Optional callback when error is detected
     */
    explicit TMR(const T& initial_value = T(), 
               std::function<void(const T&, const T&)> on_error_callback = nullptr)
        : on_error_callback_(on_error_callback) {
        // Initialize all copies with the same value
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = initial_value;
        }
    }
    
    /**
     * @brief Get the current value with majority voting
     * 
     * @return Current value based on majority voting
     */
    T get() const {
        // Use majority voting to determine the correct value
        return performMajorityVote();
    }
    
    /**
     * @brief Set a new value to all copies
     * 
     * @param value New value
     */
    void set(const T& value) {
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = value;
        }
    }
    
    /**
     * @brief Get the raw value of a specific copy (for testing)
     * 
     * @param index Index of the copy (0-2)
     * @return Raw value of the specified copy
     */
    T getRawCopy(size_t index) const {
        if (index < num_copies_) {
            return copies_[index];
        }
        return copies_[0]; // Default to first copy if out of bounds
    }
    
    /**
     * @brief Set the raw value of a specific copy (for testing)
     * 
     * @param index Index of the copy (0-2)
     * @param value New value
     */
    void setRawCopy(size_t index, const T& value) {
        if (index < num_copies_) {
            copies_[index] = value;
        }
    }
    
    /**
     * @brief Error statistics structure
     */
    struct ErrorStats {
        size_t detected_errors = 0;
        size_t corrected_errors = 0;
        size_t uncorrectable_errors = 0;
    };
    
    /**
     * @brief Get error statistics
     * 
     * @return Error statistics
     */
    ErrorStats getErrorStats() const {
        return stats_;
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        stats_ = ErrorStats();
    }
    
protected:
    static constexpr size_t num_copies_ = 3;
    T copies_[num_copies_];
    
    // Error tracking
    mutable ErrorStats stats_;
    
    // Optional error callback
    std::function<void(const T&, const T&)> on_error_callback_;
    
    /**
     * @brief Perform majority voting
     * 
     * @return Voted value
     */
    T performMajorityVote() const {
        // If all values match, return any copy
        if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
            return copies_[0];
        }
        
        // We have a disagreement - at least one copy is corrupted
        stats_.detected_errors++;
        
        // Find the majority value using voting
        if (copies_[0] == copies_[1]) {
            // First and second copies agree
            stats_.corrected_errors++;
            
            // Notify about the error if callback is registered
            if (on_error_callback_) {
                on_error_callback_(copies_[0], copies_[2]);
            }
            
            return copies_[0];
        } 
        else if (copies_[0] == copies_[2]) {
            // First and third copies agree
            stats_.corrected_errors++;
            
            // Notify about the error if callback is registered
            if (on_error_callback_) {
                on_error_callback_(copies_[0], copies_[1]);
            }
            
            return copies_[0];
        } 
        else if (copies_[1] == copies_[2]) {
            // Second and third copies agree
            stats_.corrected_errors++;
            
            // Notify about the error if callback is registered
            if (on_error_callback_) {
                on_error_callback_(copies_[1], copies_[0]);
            }
            
            return copies_[1];
        } 
        else {
            // All three values are different - cannot determine the correct value
            stats_.uncorrectable_errors++;
            
            // Return the first copy as a fallback
            return copies_[0];
        }
    }
};

} // namespace tmr
} // namespace rad_ml 
