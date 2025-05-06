#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <type_traits>
#include <unordered_map>

namespace rad_ml {
namespace tmr {

/**
 * Basic Triple Modular Redundancy (TMR) implementation.
 * Stores three copies of the data and performs majority voting.
 */
template <typename T>
class BasicTMR {
public:
    /**
     * Constructor with initial value
     * @param initial_value The initial value to store
     */
    explicit BasicTMR(const T& initial_value) {
        // Initialize all three copies with the same value
        copies_[0] = initial_value;
        copies_[1] = initial_value;
        copies_[2] = initial_value;
    }

    /**
     * Get the value with majority voting
     * @return The value determined by majority voting
     */
    T get() const {
        // Simple majority voting
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            return copies_[1];
        } else {
            // No majority - return first copy
            // In a real implementation, this could be improved
            return copies_[0];
        }
    }

    /**
     * Set a new value for all copies
     * @param value The new value
     */
    void set(const T& value) {
        copies_[0] = value;
        copies_[1] = value;
        copies_[2] = value;
    }

    /**
     * Check if there are disagreements between copies
     * @return true if disagreements detected, false otherwise
     */
    bool hasErrors() const {
        return !(copies_[0] == copies_[1] && copies_[1] == copies_[2]);
    }

    /**
     * Repair corrupted copies by setting all to the majority value
     */
    void repair() {
        T correct_value = get();
        set(correct_value);
    }

    /**
     * Get direct access to the copies for testing purposes
     * Note: This is generally not recommended in production code,
     * but useful for validating TMR functionality
     */
    const T* getCopies() const {
        return copies_;
    }

private:
    T copies_[3];  // The three redundant copies
};

/**
 * Enhanced TMR implementation with error detection and correction
 * Extends basic TMR with additional features like error checking
 */
template <typename T>
class EnhancedTMR {
public:
    /**
     * Constructor with initial value
     * @param initial_value The initial value to store
     */
    explicit EnhancedTMR(const T& initial_value) 
        : error_count_(0) {
        // Initialize all copies with the same value
        for (int i = 0; i < 4; ++i) {
            copies_[i] = initial_value;
        }
    }

    /**
     * Get the value with enhanced voting
     * @return The value determined by voting
     */
    T get() const {
        // Check if all copies are equal
        if (copies_[0] == copies_[1] && copies_[1] == copies_[2] && copies_[2] == copies_[3]) {
            return copies_[0];
        }
        
        // Count occurrences of each value
        std::unordered_map<T, int> value_counts;
        for (int i = 0; i < 4; ++i) {
            value_counts[copies_[i]]++;
        }
        
        // Find the most common value
        T most_common_value = copies_[0];
        int max_count = 0;
        
        for (const auto& pair : value_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                most_common_value = pair.first;
            }
        }
        
        return most_common_value;
    }

    /**
     * Set a new value for all copies
     * @param value The new value
     */
    void set(const T& value) {
        for (int i = 0; i < 4; ++i) {
            copies_[i] = value;
        }
        error_count_ = 0;
    }

    /**
     * Check if there are disagreements between copies
     * @return true if disagreements detected, false otherwise
     */
    bool hasErrors() const {
        for (int i = 1; i < 4; ++i) {
            if (copies_[0] != copies_[i]) {
                return true;
            }
        }
        return false;
    }

    /**
     * Repair corrupted copies by setting all to the majority value
     * @return true if repair was needed, false otherwise
     */
    bool repair() {
        if (!hasErrors()) {
            return false;
        }
        
        T correct_value = get();
        set(correct_value);
        error_count_++;
        return true;
    }

    /**
     * Get the number of errors detected and repaired
     * @return Error count
     */
    int getErrorCount() const {
        return error_count_;
    }

private:
    T copies_[4];    // Four redundant copies (enhanced over basic TMR)
    int error_count_; // Count of errors detected
};

} // namespace tmr
} // namespace rad_ml 