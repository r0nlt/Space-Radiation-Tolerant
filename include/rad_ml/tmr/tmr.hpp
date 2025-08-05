#pragma once

#include <algorithm>
#include <functional>

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
        : on_error_callback_(on_error_callback)
    {
        // Initialize all copies with the same value
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = initial_value;
        }
    }

    /**
     * @brief Copy constructor
     *
     * @param other TMR object to copy
     */
    TMR(const TMR& other) : on_error_callback_(other.on_error_callback_), stats_(other.stats_)
    {
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = other.copies_[i];
        }
    }

    /**
     * @brief Copy assignment operator
     *
     * @param other TMR object to copy
     * @return Reference to this object
     */
    TMR& operator=(const TMR& other)
    {
        if (this != &other) {
            on_error_callback_ = other.on_error_callback_;
            stats_ = other.stats_;
            for (size_t i = 0; i < num_copies_; i++) {
                copies_[i] = other.copies_[i];
            }
        }
        return *this;
    }

    /**
     * @brief Get the current value with majority voting
     *
     * @return Current value based on majority voting
     */
    T get() const
    {
        // Use majority voting to determine the correct value
        return performMajorityVote();
    }

    /**
     * @brief Get the current value (alias for get())
     *
     * @return Current value based on majority voting
     */
    T getValue() const { return get(); }

    /**
     * @brief Set a new value to all copies
     *
     * @param value New value
     */
    void set(const T& value)
    {
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = value;
        }
    }

    /**
     * @brief Set a new value (alias for set())
     *
     * @param value New value
     */
    void setValue(const T& value) { set(value); }

    /**
     * @brief Correct errors using majority voting
     *
     * @return True if correction was successful
     */
    bool correct()
    {
        T corrected_value = performMajorityVote();
        bool had_error = false;

        // Check if any copy differs from the majority
        for (size_t i = 0; i < num_copies_; i++) {
            if (copies_[i] != corrected_value) {
                copies_[i] = corrected_value;
                had_error = true;
            }
        }

        if (had_error) {
            stats_.corrected_errors++;
        }

        return true;
    }

    /**
     * @brief Check integrity of all copies
     *
     * @return True if all copies are consistent
     */
    bool checkIntegrity() const
    {
        T first_value = copies_[0];
        for (size_t i = 1; i < num_copies_; i++) {
            if (copies_[i] != first_value) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Get the raw value of a specific copy (for testing)
     *
     * @param index Index of the copy (0-2)
     * @return Raw value of the specified copy
     */
    T getRawCopy(size_t index) const
    {
        if (index < num_copies_) {
            return copies_[index];
        }
        return copies_[0];  // Default to first copy if out of bounds
    }

    /**
     * @brief Set the raw value of a specific copy (for testing)
     *
     * @param index Index of the copy (0-2)
     * @param value New value
     */
    void setRawCopy(size_t index, const T& value)
    {
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
    ErrorStats getErrorStats() const { return stats_; }

    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() { stats_ = ErrorStats(); }

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
    T performMajorityVote() const
    {
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

}  // namespace tmr
}  // namespace rad_ml
