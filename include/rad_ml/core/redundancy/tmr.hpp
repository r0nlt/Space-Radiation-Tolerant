#pragma once

#include <array>
#include <functional>
#include <type_traits>

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
class alignas(64) TripleModularRedundancy {  // PERFORMANCE: Cache line alignment
   public:
    // Default constructor
    constexpr TripleModularRedundancy() noexcept
    {
        values_[0] = T{};
        values_[1] = T{};
        values_[2] = T{};
    }

    // Constructor with initial value
    constexpr explicit TripleModularRedundancy(const T& value) noexcept
    {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
    }

    // PERFORMANCE OPTIMIZED: Get the value using fast majority voting
    [[nodiscard]] T get() const noexcept
    {
        // Fast path: if all values are equal (90%+ of cases)
        if (values_[0] == values_[1] && values_[1] == values_[2]) {
            return values_[0];
        }

        // Optimized majority voting
        if (values_[0] == values_[1]) {
            return values_[0];
        }
        else if (values_[0] == values_[2]) {
            return values_[0];
        }
        else {
            return values_[1];  // values_[1] == values_[2] OR fallback
        }
    }

    // Set the value in all three copies
    void set(const T& value) noexcept
    {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
    }

    // Repair any corrupted values
    void repair() noexcept
    {
        const T correct_value = get();
        values_[0] = correct_value;
        values_[1] = correct_value;
        values_[2] = correct_value;
    }

    // Conversion operator
    explicit operator T() const noexcept { return get(); }

    // Assignment operator
    TripleModularRedundancy& operator=(const T& value) noexcept
    {
        set(value);
        return *this;
    }

    // PERFORMANCE: Check for disagreements without full voting
    [[nodiscard]] bool hasErrors() const noexcept
    {
        return !(values_[0] == values_[1] && values_[1] == values_[2]);
    }

    // PERFORMANCE: Get raw access for testing (avoid in production)
    [[nodiscard]] const std::array<T, 3>& getRawValues() const noexcept { return values_; }

   private:
    alignas(64) std::array<T, 3> values_;  // PERFORMANCE: Cache line aligned array
};

// Template alias for convenience
template <typename T>
using TMR = TripleModularRedundancy<T>;

}  // namespace redundancy
}  // namespace core
}  // namespace rad_ml
