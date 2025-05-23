/**
 * @file space_enhanced_tmr.hpp
 * @brief Space-flight optimized Enhanced Triple Modular Redundancy
 *
 * This file provides a space-flight optimized version of enhanced TMR
 * with fixed memory allocation and status code-based error handling.
 */

#ifndef RAD_ML_SPACE_ENHANCED_TMR_HPP
#define RAD_ML_SPACE_ENHANCED_TMR_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "../error/status_code.hpp"
#include "../memory/fixed_containers.hpp"
#include "../space_flight_config.hpp"
#include "tmr.hpp"

namespace rad_ml {
namespace core {
namespace redundancy {

/**
 * CRC calculation utility optimized for space flight
 */
class SpaceCRC {
   public:
    /**
     * @brief Calculate CRC32 checksum for arbitrary data (fixed size algorithm)
     *
     * @param data Pointer to the data
     * @param size Size of the data in bytes
     * @return CRC32 checksum
     */
    static uint32_t calculate(const void* data, size_t size)
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFF;

        // Use unrolled loop for deterministic execution time
        for (size_t i = 0; i < size; ++i) {
            crc ^= bytes[i];

            // Manually unrolled loop for all 8 bits
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }

        return ~crc;
    }

    /**
     * @brief Calculate CRC32 checksum for a value of any type
     *
     * @tparam T Type of the value
     * @param value The value to calculate CRC for
     * @return CRC32 checksum
     */
    template <typename T>
    static uint32_t calculateForValue(const T& value)
    {
        return calculate(&value, sizeof(T));
    }
};

/**
 * Fault pattern categories to optimize voting strategy
 */
enum class SpaceFaultPattern {
    SINGLE_BIT,     // Single-Event Upset (SEU)
    ADJACENT_BITS,  // Multiple-Cell Upset (MCU)
    BYTE_ERROR,     // Byte-level corruption
    WORD_ERROR,     // 32-bit word corruption
    BURST_ERROR,    // Clustered errors
    UNKNOWN         // Unknown pattern (default to conservative strategy)
};

/**
 * Error statistics structure for TMR
 */
struct SpaceErrorStats {
    uint64_t detected_errors;       ///< Total number of detected errors
    uint64_t corrected_errors;      ///< Number of errors that were corrected
    uint64_t uncorrectable_errors;  ///< Number of errors that couldn't be corrected

    /**
     * Reset all counters to zero
     */
    void reset()
    {
        detected_errors = 0;
        corrected_errors = 0;
        uncorrectable_errors = 0;
    }
};

/**
 * @brief Space-flight optimized enhanced TMR implementation
 *
 * This class implements enhanced TMR with specialized voting algorithms
 * optimized for space flight with deterministic behavior, fixed memory
 * allocation, and status code-based error handling.
 *
 * @tparam T The type of the value to protect
 */
template <typename T>
class SpaceEnhancedTMR {
   public:
    /**
     * @brief Default constructor
     */
    SpaceEnhancedTMR() noexcept
    {
        values_[0] = T{};
        values_[1] = T{};
        values_[2] = T{};
        recalculateChecksums();
        error_stats_.reset();
    }

    /**
     * @brief Constructor with initial value
     *
     * @param value Initial value for all copies
     */
    explicit SpaceEnhancedTMR(const T& value) noexcept
    {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
        recalculateChecksums();
        error_stats_.reset();
    }

    /**
     * @brief Copy constructor
     *
     * @param other EnhancedTMR to copy from
     */
    SpaceEnhancedTMR(const SpaceEnhancedTMR& other) noexcept
    {
        values_ = other.values_;
        checksums_ = other.checksums_;

        // Don't copy error stats
        error_stats_.reset();
    }

    /**
     * @brief Assignment operator
     *
     * @param other EnhancedTMR to copy from
     * @return Reference to this object
     */
    SpaceEnhancedTMR& operator=(const SpaceEnhancedTMR& other) noexcept
    {
        if (this != &other) {
            values_ = other.values_;
            checksums_ = other.checksums_;

            // Don't copy error stats
            error_stats_.reset();
        }
        return *this;
    }

    /**
     * @brief Get the correct value using majority voting
     *
     * @param[out] value Reference to store the result
     * @return Status code indicating success or the specific error
     */
    error::StatusCode get(T& value) const
    {
        // First check checksums to see if any copy is definitely corrupted
        bool checksum_valid[3] = {verifyChecksum(0), verifyChecksum(1), verifyChecksum(2)};

        // Count valid copies
        int valid_count = 0;
        for (int i = 0; i < 3; ++i) {
            if (checksum_valid[i]) valid_count++;
        }

        // If all copies valid, do normal TMR voting
        if (valid_count == 3) {
            value = performMajorityVoting();
            return error::StatusCode::SUCCESS;
        }

        // If only some copies valid, use only those for voting
        if (valid_count > 0) {
            // If only one valid, return it
            if (valid_count == 1) {
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        incrementErrorStats(true);
                        value = values_[i];
                        return error::StatusCode::SUCCESS;
                    }
                }
            }

            // If two valid, compare them
            if (valid_count == 2) {
                if (checksum_valid[0] && checksum_valid[1] && values_[0] == values_[1]) {
                    incrementErrorStats(true);
                    value = values_[0];
                    return error::StatusCode::SUCCESS;
                }
                if (checksum_valid[0] && checksum_valid[2] && values_[0] == values_[2]) {
                    incrementErrorStats(true);
                    value = values_[0];
                    return error::StatusCode::SUCCESS;
                }
                if (checksum_valid[1] && checksum_valid[2] && values_[1] == values_[2]) {
                    incrementErrorStats(true);
                    value = values_[1];
                    return error::StatusCode::SUCCESS;
                }

                // Two valid but different values - uncertain which is correct
                incrementErrorStats(false);
                // Use first valid value as best guess
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        value = values_[i];
                        break;
                    }
                }
                return error::StatusCode::REDUNDANCY_FAILURE;
            }
        }

        // If no valid checksums, do a best effort majority vote anyway
        incrementErrorStats(false);
        value = performMajorityVoting();
        return error::StatusCode::REDUNDANCY_FAILURE;
    }

    /**
     * @brief Set value in all three copies
     *
     * @param value Value to set
     */
    void set(const T& value) noexcept
    {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
        recalculateChecksums();
    }

    /**
     * @brief Repair any corrupted values by majority voting
     *
     * @return Status code indicating success or failure of repair
     */
    error::StatusCode repair()
    {
        T correct_value;
        error::StatusCode status = get(correct_value);

        // Set all copies to the correct value
        values_[0] = correct_value;
        values_[1] = correct_value;
        values_[2] = correct_value;
        recalculateChecksums();

        return status;
    }

    /**
     * @brief Verify that all copies are valid according to checksums
     *
     * @return Status code indicating integrity status
     */
    error::StatusCode verify() const
    {
        bool all_valid = true;

        for (int i = 0; i < 3; ++i) {
            if (!verifyChecksum(i)) {
                all_valid = false;
                break;
            }
        }

        if (all_valid) {
            return error::StatusCode::SUCCESS;
        }

        // Check if any two copies match
        if ((values_[0] == values_[1]) || (values_[0] == values_[2]) ||
            (values_[1] == values_[2])) {
            return error::StatusCode::RADIATION_DETECTION;
        }

        return error::StatusCode::REDUNDANCY_FAILURE;
    }

    /**
     * @brief Get error statistics
     *
     * @return Error statistics structure
     */
    SpaceErrorStats getErrorStats() const { return error_stats_; }

    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() { error_stats_.reset(); }

    /**
     * @brief Assignment operator for value type
     *
     * @param value Value to assign
     * @return Reference to this object
     */
    SpaceEnhancedTMR& operator=(const T& value)
    {
        set(value);
        return *this;
    }

#ifdef ENABLE_TESTING
    /**
     * @brief Set a specific copy to a value (for testing only)
     *
     * This method is only available when ENABLE_TESTING is defined.
     * It allows direct manipulation of internal values for testing.
     *
     * @param index Index of the copy to set (0-2)
     * @param value Value to set
     */
    void setForTesting(int index, const T& value)
    {
        if (index >= 0 && index < 3) {
            values_[index] = value;
        }
    }

    /**
     * @brief Force recalculation of checksums (for testing only)
     *
     * This method is only available when ENABLE_TESTING is defined.
     */
    void recalculateChecksumsForTesting() { recalculateChecksums(); }

    /**
     * @brief Get a specific copy (for testing only)
     *
     * This method is only available when ENABLE_TESTING is defined.
     *
     * @param index Index of the copy to get (0-2)
     * @return Copy of the value
     */
    T getForTesting(int index) const
    {
        if (index >= 0 && index < 3) {
            return values_[index];
        }
        return T{};
    }
#endif  // ENABLE_TESTING

   private:
    /**
     * @brief Recalculate checksums for all three copies
     */
    void recalculateChecksums()
    {
        for (int i = 0; i < 3; ++i) {
            checksums_[i] = SpaceCRC::calculateForValue(values_[i]);
        }
    }

    /**
     * @brief Verify checksum for a specific copy
     *
     * @param index Copy index to check
     * @return True if checksum matches
     */
    bool verifyChecksum(int index) const
    {
        return SpaceCRC::calculateForValue(values_[index]) == checksums_[index];
    }

    /**
     * @brief Perform majority voting
     *
     * @return Voted value
     */
    T performMajorityVoting() const
    {
        if (values_[0] == values_[1]) {
            return values_[0];
        }

        if (values_[0] == values_[2]) {
            return values_[0];
        }

        if (values_[1] == values_[2]) {
            return values_[1];
        }

        // If no majority, apply specialized bit-level voting based on type
        if constexpr (std::is_integral<T>::value) {
            return bitLevelVote(values_[0], values_[1], values_[2]);
        }
        else if constexpr (std::is_floating_point<T>::value) {
            // For all different floating point values, use findBestFloatFallback
            // which will return the median value for finite values
            if (std::isfinite(values_[0]) && std::isfinite(values_[1]) &&
                std::isfinite(values_[2])) {
                return findBestFloatFallback(values_[0], values_[1], values_[2]);
            }
            return bitLevelVoteFloat(values_[0], values_[1], values_[2]);
        }
        else {
            // For non-integer and non-floating-point types, return first value as best guess
            return values_[0];
        }
    }

    /**
     * @brief Increment error statistics
     *
     * @param corrected Whether the error was corrected
     */
    void incrementErrorStats(bool corrected) const
    {
        error_stats_.detected_errors++;

        if (corrected) {
            error_stats_.corrected_errors++;
        }
        else {
            error_stats_.uncorrectable_errors++;
        }
    }

    /**
     * @brief Bit-level voting for integer types
     *
     * @param a First value
     * @param b Second value
     * @param c Third value
     * @return Bit-by-bit voted result
     */
    template <typename U = T>
    static typename std::enable_if<std::is_integral<U>::value, U>::type bitLevelVote(const U& a,
                                                                                     const U& b,
                                                                                     const U& c)
    {
        using UintType = typename std::make_unsigned<U>::type;

        UintType ua = static_cast<UintType>(a);
        UintType ub = static_cast<UintType>(b);
        UintType uc = static_cast<UintType>(c);

        // Perform bit-by-bit majority voting
        UintType result = 0;
        UintType mask = 1;

        for (size_t i = 0; i < sizeof(U) * 8; ++i) {
            // Count bits set at this position
            int count = 0;
            if (ua & mask) count++;
            if (ub & mask) count++;
            if (uc & mask) count++;

            // Set result bit if majority is 1
            if (count >= 2) {
                result |= mask;
            }

            mask <<= 1;
        }

        return static_cast<U>(result);
    }

    /**
     * @brief IEEE-754 aware bit-level voting for floating-point types
     *
     * This handles the special structure of IEEE-754 floating-point numbers:
     * - Sign bit (1 bit)
     * - Exponent (8 bits for float, 11 for double)
     * - Mantissa (23 bits for float, 52 for double)
     */
    template <typename U = T>
    static typename std::enable_if<std::is_floating_point<U>::value, U>::type bitLevelVoteFloat(
        const U& a, const U& b, const U& c)
    {
        using FloatBits = typename std::conditional<sizeof(U) == 4, uint32_t, uint64_t>::type;

        // Convert to bit representation
        FloatBits bits_a, bits_b, bits_c;
        std::memcpy(&bits_a, &a, sizeof(U));
        std::memcpy(&bits_b, &b, sizeof(U));
        std::memcpy(&bits_c, &c, sizeof(U));

        // Check for special values (NaN, Infinity)
        if (isSpecialFloat(a) || isSpecialFloat(b) || isSpecialFloat(c)) {
            return handleSpecialFloatVoting(a, b, c);
        }

        // Perform structured bit-level voting
        FloatBits result = 0;

        if constexpr (sizeof(U) == 4) {  // float
            // Sign bit (bit 31)
            result |= majorityBit(bits_a, bits_b, bits_c, 31);

            // Exponent bits (bits 30-23)
            FloatBits exp_result = 0;
            for (int i = 30; i >= 23; --i) {
                exp_result |= majorityBit(bits_a, bits_b, bits_c, i);
            }

            // Validate exponent (not all 0s or all 1s in most cases)
            FloatBits exp_mask = 0x7F800000;
            FloatBits exp_only = exp_result & exp_mask;

            if (exp_only == 0 || exp_only == exp_mask) {
                // Invalid exponent, use nearest valid exponent
                exp_result = findNearestValidExponent(bits_a, bits_b, bits_c);
            }

            result |= exp_result;

            // Mantissa bits (bits 22-0) - can vote normally
            for (int i = 22; i >= 0; --i) {
                result |= majorityBit(bits_a, bits_b, bits_c, i);
            }
        }
        else {  // double
            // Similar structure but with 11-bit exponent, 52-bit mantissa
            result |= majorityBit(bits_a, bits_b, bits_c, 63);  // Sign

            // Exponent (bits 62-52)
            FloatBits exp_result = 0;
            for (int i = 62; i >= 52; --i) {
                exp_result |= majorityBit(bits_a, bits_b, bits_c, i);
            }

            // Validate exponent
            FloatBits exp_mask = 0x7FF0000000000000ULL;
            FloatBits exp_only = exp_result & exp_mask;

            if (exp_only == 0 || exp_only == exp_mask) {
                // Invalid exponent, use nearest valid exponent
                exp_result = findNearestValidExponent(bits_a, bits_b, bits_c);
            }

            result |= exp_result;

            // Mantissa (bits 51-0)
            for (int i = 51; i >= 0; --i) {
                result |= majorityBit(bits_a, bits_b, bits_c, i);
            }
        }

        // Convert back to floating-point
        U final_result;
        std::memcpy(&final_result, &result, sizeof(U));

        // Final validation
        if (std::isnan(final_result) || std::isinf(final_result)) {
            return findBestFloatFallback(a, b, c);
        }

        return final_result;
    }

    /**
     * @brief Helper to get majority bit at specific position
     */
    template <typename BitType>
    static BitType majorityBit(BitType a, BitType b, BitType c, int position)
    {
        BitType mask = BitType(1) << position;
        int count = 0;

        if (a & mask) count++;
        if (b & mask) count++;
        if (c & mask) count++;

        return (count >= 2) ? mask : 0;
    }

    /**
     * @brief Check if floating-point value is special (NaN, Inf, etc.)
     */
    template <typename U>
    static bool isSpecialFloat(const U& value)
    {
        return std::isnan(value) || std::isinf(value) || value == 0.0;
    }

    /**
     * @brief Handle voting when special floating-point values are present
     */
    template <typename U>
    static U handleSpecialFloatVoting(const U& a, const U& b, const U& c)
    {
        // Count finite values
        int finite_count = 0;
        U finite_values[3];

        if (std::isfinite(a)) finite_values[finite_count++] = a;
        if (std::isfinite(b)) finite_values[finite_count++] = b;
        if (std::isfinite(c)) finite_values[finite_count++] = c;

        // If we have finite values, prefer them
        if (finite_count > 0) {
            if (finite_count == 1) {
                return finite_values[0];
            }

            // If two finite values, check if they're close
            if (finite_count >= 2) {
                if (std::abs(finite_values[0] - finite_values[1]) <
                    std::numeric_limits<U>::epsilon() * 1000) {
                    return finite_values[0];
                }
            }

            // Use bit-level voting on finite values only
            return bitLevelVoteFloat(finite_values[0],
                                     finite_count > 1 ? finite_values[1] : finite_values[0],
                                     finite_count > 2 ? finite_values[2] : finite_values[0]);
        }

        // All values are special - return the most common one
        if (a == b) return a;
        if (a == c) return a;
        if (b == c) return b;

        // Check if all values are NaN
        if (std::isnan(a) && std::isnan(b) && std::isnan(c)) {
            return std::numeric_limits<U>::quiet_NaN();
        }

        // All different special values - return zero as safest fallback
        return U(0.0);
    }

    /**
     * @brief Find nearest valid exponent when bit voting produces invalid result
     */
    template <typename BitType>
    static BitType findNearestValidExponent(BitType a, BitType b, BitType c)
    {
        // Extract exponents from the three values
        constexpr int exp_shift = (sizeof(BitType) == 4) ? 23 : 52;
        constexpr BitType exp_mask = (sizeof(BitType) == 4) ? 0xFF : 0x7FF;

        BitType exp_a = (a >> exp_shift) & exp_mask;
        BitType exp_b = (b >> exp_shift) & exp_mask;
        BitType exp_c = (c >> exp_shift) & exp_mask;

        // Find the median exponent as a reasonable compromise
        if (exp_a > exp_b) std::swap(exp_a, exp_b);
        if (exp_b > exp_c) std::swap(exp_b, exp_c);
        if (exp_a > exp_b) std::swap(exp_a, exp_b);

        return exp_b << exp_shift;  // Return median exponent
    }

    /**
     * @brief Fallback when bit-level voting produces invalid result
     */
    template <typename U>
    static U findBestFloatFallback(const U& a, const U& b, const U& c)
    {
        // Return the finite value closest to the median
        if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c)) {
            // Return median value
            if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
            if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
            return c;
        }

        // Return first finite value found
        if (std::isfinite(a)) return a;
        if (std::isfinite(b)) return b;
        if (std::isfinite(c)) return c;

        // All values are non-finite, return 0.0 as safe fallback
        return U(0.0);
    }

    // Storage for the three copies
    std::array<T, 3> values_;

    // Checksums for integrity verification
    mutable std::array<uint32_t, 3> checksums_;

    // Error statistics
    mutable SpaceErrorStats error_stats_;
};

}  // namespace redundancy
}  // namespace core
}  // namespace rad_ml

// Define ENABLE_TESTING for test builds
#define ENABLE_TESTING

#endif  // RAD_ML_SPACE_ENHANCED_TMR_HPP
