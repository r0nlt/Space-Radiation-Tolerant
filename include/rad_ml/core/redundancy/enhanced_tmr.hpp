#pragma once

#include <rad_ml/core/redundancy/tmr.hpp>
#include <array>
#include <atomic>
#include <functional>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <cassert>
#include <iostream>

namespace rad_ml {
namespace core {
namespace redundancy {

/**
 * @brief CRC calculation utility
 * 
 * Provides functions for calculating CRC checksums of arbitrary data.
 * This is used to verify the integrity of values in radiation environments.
 */
class CRC {
public:
    /**
     * @brief Calculate CRC32 checksum for arbitrary data
     * 
     * @param data Pointer to the data
     * @param size Size of the data in bytes
     * @return CRC32 checksum
     */
    static uint32_t calculate(const void* data, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFF;
        
        for (size_t i = 0; i < size; ++i) {
            crc ^= bytes[i];
            for (int j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
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
    static uint32_t calculateForValue(const T& value) {
        return calculate(&value, sizeof(T));
    }
};

/**
 * @brief Enhanced Triple Modular Redundancy implementation
 * 
 * Extends basic TMR with additional integrity verification through CRC checksums
 * and detailed error statistics. This provides stronger protection against
 * radiation-induced errors.
 * 
 * @tparam T The type of the value to protect with TMR
 * @tparam ChecksumType The type of checksum to use (default: uint32_t)
 */
template <typename T, typename ChecksumType = uint32_t>
class EnhancedTMR {
public:
    /// Statistics about detected errors
    struct ErrorStats {
        uint64_t detected_errors;       ///< Total number of detected errors
        uint64_t corrected_errors;      ///< Number of errors that were corrected
        uint64_t uncorrectable_errors;  ///< Number of errors that couldn't be corrected
    };
    
    /**
     * @brief Default constructor
     */
    EnhancedTMR() noexcept : values_{T{}, T{}, T{}} {
        recalculateChecksums();
    }
    
    /**
     * @brief Constructor with initial value
     * 
     * @param value Initial value for all copies
     */
    explicit EnhancedTMR(const T& value) noexcept : values_{value, value, value} {
        recalculateChecksums();
    }
    
    /**
     * @brief Copy constructor
     *
     * @param other EnhancedTMR to copy from
     */
    EnhancedTMR(const EnhancedTMR& other) noexcept {
        values_ = other.values_;
        checksums_ = other.checksums_;
        
        // Don't copy error stats
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }
    
    /**
     * @brief Assignment operator
     *
     * @param other EnhancedTMR to copy from
     * @return Reference to this object
     */
    EnhancedTMR& operator=(const EnhancedTMR& other) noexcept {
        if (this != &other) {
            values_ = other.values_;
            checksums_ = other.checksums_;
            
            // Don't copy error stats
            error_stats_.detected_errors = 0;
            error_stats_.corrected_errors = 0;
            error_stats_.uncorrectable_errors = 0;
        }
        return *this;
    }
    
    /**
     * @brief Get the correct value using majority voting
     * 
     * @return The correct value, or best guess if uncorrectable
     */
    T get() const {
        // First check checksums to see if any copy is definitely corrupted
        bool checksum_valid[3] = {
            verifyChecksum(0),
            verifyChecksum(1),
            verifyChecksum(2)
        };
        
        // Count valid copies
        int valid_count = 0;
        for (int i = 0; i < 3; ++i) {
            if (checksum_valid[i]) valid_count++;
        }
        
        // If all copies valid, do normal TMR voting
        if (valid_count == 3) {
            return performMajorityVoting();
        }
        
        // If only some copies valid, use only those for voting
        if (valid_count > 0) {
            // If only one valid, return it
            if (valid_count == 1) {
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        incrementErrorStats(true);
                        return values_[i];
                    }
                }
            }
            
            // If two valid, compare them
            if (valid_count == 2) {
                if (checksum_valid[0] && checksum_valid[1] && values_[0] == values_[1]) {
                    incrementErrorStats(true);
                    return values_[0];
                }
                if (checksum_valid[0] && checksum_valid[2] && values_[0] == values_[2]) {
                    incrementErrorStats(true);
                    return values_[0];
                }
                if (checksum_valid[1] && checksum_valid[2] && values_[1] == values_[2]) {
                    incrementErrorStats(true);
                    return values_[1];
                }
                
                // Two valid but different values - uncertain which is correct
                std::cerr << "WARNING: EnhancedTMR has two valid checksums but different values" << std::endl;
                incrementErrorStats(false);
                
                // Return first valid as a best guess
                for (int i = 0; i < 3; ++i) {
                    if (checksum_valid[i]) {
                        return values_[i];
                    }
                }
            }
        }
        
        // At this point all checksums are invalid
        // Resort to normal TMR voting and hope for the best
        incrementErrorStats(false);
        std::cerr << "WARNING: EnhancedTMR detected corrupted checksums for all copies" << std::endl;
        return performMajorityVoting();
    }
    
    /**
     * @brief Set the value in all three copies
     * 
     * @param value The value to set
     */
    void set(const T& value) noexcept {
        values_[0] = value;
        values_[1] = value;
        values_[2] = value;
        recalculateChecksums();
    }
    
    /**
     * @brief Repair any corrupted values and checksums
     */
    void repair() {
        T corrected_value = get(); // This will do majority voting
        
        // Forcibly correct all copies
        values_[0] = corrected_value;
        values_[1] = corrected_value;
        values_[2] = corrected_value;
        
        // Recalculate checksums
        recalculateChecksums();
    }
    
    /**
     * @brief Verify the integrity of all values
     * 
     * @return true if all values and checksums are consistent
     */
    bool verify() const {
        // Check all checksums
        bool all_valid = verifyChecksum(0) && verifyChecksum(1) && verifyChecksum(2);
        
        // If checksums valid, check values match
        if (all_valid) {
            return (values_[0] == values_[1] && values_[1] == values_[2]);
        }
        
        return false;
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Current error statistics
     */
    ErrorStats getErrorStats() const {
        return error_stats_;
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }
    
    /**
     * @brief Assignment operator for value
     * 
     * @param value The value to assign
     * @return Reference to this object
     */
    EnhancedTMR& operator=(const T& value) {
        set(value);
        return *this;
    }
    
    /**
     * @brief Value conversion operator
     * 
     * @return The protected value
     */
    explicit operator T() const {
        return get();
    }
    
private:
    /// The redundant copies of the value
    std::array<T, 3> values_;
    
    /// Checksums for each value
    mutable std::array<ChecksumType, 3> checksums_;
    
    /// Error counters (no longer using atomic for simplicity)
    mutable ErrorStats error_stats_ = {0, 0, 0};
    
    /**
     * @brief Calculate checksums for all values
     */
    void recalculateChecksums() {
        checksums_[0] = CRC::calculateForValue(values_[0]);
        checksums_[1] = CRC::calculateForValue(values_[1]);
        checksums_[2] = CRC::calculateForValue(values_[2]);
    }
    
    /**
     * @brief Verify checksum for a specific value
     * 
     * @param index Index of the value (0-2)
     * @return true if checksum is valid
     */
    bool verifyChecksum(int index) const {
        ChecksumType computed = CRC::calculateForValue(values_[index]);
        return (computed == checksums_[index]);
    }
    
    /**
     * @brief Perform majority voting on values
     * 
     * @return Majority value or best guess
     */
    T performMajorityVoting() const {
        // Majority voting
        if (values_[0] == values_[1]) {
            return values_[0];
        } else if (values_[0] == values_[2]) {
            return values_[0];
        } else if (values_[1] == values_[2]) {
            return values_[1];
        }
        
        // No majority - try to detect which one is wrong using checksums
        if (verifyChecksum(0) && !verifyChecksum(1) && !verifyChecksum(2)) {
            return values_[0];
        }
        if (!verifyChecksum(0) && verifyChecksum(1) && !verifyChecksum(2)) {
            return values_[1];
        }
        if (!verifyChecksum(0) && !verifyChecksum(1) && verifyChecksum(2)) {
            return values_[2];
        }
        
        // Still no clear answer - return first value as a last resort
        // In a real system, this would trigger a more robust recovery mechanism
        return values_[0];
    }
    
    /**
     * @brief Increment error statistics
     * 
     * @param corrected Whether the error was corrected
     */
    void incrementErrorStats(bool corrected) const {
        error_stats_.detected_errors++;
        if (corrected) {
            error_stats_.corrected_errors++;
        } else {
            error_stats_.uncorrectable_errors++;
        }
    }
};

/// Template alias for convenience
template <typename T>
using ETMR = EnhancedTMR<T>;

} // namespace redundancy
} // namespace core
} // namespace rad_ml 