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
/**
 * @file status_code.hpp
 * @brief Status code-based error handling system for space flight
 * 
 * This file defines a deterministic error handling approach using status codes
 * instead of exceptions, which is preferable for space flight software.
 */

#ifndef RAD_ML_STATUS_CODE_HPP
#define RAD_ML_STATUS_CODE_HPP

#include <string>
#include <array>
#include <cstdint>

namespace rad_ml {
namespace core {
namespace error {

/**
 * Error domains for categorizing errors
 */
enum class ErrorDomain : uint8_t {
    SYSTEM,       // System-level errors
    MEMORY,       // Memory allocation/protection errors
    RADIATION,    // Radiation-related errors
    REDUNDANCY,   // Redundancy mechanism errors
    NETWORK,      // Neural network errors
    COMPUTATION,  // Computation errors
    IO,           // Input/output errors
    VALIDATION,   // Validation errors
    APPLICATION   // Application-specific errors
};

/**
 * Status code definition for error handling
 * 
 * Provides a standardized way to report and handle errors
 * without using exceptions, following flight software best practices.
 */
class StatusCode {
public:
    // Pre-defined status codes
    static const StatusCode SUCCESS;
    static const StatusCode MEMORY_ALLOCATION_FAILURE;
    static const StatusCode REDUNDANCY_FAILURE;
    static const StatusCode RADIATION_DETECTION;
    static const StatusCode INVALID_ARGUMENT;
    static const StatusCode COMPUTATION_ERROR;
    static const StatusCode OVERFLOW_ERROR;
    static const StatusCode UNDERFLOW_ERROR;
    static const StatusCode VALIDATION_FAILURE;
    static const StatusCode CALIBRATION_ERROR;
    static const StatusCode SYSTEM_ERROR;
    
    /**
     * Constructor for creating a status code
     * 
     * @param domain Error domain
     * @param code Numeric code within domain
     * @param message Human-readable message
     */
    constexpr StatusCode(ErrorDomain domain, uint16_t code, const char* message)
        : domain_(domain), code_(code), message_(message) {}
    
    /**
     * Check if status indicates success
     * 
     * @return True if this status represents a successful operation
     */
    bool isSuccess() const {
        return domain_ == ErrorDomain::SYSTEM && code_ == 0;
    }
    
    /**
     * Check if status indicates an error
     * 
     * @return True if this status represents an error
     */
    bool isError() const {
        return !isSuccess();
    }
    
    /**
     * Get error domain
     * 
     * @return Error domain
     */
    ErrorDomain getDomain() const {
        return domain_;
    }
    
    /**
     * Get numeric error code
     * 
     * @return Error code
     */
    uint16_t getCode() const {
        return code_;
    }
    
    /**
     * Get error message
     * 
     * @return Error message
     */
    const char* getMessage() const {
        return message_;
    }
    
    /**
     * Compare status codes for equality
     * 
     * @param other Status code to compare with
     * @return True if status codes are equal
     */
    bool operator==(const StatusCode& other) const {
        return domain_ == other.domain_ && code_ == other.code_;
    }
    
    /**
     * Compare status codes for inequality
     * 
     * @param other Status code to compare with
     * @return True if status codes are not equal
     */
    bool operator!=(const StatusCode& other) const {
        return !(*this == other);
    }
    
private:
    ErrorDomain domain_;
    uint16_t code_;
    const char* message_;
};

// Define global status codes
inline constexpr StatusCode StatusCode::SUCCESS(ErrorDomain::SYSTEM, 0, "Success");
inline constexpr StatusCode StatusCode::MEMORY_ALLOCATION_FAILURE(ErrorDomain::MEMORY, 1, "Memory allocation failure");
inline constexpr StatusCode StatusCode::REDUNDANCY_FAILURE(ErrorDomain::REDUNDANCY, 1, "Redundancy mechanism failure");
inline constexpr StatusCode StatusCode::RADIATION_DETECTION(ErrorDomain::RADIATION, 1, "Radiation event detected");
inline constexpr StatusCode StatusCode::INVALID_ARGUMENT(ErrorDomain::SYSTEM, 1, "Invalid argument");
inline constexpr StatusCode StatusCode::COMPUTATION_ERROR(ErrorDomain::COMPUTATION, 1, "Computation error");
inline constexpr StatusCode StatusCode::OVERFLOW_ERROR(ErrorDomain::COMPUTATION, 2, "Overflow error");
inline constexpr StatusCode StatusCode::UNDERFLOW_ERROR(ErrorDomain::COMPUTATION, 3, "Underflow error");
inline constexpr StatusCode StatusCode::VALIDATION_FAILURE(ErrorDomain::VALIDATION, 1, "Validation failure");
inline constexpr StatusCode StatusCode::CALIBRATION_ERROR(ErrorDomain::RADIATION, 2, "Calibration error");
inline constexpr StatusCode StatusCode::SYSTEM_ERROR(ErrorDomain::SYSTEM, 2, "System error");

/**
 * Result class that combines a value with a status code
 * 
 * This is a common pattern in error-handling without exceptions.
 * The result object contains either a valid value or an error status.
 * 
 * @tparam T The result value type
 */
template<typename T>
class Result {
public:
    /**
     * Construct a successful result with a value
     * 
     * @param value The result value
     */
    Result(const T& value)
        : value_(value), status_(StatusCode::SUCCESS), has_value_(true) {}
    
    /**
     * Construct a result with an error status
     * 
     * @param status Error status
     */
    Result(const StatusCode& status)
        : status_(status), has_value_(false) {}
    
    /**
     * Check if result is successful
     * 
     * @return True if result is successful and contains a valid value
     */
    bool isSuccess() const {
        return has_value_;
    }
    
    /**
     * Check if result is an error
     * 
     * @return True if result is an error
     */
    bool isError() const {
        return !has_value_;
    }
    
    /**
     * Get the result value (only if successful)
     * 
     * @param[out] value Reference to store the value
     * @return True if value was retrieved successfully
     */
    bool getValue(T& value) const {
        if (has_value_) {
            value = value_;
            return true;
        }
        return false;
    }
    
    /**
     * Get the status code
     * 
     * @return Status code
     */
    StatusCode getStatus() const {
        return status_;
    }
    
private:
    T value_;
    StatusCode status_;
    bool has_value_;
};

// Specialization for void type (results with no value)
template<>
class Result<void> {
public:
    /**
     * Construct a successful result
     */
    Result()
        : status_(StatusCode::SUCCESS), success_(true) {}
    
    /**
     * Construct a result with an error status
     * 
     * @param status Error status
     */
    Result(const StatusCode& status)
        : status_(status), success_(status.isSuccess()) {}
    
    /**
     * Check if result is successful
     * 
     * @return True if result is successful
     */
    bool isSuccess() const {
        return success_;
    }
    
    /**
     * Check if result is an error
     * 
     * @return True if result is an error
     */
    bool isError() const {
        return !success_;
    }
    
    /**
     * Get the status code
     * 
     * @return Status code
     */
    StatusCode getStatus() const {
        return status_;
    }
    
private:
    StatusCode status_;
    bool success_;
};

} // namespace error
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_STATUS_CODE_HPP 
