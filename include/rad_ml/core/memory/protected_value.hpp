/**
 * @file protected_value.hpp
 * @brief Enhanced protected value container with error handling
 * 
 * This file defines a container for protected values with advanced error handling
 * capabilities using std::variant and monadic operations.
 */

#ifndef RAD_ML_PROTECTED_VALUE_HPP
#define RAD_ML_PROTECTED_VALUE_HPP

#include <variant>
#include <optional>
#include <string>
#include <array>
#include "../redundancy/enhanced_voting.hpp"

namespace rad_ml {
namespace core {
namespace memory {

/**
 * Error class for protected values
 */
struct ValueCorruptionError {
    redundancy::FaultPattern pattern;
    float confidence;
    std::string message;
};

/**
 * Protected value container using std::variant for error handling
 * 
 * This class wraps a value with triple redundancy and provides robust
 * error detection and correction with explicit error reporting.
 */
#if __cplusplus >= 202002L
template<redundancy::VotableType T>
#else
template<typename T>
#endif
class ProtectedValue {
public:
    /**
     * Constructor initializes with triplication
     * 
     * @param initial_value Initial value to store
     */
    ProtectedValue(const T& initial_value) 
        : copies{initial_value, initial_value, initial_value} {}
    
    /**
     * Get value with error handling
     * 
     * @return Either the corrected value or an error
     */
    std::variant<ValueCorruptionError, T> get() const {
        using namespace rad_ml::core::redundancy;
        
        // Quick return for matching values
        if (copies[0] == copies[1] && copies[1] == copies[2]) {
            return copies[0];
        }
        
        auto [pattern, confidence] = 
            EnhancedVoting::detectFaultPatternWithConfidence(
                copies[0], copies[1], copies[2]);
        
        // If confidence is too low, report error
        if (confidence < confidence_threshold) {
            return ValueCorruptionError{
                pattern, 
                confidence, 
                "Low confidence in error correction"
            };
        }
        
        return EnhancedVoting::adaptiveVote(
            copies[0], copies[1], copies[2], pattern);
    }
    
    /**
     * Get value with fallback
     * 
     * @param fallback Value to use if correction fails
     * @return Corrected value or fallback
     */
    T getOrFallback(const T& fallback) const {
        auto result = get();
        if (std::holds_alternative<ValueCorruptionError>(result)) {
            return fallback;
        }
        return std::get<T>(result);
    }
    
    /**
     * Monadic transform that applies operation safely
     * 
     * @param func Function to apply to the value
     * @return Protected value containing the transformed result or propagated error
     */
    template<typename Func>
#if __cplusplus >= 202002L
    requires std::invocable<Func, T>
#endif
    auto transform(Func&& func) const {
        using ResultType = decltype(func(std::declval<T>()));
        auto result = get();
        
        if (std::holds_alternative<ValueCorruptionError>(result)) {
            return ProtectedValue<ResultType>(
                ValueCorruptionError{
                    std::get<ValueCorruptionError>(result)
                });
        }
        
        return ProtectedValue<ResultType>(func(std::get<T>(result)));
    }
    
    /**
     * Monadic bind operation
     * 
     * @param func Function that returns a ProtectedValue
     * @return Result of applying the function or propagated error
     */
    template<typename Func>
#if __cplusplus >= 202002L
    requires std::invocable<Func, T>
#endif
    auto bind(Func&& func) const 
        -> decltype(func(std::declval<T>())) {
        
        auto result = get();
        if (std::holds_alternative<ValueCorruptionError>(result)) {
            using ReturnType = decltype(func(std::declval<T>()));
            using ValueType = typename ReturnType::value_type;
            
            return ReturnType(
                ValueCorruptionError{
                    std::get<ValueCorruptionError>(result)
                });
        }
        
        return func(std::get<T>(result));
    }
    
    /**
     * Set value with automatic replication
     * 
     * @param value New value to store
     */
    void set(const T& value) {
        copies[0] = copies[1] = copies[2] = value;
    }
    
    /**
     * Perform memory scrubbing to repair errors
     * 
     * @return True if scrubbing was performed
     */
    bool scrub() {
        using namespace rad_ml::core::redundancy;
        
        // No need to scrub if all copies match
        if (copies[0] == copies[1] && copies[1] == copies[2]) {
            return false;
        }
        
        T corrected = EnhancedVoting::adaptiveVote(
            copies[0], copies[1], copies[2],
            EnhancedVoting::detectFaultPattern(copies[0], copies[1], copies[2]));
        
        // Repair all copies
        copies[0] = copies[1] = copies[2] = corrected;
        return true;
    }
    
private:
    // Constructor for error propagation
    ProtectedValue(const ValueCorruptionError& error) 
        : error_state(error), has_error(true) {}
    
    std::array<T, 3> copies;
    ValueCorruptionError error_state;
    bool has_error = false;
    
    // Confidence threshold for error correction
    static constexpr float confidence_threshold = 0.7f;
};

} // namespace memory
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_PROTECTED_VALUE_HPP 