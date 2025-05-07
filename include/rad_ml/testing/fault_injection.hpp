/**
 * @file fault_injection.hpp
 * @brief Systematic fault injection for radiation-tolerant ML framework validation
 * 
 * This file defines classes and functions for conducting systematic fault injection
 * testing to validate the framework's robustness against different error patterns.
 */

#ifndef RAD_ML_FAULT_INJECTION_HPP
#define RAD_ML_FAULT_INJECTION_HPP

#include <string>
#include <vector>
#include <random>
#include <map>
#include <cstdint>
#include <bitset>
#include <iostream>
#include <fstream>
#include <memory>

namespace rad_ml {
namespace testing {

/**
 * Represents the results of a fault injection test
 */
struct FaultInjectionResult {
    // Test configuration
    std::string pattern_name;
    int coverage_percentage;
    bool protection_enabled;
    
    // Network parameters
    int input_size;
    int hidden_size;
    int output_size;
    
    // Error statistics
    int total_weights_affected;
    int total_biases_affected;
    int total_injected_faults;
    int detected_faults;
    int corrected_faults;
    
    // Performance metrics
    double mean_squared_error;
    double correction_rate;
    
    // Export to CSV row format
    std::string toCsvRow() const {
        return pattern_name + "," +
               std::to_string(coverage_percentage) + "," +
               (protection_enabled ? "enabled" : "disabled") + "," +
               std::to_string(input_size) + "," +
               std::to_string(hidden_size) + "," +
               std::to_string(output_size) + "," +
               std::to_string(total_weights_affected) + "," +
               std::to_string(total_biases_affected) + "," +
               std::to_string(total_injected_faults) + "," +
               std::to_string(detected_faults) + "," +
               std::to_string(corrected_faults) + "," +
               std::to_string(mean_squared_error) + "," +
               std::to_string(correction_rate);
    }
    
    // Static method to get CSV header
    static std::string getCsvHeader() {
        return "pattern,coverage,protection,input_size,hidden_size,output_size,"
               "weights_affected,biases_affected,injected_faults,detected_faults,"
               "corrected_faults,mse,correction_rate";
    }
};

/**
 * @class SystematicFaultInjector
 * @brief Performs systematic fault injection tests on neural networks
 * 
 * This class provides methods to inject faults according to specific patterns
 * and run systematic test campaigns to evaluate framework robustness.
 */
class SystematicFaultInjector {
public:
    /**
     * Fault patterns that can be injected
     */
    enum FaultPattern {
        SINGLE_BIT,          // Single bit flips (SEU)
        ADJACENT_BITS,       // 2-3 adjacent bits (MCU)
        BYTE_ERROR,          // Full byte corruption
        WORD_ERROR,          // 32-bit word corruption
        STUCK_AT_ZERO,       // Bits stuck at 0
        STUCK_AT_ONE,        // Bits stuck at 1
        ROW_COLUMN,          // Row/column pattern (common in memory)
        BURST_ERROR          // Burst of errors in time
    };
    
    /**
     * Constructor
     */
    SystematicFaultInjector();
    
    /**
     * Inject a fault according to the specified pattern
     * 
     * @param value The value to inject the fault into
     * @param pattern The fault pattern to use
     * @param bit_position Optional specific bit position to target (default: random)
     * @return The value with the fault injected
     */
    template<typename T>
    T injectFault(T value, FaultPattern pattern, int bit_position = -1);
    
    /**
     * Convert a fault pattern to its string representation
     * 
     * @param pattern The fault pattern
     * @return String representation of the pattern
     */
    static std::string patternToString(FaultPattern pattern);
    
    /**
     * Generate a sequence of bit positions to flip based on the pattern
     * 
     * @param pattern The fault pattern
     * @param total_bits Total number of bits in the value
     * @param starting_bit Starting bit position (for patterns requiring a start point)
     * @return Vector of bit positions to flip
     */
    std::vector<int> getBitsToFlip(FaultPattern pattern, int total_bits, 
                                   int starting_bit = -1);
    
    /**
     * Set the random seed for reproducible results
     * 
     * @param seed The random seed
     */
    void setSeed(unsigned int seed);
    
private:
    std::mt19937 gen;  // Mersenne Twister random generator
};

/**
 * Template implementation for injectFault
 */
template<typename T>
T SystematicFaultInjector::injectFault(T value, FaultPattern pattern, int bit_position) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    
    constexpr int total_bits = sizeof(T) * 8;
    
    // For random bit position
    if (bit_position < 0) {
        std::uniform_int_distribution<> dis(0, total_bits - 1);
        bit_position = dis(gen);
    }
    
    // Get the bits to flip based on the pattern
    std::vector<int> bits_to_flip = getBitsToFlip(pattern, total_bits, bit_position);
    
    // Convert to bitset for manipulation
    std::bitset<sizeof(T) * 8> bits = 
        *reinterpret_cast<std::bitset<sizeof(T) * 8>*>(&value);
    
    // Apply the bit flips
    for (int bit : bits_to_flip) {
        if (bit >= 0 && bit < total_bits) {
            if (pattern == STUCK_AT_ZERO) {
                bits.reset(bit);  // Set to 0
            } else if (pattern == STUCK_AT_ONE) {
                bits.set(bit);    // Set to 1
            } else {
                bits.flip(bit);   // Flip bit
            }
        }
    }
    
    // Convert back to the original type
    value = *reinterpret_cast<T*>(&bits);
    return value;
}

} // namespace testing
} // namespace rad_ml

#endif // RAD_ML_FAULT_INJECTION_HPP 