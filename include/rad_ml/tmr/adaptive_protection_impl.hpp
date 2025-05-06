/**
 * Adaptive Radiation Protection Strategies Implementation
 * 
 * This file contains implementation details for the adaptive protection strategies.
 */

#pragma once

#include "rad_ml/tmr/adaptive_protection.hpp"
#include <thread>
#include <chrono>
#include <limits>  // For std::numeric_limits
#include <type_traits>  // For std::is_arithmetic_v, std::is_integral_v

namespace rad_ml {
namespace tmr {

//------------------------------------------------------------------------------
// BasicTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> BasicTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Voting
    TMRResult<T> result;
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
        result.error_detected = false;
        result.error_corrected = false;
    } else if (result1 == result2) {
        // Majority result1/result2
        result.value = result1;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result1 == result3) {
        // Majority result1/result3
        result.value = result1;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result2 == result3) {
        // Majority result2/result3
        result.value = result2;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No majority, use first result but with low confidence
        result.value = result1;
        result.confidence = 0.33;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// EnhancedTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> EnhancedTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Compute checksums
    size_t checksum1 = 0, checksum2 = 0, checksum3 = 0;
    if constexpr (std::is_arithmetic_v<T>) {
        // For arithmetic types, use a simple hash function
        checksum1 = static_cast<size_t>(result1);
        checksum2 = static_cast<size_t>(result2);
        checksum3 = static_cast<size_t>(result3);
    } else {
        // For complex types, use a simple size-based checksum
        // This is just a placeholder - real implementation would be more robust
        checksum1 = 0; // Use type-appropriate checksum
        checksum2 = 0; 
        checksum3 = 0;
    }
    
    // Voting with checksums
    TMRResult<T> result;
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
        // Check checksums for additional verification
        if (checksum1 == checksum2 && checksum1 == checksum3) {
            result.confidence = 1.0;  // Full confidence
        } else {
            // Values match but checksums differ - possible data corruption
            result.confidence = 0.9;
            result.error_detected = true;
            result.error_corrected = true;
        }
    } else if (result1 == result2) {
        // Majority result1/result2
        result.value = result1;
        if (checksum1 == checksum2) {
            result.confidence = 0.8;  // Strong confidence due to matching checksums
        } else {
            result.confidence = 0.7;  // Reduced confidence due to checksum mismatch
        }
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result1 == result3) {
        // Majority result1/result3
        result.value = result1;
        if (checksum1 == checksum3) {
            result.confidence = 0.8;
        } else {
            result.confidence = 0.7;
        }
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result2 == result3) {
        // Majority result2/result3
        result.value = result2;
        if (checksum2 == checksum3) {
            result.confidence = 0.8;
        } else {
            result.confidence = 0.7;
        }
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No majority, check if any checksums match
        if (checksum1 == checksum2) {
            result.value = result1;
            result.confidence = 0.5;
        } else if (checksum1 == checksum3) {
            result.value = result1;
            result.confidence = 0.5;
        } else if (checksum2 == checksum3) {
            result.value = result2;
            result.confidence = 0.5;
        } else {
            // No majority and no matching checksums
            result.value = result1;
            result.confidence = 0.3;
        }
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// StuckBitTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> StuckBitTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Initialize result
    TMRResult<T> result;
    
    // Check for stuck bits (simplified version)
    bool has_stuck_bits1 = checkForStuckBits(result1);
    bool has_stuck_bits2 = checkForStuckBits(result2);
    bool has_stuck_bits3 = checkForStuckBits(result3);
    
    // Count valid results (without stuck bits)
    int valid_count = 0;
    if (!has_stuck_bits1) valid_count++;
    if (!has_stuck_bits2) valid_count++;
    if (!has_stuck_bits3) valid_count++;
    
    // Perform voting based on valid results
    if (valid_count == 3) {
        // All results are valid, perform normal TMR voting
        if (result1 == result2 && result1 == result3) {
            result.value = result1;
            result.confidence = 1.0;
            result.error_detected = false;
            result.error_corrected = false;
        } else if (result1 == result2) {
            result.value = result1;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (result1 == result3) {
            result.value = result1;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (result2 == result3) {
            result.value = result2;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else {
            // No agreement, use first result with low confidence
            result.value = result1;
            result.confidence = 0.3;
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (valid_count == 2) {
        // Two valid results
        if (!has_stuck_bits1 && !has_stuck_bits2 && result1 == result2) {
            result.value = result1;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (!has_stuck_bits1 && !has_stuck_bits3 && result1 == result3) {
            result.value = result1;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (!has_stuck_bits2 && !has_stuck_bits3 && result2 == result3) {
            result.value = result2;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else {
            // Valid results don't agree
            result.value = !has_stuck_bits1 ? result1 : (!has_stuck_bits2 ? result2 : result3);
            result.confidence = 0.5;
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (valid_count == 1) {
        // Only one valid result
        result.value = !has_stuck_bits1 ? result1 : (!has_stuck_bits2 ? result2 : result3);
        result.confidence = 0.6;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No valid results, attempt bit correction
        result.value = attemptBitCorrection(result1, result2, result3);
        result.confidence = 0.2;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

template<typename T>
bool StuckBitTMR<T>::checkForStuckBits(const T& value) {
    // Simplified check for stuck bits - in a real implementation, this would be
    // more sophisticated and hardware-specific
    
    // Generic check that works for any type
    if constexpr (std::is_arithmetic_v<T>) {
        // For arithmetic types, check for extreme values (like 0 or max value)
        if (value == T{} || value == std::numeric_limits<T>::max()) {
            return true;
        }
    }
    
    return false;
}

template<typename T>
T StuckBitTMR<T>::attemptBitCorrection(const T& value1, const T& value2, const T& value3) {
    // Simple correction strategy based on voting
    if constexpr (std::is_integral_v<T>) {
        // For integral types, use bitwise operations for majority voting
        return (value1 & value2) | (value1 & value3) | (value2 & value3);
    } else {
        // For non-integral types, just use majority voting logic
        if (value1 == value2) return value1;
        if (value1 == value3) return value1;
        if (value2 == value3) return value2;
        return value1; // Default to first value if no majority
    }
}

// Specialization for std::vector<T>
template<typename U>
bool StuckBitTMR<std::vector<U>>::checkForStuckBits(const std::vector<U>& values) {
    // For vectors, check if all elements are identical or if the vector is empty
    if (values.empty()) return true;
    
    // Check if all elements are identical
    if (values.size() > 1) {
        for (size_t i = 1; i < values.size(); ++i) {
            if (values[i] != values[0]) {
                return false; // Not all elements are identical
            }
        }
    }
    
    // Additional check for first element, if it's a suspicious value
    if constexpr (std::is_arithmetic_v<U>) {
        // For arithmetic element types, check if all elements are 0 or max
        if (values[0] == U{} || values[0] == std::numeric_limits<U>::max()) {
            return true;
        }
    }
    
    // If we reach here, all elements are identical but not obviously stuck
    return false;
}

template<typename U>
std::vector<U> StuckBitTMR<std::vector<U>>::attemptBitCorrection(
    const std::vector<U>& values1, 
    const std::vector<U>& values2, 
    const std::vector<U>& values3) {
    
    // Simple implementation: take the first valid vector or the longest one
    std::vector<U> result;
    
    if (!values1.empty() && values1.size() <= values2.size() && values1.size() <= values3.size()) {
        result = values1;
    } else if (!values2.empty() && values2.size() <= values1.size() && values2.size() <= values3.size()) {
        result = values2;
    } else {
        result = values3;
    }
    
    return result;
}

template<typename U>
TMRResult<std::vector<U>> StuckBitTMR<std::vector<U>>::execute(const std::function<std::vector<U>()>& operation) {
    // Execute operation three times
    std::vector<U> result1 = operation();
    std::vector<U> result2 = operation();
    std::vector<U> result3 = operation();
    
    // Initialize result
    TMRResult<std::vector<U>> result;
    
    // Check for stuck bits
    bool has_stuck_bits1 = checkForStuckBits(result1);
    bool has_stuck_bits2 = checkForStuckBits(result2);
    bool has_stuck_bits3 = checkForStuckBits(result3);
    
    // Count valid results (without stuck bits)
    int valid_count = 0;
    if (!has_stuck_bits1) valid_count++;
    if (!has_stuck_bits2) valid_count++;
    if (!has_stuck_bits3) valid_count++;
    
    // Function to check vector equality
    auto vectors_equal = [](const std::vector<U>& a, const std::vector<U>& b) -> bool {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    };
    
    // Perform voting based on valid results
    if (valid_count == 3) {
        // All results are valid, perform normal TMR voting
        if (vectors_equal(result1, result2) && vectors_equal(result1, result3)) {
            result.value = result1;
            result.confidence = 1.0;
            result.error_detected = false;
            result.error_corrected = false;
        } else if (vectors_equal(result1, result2)) {
            result.value = result1;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (vectors_equal(result1, result3)) {
            result.value = result1;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (vectors_equal(result2, result3)) {
            result.value = result2;
            result.confidence = 0.7;
            result.error_detected = true;
            result.error_corrected = true;
        } else {
            // No agreement, use first result with low confidence
            result.value = result1;
            result.confidence = 0.3;
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (valid_count == 2) {
        // Two valid results
        if (!has_stuck_bits1 && !has_stuck_bits2 && vectors_equal(result1, result2)) {
            result.value = result1;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (!has_stuck_bits1 && !has_stuck_bits3 && vectors_equal(result1, result3)) {
            result.value = result1;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else if (!has_stuck_bits2 && !has_stuck_bits3 && vectors_equal(result2, result3)) {
            result.value = result2;
            result.confidence = 0.8;
            result.error_detected = true;
            result.error_corrected = true;
        } else {
            // Valid results don't agree
            result.value = !has_stuck_bits1 ? result1 : (!has_stuck_bits2 ? result2 : result3);
            result.confidence = 0.5;
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (valid_count == 1) {
        // Only one valid result
        result.value = !has_stuck_bits1 ? result1 : (!has_stuck_bits2 ? result2 : result3);
        result.confidence = 0.6;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No valid results, attempt bit correction
        result.value = attemptBitCorrection(result1, result2, result3);
        result.confidence = 0.2;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// HealthWeightedTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> HealthWeightedTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Calculate weighted voting based on health scores
    double total_health = health_scores[0] + health_scores[1] + health_scores[2];
    double weight1 = health_scores[0] / total_health;
    double weight2 = health_scores[1] / total_health;
    double weight3 = health_scores[2] / total_health;
    
    // Initialize result
    TMRResult<T> result;
    result.error_detected = false;
    result.error_corrected = false;
    
    // Voting with health scores
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
    } else if (result1 == result2) {
        // Check if combined weight is high enough
        double combined_weight = weight1 + weight2;
        if (combined_weight > 0.6) {
            result.value = result1;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            // Update health scores
            updateHealthScores(2, true); // Component 3 had error
        } else {
            // Use highest health component
            if (weight1 >= weight2 && weight1 >= weight3) {
                result.value = result1;
                result.confidence = weight1;
            } else if (weight2 >= weight1 && weight2 >= weight3) {
                result.value = result2;
                result.confidence = weight2;
            } else {
                result.value = result3;
                result.confidence = weight3;
            }
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (result1 == result3) {
        // Check if combined weight is high enough
        double combined_weight = weight1 + weight3;
        if (combined_weight > 0.6) {
            result.value = result1;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            updateHealthScores(1, true); // Component 2 had error
        } else {
            // Use highest health component
            if (weight1 >= weight2 && weight1 >= weight3) {
                result.value = result1;
                result.confidence = weight1;
            } else if (weight2 >= weight1 && weight2 >= weight3) {
                result.value = result2;
                result.confidence = weight2;
            } else {
                result.value = result3;
                result.confidence = weight3;
            }
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (result2 == result3) {
        // Check if combined weight is high enough
        double combined_weight = weight2 + weight3;
        if (combined_weight > 0.6) {
            result.value = result2;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            updateHealthScores(0, true); // Component 1 had error
        } else {
            // Use highest health component
            if (weight1 >= weight2 && weight1 >= weight3) {
                result.value = result1;
                result.confidence = weight1;
            } else if (weight2 >= weight1 && weight2 >= weight3) {
                result.value = result2;
                result.confidence = weight2;
            } else {
                result.value = result3;
                result.confidence = weight3;
            }
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else {
        // No majority, use component with highest health
        if (weight1 >= weight2 && weight1 >= weight3) {
            result.value = result1;
            result.confidence = weight1;
        } else if (weight2 >= weight1 && weight2 >= weight3) {
            result.value = result2;
            result.confidence = weight2;
        } else {
            result.value = result3;
            result.confidence = weight3;
        }
        result.error_detected = true;
        result.error_corrected = false;
        
        // Update all health scores due to disagreement
        updateHealthScores(0, false);
        updateHealthScores(1, false);
        updateHealthScores(2, false);
    }
    
    return result;
}

template<typename T>
void HealthWeightedTMR<T>::updateHealthScores(int component_index, bool had_error) {
    // Update health score based on error
    if (had_error) {
        // Decrease health score, but keep it above 0.1
        health_scores[component_index] = std::max(0.1, health_scores[component_index] * 0.8);
    } else {
        // Increase health score, but keep it below 1.0
        health_scores[component_index] = std::min(1.0, health_scores[component_index] * 1.1);
    }
}

//------------------------------------------------------------------------------
// HybridRedundancy Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> HybridRedundancy<T>::execute(const std::function<T()>& operation) {
    // Spatial redundancy
    T result1 = operation();
    
    // Add temporal delay to avoid correlated errors
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(time_delay_ms)));
    
    // More spatial redundancy
    T result2 = operation();
    
    // Another temporal delay
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(time_delay_ms)));
    
    // Final redundant operation
    T result3 = operation();
    
    // Simple voting for now, can be extended with more complex strategies
    TMRResult<T> result;
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
        result.error_detected = false;
        result.error_corrected = false;
    } else if (result1 == result2) {
        // Majority result1/result2
        result.value = result1;
        result.confidence = 0.8;  // Higher confidence due to temporal separation
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result1 == result3) {
        // Majority result1/result3
        result.value = result1;
        result.confidence = 0.8;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result2 == result3) {
        // Majority result2/result3
        result.value = result2;
        result.confidence = 0.8;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No majority, use first result but with low confidence
        result.value = result1;
        result.confidence = 0.33;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// TMRStrategyFactory Implementation
//------------------------------------------------------------------------------

template<typename T>
std::unique_ptr<TMRStrategy<T>> TMRStrategyFactory::createStrategy(
    const sim::RadiationEnvironment& env,
    const core::MaterialProperties& material,
    double criticality
) {
    // Calculate temperature and stress factors
    double temperature = (env.temperature.min + env.temperature.max) / 2.0;
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    
    // Get optimal protection level
    ProtectionLevel level = calculateOptimalProtectionLevel(
        env, material, criticality, temperature, 0.0);
    
    // Create appropriate strategy
    switch (level) {
        case ProtectionLevel::NONE:
            return std::make_unique<NoProtection<T>>();
        case ProtectionLevel::BASIC_TMR:
            return std::make_unique<BasicTMR<T>>();
        case ProtectionLevel::ENHANCED_TMR:
            return std::make_unique<EnhancedTMR<T>>();
        case ProtectionLevel::STUCK_BIT_TMR:
            return std::make_unique<StuckBitTMR<T>>();
        case ProtectionLevel::HEALTH_WEIGHTED_TMR:
            return std::make_unique<HealthWeightedTMR<T>>();
        case ProtectionLevel::HYBRID_REDUNDANCY:
            return std::make_unique<HybridRedundancy<T>>();
        default:
            return std::make_unique<BasicTMR<T>>();
    }
}

ProtectionLevel TMRStrategyFactory::calculateOptimalProtectionLevel(
    const sim::RadiationEnvironment& env,
    const core::MaterialProperties& material,
    double criticality,
    double temperature_K,
    double mechanical_stress_MPa
) {
    // Calculate radiation severity factor
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    double radiation_factor = radiation_dose / (material.radiation_tolerance);
    
    // Apply criticality factor
    double protection_need = radiation_factor * criticality;
    
    // Apply environment factors
    if (env.saa_region) {
        protection_need *= 2.0;
    }
    if (env.solar_activity > 0.7) {
        protection_need *= 1.5;
    }
    
    // Choose protection level based on combined factors
    if (protection_need > 5.0 || criticality > 0.9) {
        return ProtectionLevel::HYBRID_REDUNDANCY;
    } else if (protection_need > 2.0 || criticality > 0.7) {
        return ProtectionLevel::HEALTH_WEIGHTED_TMR;
    } else if (protection_need > 1.0 || criticality > 0.5) {
        return ProtectionLevel::ENHANCED_TMR;
    } else if (protection_need > 0.5) {
        return ProtectionLevel::BASIC_TMR;
    } else {
        return ProtectionLevel::BASIC_TMR;  // Always use at least basic protection
    }
}

//------------------------------------------------------------------------------
// MissionAwareProtectionController Implementation
//------------------------------------------------------------------------------

template<typename T>
std::unique_ptr<TMRStrategy<T>> MissionAwareProtectionController::createCurrentStrategy(double criticality) {
    // Create strategy based on current global protection level
    switch (current_global_level) {
        case ProtectionLevel::BASIC_TMR:
            return std::make_unique<BasicTMR<T>>();
        case ProtectionLevel::ENHANCED_TMR:
            return std::make_unique<EnhancedTMR<T>>();
        case ProtectionLevel::STUCK_BIT_TMR:
            return std::make_unique<StuckBitTMR<T>>();
        case ProtectionLevel::HEALTH_WEIGHTED_TMR:
            return std::make_unique<HealthWeightedTMR<T>>();
        case ProtectionLevel::HYBRID_REDUNDANCY: {
            double delay = 50.0;  // Default delay
            if (current_env.saa_region) {
                delay = 100.0;  // Longer delay in SAA region
            }
            return std::make_unique<HybridRedundancy<T>>(delay);
        }
        default:
            return std::make_unique<BasicTMR<T>>();
    }
}

} // namespace tmr
} // namespace rad_ml 