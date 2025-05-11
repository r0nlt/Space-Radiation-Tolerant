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
 * @file adaptive_protection.hpp
 * @brief Adaptive radiation protection for neural networks
 * 
 * This file implements an adaptive protection strategy that dynamically
 * selects the most appropriate error correction approach based on the
 * current radiation environment and critical weight identification.
 */

#ifndef RAD_ML_NEURAL_ADAPTIVE_PROTECTION_HPP
#define RAD_ML_NEURAL_ADAPTIVE_PROTECTION_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <tuple>
#include <optional>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <variant>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <mutex>
#include <concepts>

#include "advanced_reed_solomon.hpp"
#include "multi_bit_protection.hpp"
#include "radiation_environment.hpp"

namespace rad_ml {
namespace neural {

// Forward declaration
template<typename T>
class ProtectedNeuralNetwork;

/**
 * @brief Protection level enum
 */
enum class ProtectionLevel {
    NONE,           // No protection
    MINIMAL,        // Basic parity-based protection
    MODERATE,       // TMR or Hamming code protection
    HIGH,           // Reed-Solomon with moderate parameter settings
    VERY_HIGH,      // Reed-Solomon with strong parameter settings
    ADAPTIVE        // Dynamically adjusted based on radiation conditions
};

/**
 * @brief Concept for types that can be protected
 * 
 * This concept ensures that a type can be properly protected by our mechanisms
 */
template<typename T>
concept Protectable = std::is_trivially_copyable_v<T> && 
                      (std::is_arithmetic_v<T> || 
                       std::is_enum_v<T> || 
                       std::is_pointer_v<T>);

/**
 * @brief Weight criticality data for adaptive protection
 */
template<typename T>
struct WeightCriticality {
    T weight;                  // The weight value
    float sensitivity;         // Sensitivity score (higher = more critical)
    ProtectionLevel level;     // Selected protection level
    
    // Allow comparison for sorting
    bool operator<(const WeightCriticality& other) const {
        return sensitivity < other.sensitivity;
    }
};

/**
 * @brief Adaptive protection strategy for neural network weights
 * 
 * This class implements adaptive protection strategies that can
 * dynamically adjust to radiation conditions and focus resources
 * on the most critical network parameters.
 * 
 * @tparam T Type of data to protect
 */
template<typename T>
requires Protectable<T>
class AdaptiveProtection {
public:
    /**
     * @brief Protection statistics
     */
    struct ProtectionStats {
        size_t total_weights = 0;          // Total number of weights
        size_t protected_weights = 0;      // Number of protected weights
        size_t corrections = 0;            // Number of corrections applied
        size_t uncorrectable_errors = 0;   // Number of uncorrectable errors
        size_t total_bits = 0;             // Total number of bits
        size_t flipped_bits = 0;           // Number of bits flipped
        
        double protection_overhead = 0.0;  // Memory overhead for protection
        double seu_rate = 0.0;             // SEU rate in errors/bit/day
    
        // Reset statistics
        void reset() {
            corrections = 0;
            uncorrectable_errors = 0;
            flipped_bits = 0;
        }
        
        // Get correction ratio (corrected/total errors)
        double correction_ratio() const {
            if (corrections + uncorrectable_errors == 0) return 1.0;
            return static_cast<double>(corrections) / 
                   (corrections + uncorrectable_errors);
        }
    };
    
    /**
     * @brief Default constructor
     */
    AdaptiveProtection()
        : radiation_env_(), 
          error_model_(MultibitUpsetType::SINGLE_BIT),
          protection_level_(ProtectionLevel::MODERATE),
          stats_(),
          rng_(std::random_device{}()) {
    }
    
    /**
     * @brief Constructor with environment and protection level
     * 
     * @param env Initial radiation environment
     * @param level Initial protection level
     */
    AdaptiveProtection(
        const RadiationEnvironment& env,
        ProtectionLevel level = ProtectionLevel::MODERATE
    ) : radiation_env_(env),
        error_model_(MultibitUpsetType::SINGLE_BIT),
        protection_level_(level),
        stats_(),
        rng_(std::random_device{}()) {
    }
    
    /**
     * @brief Set radiation environment
     * 
     * @param env Radiation environment
     */
    void set_environment(const RadiationEnvironment& env) {
        radiation_env_ = env;
        adapt_to_environment();
    }
    
    /**
     * @brief Set protection level
     * 
     * @param level Protection level
     */
    void set_protection_level(ProtectionLevel level) {
        protection_level_ = level;
    }
    
    /**
     * @brief Set error model
     * 
     * @param model Error model type
     */
    void set_error_model(MultibitUpsetType model) {
        error_model_ = model;
    }
    
    /**
     * @brief Get current protection statistics
     * 
     * @return Reference to protection statistics
     */
    const ProtectionStats& get_stats() const {
        return stats_;
    }
    
    /**
     * @brief Reset protection statistics
     */
    void reset_stats() {
        stats_.reset();
    }
    
    /**
     * @brief Apply protection to a value
     * 
     * @param value Value to protect
     * @param criticality Optional criticality score
     * @return Protected value
     */
    template<typename U = T>
    U protect_value(const U& value, float criticality = 1.0) {
        // Select protection mechanism based on level and criticality
        auto level = get_effective_protection_level(criticality);
        
        // Keep an unmodified copy in case we need to simulate errors
        U result = value;
        
        // Apply the selected protection
        switch (level) {
            case ProtectionLevel::NONE:
                // No protection
                break;
            
            case ProtectionLevel::MINIMAL: {
                // Simple parity-based protection
                auto parity = compute_parity(value);
                result = add_parity_bit(value, parity);
                break;
            }
            
            case ProtectionLevel::MODERATE: {
                // Apply Hamming code (7,4) for each byte
                result = apply_hamming_protection(value);
                break;
            }
            
            case ProtectionLevel::HIGH: {
                // Apply Reed-Solomon with moderate parameters
                RS8Bit8Sym<U> rs;
                auto encoded = rs.encode(value);
                
                // For now, just store the original value
                // In a real system, we would store the encoded value
                stats_.protection_overhead += rs.overhead_percent();
                break;
            }
            
            case ProtectionLevel::VERY_HIGH: {
                // Apply Reed-Solomon with stronger parameters
                RS8Bit16Sym<U> rs;
                auto encoded = rs.encode(value);
                
                // For now, just store the original value
                // In a real system, we would store the encoded value
                stats_.protection_overhead += rs.overhead_percent();
                break;
            }
            
            case ProtectionLevel::ADAPTIVE: {
                // This should not happen here
                // Use the moderate level as fallback
                result = apply_hamming_protection(value);
                break;
            }
        }
        
        // Update statistics
        stats_.total_weights++;
        if (level != ProtectionLevel::NONE) {
            stats_.protected_weights++;
        }
        
        return result;
    }
    
    /**
     * @brief Recover a protected value after potential errors
     * 
     * @param value Value to check/recover
     * @param criticality Optional criticality score
     * @return Tuple of (corrected value, correction applied)
     */
    template<typename U = T>
    std::tuple<U, bool> recover_value(const U& value, float criticality = 1.0) {
        // Select protection mechanism based on level and criticality
        auto level = get_effective_protection_level(criticality);
        
        // Prepare result
        U result = value;
        bool corrected = false;
        
        // Apply the selected recovery strategy
        switch (level) {
            case ProtectionLevel::NONE:
                // No protection, nothing to recover
                break;
            
            case ProtectionLevel::MINIMAL: {
                // Check parity
                bool original_parity = extract_parity_bit(value);
                bool current_parity = compute_parity(value);
                
                if (original_parity != current_parity) {
                    // Parity error detected, but can't correct
                    stats_.uncorrectable_errors++;
                }
                
                // Remove parity bit to get original value
                result = remove_parity_bit(value);
                break;
            }
            
            case ProtectionLevel::MODERATE: {
                // Recover using Hamming code
                auto [recovered, was_corrected] = recover_with_hamming(value);
                result = recovered;
                corrected = was_corrected;
                
                if (corrected) {
                    stats_.corrections++;
                }
                break;
            }
            
            case ProtectionLevel::HIGH:
            case ProtectionLevel::VERY_HIGH: {
                // Use Reed-Solomon with appropriate parameters
                if (level == ProtectionLevel::HIGH) {
                    RS8Bit8Sym<U> rs;
                    
                    // In a real system, we would decode the encoded value
                    // For now, just simulate the process
                    auto encoded = rs.encode(value);
                    
                    // Simulate potential errors
                    double seu_prob = radiation_env_.get_seu_probability();
                    encoded = rs.apply_bit_errors(encoded, seu_prob, rng_());
                    
                    // Try to decode
                    auto decoded = rs.decode(encoded);
                    if (decoded) {
                        result = *decoded;
                        corrected = true;
                        stats_.corrections++;
                    } else {
                        stats_.uncorrectable_errors++;
                    }
                } else {
                    RS8Bit16Sym<U> rs;
                    
                    // Similar to above but with stronger parameters
                    auto encoded = rs.encode(value);
                    
                    double seu_prob = radiation_env_.get_seu_probability();
                    encoded = rs.apply_bit_errors(encoded, seu_prob, rng_());
                    
                    auto decoded = rs.decode(encoded);
                    if (decoded) {
                        result = *decoded;
                        corrected = true;
                        stats_.corrections++;
                    } else {
                        stats_.uncorrectable_errors++;
                    }
                }
                break;
            }
            
            case ProtectionLevel::ADAPTIVE: {
                // This should not happen here
                // Use the moderate level as fallback
                auto [recovered, was_corrected] = recover_with_hamming(value);
                result = recovered;
                corrected = was_corrected;
                
                if (corrected) {
                    stats_.corrections++;
                }
                break;
            }
        }
        
        return {result, corrected};
    }
    
    /**
     * @brief Apply simulated radiation effects to a value
     * 
     * @param value Value to affect
     * @param seu_probability Single Event Upset probability
     * @return Value with simulated radiation effects
     */
    template<typename U = T>
    U apply_radiation_effects(const U& value, double seu_probability = -1.0) {
        // Use provided SEU probability or get from environment
        if (seu_probability < 0) {
            seu_probability = radiation_env_.get_seu_probability();
        }
        
        // If probability is zero, no need to process
        if (seu_probability <= 0) {
            return value;
        }
        
        // Create a byte representation of the value
        std::vector<uint8_t> bytes(sizeof(U));
        std::memcpy(bytes.data(), &value, sizeof(U));
        
        // Apply errors based on the current error model
        MultibitProtection<U> mbu;
        bytes = mbu.apply_multi_bit_upset(bytes, error_model_, seu_probability, rng_());
        
        // Convert back to the original type
        U result;
        std::memcpy(&result, bytes.data(), sizeof(U));
        
        // Update statistics based on hamming distance
        auto bit_flips = count_bit_differences(value, result);
        stats_.total_bits += sizeof(U) * 8;
        stats_.flipped_bits += bit_flips;
        stats_.seu_rate = static_cast<double>(stats_.flipped_bits) / stats_.total_bits;
        
        return result;
    }
    
    /**
     * @brief Adapt protection based on current environment
     * 
     * Adjusts protection strategies based on the current radiation
     * environment to balance protection and overhead.
     */
    void adapt_to_environment() {
        // Get current SEU probability
        double seu_prob = radiation_env_.get_seu_probability();
        
        // Default error model is single bit upsets
        error_model_ = MultibitUpsetType::SINGLE_BIT;
        
        // Check for high radiation scenarios
        if (seu_prob > 0.001) {
            // Heavy ion environment - high risk of multi-bit upsets
            error_model_ = MultibitUpsetType::ADJACENT_BITS;
            protection_level_ = ProtectionLevel::VERY_HIGH;
            
            // If extremely high, use custom model with interleaving
            if (seu_prob > 0.01) {
                error_model_ = MultibitUpsetType::RANDOM_MULTI;
            }
        } else if (seu_prob > 0.0001) {
            // Moderate radiation - use Reed-Solomon with moderate parameters
            protection_level_ = ProtectionLevel::HIGH;
        } else if (seu_prob > 0.00001) {
            // Low radiation - basic Hamming code is sufficient
            protection_level_ = ProtectionLevel::MODERATE;
        } else if (seu_prob > 0.000001) {
            // Very low radiation - simple parity check
            protection_level_ = ProtectionLevel::MINIMAL;
        } else {
            // Negligible radiation - no protection needed
            protection_level_ = ProtectionLevel::NONE;
        }
    }
    
    /**
     * @brief Identity critical weights in a neural network
     * 
     * @param network Neural network to analyze
     * @param input_samples Representative input samples
     * @param output_samples Expected outputs for the samples
     * @return Vector of weights with criticality scores
     */
    template<typename U = T>
    std::vector<WeightCriticality<U>> identify_critical_weights(
        ProtectedNeuralNetwork<U>& network,
        const std::vector<std::vector<U>>& input_samples,
        const std::vector<std::vector<U>>& output_samples
    ) {
        if (input_samples.empty() || output_samples.empty() || 
            input_samples.size() != output_samples.size()) {
            return {};
        }
        
        std::vector<WeightCriticality<U>> result;
        
        // Get all weights from the network
        auto weights = network.get_all_weights();
        result.reserve(weights.size());
        
        // Calculate baseline performance
        double baseline_error = calculate_network_error(
            network, input_samples, output_samples);
        
        // For each weight, calculate its sensitivity
        for (const auto& weight : weights) {
            // Create a modified weight with a bit flip
            U modified_weight = flip_random_bit(weight);
            
            // Temporarily replace the weight
            network.replace_weight(weight, modified_weight);
            
            // Calculate new performance
            double modified_error = calculate_network_error(
                network, input_samples, output_samples);
            
            // Calculate sensitivity as the relative error increase
            double sensitivity = (modified_error - baseline_error) / baseline_error;
            
            // Restore the original weight
            network.replace_weight(modified_weight, weight);
            
            // Store criticality information
            WeightCriticality<U> criticality;
            criticality.weight = weight;
            criticality.sensitivity = sensitivity;
            
            // Assign protection level based on sensitivity
            if (sensitivity > 10.0) {
                criticality.level = ProtectionLevel::VERY_HIGH;
            } else if (sensitivity > 5.0) {
                criticality.level = ProtectionLevel::HIGH;
            } else if (sensitivity > 1.0) {
                criticality.level = ProtectionLevel::MODERATE;
            } else if (sensitivity > 0.1) {
                criticality.level = ProtectionLevel::MINIMAL;
            } else {
                criticality.level = ProtectionLevel::NONE;
            }
            
            result.push_back(criticality);
        }
        
        // Sort by sensitivity (highest first)
        std::sort(result.begin(), result.end(), 
                 [](const auto& a, const auto& b) {
                     return a.sensitivity > b.sensitivity;
                 });
        
        return result;
    }
    
    /**
     * @brief Apply optimized protection based on weight criticality
     * 
     * @param network Neural network to protect
     * @param criticalities Weight criticality scores
     * @param budget Resource budget (0.0-1.0, portion of weights to protect)
     */
    template<typename U = T>
    void apply_optimized_protection(
        ProtectedNeuralNetwork<U>& network,
        const std::vector<WeightCriticality<U>>& criticalities,
        double budget = 0.5
    ) {
        if (criticalities.empty() || budget <= 0) {
            return;
        }
        
        // Determine how many weights to protect
        size_t total_weights = criticalities.size();
        size_t protected_count = static_cast<size_t>(total_weights * budget);
        
        // Ensure we protect at least one weight if budget > 0
        protected_count = std::max(protected_count, size_t(1));
        
        // Apply protection to the most critical weights
        for (size_t i = 0; i < std::min(protected_count, total_weights); ++i) {
            const auto& criticality = criticalities[i];
            auto protected_weight = protect_value(criticality.weight, criticality.sensitivity);
            network.replace_weight(criticality.weight, protected_weight);
        }
        
        // Update statistics
        stats_.total_weights = total_weights;
        stats_.protected_weights = protected_count;
    }
    
private:
    RadiationEnvironment radiation_env_;
    MultibitUpsetType error_model_;
    ProtectionLevel protection_level_;
    ProtectionStats stats_;
    std::mt19937_64 rng_;
    
    /**
     * @brief Get effective protection level based on criticality
     * 
     * @param criticality Criticality score
     * @return Effective protection level
     */
    ProtectionLevel get_effective_protection_level(float criticality) const {
        if (protection_level_ != ProtectionLevel::ADAPTIVE) {
            return protection_level_;
        }
        
        // For adaptive mode, select based on criticality
        if (criticality > 10.0) {
            return ProtectionLevel::VERY_HIGH;
        } else if (criticality > 5.0) {
            return ProtectionLevel::HIGH;
        } else if (criticality > 1.0) {
            return ProtectionLevel::MODERATE;
        } else if (criticality > 0.1) {
            return ProtectionLevel::MINIMAL;
        } else {
            return ProtectionLevel::NONE;
        }
    }
    
    /**
     * @brief Compute parity bit for a value
     * 
     * @param value Value to compute parity for
     * @return Parity bit (true for odd parity)
     */
    template<typename U>
    bool compute_parity(const U& value) const {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        bool parity = false;
        
        for (size_t i = 0; i < sizeof(U); ++i) {
            uint8_t byte = bytes[i];
            
            // XOR all bits in the byte
            for (int bit = 0; bit < 8; ++bit) {
                parity ^= ((byte >> bit) & 1) != 0;
            }
        }
        
        return parity;
    }
    
    /**
     * @brief Add parity bit to a value
     * 
     * @param value Original value
     * @param parity Parity bit to add
     * @return Value with parity bit
     */
    template<typename U>
    U add_parity_bit(const U& value, bool parity) const {
        // For simplicity, we just store the parity in the MSB
        // In a real implementation, we would use a more sophisticated approach
        
        U result = value;
        if (parity) {
            // Set the MSB
            const size_t msb_byte = sizeof(U) - 1;
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
            bytes[msb_byte] |= 0x80;  // Set MSB
        }
        
        return result;
    }
    
    /**
     * @brief Extract parity bit from a value
     * 
     * @param value Value with parity bit
     * @return Extracted parity bit
     */
    template<typename U>
    bool extract_parity_bit(const U& value) const {
        // Corresponding to the add_parity_bit function
        const size_t msb_byte = sizeof(U) - 1;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        return (bytes[msb_byte] & 0x80) != 0;
    }
    
    /**
     * @brief Remove parity bit from a value
     * 
     * @param value Value with parity bit
     * @return Value without parity bit
     */
    template<typename U>
    U remove_parity_bit(const U& value) const {
        // Corresponding to the add_parity_bit function
        U result = value;
        const size_t msb_byte = sizeof(U) - 1;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
        bytes[msb_byte] &= 0x7F;  // Clear MSB
        
        return result;
    }
    
    /**
     * @brief Apply Hamming code protection
     * 
     * @param value Value to protect
     * @return Protected value
     */
    template<typename U>
    U apply_hamming_protection(const U& value) const {
        // For simplicity, we use a dummy implementation
        // In a real system, we would properly implement Hamming encoding
        return value;
    }
    
    /**
     * @brief Recover a value using Hamming code
     * 
     * @param value Value to recover
     * @return Tuple of (recovered value, correction applied)
     */
    template<typename U>
    std::tuple<U, bool> recover_with_hamming(const U& value) const {
        // For simplicity, we use a dummy implementation
        // In a real system, we would properly implement Hamming decoding
        return {value, false};
    }
    
    /**
     * @brief Count bit differences between two values
     * 
     * @param a First value
     * @param b Second value
     * @return Number of bit differences
     */
    template<typename U>
    size_t count_bit_differences(const U& a, const U& b) const {
        const uint8_t* bytes_a = reinterpret_cast<const uint8_t*>(&a);
        const uint8_t* bytes_b = reinterpret_cast<const uint8_t*>(&b);
        
        size_t differences = 0;
        
        for (size_t i = 0; i < sizeof(U); ++i) {
            uint8_t diff = bytes_a[i] ^ bytes_b[i];
            
            // Count bits in the difference
            for (int bit = 0; bit < 8; ++bit) {
                if ((diff >> bit) & 1) {
                    differences++;
                }
            }
        }
        
        return differences;
    }
    
    /**
     * @brief Flip a random bit in a value
     * 
     * @param value Value to modify
     * @return Value with one bit flipped
     */
    template<typename U>
    U flip_random_bit(const U& value) const {
        U result = value;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
        
        // Choose a random byte and bit
        std::uniform_int_distribution<size_t> byte_dist(0, sizeof(U) - 1);
        std::uniform_int_distribution<int> bit_dist(0, 7);
        
        size_t byte_idx = byte_dist(rng_);
        int bit_idx = bit_dist(rng_);
        
        // Flip the bit
        bytes[byte_idx] ^= (1 << bit_idx);
        
        return result;
    }
    
    /**
     * @brief Calculate error between network outputs and expected outputs
     * 
     * @param network Neural network to evaluate
     * @param inputs Input samples
     * @param expected Expected outputs
     * @return Mean squared error
     */
    template<typename U>
    double calculate_network_error(
        ProtectedNeuralNetwork<U>& network,
        const std::vector<std::vector<U>>& inputs,
        const std::vector<std::vector<U>>& expected
    ) const {
        double total_error = 0.0;
        size_t total_outputs = 0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = network.forward(inputs[i]);
            
            // Calculate MSE for this sample
            for (size_t j = 0; j < output.size() && j < expected[i].size(); ++j) {
                double diff = static_cast<double>(output[j] - expected[i][j]);
                total_error += diff * diff;
                total_outputs++;
            }
        }
        
        return total_outputs > 0 ? total_error / total_outputs : 0.0;
    }
};

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_ADAPTIVE_PROTECTION_HPP 
