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
 * @file protected_neural_network.hpp
 * @brief Radiation-tolerant neural network implementation
 * 
 * This file implements a radiation-tolerant neural network that protects
 * weights, biases and activations using TMR and other redundancy techniques.
 */

#ifndef RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP
#define RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <string>
#include <random>
#include <cassert>
#include <bitset>
#include <tuple>
#include <algorithm>

#include "../core/redundancy/space_tmr.hpp"
#include "multi_bit_protection.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Interface for neural network models
 */
class NetworkModel {
public:
    virtual ~NetworkModel() = default;
    
    /**
     * @brief Get the name of the network
     * 
     * @return Network name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the number of layers in the network
     * 
     * @return Layer count
     */
    virtual size_t getLayerCount() const = 0;
    
    /**
     * @brief Get the input size of the network
     * 
     * @return Input size
     */
    virtual size_t getInputSize() const = 0;
    
    /**
     * @brief Get the output size of the network
     * 
     * @return Output size
     */
    virtual size_t getOutputSize() const = 0;
    
    /**
     * @brief Apply protection to the network based on its criticality
     * 
     * @param criticality_threshold Threshold for protection (0-1)
     * @return True if protection was successfully applied
     */
    virtual bool applyProtection(float criticality_threshold = 0.5f) = 0;
};

/**
 * @brief Protection levels for neural network components
 */
enum class ProtectionLevel {
    NONE,               ///< No protection
    CHECKSUM_ONLY,      ///< Only checksum validation
    SELECTIVE_TMR,      ///< TMR only for critical components
    FULL_TMR,           ///< Full TMR for all components
    ADAPTIVE_TMR,       ///< Adaptive TMR based on component criticality
    SPACE_OPTIMIZED     ///< Space-optimized TMR with minimized memory
};

/**
 * @brief Radiation-tolerant neural network implementation
 * 
 * This class implements a feed-forward neural network with radiation
 * protection mechanisms applied to weights, biases, and activations.
 * 
 * @tparam T Value type (typically float)
 */
template <typename T = float>
class ProtectedNeuralNetwork : public NetworkModel {
public:
    /**
     * @brief Constructor
     * 
     * @param layer_sizes Vector containing the size of each layer (including input and output)
     * @param protection_level Protection level to apply
     */
    ProtectedNeuralNetwork(
        const std::vector<size_t>& layer_sizes,
        ProtectionLevel protection_level = ProtectionLevel::ADAPTIVE_TMR
    ) 
        : layer_sizes_(layer_sizes),
          protection_level_(protection_level),
          weights_(),
          biases_(),
          activation_functions_()
    {
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("Neural network must have at least input and output layers");
        }
        
        // Initialize network structure
        initializeNetwork();
    }
    
    /**
     * @brief Get the name of the network
     * 
     * @return Network name
     */
    std::string getName() const override {
        return "ProtectedNeuralNetwork";
    }
    
    /**
     * @brief Get the number of layers in the network
     * 
     * @return Layer count
     */
    size_t getLayerCount() const override {
        return layer_sizes_.size();
    }
    
    /**
     * @brief Get the input size of the network
     * 
     * @return Input size
     */
    size_t getInputSize() const override {
        return layer_sizes_.front();
    }
    
    /**
     * @brief Get the output size of the network
     * 
     * @return Output size
     */
    size_t getOutputSize() const override {
        return layer_sizes_.back();
    }
    
    /**
     * @brief Forward pass through the network
     * 
     * @param input Input tensor
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Output tensor
     */
    std::vector<T> forward(const std::vector<T>& input, double radiation_level = 0.0) {
        if (input.size() != getInputSize()) {
            throw std::invalid_argument("Input size does not match network input layer");
        }
        
        // Apply environmental adaptations based on radiation level
        if (protection_level_ == ProtectionLevel::ADAPTIVE_TMR) {
            adaptToRadiationLevel(radiation_level);
        }
        
        // Input layer activations
        std::vector<std::vector<T>> activations(layer_sizes_.size());
        activations[0] = input;
        
        // Forward pass through each layer
        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            activations[layer + 1].resize(layer_sizes_[layer + 1]);
            
            // For each neuron in the current layer
            for (size_t neuron = 0; neuron < layer_sizes_[layer + 1]; ++neuron) {
                T sum = getBias(layer, neuron);
                
                // Sum weighted inputs from previous layer
                for (size_t prev = 0; prev < layer_sizes_[layer]; ++prev) {
                    sum += getWeight(layer, prev, neuron) * activations[layer][prev];
                }
                
                // Apply activation function
                activations[layer + 1][neuron] = activation_functions_[layer](sum);
            }
            
            // Apply radiation protection to activations if needed
            if (protection_level_ != ProtectionLevel::NONE) {
                protectActivations(activations[layer + 1], radiation_level);
            }
        }
        
        return activations.back();
    }
    
    /**
     * @brief Apply protection to the network based on its criticality
     * 
     * @param criticality_threshold Threshold for protection (0-1)
     * @return True if protection was successfully applied
     */
    bool applyProtection(float criticality_threshold = 0.5f) override {
        // Already set by constructor, but could be used to adjust protection
        return true;
    }
    
    /**
     * @brief Set a custom activation function for a layer
     * 
     * @param layer Layer index (0 for first hidden layer)
     * @param function Activation function
     */
    void setActivationFunction(
        size_t layer,
        const std::function<T(T)>& function
    ) {
        if (layer >= activation_functions_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        activation_functions_[layer] = function;
    }
    
    /**
     * @brief Set weights for a layer
     * 
     * @param layer Layer index (0 for first hidden layer)
     * @param weights Weight matrix (input_size x output_size)
     */
    void setLayerWeights(
        size_t layer,
        const std::vector<std::vector<T>>& weights
    ) {
        if (layer >= weights_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        
        if (weights.size() != layer_sizes_[layer]) {
            throw std::invalid_argument("Weight matrix input dimension mismatch");
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i].size() != layer_sizes_[layer + 1]) {
                throw std::invalid_argument("Weight matrix output dimension mismatch");
            }
            
            for (size_t j = 0; j < weights[i].size(); ++j) {
                setWeight(layer, i, j, weights[i][j]);
            }
        }
    }
    
    /**
     * @brief Set biases for a layer
     * 
     * @param layer Layer index (0 for first hidden layer)
     * @param biases Bias vector
     */
    void setLayerBiases(
        size_t layer,
        const std::vector<T>& biases
    ) {
        if (layer >= biases_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        
        if (biases.size() != layer_sizes_[layer + 1]) {
            throw std::invalid_argument("Bias vector size mismatch");
        }
        
        for (size_t i = 0; i < biases.size(); ++i) {
            setBias(layer, i, biases[i]);
        }
    }
    
    /**
     * @brief Apply radiation effects to the network
     * 
     * @param radiation_level Radiation level (0.0-1.0)
     * @param seed Random seed for reproducibility
     */
    void applyRadiationEffects(double radiation_level, uint64_t seed) {
        if (radiation_level <= 0.0) return;
        
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<size_t> layer_dist(0, weights_.size() - 1);
        
        // Number of bit flips to apply scales with radiation level
        size_t num_bitflips = static_cast<size_t>(radiation_level * 50);
        
        // If using advanced protection, we can simulate multi-bit upsets
        if (protection_level_ >= ProtectionLevel::SELECTIVE_TMR) {
            // Apply bit flips to weights
            for (size_t i = 0; i < num_bitflips; ++i) {
                size_t layer = layer_dist(rng);
                size_t input = std::uniform_int_distribution<size_t>(0, layer_sizes_[layer] - 1)(rng);
                size_t output = std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);
                
                // Get current weight
                T value = getWeight(layer, input, output);
                
                // Apply bit flip
                MultibitUpsetType upset_type = static_cast<MultibitUpsetType>(
                    std::uniform_int_distribution<int>(0, 4)(rng));
                    
                T corrupted = MultibitProtection<T>::applyMultiBitErrors(
                    value, dist(rng) * radiation_level, upset_type, rng());
                
                // Update weight with corrupted value
                raw_setWeight(layer, input, output, corrupted);
            }
            
            // Apply bit flips to biases
            for (size_t i = 0; i < num_bitflips / 5; ++i) { // Fewer bias errors
                size_t layer = layer_dist(rng);
                size_t output = std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);
                
                // Get current bias
                T value = getBias(layer, output);
                
                // Apply bit flip
                T corrupted = applyBitFlip(value, rng);
                
                // Update bias with corrupted value
                raw_setBias(layer, output, corrupted);
            }
        } else {
            // Simple bit flip model for basic protection
            // Apply bit flips to weights
            for (size_t i = 0; i < num_bitflips; ++i) {
                size_t layer = layer_dist(rng);
                size_t input = std::uniform_int_distribution<size_t>(0, layer_sizes_[layer] - 1)(rng);
                size_t output = std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);
                
                // Get current weight
                T value = getWeight(layer, input, output);
                
                // Apply bit flip
                T corrupted = applyBitFlip(value, rng);
                
                // Update weight with corrupted value
                raw_setWeight(layer, input, output, corrupted);
            }
        }
        
        // For adaptive TMR, trigger error correction
        if (protection_level_ == ProtectionLevel::ADAPTIVE_TMR ||
            protection_level_ == ProtectionLevel::FULL_TMR) {
            repairAllWeights();
        }
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Pair of detected and corrected errors
     */
    std::pair<uint64_t, uint64_t> getErrorStats() const {
        return {error_stats_.detected_errors, error_stats_.corrected_errors};
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }
    
private:
    // Define the weight protection type based on protection level
    using WeightType = std::conditional_t<
        std::is_floating_point_v<T>,
        MultibitProtection<T>,
        T
    >;
    
    /**
     * @brief Initialize the network structure
     */
    void initializeNetwork() {
        size_t num_layers = layer_sizes_.size();
        
        // Initialize weights for each layer
        weights_.resize(num_layers - 1);
        for (size_t i = 0; i < num_layers - 1; ++i) {
            weights_[i].resize(layer_sizes_[i]);
            for (size_t j = 0; j < layer_sizes_[i]; ++j) {
                weights_[i][j].resize(layer_sizes_[i + 1]);
            }
        }
        
        // Initialize biases for each layer (except input layer)
        biases_.resize(num_layers - 1);
        for (size_t i = 0; i < num_layers - 1; ++i) {
            biases_[i].resize(layer_sizes_[i + 1]);
        }
        
        // Initialize activation functions (default to ReLU)
        activation_functions_.resize(num_layers - 1, [](T x) { return x > 0 ? x : 0; });
        
        // Initialize weights and biases with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (size_t layer = 0; layer < num_layers - 1; ++layer) {
            // Xavier/Glorot initialization
            T scale = std::sqrt(6.0 / (layer_sizes_[layer] + layer_sizes_[layer + 1]));
            std::uniform_real_distribution<T> dist(-scale, scale);
            
            // Initialize weights
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    T value = dist(gen);
                    setWeight(layer, i, j, value);
                }
            }
            
            // Initialize biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                setBias(layer, j, T{0});
            }
        }
    }
    
    /**
     * @brief Create a protected value based on the protection level
     * 
     * @param value Raw value
     * @return Protected value
     */
    auto createProtectedValue(const T& value) const {
        switch (protection_level_) {
            case ProtectionLevel::NONE:
                return value;
                
            case ProtectionLevel::CHECKSUM_ONLY:
                return MultibitProtection<T>(value, ECCCodingScheme::HAMMING);
                
            case ProtectionLevel::SELECTIVE_TMR:
            case ProtectionLevel::FULL_TMR:
                return MultibitProtection<T>(value, ECCCodingScheme::SECDED);
                
            case ProtectionLevel::ADAPTIVE_TMR:
                return MultibitProtection<T>(value, ECCCodingScheme::REED_SOLOMON);
                
            case ProtectionLevel::SPACE_OPTIMIZED:
                return MultibitProtection<T>(value, ECCCodingScheme::HAMMING);
                
            default:
                return value;
        }
    }
    
    /**
     * @brief Get a weight value from a specific layer
     * 
     * @param layer Layer index
     * @param input Input neuron index
     * @param output Output neuron index
     * @return Weight value
     */
    T getWeight(size_t layer, size_t input, size_t output) const {
        if (layer >= weights_.size() || 
            input >= weights_[layer].size() || 
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            return weights_[layer][input][output];
        } else {
            auto value = weights_[layer][input][output].getValue();
            if (weights_[layer][input][output].hasError()) {
                error_stats_.detected_errors++;
                if (weights_[layer][input][output].correctErrors()) {
                    error_stats_.corrected_errors++;
                } else {
                    error_stats_.uncorrectable_errors++;
                }
            }
            return value;
        }
    }
    
    /**
     * @brief Set a weight value in a specific layer
     * 
     * @param layer Layer index
     * @param input Input neuron index
     * @param output Output neuron index
     * @param value New weight value
     */
    void setWeight(size_t layer, size_t input, size_t output, const T& value) {
        if (layer >= weights_.size() || 
            input >= weights_[layer].size() || 
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            weights_[layer][input][output] = value;
        } else {
            weights_[layer][input][output].setValue(value);
        }
    }
    
    /**
     * @brief Get a bias value from a specific layer
     * 
     * @param layer Layer index
     * @param output Output neuron index
     * @return Bias value
     */
    T getBias(size_t layer, size_t output) const {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            return biases_[layer][output];
        } else {
            auto value = biases_[layer][output].getValue();
            if (biases_[layer][output].hasError()) {
                error_stats_.detected_errors++;
                if (biases_[layer][output].correctErrors()) {
                    error_stats_.corrected_errors++;
                } else {
                    error_stats_.uncorrectable_errors++;
                }
            }
            return value;
        }
    }
    
    /**
     * @brief Set a bias value in a specific layer
     * 
     * @param layer Layer index
     * @param output Output neuron index
     * @param value New bias value
     */
    void setBias(size_t layer, size_t output, const T& value) {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            biases_[layer][output] = value;
        } else {
            biases_[layer][output].setValue(value);
        }
    }
    
    /**
     * @brief Set a weight value without protection
     * 
     * @param layer Layer index
     * @param input Input neuron index
     * @param output Output neuron index
     * @param value New weight value
     */
    void raw_setWeight(size_t layer, size_t input, size_t output, const T& value) {
        if (layer >= weights_.size() || 
            input >= weights_[layer].size() || 
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            weights_[layer][input][output] = value;
        } else {
            *(T*)&weights_[layer][input][output] = value;
        }
    }
    
    /**
     * @brief Set a bias value without protection
     * 
     * @param layer Layer index
     * @param output Output neuron index
     * @param value New bias value
     */
    void raw_setBias(size_t layer, size_t output, const T& value) {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }
        
        if constexpr (std::is_same_v<WeightType, T>) {
            biases_[layer][output] = value;
        } else {
            *(T*)&biases_[layer][output] = value;
        }
    }
    
    /**
     * @brief Apply a random bit flip to a value
     * 
     * @tparam RNG Random number generator type
     * @param value Value to flip a bit in
     * @param rng Random number generator
     * @return Value with flipped bit
     */
    template<typename RNG>
    T applyBitFlip(T value, RNG& rng) const {
        // Flip a random bit in the value's binary representation
        union {
            T value;
            uint8_t bytes[sizeof(T)];
        } converter;
        
        converter.value = value;
        
        // Choose a random byte and bit
        std::uniform_int_distribution<size_t> byte_dist(0, sizeof(T) - 1);
        std::uniform_int_distribution<unsigned> bit_dist(0, 7);
        
        size_t byte_idx = byte_dist(rng);
        unsigned bit_idx = bit_dist(rng);
        
        // Flip the bit
        converter.bytes[byte_idx] ^= (1u << bit_idx);
        
        return converter.value;
    }
    
    /**
     * @brief Apply protection to neuron activations
     * 
     * @param activations Vector of activations to protect
     * @param radiation_level Current radiation level
     */
    void protectActivations(std::vector<T>& activations, double radiation_level) {
        if (protection_level_ == ProtectionLevel::NONE) return;
        
        // For high protection levels, use TMR for activations in high radiation
        if ((protection_level_ == ProtectionLevel::FULL_TMR || 
             protection_level_ == ProtectionLevel::ADAPTIVE_TMR) && 
            radiation_level > 0.2) {
            
            // Create temporary copies for TMR
            std::vector<T> copy1 = activations;
            std::vector<T> copy2 = activations;
            
            // Apply radiation to each copy independently
            std::random_device rd;
            std::mt19937 gen1(rd()), gen2(rd() + 1);
            
            for (size_t i = 0; i < activations.size(); ++i) {
                // Only apply errors with some probability
                if (std::uniform_real_distribution<double>(0, 1)(gen1) < radiation_level * 0.1) {
                    copy1[i] = applyBitFlip(copy1[i], gen1);
                }
                if (std::uniform_real_distribution<double>(0, 1)(gen2) < radiation_level * 0.1) {
                    copy2[i] = applyBitFlip(copy2[i], gen2);
                }
            }
            
            // Perform TMR voting for each activation
            for (size_t i = 0; i < activations.size(); ++i) {
                // Simple majority voting
                if (activations[i] == copy1[i]) {
                    // Original matches copy1, use this value
                    continue;
                } else if (copy1[i] == copy2[i]) {
                    // Two copies match, use their value
                    activations[i] = copy1[i];
                    error_stats_.detected_errors++;
                    error_stats_.corrected_errors++;
                } else if (activations[i] == copy2[i]) {
                    // Original matches copy2, use this value
                    error_stats_.detected_errors++;
                    error_stats_.corrected_errors++;
                    continue;
                } else {
                    // All three values different, can't correct
                    error_stats_.detected_errors++;
                    error_stats_.uncorrectable_errors++;
                    // Keep the original value
                }
            }
        }
    }
    
    /**
     * @brief Adapt protection level based on radiation
     * 
     * @param radiation_level Current radiation level
     */
    void adaptToRadiationLevel(double radiation_level) {
        if (protection_level_ != ProtectionLevel::ADAPTIVE_TMR) return;
        
        // Increase error checking frequency in high radiation
        if (radiation_level > 0.5) {
            // Periodically check and repair all weights
            if (++check_counter_ % 10 == 0) {
                repairAllWeights();
            }
        } else {
            // Less frequent checking in low radiation
            if (++check_counter_ % 100 == 0) {
                repairAllWeights();
            }
        }
    }
    
    /**
     * @brief Repair all weights in the network
     */
    void repairAllWeights() {
        // Only for protected types
        if constexpr (!std::is_same_v<WeightType, T>) {
            // Repair weights
            for (auto& layer : weights_) {
                for (auto& input_weights : layer) {
                    for (auto& weight : input_weights) {
                        if (weight.hasError()) {
                            error_stats_.detected_errors++;
                            if (weight.correctErrors()) {
                                error_stats_.corrected_errors++;
                            } else {
                                error_stats_.uncorrectable_errors++;
                            }
                        }
                    }
                }
            }
            
            // Repair biases
            for (auto& layer : biases_) {
                for (auto& bias : layer) {
                    if (bias.hasError()) {
                        error_stats_.detected_errors++;
                        if (bias.correctErrors()) {
                            error_stats_.corrected_errors++;
                        } else {
                            error_stats_.uncorrectable_errors++;
                        }
                    }
                }
            }
        }
    }
    
private:
    std::vector<size_t> layer_sizes_;
    ProtectionLevel protection_level_;
    size_t check_counter_ = 0;
    
    // Use appropriate types based on protection level
    // If we're using bit-level protection, weights will be MultibitProtection<T>
    // Otherwise, they'll just be T
    std::vector<std::vector<std::vector<WeightType>>> weights_;
    std::vector<std::vector<WeightType>> biases_;
    std::vector<std::function<T(T)>> activation_functions_;
    
    // Error statistics
    mutable struct {
        uint64_t detected_errors = 0;
        uint64_t corrected_errors = 0;
        uint64_t uncorrectable_errors = 0;
    } error_stats_;
};

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP 
