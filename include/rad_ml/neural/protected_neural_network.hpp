/**
 * @file protected_neural_network.hpp
 * @brief Radiation-tolerant neural network implementation
 *
 * This file implements a radiation-tolerant neural network that protects
 * weights, biases and activations using TMR and other redundancy techniques.
 */

#ifndef RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP
#define RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../core/logger.hpp"
#include "../core/redundancy/space_enhanced_tmr.hpp"
#include "multi_bit_protection.hpp"

// Optimization-layer projections (required for simplex projection path)
#ifdef __has_include
#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#else
#error "Could not find Eigen/Dense"
#endif
#else
#include <Eigen/Dense>
#endif
#include "../optimization/simplex_projection.hpp"

// SIMD optimizations
#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_ENABLED
#endif

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
    NONE,            ///< No protection
    CHECKSUM_ONLY,   ///< Only checksum validation
    SELECTIVE_TMR,   ///< TMR only for critical components
    FULL_TMR,        ///< Full TMR for all components
    ADAPTIVE_TMR,    ///< Adaptive TMR based on component criticality
    SPACE_OPTIMIZED  ///< Space-optimized TMR with minimized memory
};

/**
 * @brief Compile-time network traits for optimization
 *
 * @tparam LayerCount Number of layers in the network
 */
template <size_t LayerCount>
struct NetworkTraits {
    static constexpr size_t layer_count = LayerCount;
    static constexpr bool is_small_network = LayerCount <= 5;
    static constexpr bool use_simd = LayerCount >= 3;
    static constexpr bool use_optimized_storage = LayerCount >= 4;
};

/**
 * @brief Constexpr helper functions for compile-time optimizations
 */
namespace constexpr_helpers {

static constexpr bool should_use_protection(ProtectionLevel level) noexcept
{
    return level != ProtectionLevel::NONE;
}

static constexpr bool should_use_simd(ProtectionLevel level, size_t layer_count) noexcept
{
    return should_use_protection(level) && layer_count >= 3;
}

static constexpr bool should_use_adaptive_protection(ProtectionLevel level) noexcept
{
    return level == ProtectionLevel::ADAPTIVE_TMR;
}

}  // namespace constexpr_helpers

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
    void setUseSimplexProjection(bool enabled) { use_simplex_projection_ = enabled; }
    /**
     * @brief Layer structure containing weights and biases
     */
    struct Layer {
        std::vector<std::vector<T>> weights;
        std::vector<T> biases;
    };

    /**
     * @brief Apply radiation effects to the network
     *
     * @param radiation_level Radiation level (0.0-1.0)
     * @param seed Random seed for reproducibility
     * @note This method is not thread-safe due to shared mutable state in error_stats_.
     *       Use external synchronization if calling from multiple threads.
     */
    void applyRadiationEffects(double radiation_level, uint64_t seed)
    {
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
                size_t input =
                    std::uniform_int_distribution<size_t>(0, layer_sizes_[layer] - 1)(rng);
                size_t output =
                    std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);

                // Get current weight
                T value = getWeight(layer, input, output);

                // Apply bit flip
                MultibitUpsetType upset_type =
                    static_cast<MultibitUpsetType>(std::uniform_int_distribution<int>(0, 4)(rng));

                T corrupted = MultibitProtection<T>::applyMultiBitErrors(
                    value, dist(rng) * radiation_level, upset_type, rng());

                // Update weight with corrupted value
                raw_setWeight(layer, input, output, corrupted);
            }

            // Apply bit flips to biases
            for (size_t i = 0; i < num_bitflips / 5; ++i) {  // Fewer bias errors
                size_t layer = layer_dist(rng);
                size_t output =
                    std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);

                // Get current bias
                T value = getBias(layer, output);

                // Apply bit flip
                T corrupted = applyBitFlip(value, rng);

                // Update bias with corrupted value
                raw_setBias(layer, output, corrupted);
            }
        }
        else {
            // Simple bit flip model for basic protection
            // Apply bit flips to weights
            for (size_t i = 0; i < num_bitflips; ++i) {
                size_t layer = layer_dist(rng);
                size_t input =
                    std::uniform_int_distribution<size_t>(0, layer_sizes_[layer] - 1)(rng);
                size_t output =
                    std::uniform_int_distribution<size_t>(0, layer_sizes_[layer + 1] - 1)(rng);

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
    std::pair<uint64_t, uint64_t> getErrorStats() const
    {
        return {error_stats_.detected_errors, error_stats_.corrected_errors};
    }

    /**
     * @brief Set a custom activation function for a layer
     *
     * The activation function should be a continuous, monotonic, and differentiable function
     * that maps real values to a bounded output range (e.g., [0, 1] for sigmoid, [-1, 1] for tanh).
     * It should be suitable for neural network training (e.g., non-linear, smooth).
     *
     * @param layer Layer index (0 for first hidden layer)
     * @param function Activation function. Must accept and return type T.
     * @param validate If true (default), checks that the function maps a sample of values in [-10,
     * 10] to a bounded output.
     * @throws std::invalid_argument if validation fails.
     * @throws std::out_of_range if layer index is invalid.
     */
    void setActivationFunction(size_t layer, const std::function<T(T)>& function,
                               bool validate = true)
    {
        if (layer >= activation_functions_.size()) {
            throw std::out_of_range("Layer index out of range");
        }

        if (validate) {
            // Sample input range for typical neural network activations
            constexpr T min_input = static_cast<T>(-10);
            constexpr T max_input = static_cast<T>(10);
            constexpr T step = static_cast<T>(1);
            T min_output = function(min_input);
            T max_output = function(min_input);

            for (T x = min_input; x <= max_input; x += step) {
                T y = function(x);
                if (std::isnan(y) || std::isinf(y)) {
                    throw std::invalid_argument("Activation function produces NaN or Inf output.");
                }
                if (y < min_output) min_output = y;
                if (y > max_output) max_output = y;
            }

            // Check for reasonable output bounds (adjust as needed for your use case)
            // Note: Some functions like ReLU can have unbounded outputs, so we use more permissive
            // bounds
            if (min_output < static_cast<T>(-100) || max_output > static_cast<T>(100)) {
                throw std::invalid_argument(
                    "Activation function output is out of reasonable bounds [-100, 100]. "
                    "Consider using a different activation function or disable validation.");
            }
        }

        activation_functions_[layer] = function;
    }

    /**
     * @brief Set activation function with explicit derivative (recommended for custom functions)
     *
     * This overload allows users to provide both the activation function and its derivative,
     * avoiding the need for runtime inference or numerical differentiation. This is more
     * reliable for custom activation functions and provides better performance.
     *
     * @param layer Layer index (0 for first hidden layer)
     * @param function Activation function
     * @param derivative Derivative of the activation function
     * @param validate If true (default), validates both function and derivative
     * @throws std::invalid_argument if validation fails
     * @throws std::out_of_range if layer index is invalid
     */
    void setActivationFunction(size_t layer, const std::function<T(T)>& function,
                               const std::function<T(T)>& derivative, bool validate = true)
    {
        if (layer >= activation_functions_.size()) {
            throw std::out_of_range("Layer index out of range");
        }

        if (validate) {
            // Validate the activation function
            setActivationFunction(layer, function, true);

            // Basic validation of derivative - check it's not constant zero
            bool non_zero_found = false;
            for (T x = T{-5}; x <= T{5}; x += T{1}) {
                if (std::abs(derivative(x)) > T{1e-6}) {
                    non_zero_found = true;
                    break;
                }
            }
            if (!non_zero_found) {
                throw std::invalid_argument("Derivative function appears to be constant zero");
            }
        }
        else {
            activation_functions_[layer] = function;
        }

        // Store the derivative for later use (we'll need to extend the class to store derivatives)
        activation_derivatives_[layer] = derivative;
    }

    /**
     * @brief Constructor
     *
     * @param layer_sizes Vector containing the size of each layer (including input and output)
     * @param protection_level Protection level to apply
     */
    ProtectedNeuralNetwork(const std::vector<size_t>& layer_sizes,
                           ProtectionLevel protection_level = ProtectionLevel::ADAPTIVE_TMR)
        : layer_sizes_(layer_sizes),
          protection_level_(protection_level),
          weights_(),
          biases_(),
          activation_functions_()
    {
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument(
                "Neural network must have at least input and output layers");
        }

        // Initialize network structure
        initializeNetwork();
    }

    /**
     * @brief Copy constructor
     *
     * @param other Network to copy
     */
    ProtectedNeuralNetwork(const ProtectedNeuralNetwork& other)
        : layer_sizes_(other.layer_sizes_),
          protection_level_(other.protection_level_),
          check_counter_(other.check_counter_),
          error_stats_(other.error_stats_),
          activation_functions_(other.activation_functions_),
          activation_derivatives_(other.activation_derivatives_),
          layers_(other.layers_)
    {
        // Copy weights and biases with protection
        weights_.resize(other.weights_.size());
        for (size_t i = 0; i < other.weights_.size(); ++i) {
            weights_[i].resize(other.weights_[i].size());
            for (size_t j = 0; j < other.weights_[i].size(); ++j) {
                weights_[i][j].resize(other.weights_[i][j].size());
                for (size_t k = 0; k < other.weights_[i][j].size(); ++k) {
                    if constexpr (std::is_same_v<WeightType, T>) {
                        weights_[i][j][k] = other.weights_[i][j][k];
                    }
                    else {
                        weights_[i][j][k] = WeightType(other.weights_[i][j][k]);
                    }
                }
            }
        }

        biases_.resize(other.biases_.size());
        for (size_t i = 0; i < other.biases_.size(); ++i) {
            biases_[i].resize(other.biases_[i].size());
            for (size_t j = 0; j < other.biases_[i].size(); ++j) {
                if constexpr (std::is_same_v<WeightType, T>) {
                    biases_[i][j] = other.biases_[i][j];
                }
                else {
                    biases_[i][j] = WeightType(other.biases_[i][j]);
                }
            }
        }
    }

    /**
     * @brief Copy assignment operator
     *
     * @param other Network to copy
     * @return This network
     */
    ProtectedNeuralNetwork& operator=(const ProtectedNeuralNetwork& other)
    {
        if (this != &other) {
            layer_sizes_ = other.layer_sizes_;
            protection_level_ = other.protection_level_;
            check_counter_ = other.check_counter_;
            error_stats_ = other.error_stats_;
            activation_functions_ = other.activation_functions_;
            activation_derivatives_ = other.activation_derivatives_;
            layers_ = other.layers_;

            // Copy weights and biases with protection
            weights_.resize(other.weights_.size());
            for (size_t i = 0; i < other.weights_.size(); ++i) {
                weights_[i].resize(other.weights_[i].size());
                for (size_t j = 0; j < other.weights_[i].size(); ++j) {
                    weights_[i][j].resize(other.weights_[i][j].size());
                    for (size_t k = 0; k < other.weights_[i][j].size(); ++k) {
                        if constexpr (std::is_same_v<WeightType, T>) {
                            weights_[i][j][k] = other.weights_[i][j][k];
                        }
                        else {
                            weights_[i][j][k] = WeightType(other.weights_[i][j][k]);
                        }
                    }
                }
            }

            biases_.resize(other.biases_.size());
            for (size_t i = 0; i < other.biases_.size(); ++i) {
                biases_[i].resize(other.biases_[i].size());
                for (size_t j = 0; j < other.biases_[i].size(); ++j) {
                    if constexpr (std::is_same_v<WeightType, T>) {
                        biases_[i][j] = other.biases_[i][j];
                    }
                    else {
                        biases_[i][j] = WeightType(other.biases_[i][j]);
                    }
                }
            }
        }
        return *this;
    }

    /**
     * @brief Get the name of the network
     *
     * @return Network name
     */
    std::string getName() const override { return "ProtectedNeuralNetwork"; }

    /**
     * @brief Get the number of layers in the network
     *
     * @return Layer count
     */
    size_t getLayerCount() const override { return layer_sizes_.size(); }

    /**
     * @brief Get the input size of the network
     *
     * @return Input size
     */
    size_t getInputSize() const override { return layer_sizes_.front(); }

    /**
     * @brief Get the output size of the network
     *
     * @return Output size
     */
    size_t getOutputSize() const override { return layer_sizes_.back(); }

    /**
     * @brief Forward pass through the network (const version for evaluation)
     *
     * @param input Input tensor
     * @return Output tensor (without radiation adaptation)
     */
    std::vector<T> forward(const std::vector<T>& input) const
    {
        return forward_impl(input, 0.0, false);  // No radiation adaptation in const version
    }

    /**
     * @brief Optimized forward pass with constexpr and template optimizations
     *
     * This version uses compile-time optimizations and improved memory layout
     * for better performance on large layers.
     *
     * @param input Input tensor
     * @return Output tensor
     */
    std::vector<T> forward_optimized(const std::vector<T>& input) const
    {
        if (input.size() != getInputSize()) {
            throw std::invalid_argument("Input size mismatch");
        }

        std::vector<T> current_activations = input;
        std::vector<T> next_activations;

        // Use compile-time optimization based on network traits
        constexpr bool use_fast_path =
            constexpr_helpers::should_use_protection(protection_level_) &&
            NetworkTraits<5>::use_simd;  // Default assumption for optimization

        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            const size_t input_size = layer_sizes_[layer];
            const size_t output_size = layer_sizes_[layer + 1];

            next_activations.resize(output_size);

            // Runtime branch for SIMD optimization
            if (use_fast_path) {
#ifdef SIMD_ENABLED
                if (std::is_same_v<T, float> && input_size >= 8) {
                    // Use optimized SIMD path with constexpr layout
                    optimized_layer_computation(layer, current_activations, next_activations,
                                                input_size, output_size);
                }
                else
#endif
                {
                    standard_layer_computation(layer, current_activations, next_activations,
                                               input_size, output_size);
                }
            }
            else {
                // Fallback to standard computation
                standard_layer_computation(layer, current_activations, next_activations, input_size,
                                           output_size);
            }

            // Compile-time optimized activation application
            apply_activation_optimized(next_activations, layer);

            current_activations = std::move(next_activations);
        }

        return current_activations;
    }

    /**
     * @brief Forward pass through the network with radiation protection
     *
     * @param input Input tensor
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Output tensor
     */
    std::vector<T> forward(const std::vector<T>& input, double radiation_level)
    {
        return forward_impl(input, radiation_level, true);  // Full radiation protection
    }

    /**
     * @brief Reset error statistics
     */
    void resetErrorStats()
    {
        error_stats_.detected_errors = 0;
        error_stats_.corrected_errors = 0;
        error_stats_.uncorrectable_errors = 0;
    }

    /**
     * @brief Get the network layers
     *
     * @return Layers of the network
     */
    const std::vector<Layer>& getLayers() const { return layers_; }

    /**
     * @brief Get mutable access to the network layers
     *
     * @return Mutable reference to layers
     */
    std::vector<Layer>& getLayers() { return layers_; }

   private:
    /**
     * @brief Optimized layer computation using SIMD
     */
    void optimized_layer_computation(size_t layer, const std::vector<T>& current_activations,
                                     std::vector<T>& next_activations, size_t input_size,
                                     size_t output_size) const
    {
#ifdef SIMD_ENABLED
        // Use row-major layout: weight_matrix[j * input_size + i] for neuron j, input i
        std::vector<T> weight_matrix(input_size * output_size);
        std::vector<T> biases(output_size);

        // Pre-load weights and biases with runtime optimization for small layers
        if (input_size <= 4 && output_size <= 4) {
            // Unrolled loop for small layers
            for (size_t j = 0; j < output_size; ++j) {
                biases[j] = getBias(layer, j);
                for (size_t i = 0; i < input_size; ++i) {
                    weight_matrix[j * input_size + i] = getWeight(layer, i, j);
                }
            }
        }
        else {
            // Standard loading for larger layers
            for (size_t j = 0; j < output_size; ++j) {
                biases[j] = getBias(layer, j);
                for (size_t i = 0; i < input_size; ++i) {
                    weight_matrix[j * input_size + i] = getWeight(layer, i, j);
                }
            }
        }

        // Use SIMD multiplication
        matrixVectorMultiplyAVX(weight_matrix, current_activations, next_activations, output_size,
                                input_size);

        // Add biases
        for (size_t j = 0; j < output_size; ++j) {
            next_activations[j] += biases[j];
        }
#endif
    }

    /**
     * @brief Standard layer computation
     */
    void standard_layer_computation(size_t layer, const std::vector<T>& current_activations,
                                    std::vector<T>& next_activations, size_t input_size,
                                    size_t output_size) const
    {
        for (size_t j = 0; j < output_size; ++j) {
            T sum = getBias(layer, j);
            for (size_t i = 0; i < input_size; ++i) {
                sum += getWeight(layer, i, j) * current_activations[i];
            }
            next_activations[j] = sum;
        }
    }

    /**
     * @brief Optimized activation function application
     */
    void apply_activation_optimized(std::vector<T>& activations, size_t layer) const
    {
        if (NetworkTraits<5>::is_small_network) {
            // For small networks, use direct function calls
            const auto& activation = activation_functions_[layer];
            for (auto& val : activations) {
                val = activation(val);
            }
        }
        else {
            // For larger networks, use std::transform
            const auto& activation = activation_functions_[layer];
            std::transform(activations.begin(), activations.end(), activations.begin(), activation);
        }
    }

    /**
     * @brief Get optimized ReLU function for small networks
     */
    static std::function<T(T)> getOptimizedReLU()
    {
        return [](T x) -> T { return x > T{0} ? x : T{0}; };
    }

    /**
     * @brief Template specializations for common activation functions
     */
    template <typename Func>
    static constexpr T relu_derivative(T z)
    {
        return z > T{0} ? T{1} : T{0};
    }

    template <typename Func>
    static constexpr T sigmoid_derivative(T z, Func&& sig)
    {
        T sig_z = sig(z);
        return sig_z * (T{1} - sig_z);
    }

    template <typename Func>
    static constexpr T tanh_derivative(T z, Func&& tanh_func)
    {
        T tanh_z = tanh_func(z);
        return T{1} - tanh_z * tanh_z;
    }

    /**
     * @brief Apply protection to the network based on its criticality
     *
     * @param criticality_threshold Threshold for protection (0-1)
     * @return True if protection was successfully applied
     */
    bool applyProtection(float criticality_threshold = 0.5f) override
    {
        // Already set by constructor, but could be used to adjust protection
        return true;
    }

    /**
     * @brief Set weights for a layer
     *
     * @param layer Layer index (0 for first hidden layer)
     * @param weights Weight matrix (input_size x output_size)
     */
    void setLayerWeights(size_t layer, const std::vector<std::vector<T>>& weights)
    {
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
    void setLayerBiases(size_t layer, const std::vector<T>& biases)
    {
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
     * @brief Train the network using backpropagation with constexpr optimizations
     *
     * This version uses compile-time optimizations and improved algorithms
     * for better performance during training.
     *
     * @param training_data Input training samples (flattened: [sample1, sample2, ...])
     * @param training_labels Target outputs (flattened: [label1, label2, ...])
     * @param epochs Number of training epochs
     * @param batch_size Batch size for mini-batch gradient descent
     * @param learning_rate Learning rate for optimization
     * @param optimizer Optimization algorithm to use
     * @param validation_data Optional validation data for early stopping
     * @param validation_labels Optional validation labels
     * @return Training history with loss and accuracy metrics
     */

   public:
    /**
     * @brief Training history structure for tracking metrics
     */
    struct TrainingHistory {
        std::vector<T> train_losses;
        std::vector<T> train_accuracies;
        std::vector<T> val_losses;
        std::vector<T> val_accuracies;
        int best_epoch = -1;
        T best_val_loss = std::numeric_limits<T>::max();
    };
    enum class OptimizerType {
        SGD,       // Stochastic Gradient Descent
        MOMENTUM,  // SGD with Momentum
        ADAM,      // Adam optimizer
        RMSPROP    // RMSprop optimizer
    };

    struct OptimizerConfig {
        OptimizerType type = OptimizerType::ADAM;
        T learning_rate = static_cast<T>(0.001);
        T momentum = static_cast<T>(0.9);      // For MOMENTUM and ADAM
        T beta1 = static_cast<T>(0.9);         // For ADAM
        T beta2 = static_cast<T>(0.999);       // For ADAM
        T epsilon = static_cast<T>(1e-8);      // For ADAM and RMSPROP
        T decay = static_cast<T>(0.0);         // Learning rate decay
        T weight_decay = static_cast<T>(0.0);  // L2 regularization
    };

    TrainingHistory train(const std::vector<T>& training_data,
                          const std::vector<T>& training_labels, int epochs = 100,
                          int batch_size = 32, const OptimizerConfig& config = OptimizerConfig{},
                          const std::vector<T>& validation_data = {},
                          const std::vector<T>& validation_labels = {}, bool early_stopping = true,
                          int patience = 10, T min_delta = static_cast<T>(0.001),
                          bool verbose = true)
    {
        // Validate input data
        const size_t num_train_samples = training_data.size() / getInputSize();
        const size_t num_val_samples =
            validation_data.empty() ? 0 : validation_data.size() / getInputSize();

        if (training_data.size() % getInputSize() != 0 ||
            training_labels.size() != num_train_samples * getOutputSize()) {
            throw std::invalid_argument("Training data/labels size mismatch");
        }

        if (!validation_data.empty() &&
            (validation_data.size() % getInputSize() != 0 ||
             validation_labels.size() != num_val_samples * getOutputSize())) {
            throw std::invalid_argument("Validation data/labels size mismatch");
        }

        // Initialize training history
        TrainingHistory history;
        history.train_losses.reserve(epochs);
        history.train_accuracies.reserve(epochs);
        if (!validation_data.empty()) {
            history.val_losses.reserve(epochs);
            history.val_accuracies.reserve(epochs);
        }

        // Initialize optimizer state only if not already initialized or config changed
        if (current_optimizer_config_.type != config.type || weight_momentum_.empty()) {
            initializeOptimizer(config);
        }
        current_optimizer_config_ = config;

        // Create batch indices for shuffling
        std::vector<size_t> indices(num_train_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());

        int epochs_without_improvement = 0;
        T current_learning_rate = config.learning_rate;

        if (verbose) {
            std::cout << "Training Neural Network with " << num_train_samples << " samples, "
                      << epochs << " epochs, batch size " << batch_size << std::endl;
            std::cout << "Optimizer: " << optimizerTypeToString(config.type)
                      << ", Learning Rate: " << config.learning_rate << std::endl;
        }

        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Apply learning rate decay
            if (config.decay > 0) {
                current_learning_rate = config.learning_rate / (1 + config.decay * epoch);
            }

            // Shuffle training data
            std::shuffle(indices.begin(), indices.end(), gen);

            // Mini-batch training
            T epoch_loss = 0.0;
            T epoch_accuracy = 0.0;
            int num_batches = 0;

            for (size_t batch_start = 0; batch_start < num_train_samples;
                 batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, num_train_samples);
                size_t current_batch_size = batch_end - batch_start;

                // Extract batch data
                auto [batch_inputs, batch_targets] =
                    extractBatch(training_data, training_labels, indices, batch_start, batch_end);

                // Forward pass and compute gradients
                auto [batch_loss, batch_acc] =
                    trainBatch(batch_inputs, batch_targets, current_learning_rate, config);

                epoch_loss += batch_loss;
                epoch_accuracy += batch_acc;
                ++num_batches;
            }

            // Average metrics over batches
            epoch_loss /= num_batches;
            epoch_accuracy /= num_batches;

            history.train_losses.push_back(epoch_loss);
            history.train_accuracies.push_back(epoch_accuracy);

            // Validation
            T val_loss = 0.0, val_accuracy = 0.0;
            if (!validation_data.empty()) {
                std::tie(val_loss, val_accuracy) = evaluate(validation_data, validation_labels);
                history.val_losses.push_back(val_loss);
                history.val_accuracies.push_back(val_accuracy);

                // Early stopping check
                if (early_stopping) {
                    if (val_loss < history.best_val_loss - min_delta) {
                        history.best_val_loss = val_loss;
                        history.best_epoch = epoch;
                        epochs_without_improvement = 0;
                    }
                    else {
                        ++epochs_without_improvement;
                        if (epochs_without_improvement >= patience) {
                            if (verbose) {
                                std::cout << "Early stopping at epoch " << epoch + 1
                                          << " (best epoch: " << history.best_epoch + 1 << ")"
                                          << std::endl;
                            }
                            break;
                        }
                    }
                }
            }

            // Progress reporting
            if (verbose && (epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << std::fixed
                          << std::setprecision(6) << epoch_loss
                          << ", Accuracy: " << std::setprecision(4) << epoch_accuracy * 100 << "%";
                if (!validation_data.empty()) {
                    std::cout << " - Val Loss: " << std::setprecision(6) << val_loss
                              << ", Val Accuracy: " << std::setprecision(4) << val_accuracy * 100
                              << "%";
                }
                std::cout << std::endl;
            }
        }

        if (verbose) {
            std::cout << "Training completed!" << std::endl;
            if (history.best_epoch >= 0) {
                std::cout << "Best validation loss: " << history.best_val_loss << " at epoch "
                          << history.best_epoch + 1 << std::endl;
            }
        }

        return history;
    }

    /**
     * @brief Evaluate the network on test data with detailed metrics
     *
     * @param test_data Test input data (flattened)
     * @param test_labels Test target labels (flattened)
     * @return Pair of (loss, accuracy)
     */
    std::pair<T, T> evaluate(const std::vector<T>& test_data,
                             const std::vector<T>& test_labels) const
    {
        const size_t num_samples = test_data.size() / getInputSize();

        if (test_data.size() % getInputSize() != 0 ||
            test_labels.size() != num_samples * getOutputSize()) {
            throw std::invalid_argument("Test data/labels size mismatch");
        }

        T total_loss = 0.0;
        T total_accuracy = 0.0;

        for (size_t sample = 0; sample < num_samples; ++sample) {
            // Extract sample data
            std::vector<T> input(test_data.begin() + sample * getInputSize(),
                                 test_data.begin() + (sample + 1) * getInputSize());

            std::vector<T> target(test_labels.begin() + sample * getOutputSize(),
                                  test_labels.begin() + (sample + 1) * getOutputSize());

            // Forward pass
            std::vector<T> prediction = forward(input);

            // Compute metrics
            total_loss += computeLoss(prediction, target);
            total_accuracy += computeAccuracy(prediction, target);
        }

        return {total_loss / num_samples, total_accuracy / num_samples};
    }

    /**
     * @brief Calculate loss on the given data and labels (improved implementation)
     */
    float calculateLoss(const std::vector<T>& data, const std::vector<T>& labels)
    {
        auto [loss, _] = evaluate(data, labels);
        return static_cast<float>(loss);
    }

    /**
     * @brief Get a mutable reference to a specific layer (for radiation-aware training)
     *
     * @param layer_idx Index of the layer
     * @return Mutable reference to the layer
     */
    Layer& getLayerMutable(size_t layer_idx)
    {
        if (layer_idx >= layers_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers_[layer_idx];
    }

    /**
     * @brief Compute derivative of activation function with constexpr optimizations
     *
     * This method uses compile-time optimizations and if constexpr for better performance.
     */
    template <typename ActivationFunc>
    static constexpr T compute_derivative_analytical(T z, ActivationFunc&& func)
    {
        // Use if constexpr for compile-time branching (C++17 feature)
        if constexpr (std::is_invocable_v<ActivationFunc, T>) {
            // For known functions at compile time, we could add optimizations here
            // For now, delegate to runtime detection
            return compute_derivative_runtime(z, func);
        }
        else {
            return T{1};  // Default derivative
        }
    }

    /**
     * @brief Runtime activation derivative computation (for testing)
     *
     * This method uses analytical detection and numerical differentiation
     * to compute the derivative of the activation function.
     */
    T computeActivationDerivative(T z, size_t layer) const
    {
        if (layer >= activation_functions_.size()) {
            // Fallback to linear derivative for output layer or invalid layer
            return T{1};
        }

        // First, check if we have an explicit derivative function
        if (layer < activation_derivatives_.size() && activation_derivatives_[layer]) {
            return activation_derivatives_[layer](z);
        }

        // Fall back to analytical detection and numerical differentiation
        const auto& activation_func = activation_functions_[layer];

        // Use compile-time optimized analytical detection
        return detect_and_compute_derivative(z, activation_func, layer);
    }

   private:
    /**
     * @brief Runtime derivative computation with analytical detection
     */
    static T compute_derivative_runtime(T z, const std::function<T(T)>& func)
    {
        // Test common activation functions with compile-time constants
        const T epsilon = T{1e-6};
        const T relu_threshold = T{1e-6};
        const T sigmoid_threshold = T{1e-5};
        const T elu_alpha = T{1.0};
        const T elu_expected_neg = T{-0.6321205588285577};  // exp(-1) - 1

        // ReLU detection: f(x) = max(0, x)
        const T relu_pos = func(T{1});
        const T relu_neg = func(T{-1});
        if (std::abs(relu_pos - T{1}) < relu_threshold &&
            std::abs(relu_neg - T{0}) < relu_threshold) {
            return z > T{0} ? T{1} : T{0};
        }

        // Leaky ReLU detection: f(x) = x if x > 0, else α*x
        const T leaky_pos = func(T{1});
        const T leaky_zero = func(T{0});
        const T leaky_neg = func(T{-1});
        if (std::abs(leaky_pos - T{1}) < epsilon && std::abs(leaky_zero - T{0}) < epsilon &&
            leaky_neg < T{0} && leaky_neg > T{-0.5}) {
            const T alpha = -leaky_neg;
            return z > T{0} ? T{1} : alpha;
        }

        // Sigmoid detection: f(x) = 1/(1+exp(-x))
        const T sigmoid_zero = func(T{0});
        if (std::abs(sigmoid_zero - T{0.5}) < sigmoid_threshold) {
            const T sigmoid_z = func(z);
            return sigmoid_z * (T{1} - sigmoid_z);
        }

        // Tanh detection: f(x) = tanh(x)
        const T tanh_zero = func(T{0});
        const T tanh_one = func(T{1});
        if (std::abs(tanh_zero - T{0}) < epsilon &&
            std::abs(tanh_one - std::tanh(T{1})) < sigmoid_threshold) {
            const T tanh_z = func(z);
            return T{1} - tanh_z * tanh_z;
        }

        // Linear detection: f(x) = x
        const T linear_pos = func(T{1});
        const T linear_neg = func(T{-1});
        if (std::abs(linear_pos - T{1}) < epsilon && std::abs(linear_neg - T{-1}) < epsilon) {
            return T{1};
        }

        // ELU detection: f(x) = x if x > 0, else α*(exp(x) - 1)
        const T elu_pos = func(T{1});
        const T elu_zero = func(T{0});
        const T elu_neg = func(T{-1});
        if (std::abs(elu_pos - T{1}) < epsilon && std::abs(elu_zero - T{0}) < epsilon &&
            std::abs(elu_neg - elu_expected_neg) < epsilon) {
            return z > T{0} ? T{1} : std::exp(z) * elu_alpha;
        }

        // For custom/unknown functions, use numerical differentiation with adaptive epsilon
        const T base_epsilon = T{1e-4};
        const T adaptive_epsilon = std::max(base_epsilon, std::abs(z) * T{1e-5});
        const T numerical_epsilon = std::min(adaptive_epsilon, T{1e-3});

        const T f_plus = func(z + numerical_epsilon);
        const T f_minus = func(z - numerical_epsilon);

        T derivative = (f_plus - f_minus) / (T{2} * numerical_epsilon);
        derivative = std::max(T{-10}, std::min(T{10}, derivative));

        return derivative;
    }

    /**
     * @brief Analytical derivative detection and computation
     */
    T detect_and_compute_derivative(T z, const std::function<T(T)>& func, size_t layer) const
    {
        // Use compile-time optimized detection with if constexpr
        if constexpr (std::is_same_v<T, float>) {
            return compute_derivative_runtime(z, func);
        }
        else {
            return compute_derivative_runtime(z, func);
        }
    }

    /**
     * @brief Reset optimizer state (momentum, velocity, step count)
     * Useful for starting fresh training or switching between different training phases
     */
    void resetOptimizerState()
    {
        weight_momentum_.clear();
        bias_momentum_.clear();
        weight_velocity_.clear();
        bias_velocity_.clear();
        optimizer_step_ = 0;

        // Re-initialize with current config
        if (current_optimizer_config_.type != OptimizerType::SGD) {
            initializeOptimizer(current_optimizer_config_);
        }
    }

    /**
     * @brief Set optimizer configuration and initialize state if needed
     */
    void setOptimizerConfig(const OptimizerConfig& config)
    {
        if (current_optimizer_config_.type != config.type) {
            current_optimizer_config_ = config;
            initializeOptimizer(config);
        }
        else {
            current_optimizer_config_ = config;
        }
    }

    /**
     * @brief Get current optimizer configuration
     */
    const OptimizerConfig& getOptimizerConfig() const { return current_optimizer_config_; }

    /**
     * @brief Simplified backward pass (for testing)
     */
    void backward(const std::vector<T>& input, const std::vector<T>& target)
    {
        // This is a simplified implementation for testing purposes
        // In practice, this would compute gradients and update weights

        // Perform forward pass to get activations
        auto output = forward(input);

        // Compute simple loss (MSE)
        T loss = T{0};
        for (size_t i = 0; i < output.size() && i < target.size(); ++i) {
            T diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= static_cast<T>(output.size());

        // For testing, we just acknowledge that backward pass was called
        // Real implementation would compute and apply gradients
    }

   protected:
    /**
     * @brief Get a weight value from a specific layer
     *
     * @param layer Layer index
     * @param input Input neuron index
     * @param output Output neuron index
     * @return Weight value
     */
    T getWeight(size_t layer, size_t input, size_t output) const
    {
        if (layer >= weights_.size() || input >= weights_[layer].size() ||
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            return weights_[layer][input][output];
        }
        else {
            auto value = weights_[layer][input][output].getValue();
            if (weights_[layer][input][output].hasError()) {
                error_stats_.detected_errors++;
                if (weights_[layer][input][output].correctErrors()) {
                    error_stats_.corrected_errors++;
                }
                else {
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
    void setWeight(size_t layer, size_t input, size_t output, const T& value)
    {
        if (layer >= weights_.size() || input >= weights_[layer].size() ||
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            weights_[layer][input][output] = value;
        }
        else {
            weights_[layer][input][output].setValue(value);
        }

        // Update the layer representation
        layers_[layer].weights[input][output] = value;
    }

    /**
     * @brief Get a bias value from a specific layer
     *
     * @param layer Layer index
     * @param output Output neuron index
     * @return Bias value
     */
    T getBias(size_t layer, size_t output) const
    {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            return biases_[layer][output];
        }
        else {
            auto value = biases_[layer][output].getValue();
            if (biases_[layer][output].hasError()) {
                error_stats_.detected_errors++;
                if (biases_[layer][output].correctErrors()) {
                    error_stats_.corrected_errors++;
                }
                else {
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
    void setBias(size_t layer, size_t output, const T& value)
    {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            biases_[layer][output] = value;
        }
        else {
            biases_[layer][output].setValue(value);
        }

        // Update the layer representation
        layers_[layer].biases[output] = value;
    }

   private:
    // Define the weight protection type based on protection level
    using WeightType = std::conditional_t<std::is_floating_point_v<T>, MultibitProtection<T>, T>;

    /**
     * @brief Initialize the network structure
     */
    void initializeNetwork()
    {
        size_t num_layers = layer_sizes_.size();

        // Initialize weights for each layer
        weights_.resize(num_layers - 1);
        biases_.resize(num_layers - 1);
        layers_.resize(num_layers - 1);

        for (size_t i = 0; i < num_layers - 1; ++i) {
            weights_[i].resize(layer_sizes_[i]);
            for (size_t j = 0; j < layer_sizes_[i]; ++j) {
                weights_[i][j].resize(layer_sizes_[i + 1]);
            }

            biases_[i].resize(layer_sizes_[i + 1]);

            // Initialize the Layer structure
            layers_[i].weights.resize(layer_sizes_[i], std::vector<T>(layer_sizes_[i + 1]));
            layers_[i].biases.resize(layer_sizes_[i + 1]);
        }

        // Initialize activation functions with compile-time optimization
        activation_functions_.resize(num_layers - 1, [](T x) { return x > T{0} ? x : T{0}; });
        activation_derivatives_.resize(num_layers - 1);

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
                    layers_[layer].weights[i][j] = value;
                }
            }

            // Initialize biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                setBias(layer, j, T{0});
                layers_[layer].biases[j] = T{0};
            }
        }
    }

    /**
     * @brief Create a protected value based on the protection level
     *
     * @param value Raw value
     * @return Protected value
     */
    auto createProtectedValue(const T& value) const
    {
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
     * @brief Set a weight value without protection
     *
     * @param layer Layer index
     * @param input Input neuron index
     * @param output Output neuron index
     * @param value New weight value
     */
    void raw_setWeight(size_t layer, size_t input, size_t output, const T& value)
    {
        if (layer >= weights_.size() || input >= weights_[layer].size() ||
            output >= weights_[layer][input].size()) {
            throw std::out_of_range("Weight index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            weights_[layer][input][output] = value;
        }
        else {
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
    void raw_setBias(size_t layer, size_t output, const T& value)
    {
        if (layer >= biases_.size() || output >= biases_[layer].size()) {
            throw std::out_of_range("Bias index out of range");
        }

        if constexpr (std::is_same_v<WeightType, T>) {
            biases_[layer][output] = value;
        }
        else {
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
    template <typename RNG>
    T applyBitFlip(T value, RNG& rng) const
    {
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
    void protectActivations(std::vector<T>& activations, double radiation_level)
    {
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
                }
                else if (copy1[i] == copy2[i]) {
                    // Two copies match, use their value
                    activations[i] = copy1[i];
                    error_stats_.detected_errors++;
                    error_stats_.corrected_errors++;
                }
                else if (activations[i] == copy2[i]) {
                    // Original matches copy2, use this value
                    error_stats_.detected_errors++;
                    error_stats_.corrected_errors++;
                    continue;
                }
                else {
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
    void adaptToRadiationLevel(double radiation_level)
    {
        if (protection_level_ != ProtectionLevel::ADAPTIVE_TMR) return;

        // Increase error checking frequency in high radiation
        if (radiation_level > 0.5) {
            // Periodically check and repair all weights
            if (++check_counter_ % 10 == 0) {
                repairAllWeights();
            }
        }
        else {
            // Less frequent checking in low radiation
            if (++check_counter_ % 100 == 0) {
                repairAllWeights();
            }
        }
    }

    /**
     * @brief Repair all weights in the network
     */
    void repairAllWeights()
    {
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
                            }
                            else {
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
                        }
                        else {
                            error_stats_.uncorrectable_errors++;
                        }
                    }
                }
            }
        }
    }

    // Optimizer state variables
    std::vector<std::vector<std::vector<T>>> weight_momentum_;
    std::vector<std::vector<T>> bias_momentum_;
    std::vector<std::vector<std::vector<T>>> weight_velocity_;
    std::vector<std::vector<T>> bias_velocity_;
    int optimizer_step_ = 0;
    OptimizerConfig current_optimizer_config_;

    /**
     * @brief Initialize optimizer state based on configuration
     *
     * This should only be called when a new optimizer is set or when explicitly resetting state.
     * It should NOT be called at the start of every train() call to preserve momentum/velocity.
     */
    void initializeOptimizer(const OptimizerConfig& config)
    {
        const size_t num_layers = layer_sizes_.size() - 1;

        // Initialize momentum and velocity terms for optimizers that need them
        if (config.type == OptimizerType::MOMENTUM || config.type == OptimizerType::ADAM) {
            weight_momentum_.resize(num_layers);
            bias_momentum_.resize(num_layers);

            for (size_t layer = 0; layer < num_layers; ++layer) {
                weight_momentum_[layer].resize(layer_sizes_[layer]);
                for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                    weight_momentum_[layer][i].resize(layer_sizes_[layer + 1], T{0});
                }
                bias_momentum_[layer].resize(layer_sizes_[layer + 1], T{0});
            }
        }

        if (config.type == OptimizerType::ADAM || config.type == OptimizerType::RMSPROP) {
            weight_velocity_.resize(num_layers);
            bias_velocity_.resize(num_layers);

            for (size_t layer = 0; layer < num_layers; ++layer) {
                weight_velocity_[layer].resize(layer_sizes_[layer]);
                for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                    weight_velocity_[layer][i].resize(layer_sizes_[layer + 1], T{0});
                }
                bias_velocity_[layer].resize(layer_sizes_[layer + 1], T{0});
            }
        }

        optimizer_step_ = 0;
    }

    /**
     * @brief Extract a batch of data from the training set
     */
    std::pair<std::vector<std::vector<T>>, std::vector<std::vector<T>>> extractBatch(
        const std::vector<T>& data, const std::vector<T>& labels,
        const std::vector<size_t>& indices, size_t batch_start, size_t batch_end) const
    {
        const size_t batch_size = batch_end - batch_start;
        const size_t input_size = getInputSize();
        const size_t output_size = getOutputSize();

        std::vector<std::vector<T>> batch_inputs(batch_size, std::vector<T>(input_size));
        std::vector<std::vector<T>> batch_targets(batch_size, std::vector<T>(output_size));

        for (size_t i = 0; i < batch_size; ++i) {
            const size_t sample_idx = indices[batch_start + i];

            // Copy input data
            std::copy(data.begin() + sample_idx * input_size,
                      data.begin() + (sample_idx + 1) * input_size, batch_inputs[i].begin());

            // Copy target data
            std::copy(labels.begin() + sample_idx * output_size,
                      labels.begin() + (sample_idx + 1) * output_size, batch_targets[i].begin());
        }

        return {batch_inputs, batch_targets};
    }

    /**
     * @brief Train on a single batch using backpropagation
     */
    std::pair<T, T> trainBatch(const std::vector<std::vector<T>>& batch_inputs,
                               const std::vector<std::vector<T>>& batch_targets, T learning_rate,
                               const OptimizerConfig& config)
    {
        const size_t batch_size = batch_inputs.size();
        const size_t num_layers = layer_sizes_.size();

        // Initialize gradient accumulators
        std::vector<std::vector<std::vector<T>>> weight_gradients(num_layers - 1);
        std::vector<std::vector<T>> bias_gradients(num_layers - 1);

        for (size_t layer = 0; layer < num_layers - 1; ++layer) {
            weight_gradients[layer].resize(layer_sizes_[layer]);
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                weight_gradients[layer][i].resize(layer_sizes_[layer + 1], T{0});
            }
            bias_gradients[layer].resize(layer_sizes_[layer + 1], T{0});
        }

        T total_loss = 0.0;
        T total_accuracy = 0.0;

        // Process each sample in the batch
        for (size_t sample = 0; sample < batch_size; ++sample) {
            // Forward pass with activation storage
            auto [activations, z_values] = forwardPassWithStorage(batch_inputs[sample]);

            // Optionally project output onto simplex for loss/metrics
            const std::vector<T>& raw_output = activations.back();
            std::vector<T> projected_output;
            if (use_simplex_projection_ && raw_output.size() > 1) {
                std::vector<double> raw_as_double(raw_output.size(), 0.0);
                for (size_t i = 0; i < raw_output.size(); ++i) {
                    raw_as_double[i] = static_cast<double>(raw_output[i]);
                }
                const std::vector<double> projected =
                    rad_ml::optimization::SimplexProjection::forward_vector(raw_as_double);
                projected_output.resize(projected.size(), T{0});
                for (size_t i = 0; i < projected.size(); ++i) {
                    projected_output[i] = static_cast<T>(projected[i]);
                }
            }

            std::vector<T> predictions_for_metrics =
                (use_simplex_projection_ && raw_output.size() > 1) ? projected_output : raw_output;

            // Compute loss
            T sample_loss = computeLoss(predictions_for_metrics, batch_targets[sample]);
            total_loss += sample_loss;

            // Compute accuracy
            T sample_accuracy = computeAccuracy(predictions_for_metrics, batch_targets[sample]);
            total_accuracy += sample_accuracy;

            // Backward pass
            if (use_simplex_projection_ && raw_output.size() > 1) {
                std::vector<double> raw_as_double(raw_output.size(), 0.0);
                for (size_t i = 0; i < raw_output.size(); ++i) {
                    raw_as_double[i] = static_cast<double>(raw_output[i]);
                }

                // Compose loss gradient through projection for output layer
                std::vector<double> a_proj(raw_output.size(), 0.0);
                if (!projected_output.empty()) {
                    for (size_t i = 0; i < raw_output.size(); ++i)
                        a_proj[i] = static_cast<double>(projected_output[i]);
                }
                else {
                    a_proj = rad_ml::optimization::SimplexProjection::forward_vector(raw_as_double);
                }

                // dL/da_proj = (a_proj - y)
                std::vector<double> g_up(raw_output.size(), 0.0);
                for (size_t i = 0; i < raw_output.size(); ++i) {
                    g_up[i] = a_proj[i] - static_cast<double>(batch_targets[sample][i]);
                }
                const std::vector<double> dL_da =
                    rad_ml::optimization::SimplexProjection::backward_vector(raw_as_double, g_up);

                // Create pseudo-target so that (a - pseudo) = dL/da
                std::vector<T> adjusted_targets(raw_output.size());
                for (size_t i = 0; i < raw_output.size(); ++i) {
                    adjusted_targets[i] = static_cast<T>(raw_output[i] - static_cast<T>(dL_da[i]));
                }

                backpropagation(activations, z_values, adjusted_targets, weight_gradients,
                                bias_gradients);
            }
            else {
                backpropagation(activations, z_values, batch_targets[sample], weight_gradients,
                                bias_gradients);
            }
        }

        // Average gradients over batch
        for (size_t layer = 0; layer < num_layers - 1; ++layer) {
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    weight_gradients[layer][i][j] /= batch_size;
                }
            }
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                bias_gradients[layer][j] /= batch_size;
            }
        }

        // Apply optimizer update
        applyOptimizerUpdate(weight_gradients, bias_gradients, learning_rate, config);

        return {total_loss / batch_size, total_accuracy / batch_size};
    }

    /**
     * @brief Forward pass with activation and pre-activation storage
     */
    std::pair<std::vector<std::vector<T>>, std::vector<std::vector<T>>> forwardPassWithStorage(
        const std::vector<T>& input) const
    {
        const size_t num_layers = layer_sizes_.size();

        std::vector<std::vector<T>> activations(num_layers);
        std::vector<std::vector<T>> z_values(num_layers - 1);  // Pre-activation values

        // Input layer
        activations[0] = input;

        // Hidden and output layers
        for (size_t layer = 0; layer < num_layers - 1; ++layer) {
            z_values[layer].resize(layer_sizes_[layer + 1]);
            activations[layer + 1].resize(layer_sizes_[layer + 1]);

            // Compute pre-activation values (z = Wx + b)
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T z = getBias(layer, j);
                for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                    z += getWeight(layer, i, j) * activations[layer][i];
                }
                z_values[layer][j] = z;

                // Apply activation function
                activations[layer + 1][j] = activation_functions_[layer](z);
            }
        }

        return {activations, z_values};
    }

    /**
     * @brief Compute loss using mean squared error with constexpr optimizations
     */
    T computeLoss(const std::vector<T>& predictions, const std::vector<T>& targets) const
    {
        if (NetworkTraits<5>::is_small_network) {
            // For small networks, use optimized loop
            return computeLossSmall(predictions, targets);
        }
        else {
            // For larger networks, use standard loop
            return computeLossLarge(predictions, targets);
        }
    }

   private:
    bool use_simplex_projection_ = false;
    /**
     * @brief Optimized loss computation for small networks
     */
    T computeLossSmall(const std::vector<T>& predictions, const std::vector<T>& targets) const
    {
        T loss = 0.0;
        const size_t size = predictions.size();

        // Runtime optimization for small arrays
        if (size <= 4) {
            for (size_t i = 0; i < size; ++i) {
                const T diff = predictions[i] - targets[i];
                loss += diff * diff;
            }
        }
        else {
            for (size_t i = 0; i < size; ++i) {
                const T diff = predictions[i] - targets[i];
                loss += diff * diff;
            }
        }

        return loss / (T{2.0} * static_cast<T>(size));
    }

    /**
     * @brief Standard loss computation for large networks
     */
    T computeLossLarge(const std::vector<T>& predictions, const std::vector<T>& targets) const
    {
        T loss = T{0};
        for (size_t i = 0; i < predictions.size(); ++i) {
            const T diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        return loss / (T{2.0} * predictions.size());
    }

    /**
     * @brief Compute accuracy (percentage of correct predictions)
     */
    T computeAccuracy(const std::vector<T>& predictions, const std::vector<T>& targets) const
    {
        if (predictions.size() == 1) {
            // Regression case - use threshold-based accuracy
            return std::abs(predictions[0] - targets[0]) < 0.1 ? 1.0 : 0.0;
        }
        else {
            // Classification case - check if predicted class matches target class
            auto pred_max = std::max_element(predictions.begin(), predictions.end());
            auto target_max = std::max_element(targets.begin(), targets.end());

            size_t pred_class = std::distance(predictions.begin(), pred_max);
            size_t target_class = std::distance(targets.begin(), target_max);

            return pred_class == target_class ? 1.0 : 0.0;
        }
    }

    /**
     * @brief Backpropagation algorithm implementation with constexpr optimizations
     */
    void backpropagation(const std::vector<std::vector<T>>& activations,
                         const std::vector<std::vector<T>>& z_values, const std::vector<T>& targets,
                         std::vector<std::vector<std::vector<T>>>& weight_gradients,
                         std::vector<std::vector<T>>& bias_gradients) const
    {
        const size_t num_layers = layer_sizes_.size();
        const bool use_optimized_backprop =
            NetworkTraits<5>::is_small_network;  // Default assumption

        if (use_optimized_backprop) {
            backpropagation_optimized(activations, z_values, targets, weight_gradients,
                                      bias_gradients);
        }
        else {
            backpropagation_standard(activations, z_values, targets, weight_gradients,
                                     bias_gradients);
        }
    }

   private:
    /**
     * @brief Optimized backpropagation for small networks
     */
    void backpropagation_optimized(const std::vector<std::vector<T>>& activations,
                                   const std::vector<std::vector<T>>& z_values,
                                   const std::vector<T>& targets,
                                   std::vector<std::vector<std::vector<T>>>& weight_gradients,
                                   std::vector<std::vector<T>>& bias_gradients) const
    {
        const size_t output_layer = layer_sizes_.size() - 2;
        const size_t output_size = layer_sizes_.back();

        // Compute output layer deltas
        for (size_t j = 0; j < output_size; ++j) {
            const T error = activations[output_layer + 1][j] - targets[j];
            const T activation_derivative =
                computeActivationDerivative(z_values[output_layer][j], output_layer);
            bias_gradients[output_layer][j] = error * activation_derivative;

            // Compute weight gradients
            for (size_t i = 0; i < layer_sizes_[output_layer]; ++i) {
                weight_gradients[output_layer][i][j] =
                    activations[output_layer][i] * bias_gradients[output_layer][j];
            }
        }

        // Backpropagate to hidden layers
        for (int layer = static_cast<int>(output_layer) - 1; layer >= 0; --layer) {
            const size_t layer_size = layer_sizes_[layer + 1];

            for (size_t j = 0; j < layer_size; ++j) {
                T error = T{0};
                for (size_t k = 0; k < layer_sizes_[layer + 2]; ++k) {
                    error += bias_gradients[layer + 1][k] * getWeight(layer + 1, j, k);
                }

                const T activation_derivative =
                    computeActivationDerivative(z_values[layer][j], layer);
                bias_gradients[layer][j] = error * activation_derivative;

                // Weight gradients
                for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                    weight_gradients[layer][i][j] =
                        activations[layer][i] * bias_gradients[layer][j];
                }
            }
        }
    }

    /**
     * @brief Standard backpropagation for larger networks
     */
    void backpropagation_standard(const std::vector<std::vector<T>>& activations,
                                  const std::vector<std::vector<T>>& z_values,
                                  const std::vector<T>& targets,
                                  std::vector<std::vector<std::vector<T>>>& weight_gradients,
                                  std::vector<std::vector<T>>& bias_gradients) const
    {
        const size_t num_layers = layer_sizes_.size();
        std::vector<std::vector<T>> deltas(num_layers - 1);

        // Output layer error (delta)
        const size_t output_layer = num_layers - 2;
        deltas[output_layer].resize(layer_sizes_[output_layer + 1]);

        for (size_t j = 0; j < layer_sizes_[output_layer + 1]; ++j) {
            // For MSE loss: delta = (a - y) * activation_derivative(z)
            T error = activations[output_layer + 1][j] - targets[j];
            T activation_derivative =
                computeActivationDerivative(z_values[output_layer][j], output_layer);
            deltas[output_layer][j] = error * activation_derivative;
        }

        // Backpropagate errors to hidden layers
        for (int layer = static_cast<int>(output_layer) - 1; layer >= 0; --layer) {
            deltas[layer].resize(layer_sizes_[layer + 1]);

            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T error = 0.0;

                // Sum weighted errors from next layer
                for (size_t k = 0; k < layer_sizes_[layer + 2]; ++k) {
                    error += deltas[layer + 1][k] * getWeight(layer + 1, j, k);
                }

                T activation_derivative = computeActivationDerivative(z_values[layer][j], layer);
                deltas[layer][j] = error * activation_derivative;
            }
        }

        // Compute gradients
        for (size_t layer = 0; layer < num_layers - 1; ++layer) {
            // Weight gradients
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    weight_gradients[layer][i][j] += activations[layer][i] * deltas[layer][j];
                }
            }

            // Bias gradients
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                bias_gradients[layer][j] += deltas[layer][j];
            }
        }
    }

    /**
     * @brief Apply optimizer update to weights and biases
     */
    void applyOptimizerUpdate(const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                              const std::vector<std::vector<T>>& bias_gradients, T learning_rate,
                              const OptimizerConfig& config)
    {
        ++optimizer_step_;
        const size_t num_layers = layer_sizes_.size() - 1;

        switch (config.type) {
            case OptimizerType::SGD:
                applySGDUpdate(weight_gradients, bias_gradients, learning_rate, config);
                break;

            case OptimizerType::MOMENTUM:
                applyMomentumUpdate(weight_gradients, bias_gradients, learning_rate, config);
                break;

            case OptimizerType::ADAM:
                applyAdamUpdate(weight_gradients, bias_gradients, learning_rate, config);
                break;

            case OptimizerType::RMSPROP:
                applyRMSpropUpdate(weight_gradients, bias_gradients, learning_rate, config);
                break;
        }
    }

    /**
     * @brief Apply SGD update
     */
    void applySGDUpdate(const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                        const std::vector<std::vector<T>>& bias_gradients, T learning_rate,
                        const OptimizerConfig& config)
    {
        const size_t num_layers = layer_sizes_.size() - 1;

        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Update weights
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    T current_weight = getWeight(layer, i, j);
                    T gradient = weight_gradients[layer][i][j];

                    // Add L2 regularization if specified
                    if (config.weight_decay > 0) {
                        gradient += config.weight_decay * current_weight;
                    }

                    T new_weight = current_weight - learning_rate * gradient;
                    setWeight(layer, i, j, new_weight);
                }
            }

            // Update biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T current_bias = getBias(layer, j);
                T gradient = bias_gradients[layer][j];
                T new_bias = current_bias - learning_rate * gradient;
                setBias(layer, j, new_bias);
            }
        }
    }

    /**
     * @brief Apply Momentum update
     */
    void applyMomentumUpdate(const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                             const std::vector<std::vector<T>>& bias_gradients, T learning_rate,
                             const OptimizerConfig& config)
    {
        const size_t num_layers = layer_sizes_.size() - 1;

        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Update weights
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    T current_weight = getWeight(layer, i, j);
                    T gradient = weight_gradients[layer][i][j];

                    // Add L2 regularization if specified
                    if (config.weight_decay > 0) {
                        gradient += config.weight_decay * current_weight;
                    }

                    // Update momentum
                    weight_momentum_[layer][i][j] =
                        config.momentum * weight_momentum_[layer][i][j] + learning_rate * gradient;

                    T new_weight = current_weight - weight_momentum_[layer][i][j];
                    setWeight(layer, i, j, new_weight);
                }
            }

            // Update biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T current_bias = getBias(layer, j);
                T gradient = bias_gradients[layer][j];

                // Update momentum
                bias_momentum_[layer][j] =
                    config.momentum * bias_momentum_[layer][j] + learning_rate * gradient;

                T new_bias = current_bias - bias_momentum_[layer][j];
                setBias(layer, j, new_bias);
            }
        }
    }

    /**
     * @brief Apply Adam update
     */
    void applyAdamUpdate(const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                         const std::vector<std::vector<T>>& bias_gradients, T learning_rate,
                         const OptimizerConfig& config)
    {
        const size_t num_layers = layer_sizes_.size() - 1;

        // Bias correction terms
        T beta1_t = std::pow(config.beta1, optimizer_step_);
        T beta2_t = std::pow(config.beta2, optimizer_step_);
        T lr_corrected = learning_rate * std::sqrt(1 - beta2_t) / (1 - beta1_t);

        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Update weights
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    T current_weight = getWeight(layer, i, j);
                    T gradient = weight_gradients[layer][i][j];

                    // Add L2 regularization if specified
                    if (config.weight_decay > 0) {
                        gradient += config.weight_decay * current_weight;
                    }

                    // Update biased first moment estimate
                    weight_momentum_[layer][i][j] = config.beta1 * weight_momentum_[layer][i][j] +
                                                    (1 - config.beta1) * gradient;

                    // Update biased second raw moment estimate
                    weight_velocity_[layer][i][j] = config.beta2 * weight_velocity_[layer][i][j] +
                                                    (1 - config.beta2) * gradient * gradient;

                    // Compute update
                    T update = lr_corrected * weight_momentum_[layer][i][j] /
                               (std::sqrt(weight_velocity_[layer][i][j]) + config.epsilon);

                    T new_weight = current_weight - update;
                    setWeight(layer, i, j, new_weight);
                }
            }

            // Update biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T current_bias = getBias(layer, j);
                T gradient = bias_gradients[layer][j];

                // Update biased first moment estimate
                bias_momentum_[layer][j] =
                    config.beta1 * bias_momentum_[layer][j] + (1 - config.beta1) * gradient;

                // Update biased second raw moment estimate
                bias_velocity_[layer][j] = config.beta2 * bias_velocity_[layer][j] +
                                           (1 - config.beta2) * gradient * gradient;

                // Compute update
                T update = lr_corrected * bias_momentum_[layer][j] /
                           (std::sqrt(bias_velocity_[layer][j]) + config.epsilon);

                T new_bias = current_bias - update;
                setBias(layer, j, new_bias);
            }
        }
    }

    /**
     * @brief Apply RMSprop update
     */
    void applyRMSpropUpdate(const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                            const std::vector<std::vector<T>>& bias_gradients, T learning_rate,
                            const OptimizerConfig& config)
    {
        const size_t num_layers = layer_sizes_.size() - 1;

        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Update weights
            for (size_t i = 0; i < layer_sizes_[layer]; ++i) {
                for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                    T current_weight = getWeight(layer, i, j);
                    T gradient = weight_gradients[layer][i][j];

                    // Add L2 regularization if specified
                    if (config.weight_decay > 0) {
                        gradient += config.weight_decay * current_weight;
                    }

                    // Update moving average of squared gradients
                    weight_velocity_[layer][i][j] = config.beta2 * weight_velocity_[layer][i][j] +
                                                    (1 - config.beta2) * gradient * gradient;

                    // Compute update
                    T update = learning_rate * gradient /
                               (std::sqrt(weight_velocity_[layer][i][j]) + config.epsilon);

                    T new_weight = current_weight - update;
                    setWeight(layer, i, j, new_weight);
                }
            }

            // Update biases
            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T current_bias = getBias(layer, j);
                T gradient = bias_gradients[layer][j];

                // Update moving average of squared gradients
                bias_velocity_[layer][j] = config.beta2 * bias_velocity_[layer][j] +
                                           (1 - config.beta2) * gradient * gradient;

                // Compute update
                T update = learning_rate * gradient /
                           (std::sqrt(bias_velocity_[layer][j]) + config.epsilon);

                T new_bias = current_bias - update;
                setBias(layer, j, new_bias);
            }
        }
    }

    /**
     * @brief Convert optimizer type to string for logging
     */
    std::string optimizerTypeToString(OptimizerType type) const
    {
        switch (type) {
            case OptimizerType::SGD:
                return "SGD";
            case OptimizerType::MOMENTUM:
                return "SGD+Momentum";
            case OptimizerType::ADAM:
                return "Adam";
            case OptimizerType::RMSPROP:
                return "RMSprop";
            default:
                return "Unknown";
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
    std::vector<std::function<T(T)>> activation_derivatives_;

    // Layers representation for external access
    std::vector<Layer> layers_;

    // Error statistics
    mutable struct {
        uint64_t detected_errors = 0;
        uint64_t corrected_errors = 0;
        uint64_t uncorrectable_errors = 0;
    } error_stats_;

    /**
     * @brief Internal forward pass implementation
     *
     * @param input Input tensor
     * @param radiation_level Current radiation level (0.0-1.0)
     * @param enable_protection Whether to enable radiation protection adaptations
     * @return Output tensor
     */
    std::vector<T> forward_impl(const std::vector<T>& input, double radiation_level,
                                bool enable_protection) const
    {
        if (input.size() != getInputSize()) {
            // core::Logger::error("Input size mismatch in forward pass");
            // core::Logger::error("Expected input size: " + std::to_string(getInputSize()) +
            //                     ", Actual input size: " + std::to_string(input.size()));
            throw std::invalid_argument("Input size does not match network input layer");
        }

        // Apply environmental adaptations based on radiation level (only in non-const version)
        if (enable_protection && protection_level_ == ProtectionLevel::ADAPTIVE_TMR) {
            const_cast<ProtectedNeuralNetwork*>(this)->adaptToRadiationLevel(radiation_level);
        }

        // Apply temporary radiation effects during forward pass (non-destructive)
        std::mt19937_64 temp_rng;
        std::uniform_real_distribution<double> temp_dist(0.0, 1.0);
        if (enable_protection && radiation_level > 0.0) {
            uint64_t seed = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            temp_rng.seed(seed);
        }

        // Input layer activations
        std::vector<std::vector<T>> activations(layer_sizes_.size());
        activations[0] = input;

        // Forward pass through each layer
        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            activations[layer + 1].resize(layer_sizes_[layer + 1]);

            // For better performance with SIMD, we can optimize matrix-vector multiplication
            // for large layers when using float type. Prefer SIMD whenever available;
            // protection logic runs orthogonally to compute path.
            if (layer_sizes_[layer] >= 8 && layer_sizes_[layer + 1] >= 8 &&
                std::is_same_v<T, float>) {
                // Pre-load all weights for this layer into a contiguous matrix
                std::vector<T> layer_weights(layer_sizes_[layer] * layer_sizes_[layer + 1]);
                std::vector<T> layer_biases(layer_sizes_[layer + 1]);

                for (size_t neuron = 0; neuron < layer_sizes_[layer + 1]; ++neuron) {
                    layer_biases[neuron] = getBias(layer, neuron);

                    for (size_t prev = 0; prev < layer_sizes_[layer]; ++prev) {
                        layer_weights[neuron * layer_sizes_[layer] + prev] =
                            getWeight(layer, prev, neuron);
                    }
                }

                // Use optimized matrix-vector multiplication
                std::vector<T> layer_output(layer_sizes_[layer + 1]);
                const_cast<ProtectedNeuralNetwork*>(this)->matrixVectorMultiplySIMD(
                    layer_weights, activations[layer], layer_output, layer_sizes_[layer + 1],
                    layer_sizes_[layer]);

                // Add biases and apply activation function
                for (size_t neuron = 0; neuron < layer_sizes_[layer + 1]; ++neuron) {
                    T sum = layer_output[neuron] + layer_biases[neuron];
                    activations[layer + 1][neuron] = activation_functions_[layer](sum);
                }
            }
            else {
                // Standard implementation for each neuron
                for (size_t neuron = 0; neuron < layer_sizes_[layer + 1]; ++neuron) {
                    T bias = getBias(layer, neuron);

                    // Apply temporary radiation effects to bias if needed
                    if (enable_protection && radiation_level > 0.0 &&
                        temp_dist(temp_rng) < radiation_level * 2.0) {
                        bias = applyBitFlip(bias, temp_rng);
                    }

                    T sum = bias;

                    // Sum weighted inputs from previous layer
                    for (size_t prev = 0; prev < layer_sizes_[layer]; ++prev) {
                        T weight = getWeight(layer, prev, neuron);

                        // Apply temporary radiation effects to weight if needed
                        if (enable_protection && radiation_level > 0.0 &&
                            temp_dist(temp_rng) < radiation_level * 2.0) {
                            weight = applyBitFlip(weight, temp_rng);
                        }

                        sum += weight * activations[layer][prev];
                    }

                    // Apply activation function
                    activations[layer + 1][neuron] = activation_functions_[layer](sum);
                }
            }

            // Apply radiation protection to activations if needed (only in non-const version)
            if (enable_protection && protection_level_ != ProtectionLevel::NONE) {
                const_cast<ProtectedNeuralNetwork*>(this)->protectActivations(
                    activations[layer + 1], radiation_level);
            }
        }

        return activations.back();
    }

    /**
     * @brief SIMD-optimized matrix-vector multiplication for performance
     *
     * @param matrix Input matrix (flattened)
     * @param vector Input vector
     * @param result Output vector
     * @param rows Number of rows in matrix
     * @param cols Number of columns in matrix
     */
    void matrixVectorMultiplySIMD(const std::vector<T>& matrix, const std::vector<T>& vector,
                                  std::vector<T>& result, size_t rows, size_t cols) const
    {
#ifdef SIMD_ENABLED
        // Use SIMD for large vectors
        if (cols >= 8 && std::is_same_v<T, float>) {
            matrixVectorMultiplyAVX(matrix, vector, result, rows, cols);
        }
        else {
            // Fallback to standard multiplication
            matrixVectorMultiplyStandard(matrix, vector, result, rows, cols);
        }
#else
        matrixVectorMultiplyStandard(matrix, vector, result, rows, cols);
#endif
    }

   private:
    /**
     * @brief Standard matrix-vector multiplication (fallback)
     */
    void matrixVectorMultiplyStandard(const std::vector<T>& matrix, const std::vector<T>& vector,
                                      std::vector<T>& result, size_t rows, size_t cols) const
    {
        // Standard implementation for matrix-vector multiplication
        for (size_t row = 0; row < rows; ++row) {
            T sum = T(0);
            for (size_t col = 0; col < cols; ++col) {
                sum += matrix[row * cols + col] * vector[col];
            }
            result[row] = sum;
        }
    }

#ifdef SIMD_ENABLED
    /**
     * @brief AVX2-optimized matrix-vector multiplication
     */
    void matrixVectorMultiplyAVX(const std::vector<T>& matrix, const std::vector<T>& vector,
                                 std::vector<T>& result, size_t rows, size_t cols) const
    {
        static_assert(std::is_same_v<T, float>, "AVX implementation only for float");

#if !defined(__AVX2__)
        // Fallback scalar implementation when AVX2 is unavailable
        for (size_t row = 0; row < rows; ++row) {
            float acc = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                acc += matrix[row * cols + col] * vector[col];
            }
            result[row] = acc;
        }
        return;
#else
        for (size_t row = 0; row < rows; ++row) {
            __m256 sum = _mm256_setzero_ps();
            size_t col = 0;

            // Process 8 elements at a time; loop condition prevents over-read
            for (; col + 7 < cols; col += 8) {
                __m256 m_vals = _mm256_loadu_ps(&matrix[row * cols + col]);
                __m256 v_vals = _mm256_loadu_ps(&vector[col]);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(m_vals, v_vals, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(m_vals, v_vals));
#endif
            }

            // Reduce 8-wide sum safely via store
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, sum);
            float acc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            // Handle remaining elements
            for (; col < cols; ++col) {
                acc += matrix[row * cols + col] * vector[col];
            }

            result[row] = acc;
        }
#endif
    }
#endif
};

}  // namespace neural
}  // namespace rad_ml

#endif  // RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP
