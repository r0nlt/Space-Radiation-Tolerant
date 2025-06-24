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
     * @brief Layer structure containing weights and biases
     */
    struct Layer {
        std::vector<std::vector<T>> weights;
        std::vector<T> biases;
    };

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
     * @brief Set a custom activation function for a layer
     *
     * @param layer Layer index (0 for first hidden layer)
     * @param function Activation function
     */
    void setActivationFunction(size_t layer, const std::function<T(T)>& function)
    {
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
     * @brief Apply radiation effects to the network
     *
     * @param radiation_level Radiation level (0.0-1.0)
     * @param seed Random seed for reproducibility
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

    /**
     * @brief Train the network using backpropagation with modern C++ features
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
     * @brief Compute derivative of activation function (for testing)
     *
     * This method uses numerical differentiation to compute the derivative
     * of the actual activation function set for each layer, ensuring
     * correct gradients regardless of the activation function used.
     */
    T computeActivationDerivative(T z, size_t layer) const
    {
        if (layer >= activation_functions_.size()) {
            // Fallback to linear derivative for output layer or invalid layer
            return T{1};
        }

        // First, try to detect common activation functions analytically for performance
        const auto& activation_func = activation_functions_[layer];

        // Test if this is ReLU: f(x) = max(0, x)
        if (std::abs(activation_func(T{1}) - T{1}) < T{1e-6} &&
            std::abs(activation_func(T{-1}) - T{0}) < T{1e-6}) {
            return z > 0 ? T{1} : T{0};
        }

        // Test if this is Leaky ReLU: f(x) = x if x > 0, else α*x
        T pos_test = activation_func(T{1});
        T neg_test = activation_func(T{-1});
        T zero_test = activation_func(T{0});
        // For Leaky ReLU: f(-1) should be exactly -α, and f(0) should be 0
        if (std::abs(pos_test - T{1}) < T{1e-6} && std::abs(zero_test - T{0}) < T{1e-6} &&
            neg_test < T{0} && neg_test > T{-0.5}) {  // Restrict range to typical Leaky ReLU values
            // This looks like Leaky ReLU with slope α = neg_test / -1
            T alpha = -neg_test;
            return z > 0 ? T{1} : alpha;
        }

        // Test if this is sigmoid: f(x) = 1/(1+exp(-x))
        T sigmoid_test = activation_func(T{0});
        if (std::abs(sigmoid_test - T{0.5}) < T{1e-5}) {
            T sigmoid_z = activation_func(z);
            return sigmoid_z * (T{1} - sigmoid_z);
        }

        // Test if this is tanh: f(x) = tanh(x)
        if (std::abs(activation_func(T{0}) - T{0}) < T{1e-6} &&
            std::abs(activation_func(T{1}) - std::tanh(T{1})) < T{1e-5}) {
            T tanh_z = activation_func(z);
            return T{1} - tanh_z * tanh_z;
        }

        // Test if this is linear: f(x) = x
        if (std::abs(activation_func(T{1}) - T{1}) < T{1e-6} &&
            std::abs(activation_func(T{-1}) - T{-1}) < T{1e-6}) {
            return T{1};
        }

        // Test if this is ELU: f(x) = x if x > 0, else α*(exp(x) - 1)
        T pos_test_elu = activation_func(T{1});
        T zero_test_elu = activation_func(T{0});
        T neg_test_elu = activation_func(T{-1});
        T expected_neg_elu = std::exp(T{-1}) - T{1};  // ≈ -0.632

        if (std::abs(pos_test_elu - T{1}) < T{1e-6} && std::abs(zero_test_elu - T{0}) < T{1e-6} &&
            std::abs(neg_test_elu - expected_neg_elu) < T{1e-6}) {
            // This looks like ELU with α = 1, derivative: f'(x) = 1 if x > 0, else α*exp(x)
            return z > T{0} ? T{1} : std::exp(z);  // For α=1.0
        }

        // For custom/unknown activation functions, use numerical differentiation
        // Use adaptive epsilon based on the magnitude of z for better precision
        const T base_epsilon = static_cast<T>(1e-4);  // Increased for better stability
        const T adaptive_epsilon = std::max(base_epsilon, std::abs(z) * static_cast<T>(1e-5));
        const T epsilon = std::min(adaptive_epsilon, static_cast<T>(1e-3));  // Cap maximum epsilon

        const T f_plus = activation_func(z + epsilon);
        const T f_minus = activation_func(z - epsilon);

        // Central difference approximation: f'(z) ≈ (f(z+ε) - f(z-ε)) / (2ε)
        T derivative = (f_plus - f_minus) / (2 * epsilon);

        // Clamp extreme values to prevent numerical instability
        derivative = std::max(static_cast<T>(-10), std::min(static_cast<T>(10), derivative));

        return derivative;
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

            // Compute loss
            T sample_loss = computeLoss(activations.back(), batch_targets[sample]);
            total_loss += sample_loss;

            // Compute accuracy
            T sample_accuracy = computeAccuracy(activations.back(), batch_targets[sample]);
            total_accuracy += sample_accuracy;

            // Backward pass
            backpropagation(activations, z_values, batch_targets[sample], weight_gradients,
                            bias_gradients);
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
     * @brief Compute loss using mean squared error
     */
    T computeLoss(const std::vector<T>& predictions, const std::vector<T>& targets) const
    {
        T loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            T diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        return loss / (2.0 * predictions.size());  // MSE with 1/2 factor for cleaner derivatives
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
     * @brief Backpropagation algorithm implementation
     */
    void backpropagation(const std::vector<std::vector<T>>& activations,
                         const std::vector<std::vector<T>>& z_values, const std::vector<T>& targets,
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

            // Apply radiation protection to activations if needed (only in non-const version)
            if (enable_protection && protection_level_ != ProtectionLevel::NONE) {
                const_cast<ProtectedNeuralNetwork*>(this)->protectActivations(
                    activations[layer + 1], radiation_level);
            }
        }

        return activations.back();
    }
};

}  // namespace neural
}  // namespace rad_ml

#endif  // RAD_ML_NEURAL_PROTECTED_NEURAL_NETWORK_HPP
