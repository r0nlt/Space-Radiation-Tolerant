/**
 * @file variational_autoencoder.hpp
 * @brief Production-ready Radiation-tolerant Variational Autoencoder implementation
 *
 * This file implements a VAE with radiation protection that can operate reliably
 * in space environments while maintaining generative capabilities.
 *
 * PRODUCTION VERSION: Includes proper gradient computation, optimizers,
 * checkpointing, validation, and performance optimizations.
 */

#ifndef RAD_ML_RESEARCH_VARIATIONAL_AUTOENCODER_HPP
#define RAD_ML_RESEARCH_VARIATIONAL_AUTOENCODER_HPP

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/logger.hpp"
#include "../neural/activation.hpp"
#include "../neural/multibit_protection.hpp"
#include "../neural/protected_neural_network.hpp"

namespace rad_ml {
namespace research {

/**
 * @brief Sampling techniques for VAE latent space
 */
enum class SamplingTechnique {
    STANDARD,         ///< Standard Gaussian sampling
    REPARAMETERIZED,  ///< Reparameterization trick
    IMPORTANCE,       ///< Importance sampling
    ADVERSARIAL       ///< Adversarial sampling
};

/**
 * @brief VAE loss computation methods
 */
enum class VAELossType {
    STANDARD_ELBO,  ///< Standard Evidence Lower BOund
    BETA_VAE,       ///< β-VAE with weighted KL divergence
    FACTOR_VAE,     ///< Factor-VAE for disentanglement
    CONTROLLED_VAE  ///< Controlled VAE with explicit control
};

/**
 * @brief Optimizer types for training
 */
enum class OptimizerType {
    SGD,      ///< Stochastic Gradient Descent
    ADAM,     ///< Adam optimizer
    RMSPROP,  ///< RMSprop optimizer
    ADAGRAD   ///< AdaGrad optimizer
};

/**
 * @brief Training metrics and statistics
 */
struct TrainingMetrics {
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> kl_losses;
    std::vector<float> reconstruction_losses;
    float best_val_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;
    int epochs_without_improvement = 0;
};

/**
 * @brief Configuration for VAE training and architecture
 */
struct VAEConfig {
    // Architecture parameters
    size_t latent_dim = 10;             ///< Latent space dimensionality
    float beta = 1.0f;                  ///< β parameter for β-VAE
    bool use_interpolation = true;      ///< Enable interpolation capabilities
    float interpolation_weight = 0.1f;  ///< Weight for interpolation loss

    // Training parameters
    float learning_rate = 0.001f;  ///< Learning rate
    int batch_size = 32;           ///< Batch size for training
    int epochs = 100;              ///< Number of training epochs
    OptimizerType optimizer = OptimizerType::ADAM;

    // Adam optimizer parameters
    float adam_beta1 = 0.9f;     ///< Adam β₁ parameter
    float adam_beta2 = 0.999f;   ///< Adam β₂ parameter
    float adam_epsilon = 1e-8f;  ///< Adam ε parameter

    // Regularization
    float weight_decay = 0.0f;  ///< L2 weight decay
    float dropout_rate = 0.0f;  ///< Dropout rate (if supported)

    // Validation and early stopping
    float validation_split = 0.2f;           ///< Fraction of data for validation
    int early_stopping_patience = 10;        ///< Epochs to wait before early stopping
    float early_stopping_min_delta = 1e-4f;  ///< Minimum improvement for early stopping

    // Checkpointing
    bool enable_checkpointing = true;               ///< Save model checkpoints
    int checkpoint_frequency = 10;                  ///< Save checkpoint every N epochs
    std::string checkpoint_dir = "./checkpoints/";  ///< Directory for checkpoints

    // Advanced options
    SamplingTechnique sampling = SamplingTechnique::REPARAMETERIZED;
    VAELossType loss_type = VAELossType::STANDARD_ELBO;
    bool use_learning_rate_decay = true;  ///< Enable learning rate decay
    float lr_decay_factor = 0.95f;        ///< Decay factor per epoch
    int lr_decay_frequency = 20;          ///< Apply decay every N epochs
};

/**
 * @brief Production-ready Radiation-tolerant Variational Autoencoder
 *
 * This class implements a VAE with radiation protection capabilities,
 * including proper gradient computation, optimizers, validation, and checkpointing.
 *
 * @tparam T Value type (typically float)
 */
template <typename T = float>
class VariationalAutoencoder {
   public:
    /**
     * @brief Constructor
     *
     * @param input_dim Input dimension
     * @param latent_dim Latent space dimension
     * @param hidden_dims Hidden layer dimensions for encoder/decoder
     * @param protection_level Protection level to apply
     * @param config VAE configuration
     */
    VariationalAutoencoder(
        size_t input_dim, size_t latent_dim, const std::vector<size_t>& hidden_dims,
        neural::ProtectionLevel protection_level = neural::ProtectionLevel::ADAPTIVE_TMR,
        const VAEConfig& config = VAEConfig{});

    /**
     * @brief Copy constructor
     */
    VariationalAutoencoder(const VariationalAutoencoder& other);

    /**
     * @brief Copy assignment operator
     */
    VariationalAutoencoder& operator=(const VariationalAutoencoder& other);

    /**
     * @brief Destructor
     */
    ~VariationalAutoencoder() = default;

    /**
     * @brief Encode input to latent space parameters
     *
     * @param input Input data
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Pair of (mean, log_variance) vectors
     */
    std::pair<std::vector<T>, std::vector<T>> encode(const std::vector<T>& input,
                                                     double radiation_level = 0.0);

    /**
     * @brief Sample from latent space using reparameterization trick
     *
     * @param mean Mean vector
     * @param log_var Log variance vector
     * @param seed Random seed for reproducibility
     * @return Sampled latent vector
     */
    std::vector<T> sample(const std::vector<T>& mean, const std::vector<T>& log_var,
                          uint64_t seed = 0);

    /**
     * @brief Decode latent representation to output
     *
     * @param latent Latent vector
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Decoded output
     */
    std::vector<T> decode(const std::vector<T>& latent, double radiation_level = 0.0);

    /**
     * @brief Full forward pass (encode + sample + decode)
     *
     * @param input Input data
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Reconstructed output
     */
    std::vector<T> forward(const std::vector<T>& input, double radiation_level = 0.0);

    /**
     * @brief Interpolate between two points in latent space
     *
     * @param latent1 First latent vector
     * @param latent2 Second latent vector
     * @param alpha Interpolation factor (0.0 to 1.0)
     * @param radiation_level Current radiation level (0.0-1.0)
     * @return Interpolated output
     */
    std::vector<T> interpolate(const std::vector<T>& latent1, const std::vector<T>& latent2,
                               T alpha, double radiation_level = 0.0);

    /**
     * @brief Generate new samples from prior distribution
     *
     * @param num_samples Number of samples to generate
     * @param radiation_level Current radiation level (0.0-1.0)
     * @param seed Random seed
     * @return Generated samples
     */
    std::vector<std::vector<T>> generate(size_t num_samples, double radiation_level = 0.0,
                                         uint64_t seed = 0);

    /**
     * @brief Production training with validation, early stopping, and checkpointing
     *
     * @param train_data Training data
     * @param val_data Validation data (optional, will split from train_data if not provided)
     * @return Training metrics
     */
    TrainingMetrics trainProduction(const std::vector<std::vector<T>>& train_data,
                                    const std::vector<std::vector<T>>& val_data = {});

    /**
     * @brief Batch training for better performance
     *
     * @param batches Pre-batched training data
     * @param val_batches Pre-batched validation data
     * @return Training metrics
     */
    TrainingMetrics trainBatched(const std::vector<std::vector<std::vector<T>>>& batches,
                                 const std::vector<std::vector<std::vector<T>>>& val_batches = {});

    /**
     * @brief Comprehensive evaluation with multiple metrics
     *
     * @param data Test data
     * @return Map of metric names to values
     */
    std::unordered_map<std::string, float> evaluateComprehensive(
        const std::vector<std::vector<T>>& data);

    /**
     * @brief Save complete model state including optimizer state
     *
     * @param filepath Path to save the model
     * @return Success status
     */
    bool saveModel(const std::string& filepath) const;

    /**
     * @brief Load complete model state including optimizer state
     *
     * @param filepath Path to load the model from
     * @return Success status
     */
    bool loadModel(const std::string& filepath);

    /**
     * @brief Save checkpoint during training
     *
     * @param epoch Current epoch
     * @param metrics Current training metrics
     * @return Success status
     */
    bool saveCheckpoint(int epoch, const TrainingMetrics& metrics) const;

    /**
     * @brief Load from checkpoint
     *
     * @param checkpoint_path Path to checkpoint file
     * @return Success status and loaded epoch number
     */
    std::pair<bool, int> loadCheckpoint(const std::string& checkpoint_path);

    /**
     * @brief Get encoder network
     *
     * @return Reference to encoder
     */
    const neural::ProtectedNeuralNetwork<T>& getEncoder() const { return *encoder_; }

    /**
     * @brief Get decoder network
     *
     * @return Reference to decoder
     */
    const neural::ProtectedNeuralNetwork<T>& getDecoder() const { return *decoder_; }

    /**
     * @brief Get VAE configuration
     *
     * @return Current configuration
     */
    const VAEConfig& getConfig() const { return config_; }

    /**
     * @brief Update VAE configuration
     *
     * @param config New configuration
     */
    void setConfig(const VAEConfig& config)
    {
        config_ = config;
        initializeOptimizer();
    }

    /**
     * @brief Get latent dimension
     *
     * @return Latent space dimensionality
     */
    size_t getLatentDim() const { return latent_dim_; }

    /**
     * @brief Get input dimension
     *
     * @return Input dimensionality
     */
    size_t getInputDim() const { return input_dim_; }

    /**
     * @brief Training metrics
     *
     * @return Current training metrics
     */
    TrainingMetrics getTrainingMetrics() const { return training_metrics_; }

    /**
     * @brief Apply radiation effects to the VAE
     *
     * @param radiation_level Radiation intensity (0.0-1.0)
     * @param seed Random seed for reproducibility
     */
    void applyRadiationEffects(double radiation_level, uint64_t seed);

    /**
     * @brief Get error statistics
     *
     * @return Pair of (detected_errors, corrected_errors)
     */
    std::pair<uint64_t, uint64_t> getErrorStats() const;

    /**
     * @brief Reset error statistics
     */
    void resetErrorStats();

    // Legacy training method (kept for compatibility)
    float train(const std::vector<std::vector<T>>& data, int epochs = -1, int batch_size = -1,
                float learning_rate = -1.0f);
    float evaluate(const std::vector<std::vector<T>>& data);
    float calculateLoss(const std::vector<T>& input, const std::vector<T>& reconstruction,
                        const std::vector<T>& mean, const std::vector<T>& log_var);

    // File I/O (legacy, use saveModel/loadModel for production)
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);

   private:
    // Network components
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> encoder_;
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> decoder_;
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> interpolator_;

    // Architecture parameters
    size_t input_dim_;
    size_t latent_dim_;
    std::vector<size_t> hidden_dims_;

    // Configuration and state
    neural::ProtectionLevel protection_level_;
    VAEConfig config_;
    TrainingMetrics training_metrics_;

    // Error tracking
    mutable struct {
        uint64_t detected_errors = 0;
        uint64_t corrected_errors = 0;
        uint64_t uncorrectable_errors = 0;
    } error_stats_;

    // Random number generation
    mutable std::mt19937 rng_;

    // Optimizer state
    struct OptimizerState {
        // Adam optimizer state
        std::vector<std::vector<T>> m_encoder, v_encoder;  // Moment estimates for encoder
        std::vector<std::vector<T>> m_decoder, v_decoder;  // Moment estimates for decoder
        std::vector<std::vector<T>> m_interpolator,
            v_interpolator;  // Moment estimates for interpolator
        int step_count = 0;
        float current_lr;
    } optimizer_state_;

    // Network initialization
    void initializeEncoder();
    void initializeDecoder();
    void initializeInterpolator();
    void initializeOptimizer();

    // Data preprocessing
    std::vector<T> applyLogarithmics(const std::vector<T>& input) const;
    std::vector<T> applyInverseLogarithmics(const std::vector<T>& input) const;
    std::vector<T> applyStandardization(const std::vector<T>& input) const;
    std::vector<T> applyInverseStandardization(const std::vector<T>& input) const;

    // Loss computation
    T computeKLDivergence(const std::vector<T>& mean, const std::vector<T>& log_var) const;
    T computeReconstructionLoss(const std::vector<T>& input,
                                const std::vector<T>& reconstruction) const;
    T computeInterpolationLoss(const std::vector<T>& latent1, const std::vector<T>& latent2,
                               const std::vector<T>& interpolated) const;

    // Production training methods
    void updateWeightsProduction(const std::vector<T>& encoder_grad,
                                 const std::vector<T>& decoder_grad,
                                 const std::vector<T>& interpolator_grad = {});

    std::vector<T> computeEncoderGradients(const std::vector<T>& input,
                                           const std::vector<T>& reconstruction,
                                           const std::vector<T>& mean,
                                           const std::vector<T>& log_var);

    std::vector<T> computeDecoderGradients(const std::vector<T>& input,
                                           const std::vector<T>& reconstruction,
                                           const std::vector<T>& latent);

    // Data splitting and batching
    std::pair<std::vector<std::vector<T>>, std::vector<std::vector<T>>> splitTrainValidation(
        const std::vector<std::vector<T>>& data);

    std::vector<std::vector<std::vector<T>>> createBatches(const std::vector<std::vector<T>>& data,
                                                           int batch_size);

    // Radiation protection
    void protectLatentVariables(std::vector<T>& latent, double radiation_level) const;

    // Legacy method (kept for compatibility)
    void updateWeights(T loss, T learning_rate);
};

}  // namespace research
}  // namespace rad_ml

#endif  // RAD_ML_RESEARCH_VARIATIONAL_AUTOENCODER_HPP
