/**
 * @file variational_autoencoder.hpp
 * @brief Radiation-tolerant Variational Autoencoder implementation
 *
 * This file implements a VAE with radiation protection that can operate reliably
 * in space environments while maintaining generative capabilities.
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
 * @brief Configuration for VAE training
 */
struct VAEConfig {
    size_t latent_dim = 10;        ///< Latent space dimensionality
    float beta = 1.0f;             ///< β parameter for β-VAE
    float learning_rate = 0.001f;  ///< Learning rate
    int batch_size = 32;           ///< Batch size for training
    int epochs = 100;              ///< Number of training epochs
    SamplingTechnique sampling = SamplingTechnique::REPARAMETERIZED;
    VAELossType loss_type = VAELossType::STANDARD_ELBO;
    bool use_interpolation = true;      ///< Enable interpolation capabilities
    float interpolation_weight = 0.1f;  ///< Weight for interpolation loss
};

/**
 * @brief Radiation-tolerant Variational Autoencoder
 *
 * This class implements a VAE with radiation protection capabilities,
 * including encoder, decoder, and interpolation components as shown in the architecture.
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
     * @brief Train the VAE using provided data
     *
     * @param data Training data
     * @param epochs Number of training epochs
     * @param batch_size Batch size
     * @param learning_rate Learning rate
     * @return Final training loss
     */
    float train(const std::vector<std::vector<T>>& data, int epochs = -1, int batch_size = -1,
                float learning_rate = -1.0f);

    /**
     * @brief Evaluate the VAE on test data
     *
     * @param data Test data
     * @return Reconstruction error
     */
    float evaluate(const std::vector<std::vector<T>>& data);

    /**
     * @brief Calculate VAE loss (ELBO)
     *
     * @param input Original input
     * @param reconstruction Reconstructed output
     * @param mean Latent mean
     * @param log_var Latent log variance
     * @return Total loss
     */
    float calculateLoss(const std::vector<T>& input, const std::vector<T>& reconstruction,
                        const std::vector<T>& mean, const std::vector<T>& log_var);

    /**
     * @brief Save VAE model to file
     *
     * @param filename Output filename
     * @return Success status
     */
    bool saveToFile(const std::string& filename) const;

    /**
     * @brief Load VAE model from file
     *
     * @param filename Input filename
     * @return Success status
     */
    bool loadFromFile(const std::string& filename);

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
    void setConfig(const VAEConfig& config) { config_ = config; }

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

   private:
    // Network components
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> encoder_;
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> decoder_;
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> interpolator_;

    // Network dimensions
    size_t input_dim_;
    size_t latent_dim_;
    std::vector<size_t> hidden_dims_;

    // Protection and configuration
    neural::ProtectionLevel protection_level_;
    VAEConfig config_;

    // Error tracking
    mutable struct {
        uint64_t detected_errors = 0;
        uint64_t corrected_errors = 0;
        uint64_t uncorrectable_errors = 0;
    } error_stats_;

    // Random number generator
    mutable std::mt19937 rng_;

    /**
     * @brief Initialize encoder network
     */
    void initializeEncoder();

    /**
     * @brief Initialize decoder network
     */
    void initializeDecoder();

    /**
     * @brief Initialize interpolator network
     */
    void initializeInterpolator();

    /**
     * @brief Apply logarithmic transformation (as shown in architecture)
     *
     * @param input Input vector
     * @return Transformed vector
     */
    std::vector<T> applyLogarithmics(const std::vector<T>& input) const;

    /**
     * @brief Apply inverse logarithmic transformation
     *
     * @param input Input vector
     * @return Transformed vector
     */
    std::vector<T> applyInverseLogarithmics(const std::vector<T>& input) const;

    /**
     * @brief Apply standardization (as shown in architecture)
     *
     * @param input Input vector
     * @return Standardized vector
     */
    std::vector<T> applyStandardization(const std::vector<T>& input) const;

    /**
     * @brief Apply inverse standardization
     *
     * @param input Input vector
     * @return De-standardized vector
     */
    std::vector<T> applyInverseStandardization(const std::vector<T>& input) const;

    /**
     * @brief Compute KL divergence for VAE loss
     *
     * @param mean Mean vector
     * @param log_var Log variance vector
     * @return KL divergence value
     */
    T computeKLDivergence(const std::vector<T>& mean, const std::vector<T>& log_var) const;

    /**
     * @brief Compute reconstruction loss
     *
     * @param input Original input
     * @param reconstruction Reconstructed output
     * @return Reconstruction loss
     */
    T computeReconstructionLoss(const std::vector<T>& input,
                                const std::vector<T>& reconstruction) const;

    /**
     * @brief Compute interpolation loss (L_INF from architecture)
     *
     * @param latent1 First latent vector
     * @param latent2 Second latent vector
     * @param interpolated Interpolated result
     * @return Interpolation loss
     */
    T computeInterpolationLoss(const std::vector<T>& latent1, const std::vector<T>& latent2,
                               const std::vector<T>& interpolated) const;

    /**
     * @brief Update network weights (simplified training step)
     *
     * @param loss Current loss value
     * @param learning_rate Learning rate
     */
    void updateWeights(T loss, T learning_rate);

    /**
     * @brief Protect latent variables with radiation tolerance
     *
     * @param latent Latent vector to protect
     * @param radiation_level Current radiation level
     */
    void protectLatentVariables(std::vector<T>& latent, double radiation_level) const;
};

}  // namespace research
}  // namespace rad_ml

#endif  // RAD_ML_RESEARCH_VARIATIONAL_AUTOENCODER_HPP
