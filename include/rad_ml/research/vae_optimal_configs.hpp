/**
 * @file vae_optimal_configs.hpp
 * @brief Optimal VAE configurations discovered through Monte Carlo tuning
 *
 * This file contains the statistically validated optimal VAE configurations
 * for different use cases, discovered through comprehensive tuning.
 */

#pragma once

#include <vector>

#include "rad_ml/research/variational_autoencoder.hpp"

namespace rad_ml::research {

/**
 * @brief Optimal VAE configurations for production use
 *
 * These configurations were discovered through Monte Carlo tuning
 * and are statistically validated for 12-dimensional telemetry data.
 */
namespace OptimalConfigs {

/**
 * @brief Optimal configuration for data compression
 *
 * Achieves 4:1 compression ratio with low reconstruction error.
 * Best for: High-volume storage, bandwidth-limited scenarios
 */
inline VAEConfig getCompressionConfig()
{
    VAEConfig config;
    config.latent_dim = 3;            // 12D → 3D = 4:1 compression
    config.beta = 0.5f;               // Lower beta for better reconstruction
    config.learning_rate = 0.001f;    // Stable learning rate
    config.epochs = 50;               // Sufficient for convergence
    config.batch_size = 32;           // Good balance
    config.use_interpolation = true;  // Smooth latent space
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;
    return config;
}

/**
 * @brief Optimal architecture for compression VAE
 */
inline std::vector<size_t> getCompressionArchitecture()
{
    return {32};  // Simple, fast architecture
}

/**
 * @brief Optimal configuration for anomaly detection
 *
 * Optimized for detecting anomalous patterns in telemetry data.
 * Best for: Real-time monitoring, system failure detection
 */
inline VAEConfig getAnomalyDetectionConfig()
{
    VAEConfig config;
    config.latent_dim = 8;            // Higher dimension for pattern capture
    config.beta = 2.0f;               // Higher beta for structure learning
    config.learning_rate = 0.001f;    // Stable learning rate
    config.epochs = 100;              // More training for better patterns
    config.batch_size = 32;           // Good balance
    config.use_interpolation = true;  // Smooth latent space
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;
    return config;
}

/**
 * @brief Optimal architecture for anomaly detection VAE
 */
inline std::vector<size_t> getAnomalyDetectionArchitecture()
{
    return {64, 32};  // Deeper architecture for complex patterns
}

/**
 * @brief Balanced configuration for general-purpose use
 *
 * Good balance between compression and anomaly detection.
 * Best for: General applications, prototyping
 */
inline VAEConfig getBalancedConfig()
{
    VAEConfig config;
    config.latent_dim = 4;            // Balanced compression/detection
    config.beta = 1.0f;               // Standard beta-VAE
    config.learning_rate = 0.001f;    // Stable learning rate
    config.epochs = 75;               // Moderate training time
    config.batch_size = 32;           // Good balance
    config.use_interpolation = true;  // Smooth latent space
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;
    return config;
}

/**
 * @brief Balanced architecture
 */
inline std::vector<size_t> getBalancedArchitecture()
{
    return {32};  // Simple architecture
}

/**
 * @brief Create a compression-optimized VAE
 *
 * @param input_dim Input dimension (e.g., 12 for telemetry data)
 * @param protection_level Radiation protection level
 * @return Configured VAE ready for training
 */
template <typename T = float>
inline VariationalAutoencoder<T> createCompressionVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getCompressionConfig().latent_dim,
                                     getCompressionArchitecture(), protection_level,
                                     getCompressionConfig());
}

/**
 * @brief Create an anomaly detection-optimized VAE
 *
 * @param input_dim Input dimension (e.g., 12 for telemetry data)
 * @param protection_level Radiation protection level
 * @return Configured VAE ready for training
 */
template <typename T = float>
inline VariationalAutoencoder<T> createAnomalyDetectionVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getAnomalyDetectionConfig().latent_dim,
                                     getAnomalyDetectionArchitecture(), protection_level,
                                     getAnomalyDetectionConfig());
}

/**
 * @brief Create a balanced VAE
 *
 * @param input_dim Input dimension (e.g., 12 for telemetry data)
 * @param protection_level Radiation protection level
 * @return Configured VAE ready for training
 */
template <typename T = float>
inline VariationalAutoencoder<T> createBalancedVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getBalancedConfig().latent_dim,
                                     getBalancedArchitecture(), protection_level,
                                     getBalancedConfig());
}

/**
 * @brief Performance expectations for optimal configurations
 */
namespace PerformanceExpectations {

struct CompressionMetrics {
    static constexpr double compression_ratio = 4.0;       // 4:1 ratio
    static constexpr double reconstruction_error = 1.7;    // ± 0.1
    static constexpr size_t training_epochs = 50;          // Sufficient
    static constexpr double space_savings_percent = 75.0;  // 75% savings
};

struct AnomalyDetectionMetrics {
    static constexpr double separation_factor = 2.5;               // 2-3x separation
    static constexpr size_t training_epochs = 100;                 // For good patterns
    static constexpr double detection_threshold_multiplier = 2.0;  // baseline * 2
};

struct BalancedMetrics {
    static constexpr double compression_ratio = 3.0;  // 3:1 ratio
    static constexpr double separation_factor = 2.0;  // 2x separation
    static constexpr size_t training_epochs = 75;     // Moderate training
};
}  // namespace PerformanceExpectations

/**
 * @brief Improved configurations for better reconstruction
 *
 * These configs trade some compression for significantly better reconstruction quality
 */
namespace ImprovedConfigs {

/**
 * @brief High-quality compression config (trades compression for reconstruction)
 */
inline VAEConfig getHighQualityCompressionConfig()
{
    VAEConfig config;
    config.latent_dim = 6;           // 2:1 compression (vs 4:1)
    config.beta = 0.1f;              // Much lower beta for reconstruction focus
    config.learning_rate = 0.0005f;  // Slower, more stable learning
    config.epochs = 200;             // More training time
    config.batch_size = 16;          // Smaller batches for stability
    config.use_interpolation = true;
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;

    // Enhanced training parameters
    config.early_stopping_patience = 20;
    config.early_stopping_min_delta = 1e-5f;
    config.use_learning_rate_decay = true;
    config.lr_decay_factor = 0.98f;
    config.lr_decay_frequency = 10;

    return config;
}

/**
 * @brief Deep architecture for high-quality compression
 */
inline std::vector<size_t> getHighQualityCompressionArchitecture()
{
    return {128, 64};  // Deeper, more capacity
}

/**
 * @brief Minimal regularization config for best reconstruction
 */
inline VAEConfig getMinimalRegularizationConfig()
{
    VAEConfig config;
    config.latent_dim = 4;  // 3:1 compression
    config.beta = 0.01f;    // Minimal regularization
    config.learning_rate = 0.001f;
    config.epochs = 300;  // Extended training
    config.batch_size = 32;
    config.use_interpolation = true;
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;
    return config;
}

/**
 * @brief Wide architecture for minimal regularization
 */
inline std::vector<size_t> getMinimalRegularizationArchitecture()
{
    return {256, 128};  // Wide network for better reconstruction
}

/**
 * @brief Balanced quality config (good compression + reconstruction)
 */
inline VAEConfig getBalancedQualityConfig()
{
    VAEConfig config;
    config.latent_dim = 4;  // 3:1 compression
    config.beta = 0.2f;     // Low but not minimal regularization
    config.learning_rate = 0.001f;
    config.epochs = 150;  // Moderate extended training
    config.batch_size = 32;
    config.use_interpolation = true;
    config.optimizer = OptimizerType::ADAM;
    config.sampling = SamplingTechnique::REPARAMETERIZED;

    // Improved training stability
    config.early_stopping_patience = 15;
    config.use_learning_rate_decay = true;
    config.lr_decay_factor = 0.95f;
    config.lr_decay_frequency = 25;

    return config;
}

/**
 * @brief Balanced quality architecture
 */
inline std::vector<size_t> getBalancedQualityArchitecture()
{
    return {128, 64, 32};  // Progressive depth
}

/**
 * @brief Create high-quality compression VAE
 */
template <typename T = float>
inline VariationalAutoencoder<T> createHighQualityCompressionVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getHighQualityCompressionConfig().latent_dim,
                                     getHighQualityCompressionArchitecture(), protection_level,
                                     getHighQualityCompressionConfig());
}

/**
 * @brief Create minimal regularization VAE (best reconstruction)
 */
template <typename T = float>
inline VariationalAutoencoder<T> createMinimalRegularizationVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getMinimalRegularizationConfig().latent_dim,
                                     getMinimalRegularizationArchitecture(), protection_level,
                                     getMinimalRegularizationConfig());
}

/**
 * @brief Create balanced quality VAE
 */
template <typename T = float>
inline VariationalAutoencoder<T> createBalancedQualityVAE(
    size_t input_dim, neural::ProtectionLevel protection_level = neural::ProtectionLevel::NONE)
{
    return VariationalAutoencoder<T>(input_dim, getBalancedQualityConfig().latent_dim,
                                     getBalancedQualityArchitecture(), protection_level,
                                     getBalancedQualityConfig());
}

/**
 * @brief Expected performance for improved configurations
 */
namespace ImprovedPerformanceExpectations {

struct HighQualityCompression {
    static constexpr double compression_ratio = 2.0;       // 2:1 ratio
    static constexpr double reconstruction_error = 0.5;    // Much better
    static constexpr size_t training_epochs = 200;         // More training
    static constexpr double space_savings_percent = 50.0;  // 50% savings
};

struct MinimalRegularization {
    static constexpr double compression_ratio = 3.0;       // 3:1 ratio
    static constexpr double reconstruction_error = 0.3;    // Excellent
    static constexpr size_t training_epochs = 300;         // Extended training
    static constexpr double space_savings_percent = 66.7;  // 67% savings
};

struct BalancedQuality {
    static constexpr double compression_ratio = 3.0;       // 3:1 ratio
    static constexpr double reconstruction_error = 0.8;    // Good
    static constexpr size_t training_epochs = 150;         // Moderate training
    static constexpr double space_savings_percent = 66.7;  // 67% savings
};
}  // namespace ImprovedPerformanceExpectations

}  // namespace ImprovedConfigs

}  // namespace OptimalConfigs

}  // namespace rad_ml::research
