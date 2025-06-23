#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../storage/ai_native_database.hpp"
#include "variational_autoencoder.hpp"

namespace rad_ml::research {

/**
 * @brief VAE configuration for tuning experiments
 */
struct VAETuningConfig {
    size_t input_dim;
    size_t latent_dim;
    std::vector<size_t> hidden_dims;
    float beta;
    float learning_rate;
    int epochs;
    int batch_size;
    bool use_interpolation;

    // For database integration
    std::string data_type = "default";

    bool operator<(const VAETuningConfig& other) const
    {
        if (input_dim != other.input_dim) return input_dim < other.input_dim;
        if (latent_dim != other.latent_dim) return latent_dim < other.latent_dim;
        if (hidden_dims != other.hidden_dims) return hidden_dims < other.hidden_dims;
        if (std::abs(beta - other.beta) >= 1e-6) return beta < other.beta;
        if (std::abs(learning_rate - other.learning_rate) >= 1e-6)
            return learning_rate < other.learning_rate;
        return epochs < other.epochs;
    }
};

/**
 * @brief Results from VAE tuning experiment
 */
struct VAETuningResult {
    double compression_ratio;
    double reconstruction_error;
    double kl_divergence;
    double total_loss;
    double training_time_ms;
    double inference_time_ms;
    bool converged;

    // Database-specific metrics
    double storage_efficiency;
    double retrieval_accuracy;
};

/**
 * @brief VAE Tuner - systematic optimization of VAE parameters
 */
class VAETuner {
   public:
    VAETuner(const std::vector<std::vector<float>>& training_data,
             const std::vector<std::vector<float>>& validation_data,
             const std::string& results_file = "vae_tuning_results.csv");

    // Grid search over parameter space
    VAETuningResult gridSearch(const std::vector<size_t>& latent_dims = {4, 8, 16, 32},
                               const std::vector<float>& beta_values = {0.1f, 0.5f, 1.0f, 2.0f},
                               const std::vector<std::vector<size_t>>& architectures = {
                                   {64, 32}, {128, 64, 32}, {256, 128, 64}});

    // Test VAE integration with AI Native Database
    VAETuningResult testDatabaseIntegration(const VAETuningConfig& config);

    // Export results for analysis
    void exportResults(const std::string& filename) const;

   private:
    std::vector<std::vector<float>> training_data_;
    std::vector<std::vector<float>> validation_data_;
    std::map<VAETuningConfig, VAETuningResult> tested_configs_;
    std::string results_file_;

    VAETuningResult evaluateConfig(const VAETuningConfig& config);
};

}  // namespace rad_ml::research
