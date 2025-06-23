#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../core/logger.hpp"
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
    OptimizerType optimizer;
    SamplingTechnique sampling;

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
        if (epochs != other.epochs) return epochs < other.epochs;
        if (batch_size != other.batch_size) return batch_size < other.batch_size;
        if (use_interpolation != other.use_interpolation)
            return !use_interpolation && other.use_interpolation;
        if (optimizer != other.optimizer) return optimizer < other.optimizer;
        return sampling < other.sampling;
    }
};

/**
 * @brief Results from VAE tuning experiment with Monte Carlo statistics
 */
struct VAETuningResult {
    // Core metrics
    double compression_ratio_mean = 0.0;
    double compression_ratio_stddev = 0.0;
    double reconstruction_error_mean = 0.0;
    double reconstruction_error_stddev = 0.0;
    double kl_divergence_mean = 0.0;
    double kl_divergence_stddev = 0.0;
    double total_loss_mean = 0.0;
    double total_loss_stddev = 0.0;

    // Performance metrics
    double training_time_ms_mean = 0.0;
    double training_time_ms_stddev = 0.0;
    double inference_time_ms_mean = 0.0;
    double inference_time_ms_stddev = 0.0;

    // Database-specific metrics
    double storage_efficiency_mean = 0.0;
    double storage_efficiency_stddev = 0.0;
    double retrieval_accuracy_mean = 0.0;
    double retrieval_accuracy_stddev = 0.0;

    // Anomaly detection metrics
    double anomaly_detection_score_mean = 0.0;
    double anomaly_detection_score_stddev = 0.0;
    double false_positive_rate_mean = 0.0;
    double false_positive_rate_stddev = 0.0;
    double true_positive_rate_mean = 0.0;
    double true_positive_rate_stddev = 0.0;

    // Stability metrics
    bool converged = false;
    double convergence_rate = 0.0;
    size_t monte_carlo_trials = 0;

    // Composite scores
    double compression_score = 0.0;  // For compression use case
    double anomaly_score = 0.0;      // For anomaly detection use case
    double balanced_score = 0.0;     // Balanced for both use cases
};

/**
 * @brief Search result containing the best VAE configuration and its performance
 */
struct VAESearchResult {
    VAETuningConfig config;
    VAETuningResult result;
    std::string use_case;  // "compression", "anomaly_detection", "balanced"
    size_t search_iterations;

    VAESearchResult() : search_iterations(0) {}

    VAESearchResult(const VAETuningConfig& cfg, const VAETuningResult& res,
                    const std::string& case_type, size_t iterations)
        : config(cfg), result(res), use_case(case_type), search_iterations(iterations)
    {
    }
};

/**
 * @brief VAE Auto Tuner - systematic optimization of VAE parameters with Monte Carlo testing
 *
 * This class implements comprehensive VAE hyperparameter optimization using:
 * - Grid search for exhaustive parameter space exploration
 * - Random search for efficient sampling
 * - Evolutionary search for intelligent optimization
 * - Monte Carlo testing for statistical reliability
 * - Database integration testing for real-world performance
 */
class VAEAutoTuner {
   public:
    /**
     * @brief Constructor with training and validation datasets
     *
     * @param training_data Training dataset for VAE optimization
     * @param validation_data Validation dataset for performance evaluation
     * @param test_data Optional test dataset for final evaluation
     * @param results_file File to save detailed results
     */
    VAEAutoTuner(const std::vector<std::vector<float>>& training_data,
                 const std::vector<std::vector<float>>& validation_data,
                 const std::vector<std::vector<float>>& test_data = {},
                 const std::string& results_file = "vae_tuning_results.csv");

    /**
     * @brief Grid search over parameter space with Monte Carlo testing
     *
     * @param latent_dims Latent dimensions to test
     * @param beta_values Beta parameters for β-VAE
     * @param architectures Hidden layer architectures
     * @param learning_rates Learning rates to test
     * @param monte_carlo_trials Number of Monte Carlo trials per configuration
     * @param use_case Target use case ("compression", "anomaly_detection", "balanced")
     * @return Best configuration found
     */
    VAESearchResult gridSearch(const std::vector<size_t>& latent_dims = {4, 8, 16, 32},
                               const std::vector<float>& beta_values = {0.1f, 0.5f, 1.0f, 2.0f},
                               const std::vector<std::vector<size_t>>& architectures =
                                   {{64, 32}, {128, 64, 32}, {256, 128, 64}},
                               const std::vector<float>& learning_rates = {0.001f, 0.01f},
                               size_t monte_carlo_trials = 10,
                               const std::string& use_case = "balanced");

    /**
     * @brief Random search with Monte Carlo testing
     *
     * @param max_iterations Maximum search iterations
     * @param monte_carlo_trials Number of Monte Carlo trials per configuration
     * @param use_case Target use case
     * @return Best configuration found
     */
    VAESearchResult randomSearch(size_t max_iterations = 50, size_t monte_carlo_trials = 10,
                                 const std::string& use_case = "balanced");

    /**
     * @brief Evolutionary search with Monte Carlo testing
     *
     * @param population_size Size of the population
     * @param generations Number of generations
     * @param mutation_rate Mutation rate
     * @param monte_carlo_trials Number of Monte Carlo trials per configuration
     * @param use_case Target use case
     * @return Best configuration found
     */
    VAESearchResult evolutionarySearch(size_t population_size = 20, size_t generations = 10,
                                       double mutation_rate = 0.2, size_t monte_carlo_trials = 5,
                                       const std::string& use_case = "balanced");

    /**
     * @brief Test VAE configuration with AI Native Database integration
     *
     * @param config VAE configuration to test
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Database integration results
     */
    VAETuningResult testDatabaseIntegration(const VAETuningConfig& config,
                                            size_t monte_carlo_trials = 10);

    /**
     * @brief Test anomaly detection performance
     *
     * @param config VAE configuration to test
     * @param anomaly_rate Rate of anomalies to inject (0.0-1.0)
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Anomaly detection results
     */
    VAETuningResult testAnomalyDetection(const VAETuningConfig& config, float anomaly_rate = 0.1f,
                                         size_t monte_carlo_trials = 10);

    /**
     * @brief Comprehensive evaluation of a configuration
     *
     * @param config Configuration to evaluate
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @param include_database_test Include database integration testing
     * @param include_anomaly_test Include anomaly detection testing
     * @return Comprehensive results
     */
    VAETuningResult evaluateConfiguration(const VAETuningConfig& config,
                                          size_t monte_carlo_trials = 10,
                                          bool include_database_test = true,
                                          bool include_anomaly_test = true);

    /**
     * @brief Set random seed for reproducible results
     *
     * @param seed Random seed value
     */
    void setSeed(unsigned int seed);

    /**
     * @brief Set parameter search ranges
     *
     * @param latent_dims Range of latent dimensions
     * @param beta_values Range of beta values
     * @param learning_rates Range of learning rates
     * @param epoch_options Range of training epochs
     */
    void setSearchRanges(const std::vector<size_t>& latent_dims,
                         const std::vector<float>& beta_values,
                         const std::vector<float>& learning_rates,
                         const std::vector<int>& epoch_options);

    /**
     * @brief Export detailed results to CSV file
     *
     * @param filename Output filename
     */
    void exportResults(const std::string& filename) const;

    /**
     * @brief Get all tested configurations
     *
     * @return Map of configurations and their results
     */
    const std::map<VAETuningConfig, VAETuningResult>& getTestedConfigurations() const;

    /**
     * @brief Generate report with recommendations
     *
     * @param filename Output filename for report
     */
    void generateReport(const std::string& filename = "vae_tuning_report.md") const;

   private:
    // Dataset storage
    std::vector<std::vector<float>> training_data_;
    std::vector<std::vector<float>> validation_data_;
    std::vector<std::vector<float>> test_data_;

    // Configuration storage
    std::map<VAETuningConfig, VAETuningResult> tested_configs_;
    std::string results_file_;

    // Search parameters
    std::vector<size_t> latent_dim_options_;
    std::vector<float> beta_options_;
    std::vector<float> learning_rate_options_;
    std::vector<int> epoch_options_;
    std::vector<std::vector<size_t>> architecture_options_;

    // Random number generation
    std::mt19937 random_generator_;

    // Helper methods
    VAETuningConfig generateRandomConfig();
    VAETuningConfig mutateConfig(const VAETuningConfig& config, double mutation_rate);
    VAETuningConfig crossoverConfigs(const VAETuningConfig& parent1,
                                     const VAETuningConfig& parent2);

    std::vector<VAETuningConfig> generateAllConfigs();

    VAETuningResult runMonteCarloTrial(const VAETuningConfig& config, size_t trials,
                                       bool include_database_test = true,
                                       bool include_anomaly_test = true);

    VAETuningResult runSingleTrial(const VAETuningConfig& config, uint64_t seed);

    std::vector<std::vector<float>> injectAnomalies(const std::vector<std::vector<float>>& data,
                                                    float anomaly_rate, uint64_t seed);

    double calculateCompressionScore(const VAETuningResult& result);
    double calculateAnomalyScore(const VAETuningResult& result);
    double calculateBalancedScore(const VAETuningResult& result);

    void saveResultsToFile() const;
    void updateBestConfiguration(const VAETuningConfig& config, const VAETuningResult& result,
                                 const std::string& use_case);

    // Best configurations tracking
    std::map<std::string, VAESearchResult> best_configs_;
};

}  // namespace rad_ml::research
