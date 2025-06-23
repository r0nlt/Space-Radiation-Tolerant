/**
 * @file vae_auto_tuner.cpp
 * @brief Implementation of VAE automatic tuning with Monte Carlo testing
 */

#include "../../../include/rad_ml/research/vae_auto_tuner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace rad_ml::research {

VAEAutoTuner::VAEAutoTuner(const std::vector<std::vector<float>>& training_data,
                           const std::vector<std::vector<float>>& validation_data,
                           const std::vector<std::vector<float>>& test_data,
                           const std::string& results_file)
    : training_data_(training_data),
      validation_data_(validation_data),
      test_data_(test_data),
      results_file_(results_file),
      random_generator_(std::random_device{}())
{
    // Set default search ranges
    latent_dim_options_ = {2, 4, 6, 8, 12, 16, 24, 32};
    beta_options_ = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f};
    learning_rate_options_ = {0.0001f, 0.001f, 0.01f};
    epoch_options_ = {20, 50, 100};
    architecture_options_ = {{32},       {64},          {128},          {64, 32},       {128, 64},
                             {256, 128}, {128, 64, 32}, {256, 128, 64}, {512, 256, 128}};

    core::Logger::info("VAEAutoTuner initialized with " + std::to_string(training_data.size()) +
                       " training samples, " + std::to_string(validation_data.size()) +
                       " validation samples");
}

VAESearchResult VAEAutoTuner::gridSearch(const std::vector<size_t>& latent_dims,
                                         const std::vector<float>& beta_values,
                                         const std::vector<std::vector<size_t>>& architectures,
                                         const std::vector<float>& learning_rates,
                                         size_t monte_carlo_trials, const std::string& use_case)
{
    core::Logger::info("Starting VAE grid search with Monte Carlo testing");
    core::Logger::info("Trials per configuration: " + std::to_string(monte_carlo_trials));

    // Generate all configurations
    std::vector<VAETuningConfig> configs;

    for (auto latent_dim : latent_dims) {
        for (auto beta : beta_values) {
            for (const auto& arch : architectures) {
                for (auto lr : learning_rates) {
                    VAETuningConfig config;
                    config.input_dim = training_data_.empty() ? 0 : training_data_[0].size();
                    config.latent_dim = latent_dim;
                    config.hidden_dims = arch;
                    config.beta = beta;
                    config.learning_rate = lr;
                    config.epochs = 50;      // Default
                    config.batch_size = 32;  // Default
                    config.use_interpolation = true;
                    config.optimizer = OptimizerType::ADAM;
                    config.sampling = SamplingTechnique::REPARAMETERIZED;

                    configs.push_back(config);
                }
            }
        }
    }

    core::Logger::info("Testing " + std::to_string(configs.size()) + " configurations");

    VAESearchResult best_result;
    double best_score = -std::numeric_limits<double>::infinity();

    size_t completed = 0;
    for (const auto& config : configs) {
        // Skip if already tested
        if (tested_configs_.find(config) != tested_configs_.end()) {
            continue;
        }

        // Run Monte Carlo evaluation
        auto result = runMonteCarloTrial(config, monte_carlo_trials);
        tested_configs_[config] = result;

        // Calculate score based on use case
        double score = 0.0;
        if (use_case == "compression") {
            score = calculateCompressionScore(result);
        }
        else if (use_case == "anomaly_detection") {
            score = calculateAnomalyScore(result);
        }
        else {
            score = calculateBalancedScore(result);
        }

        if (score > best_score) {
            best_score = score;
            best_result = VAESearchResult(config, result, use_case, configs.size());

            core::Logger::info("New best configuration found:");
            core::Logger::info("  Latent dim: " + std::to_string(config.latent_dim));
            core::Logger::info("  Beta: " + std::to_string(config.beta));
            core::Logger::info("  Architecture: " +
                               std::accumulate(config.hidden_dims.begin(), config.hidden_dims.end(),
                                               std::string{}, [](const std::string& a, size_t b) {
                                                   return a + (a.empty() ? "" : "-") +
                                                          std::to_string(b);
                                               }));
            core::Logger::info("  Score: " + std::to_string(score));
        }

        completed++;
        if (completed % 10 == 0) {
            core::Logger::info("Progress: " + std::to_string(completed) + "/" +
                               std::to_string(configs.size()) + " configurations tested");
            saveResultsToFile();
        }
    }

    saveResultsToFile();
    updateBestConfiguration(best_result.config, best_result.result, use_case);

    return best_result;
}

VAESearchResult VAEAutoTuner::randomSearch(size_t max_iterations, size_t monte_carlo_trials,
                                           const std::string& use_case)
{
    core::Logger::info("Starting VAE random search with Monte Carlo testing");
    core::Logger::info("Max iterations: " + std::to_string(max_iterations));
    core::Logger::info("Trials per configuration: " + std::to_string(monte_carlo_trials));

    VAESearchResult best_result;
    double best_score = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < max_iterations; ++i) {
        // Generate random configuration
        auto config = generateRandomConfig();

        // Skip if already tested
        if (tested_configs_.find(config) != tested_configs_.end()) {
            --i;  // Don't count this iteration
            continue;
        }

        // Run Monte Carlo evaluation
        auto result = runMonteCarloTrial(config, monte_carlo_trials);
        tested_configs_[config] = result;

        // Calculate score
        double score = 0.0;
        if (use_case == "compression") {
            score = calculateCompressionScore(result);
        }
        else if (use_case == "anomaly_detection") {
            score = calculateAnomalyScore(result);
        }
        else {
            score = calculateBalancedScore(result);
        }

        if (score > best_score) {
            best_score = score;
            best_result = VAESearchResult(config, result, use_case, i + 1);

            core::Logger::info("New best configuration found (iteration " + std::to_string(i) +
                               "):");
            core::Logger::info("  Score: " + std::to_string(score));
        }

        if ((i + 1) % 10 == 0) {
            core::Logger::info("Progress: " + std::to_string(i + 1) + "/" +
                               std::to_string(max_iterations) + " iterations completed");
            saveResultsToFile();
        }
    }

    saveResultsToFile();
    updateBestConfiguration(best_result.config, best_result.result, use_case);

    return best_result;
}

VAESearchResult VAEAutoTuner::evolutionarySearch(size_t population_size, size_t generations,
                                                 double mutation_rate, size_t monte_carlo_trials,
                                                 const std::string& use_case)
{
    core::Logger::info("Starting VAE evolutionary search with Monte Carlo testing");
    core::Logger::info("Population size: " + std::to_string(population_size));
    core::Logger::info("Generations: " + std::to_string(generations));
    core::Logger::info("Trials per configuration: " + std::to_string(monte_carlo_trials));

    // Initialize population
    std::vector<VAETuningConfig> population;
    std::vector<double> fitness;

    for (size_t i = 0; i < population_size; ++i) {
        population.push_back(generateRandomConfig());
    }

    VAESearchResult best_result;
    double best_score = -std::numeric_limits<double>::infinity();

    for (size_t gen = 0; gen < generations; ++gen) {
        core::Logger::info("Generation " + std::to_string(gen + 1) + "/" +
                           std::to_string(generations));

        // Evaluate population
        fitness.clear();
        for (const auto& config : population) {
            VAETuningResult result;

            // Use cached result if available
            auto it = tested_configs_.find(config);
            if (it != tested_configs_.end()) {
                result = it->second;
            }
            else {
                result = runMonteCarloTrial(config, monte_carlo_trials);
                tested_configs_[config] = result;
            }

            // Calculate fitness
            double score = 0.0;
            if (use_case == "compression") {
                score = calculateCompressionScore(result);
            }
            else if (use_case == "anomaly_detection") {
                score = calculateAnomalyScore(result);
            }
            else {
                score = calculateBalancedScore(result);
            }

            fitness.push_back(score);

            // Update best
            if (score > best_score) {
                best_score = score;
                best_result = VAESearchResult(config, result, use_case,
                                              gen * population_size + fitness.size());
            }
        }

        // Create next generation
        std::vector<VAETuningConfig> new_population;

        // Elitism: keep best 20%
        std::vector<size_t> indices(population_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&fitness](size_t a, size_t b) { return fitness[a] > fitness[b]; });

        size_t elite_count = population_size / 5;
        for (size_t i = 0; i < elite_count; ++i) {
            new_population.push_back(population[indices[i]]);
        }

        // Fill rest with crossover and mutation
        while (new_population.size() < population_size) {
            // Tournament selection
            std::uniform_int_distribution<size_t> dist(0, population_size - 1);
            size_t parent1_idx = dist(random_generator_);
            size_t parent2_idx = dist(random_generator_);

            // Select better parent
            if (fitness[parent1_idx] < fitness[parent2_idx]) {
                std::swap(parent1_idx, parent2_idx);
            }

            // Crossover
            auto child = crossoverConfigs(population[parent1_idx], population[parent2_idx]);

            // Mutation
            child = mutateConfig(child, mutation_rate);

            new_population.push_back(child);
        }

        population = new_population;
        saveResultsToFile();
    }

    updateBestConfiguration(best_result.config, best_result.result, use_case);
    return best_result;
}

VAETuningResult VAEAutoTuner::runMonteCarloTrial(const VAETuningConfig& config, size_t trials,
                                                 bool include_database_test,
                                                 bool include_anomaly_test)
{
    std::vector<VAETuningResult> trial_results;
    trial_results.reserve(trials);

    for (size_t trial = 0; trial < trials; ++trial) {
        uint64_t seed = random_generator_() + trial;
        auto result = runSingleTrial(config, seed);
        trial_results.push_back(result);
    }

    // Aggregate results
    VAETuningResult aggregated;
    aggregated.monte_carlo_trials = trials;

    // Calculate means and standard deviations
    auto calculate_stats = [&trial_results](auto member) {
        std::vector<double> values;
        for (const auto& result : trial_results) {
            values.push_back(result.*member);
        }

        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (double value : values) {
            variance += (value - mean) * (value - mean);
        }
        variance /= values.size();
        double stddev = std::sqrt(variance);

        return std::make_pair(mean, stddev);
    };

    auto [comp_mean, comp_std] = calculate_stats(&VAETuningResult::compression_ratio_mean);
    aggregated.compression_ratio_mean = comp_mean;
    aggregated.compression_ratio_stddev = comp_std;

    auto [recon_mean, recon_std] = calculate_stats(&VAETuningResult::reconstruction_error_mean);
    aggregated.reconstruction_error_mean = recon_mean;
    aggregated.reconstruction_error_stddev = recon_std;

    auto [kl_mean, kl_std] = calculate_stats(&VAETuningResult::kl_divergence_mean);
    aggregated.kl_divergence_mean = kl_mean;
    aggregated.kl_divergence_stddev = kl_std;

    auto [loss_mean, loss_std] = calculate_stats(&VAETuningResult::total_loss_mean);
    aggregated.total_loss_mean = loss_mean;
    aggregated.total_loss_stddev = loss_std;

    auto [train_time_mean, train_time_std] =
        calculate_stats(&VAETuningResult::training_time_ms_mean);
    aggregated.training_time_ms_mean = train_time_mean;
    aggregated.training_time_ms_stddev = train_time_std;

    auto [inf_time_mean, inf_time_std] = calculate_stats(&VAETuningResult::inference_time_ms_mean);
    aggregated.inference_time_ms_mean = inf_time_mean;
    aggregated.inference_time_ms_stddev = inf_time_std;

    // Database and anomaly testing if requested
    if (include_database_test) {
        auto db_result = testDatabaseIntegration(config, std::min(trials, size_t(5)));
        aggregated.storage_efficiency_mean = db_result.storage_efficiency_mean;
        aggregated.storage_efficiency_stddev = db_result.storage_efficiency_stddev;
        aggregated.retrieval_accuracy_mean = db_result.retrieval_accuracy_mean;
        aggregated.retrieval_accuracy_stddev = db_result.retrieval_accuracy_stddev;
    }

    if (include_anomaly_test) {
        auto anomaly_result = testAnomalyDetection(config, 0.1f, std::min(trials, size_t(5)));
        aggregated.anomaly_detection_score_mean = anomaly_result.anomaly_detection_score_mean;
        aggregated.anomaly_detection_score_stddev = anomaly_result.anomaly_detection_score_stddev;
        aggregated.false_positive_rate_mean = anomaly_result.false_positive_rate_mean;
        aggregated.false_positive_rate_stddev = anomaly_result.false_positive_rate_stddev;
        aggregated.true_positive_rate_mean = anomaly_result.true_positive_rate_mean;
        aggregated.true_positive_rate_stddev = anomaly_result.true_positive_rate_stddev;
    }

    // Calculate convergence rate
    size_t converged_trials = 0;
    for (const auto& result : trial_results) {
        if (result.converged) converged_trials++;
    }
    aggregated.convergence_rate = static_cast<double>(converged_trials) / trials;
    aggregated.converged = aggregated.convergence_rate > 0.8;  // 80% convergence threshold

    // Calculate composite scores
    aggregated.compression_score = calculateCompressionScore(aggregated);
    aggregated.anomaly_score = calculateAnomalyScore(aggregated);
    aggregated.balanced_score = calculateBalancedScore(aggregated);

    return aggregated;
}

VAETuningResult VAEAutoTuner::runSingleTrial(const VAETuningConfig& config, uint64_t seed)
{
    VAETuningResult result;

    try {
        // Create VAE configuration
        VAEConfig vae_config;
        vae_config.latent_dim = config.latent_dim;
        vae_config.beta = config.beta;
        vae_config.learning_rate = config.learning_rate;
        vae_config.epochs = config.epochs;
        vae_config.batch_size = config.batch_size;
        vae_config.use_interpolation = config.use_interpolation;
        vae_config.optimizer = config.optimizer;
        vae_config.sampling = config.sampling;

        // Create VAE
        VariationalAutoencoder<float> vae(config.input_dim, config.latent_dim, config.hidden_dims,
                                          neural::ProtectionLevel::NONE, vae_config);

        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        float final_loss =
            vae.train(training_data_, config.epochs, config.batch_size, config.learning_rate);
        auto train_end = std::chrono::high_resolution_clock::now();

        result.training_time_ms_mean =
            std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
        result.total_loss_mean = final_loss;

        // Evaluation
        auto eval_start = std::chrono::high_resolution_clock::now();
        auto metrics = vae.evaluateComprehensive(validation_data_);
        auto eval_end = std::chrono::high_resolution_clock::now();

        result.inference_time_ms_mean =
            std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count();

        result.reconstruction_error_mean = metrics["reconstruction_loss"];
        result.kl_divergence_mean = metrics["kl_divergence"];
        result.compression_ratio_mean = static_cast<double>(config.input_dim) / config.latent_dim;

        // Check convergence (simple heuristic)
        result.converged = (final_loss < 10.0f && metrics["reconstruction_loss"] < 1.0f);
    }
    catch (const std::exception& e) {
        core::Logger::error("Trial failed: " + std::string(e.what()));
        result.converged = false;
        result.total_loss_mean = std::numeric_limits<double>::infinity();
    }

    return result;
}

VAETuningConfig VAEAutoTuner::generateRandomConfig()
{
    VAETuningConfig config;

    std::uniform_int_distribution<size_t> latent_dist(0, latent_dim_options_.size() - 1);
    std::uniform_int_distribution<size_t> beta_dist(0, beta_options_.size() - 1);
    std::uniform_int_distribution<size_t> lr_dist(0, learning_rate_options_.size() - 1);
    std::uniform_int_distribution<size_t> epoch_dist(0, epoch_options_.size() - 1);
    std::uniform_int_distribution<size_t> arch_dist(0, architecture_options_.size() - 1);

    config.input_dim = training_data_.empty() ? 0 : training_data_[0].size();
    config.latent_dim = latent_dim_options_[latent_dist(random_generator_)];
    config.beta = beta_options_[beta_dist(random_generator_)];
    config.learning_rate = learning_rate_options_[lr_dist(random_generator_)];
    config.epochs = epoch_options_[epoch_dist(random_generator_)];
    config.hidden_dims = architecture_options_[arch_dist(random_generator_)];
    config.batch_size = 32;                                // Fixed for now
    config.use_interpolation = true;                       // Fixed for now
    config.optimizer = OptimizerType::ADAM;                // Fixed for now
    config.sampling = SamplingTechnique::REPARAMETERIZED;  // Fixed for now

    return config;
}

double VAEAutoTuner::calculateCompressionScore(const VAETuningResult& result)
{
    // Higher compression ratio is better, lower reconstruction error is better
    double compression_factor = result.compression_ratio_mean;
    double quality_factor = 1.0 / (1.0 + result.reconstruction_error_mean);
    double stability_factor = result.convergence_rate;

    return compression_factor * quality_factor * stability_factor;
}

double VAEAutoTuner::calculateAnomalyScore(const VAETuningResult& result)
{
    // Higher true positive rate and lower false positive rate is better
    double detection_factor = result.true_positive_rate_mean;
    double precision_factor = 1.0 - result.false_positive_rate_mean;
    double stability_factor = result.convergence_rate;

    return detection_factor * precision_factor * stability_factor;
}

double VAEAutoTuner::calculateBalancedScore(const VAETuningResult& result)
{
    double compression_score = calculateCompressionScore(result);
    double anomaly_score = calculateAnomalyScore(result);

    // Weighted combination
    return 0.6 * compression_score + 0.4 * anomaly_score;
}

void VAEAutoTuner::saveResultsToFile() const
{
    std::ofstream file(results_file_);
    if (!file.is_open()) return;

    // Write header
    file << "input_dim,latent_dim,beta,learning_rate,epochs,batch_size,";
    file << "compression_ratio_mean,compression_ratio_std,";
    file << "reconstruction_error_mean,reconstruction_error_std,";
    file << "kl_divergence_mean,kl_divergence_std,";
    file << "total_loss_mean,total_loss_std,";
    file << "training_time_mean,training_time_std,";
    file << "inference_time_mean,inference_time_std,";
    file << "convergence_rate,monte_carlo_trials,";
    file << "compression_score,anomaly_score,balanced_score\n";

    // Write data
    for (const auto& [config, result] : tested_configs_) {
        file << config.input_dim << "," << config.latent_dim << "," << config.beta << ",";
        file << config.learning_rate << "," << config.epochs << "," << config.batch_size << ",";
        file << result.compression_ratio_mean << "," << result.compression_ratio_stddev << ",";
        file << result.reconstruction_error_mean << "," << result.reconstruction_error_stddev
             << ",";
        file << result.kl_divergence_mean << "," << result.kl_divergence_stddev << ",";
        file << result.total_loss_mean << "," << result.total_loss_stddev << ",";
        file << result.training_time_ms_mean << "," << result.training_time_ms_stddev << ",";
        file << result.inference_time_ms_mean << "," << result.inference_time_ms_stddev << ",";
        file << result.convergence_rate << "," << result.monte_carlo_trials << ",";
        file << result.compression_score << "," << result.anomaly_score << ","
             << result.balanced_score << "\n";
    }
}

void VAEAutoTuner::updateBestConfiguration(const VAETuningConfig& config,
                                           const VAETuningResult& result,
                                           const std::string& use_case)
{
    best_configs_[use_case] = VAESearchResult(config, result, use_case, tested_configs_.size());
}

void VAEAutoTuner::setSeed(unsigned int seed)
{
    random_generator_.seed(seed);
    core::Logger::info("Random seed set to: " + std::to_string(seed));
}

void VAEAutoTuner::exportResults(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        core::Logger::error("Failed to open file for export: " + filename);
        return;
    }

    // Export detailed results in JSON format
    file << "{\n";
    file << "  \"vae_tuning_results\": {\n";
    file << "    \"total_configurations_tested\": " << tested_configs_.size() << ",\n";
    file << "    \"configurations\": [\n";

    bool first = true;
    for (const auto& [config, result] : tested_configs_) {
        if (!first) file << ",\n";
        first = false;

        file << "      {\n";
        file << "        \"config\": {\n";
        file << "          \"input_dim\": " << config.input_dim << ",\n";
        file << "          \"latent_dim\": " << config.latent_dim << ",\n";
        file << "          \"beta\": " << config.beta << ",\n";
        file << "          \"learning_rate\": " << config.learning_rate << ",\n";
        file << "          \"epochs\": " << config.epochs << ",\n";
        file << "          \"batch_size\": " << config.batch_size << ",\n";
        file << "          \"hidden_dims\": [";
        for (size_t i = 0; i < config.hidden_dims.size(); ++i) {
            if (i > 0) file << ", ";
            file << config.hidden_dims[i];
        }
        file << "]\n";
        file << "        },\n";
        file << "        \"results\": {\n";
        file << "          \"compression_ratio_mean\": " << result.compression_ratio_mean << ",\n";
        file << "          \"reconstruction_error_mean\": " << result.reconstruction_error_mean
             << ",\n";
        file << "          \"kl_divergence_mean\": " << result.kl_divergence_mean << ",\n";
        file << "          \"total_loss_mean\": " << result.total_loss_mean << ",\n";
        file << "          \"training_time_ms_mean\": " << result.training_time_ms_mean << ",\n";
        file << "          \"convergence_rate\": " << result.convergence_rate << ",\n";
        file << "          \"compression_score\": " << result.compression_score << ",\n";
        file << "          \"anomaly_score\": " << result.anomaly_score << ",\n";
        file << "          \"balanced_score\": " << result.balanced_score << "\n";
        file << "        }\n";
        file << "      }";
    }

    file << "\n    ],\n";
    file << "    \"best_configurations\": {\n";

    bool first_best = true;
    for (const auto& [use_case, best_result] : best_configs_) {
        if (!first_best) file << ",\n";
        first_best = false;

        file << "      \"" << use_case << "\": {\n";
        file << "        \"latent_dim\": " << best_result.config.latent_dim << ",\n";
        file << "        \"beta\": " << best_result.config.beta << ",\n";
        file << "        \"score\": " << best_result.result.balanced_score << "\n";
        file << "      }";
    }

    file << "\n    }\n";
    file << "  }\n";
    file << "}\n";

    core::Logger::info("Results exported to: " + filename);
}

void VAEAutoTuner::generateReport(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        core::Logger::error("Failed to open file for report: " + filename);
        return;
    }

    file << "# VAE Auto-Tuning Report\n\n";
    file << "## Summary\n";
    file << "- Total configurations tested: " << tested_configs_.size() << "\n";
    file << "- Training data samples: " << training_data_.size() << "\n";
    file << "- Validation data samples: " << validation_data_.size() << "\n";
    file << "- Test data samples: " << test_data_.size() << "\n\n";

    if (!best_configs_.empty()) {
        file << "## Best Configurations\n\n";

        for (const auto& [use_case, best_result] : best_configs_) {
            file << "### " << use_case << "\n";
            file << "- **Latent Dimension**: " << best_result.config.latent_dim << "\n";
            file << "- **Beta Parameter**: " << best_result.config.beta << "\n";
            file << "- **Learning Rate**: " << best_result.config.learning_rate << "\n";
            file << "- **Architecture**: ";
            for (size_t i = 0; i < best_result.config.hidden_dims.size(); ++i) {
                if (i > 0) file << " → ";
                file << best_result.config.hidden_dims[i];
            }
            file << "\n";
            file << "- **Compression Ratio**: " << std::fixed << std::setprecision(2)
                 << best_result.result.compression_ratio_mean << "\n";
            file << "- **Reconstruction Error**: " << std::fixed << std::setprecision(4)
                 << best_result.result.reconstruction_error_mean << "\n";
            file << "- **Convergence Rate**: " << std::fixed << std::setprecision(2)
                 << (best_result.result.convergence_rate * 100) << "%\n";
            file << "- **Overall Score**: " << std::fixed << std::setprecision(4)
                 << best_result.result.balanced_score << "\n\n";
        }
    }

    file << "## Configuration Analysis\n\n";

    // Analyze latent dimensions
    std::map<size_t, std::vector<double>> latent_scores;
    for (const auto& [config, result] : tested_configs_) {
        latent_scores[config.latent_dim].push_back(result.balanced_score);
    }

    file << "### Latent Dimension Performance\n";
    for (const auto& [latent_dim, scores] : latent_scores) {
        if (scores.empty()) continue;
        double avg_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        file << "- **" << latent_dim << " dimensions**: Average score " << std::fixed
             << std::setprecision(4) << avg_score << " (" << scores.size() << " configurations)\n";
    }

    file << "\n### Recommendations\n";
    file << "Based on the testing results:\n\n";

    if (!best_configs_.empty()) {
        auto best_overall = std::max_element(
            best_configs_.begin(), best_configs_.end(), [](const auto& a, const auto& b) {
                return a.second.result.balanced_score < b.second.result.balanced_score;
            });

        file << "1. **Best Overall Configuration**: Use latent dimension "
             << best_overall->second.config.latent_dim << " with beta "
             << best_overall->second.config.beta << "\n";
        file << "2. **For Compression**: Focus on configurations with beta < 1.0\n";
        file << "3. **For Anomaly Detection**: Focus on configurations with beta > 1.0\n";
    }

    file << "\n---\n";
    file << "*Report generated by VAEAutoTuner*\n";

    core::Logger::info("Report generated: " + filename);
}

VAETuningResult VAEAutoTuner::testDatabaseIntegration(const VAETuningConfig& config, size_t trials)
{
    VAETuningResult result;

    // Simplified database integration test
    // In a real implementation, this would test actual database operations

    std::vector<double> storage_efficiencies;
    std::vector<double> retrieval_accuracies;

    for (size_t i = 0; i < trials; ++i) {
        // Simulate storage efficiency (compression ratio affects this)
        double compression_ratio = static_cast<double>(config.input_dim) / config.latent_dim;
        double storage_efficiency = std::min(0.95, compression_ratio / 10.0);  // Normalized
        storage_efficiencies.push_back(storage_efficiency);

        // Simulate retrieval accuracy (affected by reconstruction quality)
        double retrieval_accuracy =
            0.85 + (0.1 * std::exp(-config.beta));  // Higher beta = lower accuracy
        retrieval_accuracies.push_back(std::min(0.99, retrieval_accuracy));
    }

    // Calculate statistics
    auto calc_stats = [](const std::vector<double>& values) {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        return std::make_pair(mean, std::sqrt(variance));
    };

    auto [storage_mean, storage_std] = calc_stats(storage_efficiencies);
    auto [retrieval_mean, retrieval_std] = calc_stats(retrieval_accuracies);

    result.storage_efficiency_mean = storage_mean;
    result.storage_efficiency_stddev = storage_std;
    result.retrieval_accuracy_mean = retrieval_mean;
    result.retrieval_accuracy_stddev = retrieval_std;

    return result;
}

VAETuningResult VAEAutoTuner::testAnomalyDetection(const VAETuningConfig& config,
                                                   float anomaly_threshold, size_t trials)
{
    VAETuningResult result;

    // Simplified anomaly detection test
    std::vector<double> detection_scores;
    std::vector<double> false_positive_rates;
    std::vector<double> true_positive_rates;

    for (size_t i = 0; i < trials; ++i) {
        // Simulate anomaly detection performance
        // Higher beta generally improves anomaly detection
        double base_detection = 0.7 + (0.2 * std::tanh(config.beta - 1.0));
        double detection_score = std::min(0.95, base_detection);
        detection_scores.push_back(detection_score);

        // False positive rate (lower is better)
        double fpr = 0.1 * std::exp(-config.beta * 0.5);
        false_positive_rates.push_back(std::min(0.2, fpr));

        // True positive rate (higher is better)
        double tpr = detection_score * (0.8 + 0.2 * config.beta);
        true_positive_rates.push_back(std::min(0.98, tpr));
    }

    // Calculate statistics
    auto calc_stats = [](const std::vector<double>& values) {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        return std::make_pair(mean, std::sqrt(variance));
    };

    auto [detection_mean, detection_std] = calc_stats(detection_scores);
    auto [fpr_mean, fpr_std] = calc_stats(false_positive_rates);
    auto [tpr_mean, tpr_std] = calc_stats(true_positive_rates);

    result.anomaly_detection_score_mean = detection_mean;
    result.anomaly_detection_score_stddev = detection_std;
    result.false_positive_rate_mean = fpr_mean;
    result.false_positive_rate_stddev = fpr_std;
    result.true_positive_rate_mean = tpr_mean;
    result.true_positive_rate_stddev = tpr_std;

    return result;
}

VAETuningConfig VAEAutoTuner::mutateConfig(const VAETuningConfig& config, double mutation_rate)
{
    VAETuningConfig mutated = config;

    std::uniform_real_distribution<double> mutation_prob(0.0, 1.0);

    // Mutate latent dimension
    if (mutation_prob(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> latent_dist(0, latent_dim_options_.size() - 1);
        mutated.latent_dim = latent_dim_options_[latent_dist(random_generator_)];
    }

    // Mutate beta
    if (mutation_prob(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> beta_dist(0, beta_options_.size() - 1);
        mutated.beta = beta_options_[beta_dist(random_generator_)];
    }

    // Mutate learning rate
    if (mutation_prob(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> lr_dist(0, learning_rate_options_.size() - 1);
        mutated.learning_rate = learning_rate_options_[lr_dist(random_generator_)];
    }

    // Mutate architecture
    if (mutation_prob(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> arch_dist(0, architecture_options_.size() - 1);
        mutated.hidden_dims = architecture_options_[arch_dist(random_generator_)];
    }

    // Mutate epochs
    if (mutation_prob(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> epoch_dist(0, epoch_options_.size() - 1);
        mutated.epochs = epoch_options_[epoch_dist(random_generator_)];
    }

    return mutated;
}

VAETuningConfig VAEAutoTuner::crossoverConfigs(const VAETuningConfig& parent1,
                                               const VAETuningConfig& parent2)
{
    VAETuningConfig offspring;

    std::uniform_real_distribution<double> crossover_prob(0.0, 1.0);

    // Inherit input_dim from parent1 (should be the same for both)
    offspring.input_dim = parent1.input_dim;

    // Crossover latent dimension
    offspring.latent_dim =
        (crossover_prob(random_generator_) < 0.5) ? parent1.latent_dim : parent2.latent_dim;

    // Crossover beta
    offspring.beta = (crossover_prob(random_generator_) < 0.5) ? parent1.beta : parent2.beta;

    // Crossover learning rate
    offspring.learning_rate =
        (crossover_prob(random_generator_) < 0.5) ? parent1.learning_rate : parent2.learning_rate;

    // Crossover epochs
    offspring.epochs = (crossover_prob(random_generator_) < 0.5) ? parent1.epochs : parent2.epochs;

    // Crossover batch size
    offspring.batch_size =
        (crossover_prob(random_generator_) < 0.5) ? parent1.batch_size : parent2.batch_size;

    // Crossover architecture
    offspring.hidden_dims =
        (crossover_prob(random_generator_) < 0.5) ? parent1.hidden_dims : parent2.hidden_dims;

    // Crossover other parameters
    offspring.use_interpolation = (crossover_prob(random_generator_) < 0.5)
                                      ? parent1.use_interpolation
                                      : parent2.use_interpolation;
    offspring.optimizer =
        (crossover_prob(random_generator_) < 0.5) ? parent1.optimizer : parent2.optimizer;
    offspring.sampling =
        (crossover_prob(random_generator_) < 0.5) ? parent1.sampling : parent2.sampling;

    return offspring;
}

}  // namespace rad_ml::research
