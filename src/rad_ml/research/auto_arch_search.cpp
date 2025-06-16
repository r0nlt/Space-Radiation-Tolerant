/**
 * @file auto_arch_search.cpp
 * @brief Implementation of the automatic architecture search functionality
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <rad_ml/research/auto_arch_search.hpp>
#include <set>

namespace rad_ml {
namespace research {

// Constructor
AutoArchSearch::AutoArchSearch(const std::vector<float>& train_data,
                               const std::vector<float>& train_labels,
                               const std::vector<float>& test_data,
                               const std::vector<float>& test_labels, sim::Environment environment,
                               const std::vector<size_t>& width_options,
                               const std::vector<double>& dropout_options,
                               const std::string& results_file)
    : train_data_(train_data),
      train_labels_(train_labels),
      test_data_(test_data),
      test_labels_(test_labels),
      environment_(environment),
      width_options_(width_options),
      dropout_options_(dropout_options),
      results_file_(results_file),
      test_residual_connections_(true),
      fixed_hidden_layers_(0)
{
    // Initialize random generator with time-based seed
    std::random_device rd;
    random_generator_ = std::mt19937(rd());

    // Determine input and output sizes from data
    // Assume input_size is the size of one training example
    // and output_size is the size of one label
    input_size_ = train_data.size() / train_labels.size();
    output_size_ = 1;  // Default to 1, will be adjusted based on labels

    // Try to infer output_size from labels if they are one-hot encoded
    std::set<float> unique_labels;
    for (const auto& label : train_labels) {
        unique_labels.insert(label);
    }

    // If number of unique labels is small, it's probably classification
    if (unique_labels.size() > 1 && unique_labels.size() < 100) {
        output_size_ = unique_labels.size();
    }

    // Set default protection levels to test
    protection_levels_ = {neural::ProtectionLevel::NONE, neural::ProtectionLevel::CHECKSUM_ONLY,
                          neural::ProtectionLevel::SELECTIVE_TMR,
                          neural::ProtectionLevel::FULL_TMR};

    // Create the architecture tester
    tester_ = std::make_unique<ArchitectureTester>(train_data, train_labels, test_data, test_labels,
                                                   input_size_, output_size_, results_file_);

    std::cout << "AutoArchSearch initialized with input_size=" << input_size_
              << ", output_size=" << output_size_ << std::endl;
}

// Find optimal architecture (grid search by default)
SearchResult AutoArchSearch::findOptimalArchitecture(size_t max_epochs, bool use_monte_carlo,
                                                     size_t monte_carlo_trials)
{
    std::cout << "Starting grid search for optimal architecture..." << std::endl;
    if (use_monte_carlo) {
        std::cout << "Using Monte Carlo testing with " << monte_carlo_trials
                  << " trials per configuration" << std::endl;
    }

    // Generate all possible configurations
    auto configs = generateAllConfigs();

    std::cout << "Testing " << configs.size() << " configurations" << std::endl;

    // Test each configuration
    size_t iteration = 0;
    double best_preservation = 0.0;
    NetworkConfig best_config;
    ArchitectureTestResult best_result;

    for (const auto& config : configs) {
        // Test this configuration
        auto result = testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);

        // Store in tested_configs_
        tested_configs_[config] = result;

        // Check if this is the best so far
        if (result.accuracy_preservation > best_preservation) {
            best_preservation = result.accuracy_preservation;
            best_config = config;
            best_result = result;

            std::cout << "New best configuration found:" << std::endl;
            std::string arch_str = "Architecture: ";
            for (auto size : config.layer_sizes) {
                arch_str += std::to_string(size) + "-";
            }
            std::cout << arch_str << std::endl;
            std::cout << "Dropout: " << config.dropout_rate << std::endl;
            std::cout << "Residual: " << (config.has_residual_connections ? "Yes" : "No")
                      << std::endl;
            std::cout << "Protection: " << static_cast<int>(config.protection_level) << std::endl;

            if (use_monte_carlo) {
                std::cout << "Accuracy preservation: " << best_preservation << "% ± "
                          << result.accuracy_preservation_stddev << "% (over "
                          << result.monte_carlo_trials << " trials)" << std::endl;
            }
            else {
                std::cout << "Accuracy preservation: " << best_preservation << "%" << std::endl;
            }
        }

        // Increment iteration counter
        ++iteration;

        // Save results periodically
        if (iteration % 10 == 0) {
            saveResultsToFile();
        }
    }

    // Save final results
    saveResultsToFile();

    // Return the best configuration found
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, iteration,
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

// Random search implementation
SearchResult AutoArchSearch::randomSearch(size_t max_iterations, size_t max_epochs,
                                          bool use_monte_carlo, size_t monte_carlo_trials)
{
    std::cout << "Starting random search for optimal architecture..." << std::endl;
    if (use_monte_carlo) {
        std::cout << "Using Monte Carlo testing with " << monte_carlo_trials
                  << " trials per configuration" << std::endl;
    }

    double best_preservation = 0.0;
    NetworkConfig best_config;
    ArchitectureTestResult best_result;

    for (size_t i = 0; i < max_iterations; ++i) {
        // Generate a random configuration
        auto config = generateRandomConfig();

        // Skip if we've already tested this
        if (tested_configs_.count(config) > 0) {
            --i;  // Don't count this as an iteration
            continue;
        }

        // Test this configuration
        auto result = testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);

        // Store in tested_configs_
        tested_configs_[config] = result;

        // Check if this is the best so far
        if (result.accuracy_preservation > best_preservation) {
            best_preservation = result.accuracy_preservation;
            best_config = config;
            best_result = result;

            std::cout << "New best configuration found (iteration " << i << "):" << std::endl;
            std::string arch_str = "Architecture: ";
            for (auto size : config.layer_sizes) {
                arch_str += std::to_string(size) + "-";
            }
            std::cout << arch_str << std::endl;
            std::cout << "Dropout: " << config.dropout_rate << std::endl;
            std::cout << "Residual: " << (config.has_residual_connections ? "Yes" : "No")
                      << std::endl;
            std::cout << "Protection: " << static_cast<int>(config.protection_level) << std::endl;

            if (use_monte_carlo) {
                std::cout << "Accuracy preservation: " << best_preservation << "% ± "
                          << result.accuracy_preservation_stddev << "% (over "
                          << result.monte_carlo_trials << " trials)" << std::endl;
            }
            else {
                std::cout << "Accuracy preservation: " << best_preservation << "%" << std::endl;
            }
        }

        // Save results periodically
        if (i % 10 == 0) {
            saveResultsToFile();
        }
    }

    // Save final results
    saveResultsToFile();

    // Return the best configuration found
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, max_iterations,
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

// Evolutionary search implementation
SearchResult AutoArchSearch::evolutionarySearch(size_t population_size, size_t generations,
                                                double mutation_rate, size_t max_epochs,
                                                bool use_monte_carlo, size_t monte_carlo_trials)
{
    std::cout << "Starting evolutionary search for optimal architecture..." << std::endl;
    if (use_monte_carlo) {
        std::cout << "Using Monte Carlo testing with " << monte_carlo_trials
                  << " trials per configuration" << std::endl;
    }

    // Initialize random population
    std::vector<NetworkConfig> population;
    std::vector<double> fitness;

    // Generate initial population
    for (size_t i = 0; i < population_size; ++i) {
        population.push_back(generateRandomConfig());
    }

    double best_preservation = 0.0;
    NetworkConfig best_config;
    ArchitectureTestResult best_result;

    // Evolve for specified number of generations
    for (size_t gen = 0; gen < generations; ++gen) {
        std::cout << "Generation " << (gen + 1) << "/" << generations << std::endl;

        // Evaluate fitness for each individual
        fitness.clear();
        for (auto& config : population) {
            // Test this configuration if we haven't already
            if (tested_configs_.count(config) == 0) {
                auto result =
                    testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);
                tested_configs_[config] = result;
            }

            // Get the fitness (accuracy preservation)
            double preservation = tested_configs_[config].accuracy_preservation;
            fitness.push_back(preservation);

            // Check if this is the best so far
            if (preservation > best_preservation) {
                best_preservation = preservation;
                best_config = config;
                best_result = tested_configs_[config];

                std::cout << "New best configuration found (generation " << (gen + 1)
                          << "):" << std::endl;
                std::string arch_str = "Architecture: ";
                for (auto size : config.layer_sizes) {
                    arch_str += std::to_string(size) + "-";
                }
                std::cout << arch_str << std::endl;
                std::cout << "Dropout: " << config.dropout_rate << std::endl;
                std::cout << "Residual: " << (config.has_residual_connections ? "Yes" : "No")
                          << std::endl;
                std::cout << "Protection: " << static_cast<int>(config.protection_level)
                          << std::endl;

                if (use_monte_carlo) {
                    std::cout << "Accuracy preservation: " << best_preservation << "% ± "
                              << best_result.accuracy_preservation_stddev << "% (over "
                              << best_result.monte_carlo_trials << " trials)" << std::endl;
                }
                else {
                    std::cout << "Accuracy preservation: " << best_preservation << "%" << std::endl;
                }
            }
        }

        // Create new population through selection, crossover, and mutation
        std::vector<NetworkConfig> new_population;

        // Elitism: keep the best individual
        size_t best_idx =
            std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
        new_population.push_back(population[best_idx]);

        // Generate the rest through selection and crossover
        while (new_population.size() < population_size) {
            // Selection: tournament selection (k=2)
            std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
            size_t idx1 = dist(random_generator_);
            size_t idx2 = dist(random_generator_);

            size_t parent1_idx = (fitness[idx1] > fitness[idx2]) ? idx1 : idx2;

            idx1 = dist(random_generator_);
            idx2 = dist(random_generator_);

            size_t parent2_idx = (fitness[idx1] > fitness[idx2]) ? idx1 : idx2;

            // Crossover
            auto child = crossoverConfigs(population[parent1_idx], population[parent2_idx]);

            // Mutation
            child = mutateConfig(child, mutation_rate);

            // Add to new population
            new_population.push_back(child);
        }

        // Replace old population
        population = new_population;

        // Save results for this generation
        saveResultsToFile();
    }

    // Return the best configuration found
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, generations * population_size,
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

// Set protection levels to test
void AutoArchSearch::setProtectionLevels(const std::vector<neural::ProtectionLevel>& levels)
{
    protection_levels_ = levels;
}

// Set whether to test residual connections
void AutoArchSearch::setTestResidualConnections(bool test_residual)
{
    test_residual_connections_ = test_residual;
}

// Get all tested configurations
const std::map<NetworkConfig, ArchitectureTestResult>& AutoArchSearch::getTestedConfigurations()
    const
{
    return tested_configs_;
}

// Set fixed parameters for architecture
void AutoArchSearch::setFixedParameters(size_t input_size, size_t output_size,
                                        size_t num_hidden_layers)
{
    input_size_ = input_size;
    output_size_ = output_size;
    fixed_hidden_layers_ = num_hidden_layers;

    // Recreate tester with updated input/output sizes
    tester_ =
        std::make_unique<ArchitectureTester>(train_data_, train_labels_, test_data_, test_labels_,
                                             input_size_, output_size_, results_file_);
}

// Export results to CSV
void AutoArchSearch::exportResults(const std::string& filename) const
{
    std::ofstream out_file(filename);

    if (!out_file) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return;
    }

    // Write header
    out_file << "Architecture,Dropout,HasResidual,ProtectionLevel,Environment,"
             << "BaselineAccuracy,RadiationAccuracy,AccuracyPreservation,"
             << "ExecutionTime,ErrorsDetected,ErrorsCorrected,UncorrectableErrors,"
             << "BaselineAccuracyStdDev,RadiationAccuracyStdDev,AccuracyPreservationStdDev,"
             << "MonteCarloTrials\n";

    // Write each result
    for (const auto& [config, result] : tested_configs_) {
        // Format architecture string
        std::string arch_str;
        for (auto size : config.layer_sizes) {
            arch_str += std::to_string(size) + "-";
        }
        if (!arch_str.empty()) {
            arch_str.pop_back();  // Remove trailing dash
        }

        // Protection level string
        std::string protection_str;
        switch (config.protection_level) {
            case neural::ProtectionLevel::NONE:
                protection_str = "None";
                break;
            case neural::ProtectionLevel::CHECKSUM_ONLY:
                protection_str = "ChecksumOnly";
                break;
            case neural::ProtectionLevel::SELECTIVE_TMR:
                protection_str = "SelectiveTMR";
                break;
            case neural::ProtectionLevel::FULL_TMR:
                protection_str = "FullTMR";
                break;
            case neural::ProtectionLevel::ADAPTIVE_TMR:
                protection_str = "AdaptiveTMR";
                break;
            case neural::ProtectionLevel::SPACE_OPTIMIZED:
                protection_str = "SpaceOptimized";
                break;
            default:
                protection_str = "Unknown";
        }

        // Write row
        out_file << arch_str << "," << config.dropout_rate << ","
                 << (config.has_residual_connections ? "Yes" : "No") << "," << protection_str << ","
                 << static_cast<int>(result.environment) << "," << std::fixed
                 << std::setprecision(2) << result.baseline_accuracy << "," << std::fixed
                 << std::setprecision(2) << result.radiation_accuracy << "," << std::fixed
                 << std::setprecision(2) << result.accuracy_preservation << "," << std::fixed
                 << std::setprecision(2) << result.execution_time_ms << ","
                 << result.errors_detected << "," << result.errors_corrected << ","
                 << result.uncorrectable_errors << "," << std::fixed << std::setprecision(2)
                 << result.baseline_accuracy_stddev << "," << std::fixed << std::setprecision(2)
                 << result.radiation_accuracy_stddev << "," << std::fixed << std::setprecision(2)
                 << result.accuracy_preservation_stddev << "," << result.monte_carlo_trials << "\n";
    }

    std::cout << "Results exported to " << filename << std::endl;
}

// Test a specific configuration
ArchitectureTestResult AutoArchSearch::testConfiguration(const NetworkConfig& config, size_t epochs,
                                                         bool use_monte_carlo,
                                                         size_t monte_carlo_trials)
{
    std::cout << "Testing configuration:" << std::endl;
    std::string arch_str = "Architecture: ";
    for (auto size : config.layer_sizes) {
        arch_str += std::to_string(size) + "-";
    }
    std::cout << arch_str << std::endl;
    std::cout << "Dropout: " << config.dropout_rate << std::endl;
    std::cout << "Residual: " << (config.has_residual_connections ? "Yes" : "No") << std::endl;
    std::cout << "Protection: " << static_cast<int>(config.protection_level) << std::endl;

    ArchitectureTestResult result;

    if (use_monte_carlo) {
        // Use Monte Carlo testing with multiple trials
        result = tester_->testArchitectureMonteCarlo(
            config.layer_sizes, config.dropout_rate, config.has_residual_connections,
            config.protection_level, epochs, environment_, monte_carlo_trials);

        std::cout << "Results: Baseline accuracy = " << std::fixed << std::setprecision(2)
                  << result.baseline_accuracy << "% ± " << std::fixed << std::setprecision(2)
                  << result.baseline_accuracy_stddev << "%, Radiation accuracy = " << std::fixed
                  << std::setprecision(2) << result.radiation_accuracy << "% ± " << std::fixed
                  << std::setprecision(2) << result.radiation_accuracy_stddev
                  << "%, Preservation = " << std::fixed << std::setprecision(2)
                  << result.accuracy_preservation << "% ± " << std::fixed << std::setprecision(2)
                  << result.accuracy_preservation_stddev << "%" << std::endl;
    }
    else {
        // Use standard single-run testing
        result = tester_->testArchitecture(config.layer_sizes, config.dropout_rate,
                                           config.has_residual_connections, config.protection_level,
                                           epochs, environment_);

        std::cout << "Results: Baseline accuracy = " << std::fixed << std::setprecision(2)
                  << result.baseline_accuracy << "%, Radiation accuracy = " << std::fixed
                  << std::setprecision(2) << result.radiation_accuracy
                  << "%, Preservation = " << std::fixed << std::setprecision(2)
                  << result.accuracy_preservation << "%" << std::endl;
    }

    return result;
}

// Generate a random configuration
NetworkConfig AutoArchSearch::generateRandomConfig()
{
    // Choose number of hidden layers
    size_t num_hidden_layers;
    if (fixed_hidden_layers_ > 0) {
        num_hidden_layers = fixed_hidden_layers_;
    }
    else {
        std::uniform_int_distribution<size_t> layers_dist(1, 3);  // 1-3 hidden layers
        num_hidden_layers = layers_dist(random_generator_);
    }

    // Generate layer sizes
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(input_size_);  // Input layer

    std::uniform_int_distribution<size_t> width_idx_dist(0, width_options_.size() - 1);

    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layer_sizes.push_back(width_options_[width_idx_dist(random_generator_)]);
    }

    layer_sizes.push_back(output_size_);  // Output layer

    // Choose dropout rate
    std::uniform_int_distribution<size_t> dropout_idx_dist(0, dropout_options_.size() - 1);
    double dropout_rate = dropout_options_[dropout_idx_dist(random_generator_)];

    // Choose whether to use residual connections
    bool use_residual = false;
    if (test_residual_connections_) {
        std::uniform_int_distribution<int> residual_dist(0, 1);
        use_residual = residual_dist(random_generator_) > 0;
    }

    // Choose protection level
    std::uniform_int_distribution<size_t> protection_idx_dist(0, protection_levels_.size() - 1);
    auto protection_level = protection_levels_[protection_idx_dist(random_generator_)];

    return NetworkConfig(layer_sizes, dropout_rate, use_residual, protection_level);
}

// Mutate a configuration
NetworkConfig AutoArchSearch::mutateConfig(const NetworkConfig& config, double mutation_rate)
{
    // Clone the configuration
    NetworkConfig mutated = config;

    // Uniform distribution for mutation decisions
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Potentially mutate layer sizes
    if (mutation_dist(random_generator_) < mutation_rate) {
        // Choose a hidden layer to mutate (exclude input and output)
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(random_generator_);

            // Choose a new width for this layer
            std::uniform_int_distribution<size_t> width_idx_dist(0, width_options_.size() - 1);
            mutated.layer_sizes[layer_idx] = width_options_[width_idx_dist(random_generator_)];
        }
    }

    // Potentially mutate dropout rate
    if (mutation_dist(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> dropout_idx_dist(0, dropout_options_.size() - 1);
        mutated.dropout_rate = dropout_options_[dropout_idx_dist(random_generator_)];
    }

    // Potentially flip residual connections
    if (test_residual_connections_ && mutation_dist(random_generator_) < mutation_rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    // Potentially mutate protection level
    if (mutation_dist(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> protection_idx_dist(0, protection_levels_.size() - 1);
        mutated.protection_level = protection_levels_[protection_idx_dist(random_generator_)];
    }

    return mutated;
}

// Crossover two configurations
NetworkConfig AutoArchSearch::crossoverConfigs(const NetworkConfig& parent1,
                                               const NetworkConfig& parent2)
{
    // Create a child config
    NetworkConfig child;

    // Crossover layer sizes
    // If different number of layers, choose one parent's architecture
    if (parent1.layer_sizes.size() != parent2.layer_sizes.size()) {
        std::uniform_int_distribution<int> parent_choice(0, 1);
        child.layer_sizes =
            parent_choice(random_generator_) == 0 ? parent1.layer_sizes : parent2.layer_sizes;
    }
    else {
        // If same number of layers, perform layer-by-layer crossover
        child.layer_sizes.push_back(input_size_);  // Input layer

        // For each hidden layer, randomly choose from either parent
        for (size_t i = 1; i < parent1.layer_sizes.size() - 1; ++i) {
            std::uniform_int_distribution<int> parent_choice(0, 1);
            child.layer_sizes.push_back(parent_choice(random_generator_) == 0
                                            ? parent1.layer_sizes[i]
                                            : parent2.layer_sizes[i]);
        }

        child.layer_sizes.push_back(output_size_);  // Output layer
    }

    // Crossover dropout rate
    std::uniform_int_distribution<int> parent_choice(0, 1);
    child.dropout_rate =
        parent_choice(random_generator_) == 0 ? parent1.dropout_rate : parent2.dropout_rate;

    // Crossover residual connections
    child.has_residual_connections = parent_choice(random_generator_) == 0
                                         ? parent1.has_residual_connections
                                         : parent2.has_residual_connections;

    // Crossover protection level
    child.protection_level =
        parent_choice(random_generator_) == 0 ? parent1.protection_level : parent2.protection_level;

    return child;
}

// Generate all possible configurations for grid search
std::vector<NetworkConfig> AutoArchSearch::generateAllConfigs()
{
    std::vector<NetworkConfig> configs;

    // Define the layer patterns to test based on fixed_hidden_layers_
    std::vector<std::vector<size_t>> layer_patterns;

    if (fixed_hidden_layers_ == 0) {
        // Try different numbers of hidden layers
        // For each width option, create a simple 1-hidden-layer architecture
        for (auto& width : width_options_) {
            layer_patterns.push_back({input_size_, width, output_size_});
        }

        // Add some 2-hidden-layer architectures
        for (auto& width1 : width_options_) {
            for (auto& width2 : width_options_) {
                // Skip if both layers have the same width
                if (width1 != width2) {
                    layer_patterns.push_back({input_size_, width1, width2, output_size_});
                }
            }
        }
    }
    else {
        // Use the fixed number of hidden layers

        // Start with just the input and output sizes
        std::vector<size_t> base_pattern = {input_size_};

        // Generate all combinations of layer sizes for the hidden layers
        std::vector<std::vector<size_t>> hidden_layer_combinations;
        generateLayerSizeCombinations(hidden_layer_combinations, {}, fixed_hidden_layers_);

        // For each combination, create a complete layer pattern
        for (const auto& hidden_layers : hidden_layer_combinations) {
            std::vector<size_t> pattern = base_pattern;
            pattern.insert(pattern.end(), hidden_layers.begin(), hidden_layers.end());
            pattern.push_back(output_size_);
            layer_patterns.push_back(pattern);
        }
    }

    // Generate configurations for each layer pattern
    for (const auto& layer_sizes : layer_patterns) {
        for (auto& dropout : dropout_options_) {
            for (auto& protection : protection_levels_) {
                // Without residual connections
                configs.push_back(NetworkConfig(layer_sizes, dropout, false, protection));

                // With residual connections (if enabled and architecture has 4+ layers)
                if (test_residual_connections_ && layer_sizes.size() >= 4) {
                    configs.push_back(NetworkConfig(layer_sizes, dropout, true, protection));
                }
            }
        }
    }

    return configs;
}

// Helper method to generate layer size combinations recursively
void AutoArchSearch::generateLayerSizeCombinations(std::vector<std::vector<size_t>>& result,
                                                   std::vector<size_t> current,
                                                   size_t layers_remaining)
{
    if (layers_remaining == 0) {
        result.push_back(current);
        return;
    }

    for (auto& width : width_options_) {
        std::vector<size_t> new_current = current;
        new_current.push_back(width);
        generateLayerSizeCombinations(result, new_current, layers_remaining - 1);
    }
}

// Save results to file
void AutoArchSearch::saveResultsToFile() const { exportResults(results_file_); }

// Add setSeed method for deterministic results
void AutoArchSearch::setSeed(unsigned int seed) { random_generator_.seed(seed); }

}  // namespace research
}  // namespace rad_ml
