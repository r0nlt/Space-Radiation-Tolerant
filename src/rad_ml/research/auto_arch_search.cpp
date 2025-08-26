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
      fixed_hidden_layers_(0),
      adaptive_mutation_enabled_(false),
      adaptive_base_rate_(0.1),
      diversity_threshold_(0.3),
      adaptive_max_rate_(0.5),
      adaptive_min_rate_(0.01)
{
    // Initialize random generator with time-based seed
    std::random_device rd;
    random_generator_ = std::mt19937(rd());

    // Determine input and output sizes from data
    // Assume input_size is the size of one training example
    // and output_size is the size of one label
    if (!train_labels.empty()) {
        input_size_ = train_data.size() / train_labels.size();
    }
    else {
        input_size_ = 8;  // Default input size for testing
    }
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

    // Initialize the advanced adaptive mutation controller
    adaptive_controller_ = std::make_unique<AdaptiveMutationController>(this, random_generator_);

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

        // Calculate adaptive mutation rate for this generation
        double current_mutation_rate =
            calculateAdaptiveMutationRate(population, fitness, gen, generations);

        if (adaptive_mutation_enabled_) {
            std::cout << "  Diversity: " << std::fixed << std::setprecision(3)
                      << calculatePopulationDiversity(population)
                      << ", Adaptive mutation rate: " << std::fixed << std::setprecision(3)
                      << current_mutation_rate << std::endl;
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

            // Mutation with adaptive rate
            child = mutateConfig(child, current_mutation_rate);

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
    // Use the advanced adaptive mutation controller if available
    if (adaptive_controller_ && adaptive_mutation_enabled_) {
        // Create a version of the controller that can access our data
        // For now, we'll use the default implementation
        // TODO: Create a proper interface to pass width_options_, dropout_options_, etc.
        return adaptive_controller_->adaptiveMutate(config, mutation_rate);
    }

    // Fall back to the original implementation if adaptive controller is not available
    // or if adaptive mutation is disabled
    return mutateConfigBasic(config, mutation_rate);
}

NetworkConfig AutoArchSearch::mutateConfigBasic(const NetworkConfig& config, double mutation_rate)
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

// Set adaptive mutation parameters
void AutoArchSearch::setAdaptiveMutation(bool enable, double base_rate, double diversity_threshold,
                                         double max_rate, double min_rate)
{
    adaptive_mutation_enabled_ = enable;
    adaptive_base_rate_ = base_rate;
    diversity_threshold_ = diversity_threshold;
    adaptive_max_rate_ = max_rate;
    adaptive_min_rate_ = min_rate;

    if (enable) {
        std::cout << "Adaptive mutation enabled with:" << std::endl;
        std::cout << "  Base rate: " << base_rate << std::endl;
        std::cout << "  Diversity threshold: " << diversity_threshold << std::endl;
        std::cout << "  Max rate: " << max_rate << std::endl;
        std::cout << "  Min rate: " << min_rate << std::endl;
    }
}

// Calculate population diversity based on configuration differences
double AutoArchSearch::calculatePopulationDiversity(
    const std::vector<NetworkConfig>& population) const
{
    // Clean implementation without debug output

    if (population.size() <= 1) {
        // No diversity in population of size 1 or less
        return 0.0;  // No diversity in population of size 1 or less
    }

    double total_distance = 0.0;
    size_t pair_count = 0;

    // Calculate average distance between all pairs
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t j = i + 1; j < population.size(); ++j) {
            double dist = calculateConfigDistance(population[i], population[j]);
            total_distance += dist;
            pair_count++;
        }
    }

    if (pair_count == 0) {
        return 0.0;
    }

    // Normalize by maximum possible distance and population size
    double avg_distance = total_distance / pair_count;

    // Calculate diversity as a normalized score
    // Maximum possible distance is roughly 4.0 (max differences across all parameters)
    const double max_possible_distance = 4.0;

    if (max_possible_distance <= 0) {
        return 0.0;
    }

    double normalized_diversity = avg_distance / max_possible_distance;

    // Ensure diversity is in [0, 1] range
    return std::max(0.0, std::min(1.0, normalized_diversity));
}

// Calculate configuration distance between two network configs
double AutoArchSearch::calculateConfigDistance(const NetworkConfig& config1,
                                               const NetworkConfig& config2) const
{
    double distance = 0.0;

    // Architecture distance - compare layer sizes
    const auto& layers1 = config1.layer_sizes;
    const auto& layers2 = config2.layer_sizes;

    // Handle different architecture depths
    size_t max_layers = std::max(layers1.size(), layers2.size());
    size_t min_layers = std::min(layers1.size(), layers2.size());

    // Compare common layers
    if (!width_options_.empty()) {
        double max_width = *std::max_element(width_options_.begin(), width_options_.end());
        for (size_t i = 0; i < min_layers; ++i) {
            if (i < layers1.size() && i < layers2.size() && max_width > 0) {
                // Normalize layer size difference by maximum possible width
                double width_diff =
                    std::abs(static_cast<double>(layers1[i]) - static_cast<double>(layers2[i]));
                double layer_distance = width_diff / max_width;
                distance += layer_distance;
            }
        }
    }

    // Penalize different number of layers
    if (layers1.size() != layers2.size()) {
        distance +=
            std::abs(static_cast<double>(layers1.size()) - static_cast<double>(layers2.size()));
    }

    // Dropout rate distance
    if (!dropout_options_.empty()) {
        double dropout_range = dropout_options_.back() - dropout_options_.front();
        if (dropout_range > 0) {
            double dropout_diff = std::abs(config1.dropout_rate - config2.dropout_rate);
            double dropout_distance = dropout_diff / dropout_range;
            distance += dropout_distance;
        }
    }

    // Residual connections difference
    if (config1.has_residual_connections != config2.has_residual_connections) {
        distance += 0.5;  // Binary difference
    }

    // Protection level difference
    if (config1.protection_level != config2.protection_level) {
        distance += 0.25;  // Categorical difference
    }

    return distance;
}

// Calculate adaptive mutation rate based on population diversity
double AutoArchSearch::calculateAdaptiveMutationRate(const std::vector<NetworkConfig>& population,
                                                     const std::vector<double>& fitness,
                                                     size_t generation,
                                                     size_t total_generations) const
{
    if (!adaptive_mutation_enabled_ || population.empty()) {
        return adaptive_base_rate_;
    }

    double diversity = calculatePopulationDiversity(population);

    // Calculate fitness variance to detect convergence
    double fitness_mean = 0.0;
    double fitness_variance = 0.0;

    if (!fitness.empty()) {
        fitness_mean = std::accumulate(fitness.begin(), fitness.end(), 0.0) / fitness.size();
        fitness_variance = std::accumulate(fitness.begin(), fitness.end(), 0.0,
                                           [fitness_mean](double acc, double f) {
                                               return acc + (f - fitness_mean) * (f - fitness_mean);
                                           }) /
                           fitness.size();
    }

    // Adaptive mutation rate calculation
    double adaptive_rate = adaptive_base_rate_;

    // If diversity is low, increase mutation rate to explore more
    if (diversity < diversity_threshold_) {
        double diversity_factor = (diversity_threshold_ - diversity) / diversity_threshold_;
        adaptive_rate =
            adaptive_base_rate_ + diversity_factor * (adaptive_max_rate_ - adaptive_base_rate_);
    }
    // If diversity is high, slightly decrease mutation rate to exploit good solutions
    else if (diversity > diversity_threshold_ * 1.5) {
        double diversity_factor = (diversity - diversity_threshold_ * 1.5) / diversity_threshold_;
        adaptive_rate = adaptive_base_rate_ * (1.0 - diversity_factor * 0.3);
    }

    // If fitness variance is low (population converged), increase mutation
    if (fitness_variance < 10.0 && !fitness.empty()) {
        double convergence_factor = (10.0 - fitness_variance) / 10.0;
        adaptive_rate += convergence_factor * (adaptive_max_rate_ - adaptive_base_rate_) * 0.5;
    }

    // Progressive adjustment based on generation
    double generation_factor = static_cast<double>(generation) / total_generations;
    if (generation_factor > 0.7) {  // In later generations, increase exploration
        adaptive_rate +=
            (generation_factor - 0.7) * (adaptive_max_rate_ - adaptive_base_rate_) * 0.3;
    }

    // Ensure rate stays within bounds
    adaptive_rate = std::max(adaptive_min_rate_, std::min(adaptive_max_rate_, adaptive_rate));

    return adaptive_rate;
}

//==============================================================================
// ADAPTIVE MUTATION CONTROLLER IMPLEMENTATION
//==============================================================================

AdaptiveMutationController::AdaptiveMutationController(AutoArchSearch* parent, std::mt19937& rng)
    : parent_(parent), random_generator_(&rng)
{
    // Initialize with default operators
    initializeDefaultOperators();
}

void AdaptiveMutationController::initializeDefaultOperators()
{
    // For now, let's use a simpler approach with static functions
    // We'll create a more sophisticated approach later if needed

    // Store function pointers to member functions
    using MutationFunc =
        NetworkConfig (AdaptiveMutationController::*)(const NetworkConfig&, double);

    // Create lambda wrappers that call the member functions
    auto create_wrapper = [this](MutationFunc func) {
        return [this, func](const NetworkConfig& config, double rate) -> NetworkConfig {
            return (this->*func)(config, rate);
        };
    };

    // Add the operators
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateArchitectureFocused),
                        "Architecture");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateParameterFocused),
                        "Parameters");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateProtectionFocused),
                        "Protection");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateBalanced), "Balanced");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateAggressive),
                        "Aggressive");
}

NetworkConfig AdaptiveMutationController::mutateArchitectureFocused(const NetworkConfig& config,
                                                                    double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Higher probability for layer size mutations
    if (mutation_dist(*random_generator_) < rate * 1.5) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    // Lower probability for structural changes
    if (mutation_dist(*random_generator_) < rate * 0.7) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateParameterFocused(const NetworkConfig& config,
                                                                 double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Higher probability for dropout rate mutations
    if (mutation_dist(*random_generator_) < rate * 1.5) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    // Lower probability for other parameters
    if (mutation_dist(*random_generator_) < rate * 0.5) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateProtectionFocused(const NetworkConfig& config,
                                                                  double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Higher probability for protection level mutations
    if (mutation_dist(*random_generator_) < rate * 1.5) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    // Lower probability for other changes
    if (mutation_dist(*random_generator_) < rate * 0.5) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateBalanced(const NetworkConfig& config, double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Equal probability for all genes
    if (mutation_dist(*random_generator_) < rate) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateAggressive(const NetworkConfig& config, double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    // Very high probability for multiple mutations
    double aggressive_rate = rate * 2.0;

    // Try to mutate all genes with high probability
    if (mutation_dist(*random_generator_) < aggressive_rate) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    return mutated;
}

void AdaptiveMutationController::addMutationOperator(
    std::function<NetworkConfig(const NetworkConfig&, double)> op, const std::string& name)
{
    mutation_operators_.emplace_back(op, name);
    updateProbabilities();
}

NetworkConfig AdaptiveMutationController::adaptiveMutate(const NetworkConfig& config,
                                                         double base_rate)
{
    if (mutation_operators_.empty()) {
        return config;  // No mutation if no operators
    }

    size_t selected_operator = selectOperatorDynamically();
    NetworkConfig mutated = mutation_operators_[selected_operator].operator_func(config, base_rate);

    // Track operator usage
    mutation_operators_[selected_operator].applications++;

    return mutated;
}

void AdaptiveMutationController::updateOperatorCredits(
    const std::vector<double>& improvement_scores, const std::vector<size_t>& used_operators)
{
    if (improvement_scores.size() != used_operators.size()) {
        return;  // Invalid input
    }

    for (size_t i = 0; i < improvement_scores.size(); ++i) {
        size_t op_idx = used_operators[i];
        if (op_idx < mutation_operators_.size()) {
            double improvement = improvement_scores[i];
            mutation_operators_[op_idx].total_improvement += improvement;

            // Update success rate (simple moving average)
            double success = (improvement > 0.0) ? 1.0 : 0.0;
            mutation_operators_[op_idx].success_rate =
                0.9 * mutation_operators_[op_idx].success_rate + 0.1 * success;

            // Update credit score based on improvement
            double avg_improvement = mutation_operators_[op_idx].applications > 0
                                         ? mutation_operators_[op_idx].total_improvement /
                                               mutation_operators_[op_idx].applications
                                         : 0.0;

            mutation_operators_[op_idx].credit_score =
                learning_rate_ * avg_improvement +
                (1.0 - learning_rate_) * mutation_operators_[op_idx].credit_score;
        }
    }

    updateProbabilities();
}

std::vector<std::tuple<std::string, int, double, double>>
AdaptiveMutationController::getOperatorStatistics() const
{
    std::vector<std::tuple<std::string, int, double, double>> stats;
    for (const auto& op : mutation_operators_) {
        stats.emplace_back(op.name, op.applications, op.success_rate, op.credit_score);
    }
    return stats;
}

void AdaptiveMutationController::resetStatistics()
{
    for (auto& op : mutation_operators_) {
        op.success_rate = 0.5;
        op.credit_score = 0.0;
        op.applications = 0;
        op.total_improvement = 0.0;
    }
    updateProbabilities();
}

size_t AdaptiveMutationController::selectOperatorDynamically()
{
    if (mutation_operators_.empty()) {
        return 0;
    }

    std::uniform_real_distribution<double> explore_dist(0.0, 1.0);

    // Epsilon-greedy exploration
    if (explore_dist(*random_generator_) < exploration_factor_) {
        // Random exploration
        std::uniform_int_distribution<size_t> op_dist(0, mutation_operators_.size() - 1);
        return op_dist(*random_generator_);
    }
    else {
        // Exploitation - choose best operator
        if (!operator_probabilities_.empty()) {
            std::discrete_distribution<size_t> prob_dist(operator_probabilities_.begin(),
                                                         operator_probabilities_.end());
            return prob_dist(*random_generator_);
        }
        else {
            // Fallback to uniform selection
            std::uniform_int_distribution<size_t> op_dist(0, mutation_operators_.size() - 1);
            return op_dist(*random_generator_);
        }
    }
}

void AdaptiveMutationController::updateProbabilities()
{
    if (mutation_operators_.empty()) {
        operator_probabilities_.clear();
        return;
    }

    std::vector<double> credit_scores;
    credit_scores.reserve(mutation_operators_.size());

    for (const auto& op : mutation_operators_) {
        credit_scores.push_back(op.credit_score);
    }

    operator_probabilities_ = softmax(credit_scores);
}

std::vector<double> AdaptiveMutationController::softmax(
    const std::vector<double>& credit_scores) const
{
    if (credit_scores.empty()) {
        return {};
    }

    // Find maximum value for numerical stability
    double max_val = *std::max_element(credit_scores.begin(), credit_scores.end());

    // Compute exponentials
    std::vector<double> exp_scores;
    exp_scores.reserve(credit_scores.size());
    double sum_exp = 0.0;

    for (double score : credit_scores) {
        double exp_score = std::exp(score - max_val);  // Subtract max for stability
        exp_scores.push_back(exp_score);
        sum_exp += exp_score;
    }

    // Normalize to probabilities
    std::vector<double> probabilities;
    probabilities.reserve(exp_scores.size());

    for (double exp_score : exp_scores) {
        probabilities.push_back(exp_score / sum_exp);
    }

    return probabilities;
}

//==============================================================================
// ADAPTIVE MUTATION CONTROLLER HELPER METHODS
//==============================================================================

// Note: These methods will need to be updated to access AutoArchSearch data
size_t AdaptiveMutationController::getRandomLayerSize()
{
    if (parent_ && !parent_->width_options_.empty()) {
        std::uniform_int_distribution<size_t> width_idx_dist(0, parent_->width_options_.size() - 1);
        return parent_->width_options_[width_idx_dist(*random_generator_)];
    }
    // Fallback to default
    std::uniform_int_distribution<size_t> size_dist(32, 256);
    return size_dist(*random_generator_);
}

double AdaptiveMutationController::getRandomDropoutRate()
{
    if (parent_ && !parent_->dropout_options_.empty()) {
        std::uniform_int_distribution<size_t> dropout_idx_dist(
            0, parent_->dropout_options_.size() - 1);
        return parent_->dropout_options_[dropout_idx_dist(*random_generator_)];
    }
    // Fallback to default
    std::uniform_real_distribution<double> dropout_dist(0.1, 0.6);
    return dropout_dist(*random_generator_);
}

neural::ProtectionLevel AdaptiveMutationController::getRandomProtectionLevel()
{
    if (parent_ && !parent_->protection_levels_.empty()) {
        std::uniform_int_distribution<size_t> protection_idx_dist(
            0, parent_->protection_levels_.size() - 1);
        return parent_->protection_levels_[protection_idx_dist(*random_generator_)];
    }
    // Fallback to default
    std::uniform_int_distribution<int> prot_dist(0, 4);
    return static_cast<neural::ProtectionLevel>(prot_dist(*random_generator_));
}

//==============================================================================
// AUTO ARCH SEARCH CLASS METHODS FOR ADAPTIVE CONTROLLER
//==============================================================================

std::vector<std::tuple<std::string, int, double, double>>
AutoArchSearch::getMutationOperatorStatistics() const
{
    if (adaptive_controller_) {
        return adaptive_controller_->getOperatorStatistics();
    }
    return {};
}

void AutoArchSearch::resetMutationOperatorStatistics()
{
    if (adaptive_controller_) {
        adaptive_controller_->resetStatistics();
    }
}

}  // namespace research
}  // namespace rad_ml
