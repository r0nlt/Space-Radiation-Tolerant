/**
 * @file auto_arch_search.hpp
 * @brief Automatic architecture search for radiation-tolerant neural networks
 *
 * This file defines the AutoArchSearch class that automatically searches for
 * optimal neural network architectures in specific radiation environments.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/research/architecture_tester.hpp>
#include <rad_ml/sim/environment.hpp>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace rad_ml {
namespace research {

/**
 * @brief Configuration of a neural network architecture
 */
struct NetworkConfig {
    std::vector<size_t> layer_sizes;           ///< Sizes of network layers
    double dropout_rate;                       ///< Dropout rate
    bool has_residual_connections;             ///< Whether architecture has residual connections
    neural::ProtectionLevel protection_level;  ///< Protection level

    // Constructor
    NetworkConfig(const std::vector<size_t>& sizes = {}, double dropout = 0.5,
                  bool residual = false,
                  neural::ProtectionLevel protection = neural::ProtectionLevel::NONE)
        : layer_sizes(sizes),
          dropout_rate(dropout),
          has_residual_connections(residual),
          protection_level(protection)
    {
    }

    // Equality operator for configs (needed for sets)
    bool operator==(const NetworkConfig& other) const
    {
        return layer_sizes == other.layer_sizes &&
               std::abs(dropout_rate - other.dropout_rate) < 1e-6 &&
               has_residual_connections == other.has_residual_connections &&
               protection_level == other.protection_level;
    }

    // Less than operator for configs (needed for maps)
    bool operator<(const NetworkConfig& other) const
    {
        if (layer_sizes != other.layer_sizes) {
            return layer_sizes < other.layer_sizes;
        }
        if (std::abs(dropout_rate - other.dropout_rate) >= 1e-6) {
            return dropout_rate < other.dropout_rate;
        }
        if (has_residual_connections != other.has_residual_connections) {
            return !has_residual_connections && other.has_residual_connections;
        }
        return protection_level < other.protection_level;
    }
};

/**
 * @brief Search result containing the best architecture and its performance
 */
struct SearchResult {
    NetworkConfig config;          ///< Best network configuration
    double baseline_accuracy;      ///< Accuracy without radiation
    double radiation_accuracy;     ///< Accuracy under radiation
    double accuracy_preservation;  ///< Preservation percentage
    size_t iterations;             ///< Number of iterations to find

    // Statistical data from Monte Carlo testing
    double baseline_accuracy_stddev;      ///< Standard deviation of baseline accuracy
    double radiation_accuracy_stddev;     ///< Standard deviation of radiation accuracy
    double accuracy_preservation_stddev;  ///< Standard deviation of preservation
    size_t monte_carlo_trials;            ///< Number of Monte Carlo trials

    // Constructor
    SearchResult()
        : baseline_accuracy(0),
          radiation_accuracy(0),
          accuracy_preservation(0),
          iterations(0),
          baseline_accuracy_stddev(0),
          radiation_accuracy_stddev(0),
          accuracy_preservation_stddev(0),
          monte_carlo_trials(1)
    {
    }

    // Constructor with values
    SearchResult(const NetworkConfig& cfg, double baseline, double radiation, double preservation,
                 size_t iters, double baseline_stddev = 0.0, double radiation_stddev = 0.0,
                 double preservation_stddev = 0.0, size_t num_trials = 1)
        : config(cfg),
          baseline_accuracy(baseline),
          radiation_accuracy(radiation),
          accuracy_preservation(preservation),
          iterations(iters),
          baseline_accuracy_stddev(baseline_stddev),
          radiation_accuracy_stddev(radiation_stddev),
          accuracy_preservation_stddev(preservation_stddev),
          monte_carlo_trials(num_trials)
    {
    }
};

/**
 * @brief Class for automatic search of optimal neural network architectures
 *
 * This class implements different search strategies to find optimal
 * neural network architectures under radiation conditions.
 */
class AutoArchSearch {
   public:
    /**
     * @brief Constructor with dataset and search parameters
     *
     * @param train_data Training data vector
     * @param train_labels Training labels vector
     * @param test_data Test data vector
     * @param test_labels Test labels vector
     * @param environment Target radiation environment
     * @param width_options Available layer width options
     * @param dropout_options Available dropout rate options
     * @param results_file File to save results (optional)
     */
    AutoArchSearch(const std::vector<float>& train_data, const std::vector<float>& train_labels,
                   const std::vector<float>& test_data, const std::vector<float>& test_labels,
                   sim::Environment environment,
                   const std::vector<size_t>& width_options = {32, 64, 128, 256},
                   const std::vector<double>& dropout_options = {0.3, 0.4, 0.5, 0.6, 0.7},
                   const std::string& results_file = "auto_search_results.csv");

    /**
     * @brief Find optimal architecture using grid search
     *
     * @param max_epochs Training epochs for each architecture
     * @param use_monte_carlo Whether to use Monte Carlo testing
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Best architecture configuration and performance
     */
    SearchResult findOptimalArchitecture(size_t max_epochs = 50, bool use_monte_carlo = false,
                                         size_t monte_carlo_trials = 50);

    /**
     * @brief Find optimal architecture using random search
     *
     * @param max_iterations Maximum search iterations
     * @param max_epochs Training epochs for each architecture
     * @param use_monte_carlo Whether to use Monte Carlo testing
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Best architecture configuration and performance
     */
    SearchResult randomSearch(size_t max_iterations = 30, size_t max_epochs = 20,
                              bool use_monte_carlo = false, size_t monte_carlo_trials = 50);

    /**
     * @brief Find optimal architecture using evolutionary search
     *
     * @param population_size Size of the population
     * @param generations Number of generations
     * @param mutation_rate Mutation rate
     * @param max_epochs Training epochs for each architecture
     * @param use_monte_carlo Whether to use Monte Carlo testing
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Best architecture configuration and performance
     */
    SearchResult evolutionarySearch(size_t population_size = 10, size_t generations = 10,
                                    double mutation_rate = 0.1, size_t max_epochs = 10,
                                    bool use_monte_carlo = false, size_t monte_carlo_trials = 50);

    /**
     * @brief Set the protection levels to test
     *
     * @param levels Vector of protection levels
     */
    void setProtectionLevels(const std::vector<neural::ProtectionLevel>& levels);

    /**
     * @brief Set whether to test residual connections
     *
     * @param test_residual Whether to test residual connections
     */
    void setTestResidualConnections(bool test_residual);

    /**
     * @brief Set fixed network parameters
     *
     * @param input_size Input layer size
     * @param output_size Output layer size
     * @param hidden_layers Number of hidden layers
     */
    void setFixedParameters(size_t input_size, size_t output_size, size_t hidden_layers = 2);

    /**
     * @brief Export search results to CSV file
     *
     * @param filename Output filename
     */
    void exportResults(const std::string& filename) const;

    /**
     * @brief Set random seed for deterministic results
     *
     * @param seed Random seed value
     */
    void setSeed(unsigned int seed);

    /**
     * @brief Get all tested configurations
     *
     * @return Map of configurations and their results
     */
    const std::map<NetworkConfig, ArchitectureTestResult>& getTestedConfigurations() const;

   private:
    // Dataset fields
    std::vector<float> train_data_;
    std::vector<float> train_labels_;
    std::vector<float> test_data_;
    std::vector<float> test_labels_;

    // Target environment
    sim::Environment environment_;

    // Architecture options
    std::vector<size_t> width_options_;
    std::vector<double> dropout_options_;
    std::vector<neural::ProtectionLevel> protection_levels_;

    // Fixed parameters
    size_t input_size_;
    size_t output_size_;
    size_t fixed_hidden_layers_;

    // Search options
    bool test_residual_connections_;

    // Architecture tester
    std::unique_ptr<ArchitectureTester> tester_;

    // Results storage
    std::map<NetworkConfig, ArchitectureTestResult> tested_configs_;
    std::string results_file_;

    // Random number generator
    std::mt19937 random_generator_;

    /**
     * @brief Test a specific configuration
     *
     * @param config Network configuration to test
     * @param epochs Number of training epochs
     * @param use_monte_carlo Whether to use Monte Carlo testing
     * @param monte_carlo_trials Number of Monte Carlo trials
     * @return Test result
     */
    ArchitectureTestResult testConfiguration(const NetworkConfig& config, size_t epochs,
                                             bool use_monte_carlo = false,
                                             size_t monte_carlo_trials = 50);

    /**
     * @brief Generate a random architecture configuration
     *
     * @return Random configuration
     */
    NetworkConfig generateRandomConfig();

    /**
     * @brief Mutate an existing architecture configuration
     *
     * @param config Original configuration
     * @param mutation_rate Mutation rate
     * @return Mutated configuration
     */
    NetworkConfig mutateConfig(const NetworkConfig& config, double mutation_rate);

    /**
     * @brief Crossover two configurations to create a new one
     *
     * @param parent1 First parent configuration
     * @param parent2 Second parent configuration
     * @return Child configuration
     */
    NetworkConfig crossoverConfigs(const NetworkConfig& parent1, const NetworkConfig& parent2);

    /**
     * @brief Generate all possible configs for grid search
     *
     * @return Vector of all configurations to test
     */
    std::vector<NetworkConfig> generateAllConfigs();

    /**
     * @brief Helper method to generate layer size combinations recursively
     *
     * @param result Vector to store all combinations
     * @param current Current combination
     * @param layers_remaining Number of layers left to add
     */
    void generateLayerSizeCombinations(std::vector<std::vector<size_t>>& result,
                                       std::vector<size_t> current, size_t layers_remaining);

    /**
     * @brief Save results to file
     */
    void saveResultsToFile() const;
};

}  // namespace research
}  // namespace rad_ml
