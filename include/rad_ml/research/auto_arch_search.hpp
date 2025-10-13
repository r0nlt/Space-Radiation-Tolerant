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
#include <optional>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/research/architecture_tester.hpp>
#include <rad_ml/research/auto_arch/advanced_quality_diversity.hpp>
#include <rad_ml/research/auto_arch/genetic_operators.hpp>
#include <rad_ml/research/auto_arch/quality_diversity.hpp>
#include <rad_ml/research/auto_arch/types.hpp>
#include <rad_ml/sim/environment.hpp>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace rad_ml {
namespace research {

// Forward declarations now in headers included above

// NetworkConfig, SearchResult, and AdaptiveMutationController are provided by
// headers in rad_ml/research/auto_arch/*

/**
 * @brief Class for automatic search of optimal neural network architectures
 *
 * This class implements different search strategies to find optimal
 * neural network architectures under radiation conditions.
 */
class AutoArchSearch {
   public:
    /**
     * @brief Crossover strategy options
     */
    enum class CrossoverStrategy { UNIFORM, SINGLE_POINT };
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
     * @brief Enable adaptive mutation rates based on population diversity
     *
     * @param enable Whether to enable adaptive mutation
     * @param base_rate Base mutation rate (default 0.1)
     * @param diversity_threshold Threshold for triggering rate adjustment (default 0.3)
     * @param max_rate Maximum mutation rate (default 0.5)
     * @param min_rate Minimum mutation rate (default 0.01)
     */
    void setAdaptiveMutation(bool enable, double base_rate = 0.1, double diversity_threshold = 0.3,
                             double max_rate = 0.5, double min_rate = 0.01);

    /**
     * @brief Configure crossover behavior
     *
     * @param rate Probability of applying crossover when producing offspring
     * @param strategy Strategy to use when crossing architecture genes
     */
    void setCrossoverSettings(double rate = 0.8,
                              CrossoverStrategy strategy = CrossoverStrategy::UNIFORM)
    {
        crossover_rate_ = std::max(0.0, std::min(1.0, rate));
        crossover_strategy_ = strategy;
    }

    /**
     * @brief Enable random immigrants injection to preserve diversity
     *
     * @param enable Toggle injection
     * @param fraction Fraction of population to replace when diversity collapses
     */
    void setRandomImmigrants(bool enable, double fraction = 0.1)
    {
        random_immigrants_enabled_ = enable;
        random_immigrants_fraction_ = std::max(0.0, std::min(1.0, fraction));
    }

    /**
     * @brief Enable genetics metrics CSV logging and set output filename
     */
    void setGeneticsMetricsFile(const std::string& filename)
    {
        // If a bare filename is provided, place it under results/genetic_algorithm/
        if (filename.find('/') == std::string::npos && filename.find('\\') == std::string::npos) {
            genetics_metrics_file_ = std::string("results/genetic_algorithm/") + filename;
        }
        else {
            genetics_metrics_file_ = filename;
        }
        genetics_metrics_enabled_ = !genetics_metrics_file_.empty();
        genetics_metrics_header_written_ = false;
    }

    /**
     * @brief Configure save intervals for results persistence
     *
     * @param generations Interval (in generations) to save during evolutionary search (0 = never)
     * @param iterations Interval (in iterations) to save during grid/random search (0 = never)
     */
    void setSaveIntervals(size_t generations, size_t iterations)
    {
        save_interval_generations_ = generations;
        save_interval_iterations_ = iterations;
    }

    /**
     * @brief Decouple policy: compute mutation rate only every K generations (0 = every gen)
     */
    void setMutationRateSchedule(size_t schedule_interval)
    {
        mutation_rate_schedule_interval_ = schedule_interval;
    }

    /**
     * @brief Decouple policy: freeze mutation rate after this generation index (SIZE_MAX = never)
     */
    void setMutationRateFreezeGeneration(size_t freeze_after_gen)
    {
        mutation_rate_freeze_after_gen_ = freeze_after_gen;
    }

    /**
     * @brief Enable or disable Quality-Diversity (QD) assistance
     */
    void enableQualityDiversity(bool enable) { qd_enabled_ = enable; }

    /**
     * @brief Enable or disable Advanced Quality-Diversity (MAP-Elites + Novelty)
     */
    void enableAdvancedQualityDiversity(bool enable) { advanced_qd_enabled_ = enable; }

    /**
     * @brief Get all tested configurations
     *
     * @return Map of configurations and their results
     */
    const std::map<NetworkConfig, ArchitectureTestResult>& getTestedConfigurations() const;

    // Public access to core adaptive functions for testing
    double calculatePopulationDiversity_PUBLIC(const std::vector<NetworkConfig>& population) const
    {
        return calculatePopulationDiversity(population);
    }
    double calculateAdaptiveMutationRate_PUBLIC(const std::vector<NetworkConfig>& population,
                                                const std::vector<double>& fitness,
                                                size_t generation, size_t total_generations) const
    {
        return calculateAdaptiveMutationRate(population, fitness, generation, total_generations);
    }
    double calculateConfigDistance_PUBLIC(const NetworkConfig& config1,
                                          const NetworkConfig& config2) const
    {
        return calculateConfigDistance(config1, config2);
    }

    // Public wrappers for genetic operators for testing
    NetworkConfig mutateConfig_PUBLIC(const NetworkConfig& config, double mutation_rate)
    {
        return mutateConfig(config, mutation_rate);
    }
    NetworkConfig crossoverConfigs_PUBLIC(const NetworkConfig& parent1,
                                          const NetworkConfig& parent2)
    {
        return crossoverConfigs(parent1, parent2);
    }

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

    // Adaptive mutation settings
    bool adaptive_mutation_enabled_;
    double adaptive_base_rate_;
    double diversity_threshold_;
    double adaptive_max_rate_;
    double adaptive_min_rate_;

    // Architecture tester
    std::unique_ptr<ArchitectureTester> tester_;

    // Results storage
    std::map<NetworkConfig, ArchitectureTestResult> tested_configs_;
    std::string results_file_;

    // Random number generator
    std::mt19937 random_generator_;

    // Advanced adaptive mutation controller
    std::unique_ptr<AdaptiveMutationController> adaptive_controller_;
    std::unique_ptr<QualityDiversityManager> qd_manager_;
    std::unique_ptr<AdvancedQualityDiversityManager> advanced_qd_manager_;

    // Allow AdaptiveMutationController to access private members
    friend class AdaptiveMutationController;

    // Decoupling controls
    size_t mutation_rate_schedule_interval_ = 0;        // 0 = compute every generation
    size_t mutation_rate_freeze_after_gen_ = SIZE_MAX;  // freeze never by default
    std::optional<double> last_computed_mutation_rate_ = std::nullopt;  // cached rate

    // Persistence controls
    size_t save_interval_generations_ = 2;  // Save every N generations in evolutionary search
    size_t save_interval_iterations_ = 10;  // Save every N iterations in grid/random search

    // Quality Diversity controls
    bool qd_enabled_ = false;
    bool advanced_qd_enabled_ = false;

    // Crossover controls
    double crossover_rate_ = 0.8;
    CrossoverStrategy crossover_strategy_ = CrossoverStrategy::UNIFORM;

    // Random immigrants controls
    bool random_immigrants_enabled_ = false;
    double random_immigrants_fraction_ = 0.1;

    // Genetics metrics logging
    bool genetics_metrics_enabled_ = false;
    bool genetics_metrics_header_written_ = false;
    std::string genetics_metrics_file_;

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
     * @brief Basic mutation implementation (fallback when adaptive controller is disabled)
     *
     * @param config Original configuration
     * @param mutation_rate Mutation rate
     * @return Mutated configuration
     */
    NetworkConfig mutateConfigBasic(const NetworkConfig& config, double mutation_rate);

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
     * @brief Calculate population diversity based on configuration differences
     *
     * @param population Current population
     * @return Diversity score between 0.0 (identical) and 1.0 (maximally diverse)
     */
    double calculatePopulationDiversity(const std::vector<NetworkConfig>& population) const;

    /**
     * @brief Calculate adaptive mutation rate based on population diversity
     *
     * @param population Current population
     * @param fitness Current fitness values
     * @param generation Current generation number
     * @param total_generations Total number of generations
     * @return Adaptive mutation rate
     */
    double calculateAdaptiveMutationRate(const std::vector<NetworkConfig>& population,
                                         const std::vector<double>& fitness, size_t generation,
                                         size_t total_generations) const;

    /**
     * @brief Calculate configuration distance between two network configs
     *
     * @param config1 First configuration
     * @param config2 Second configuration
     * @return Distance score (0.0 = identical, higher = more different)
     */
    double calculateConfigDistance(const NetworkConfig& config1,
                                   const NetworkConfig& config2) const;

    /**
     * @brief Get adaptive mutation operator statistics
     * @return Vector of operator statistics (name, applications, success_rate, credit_score)
     */
    std::vector<std::tuple<std::string, int, double, double>> getMutationOperatorStatistics() const;

    /**
     * @brief Reset adaptive mutation controller statistics
     */
    void resetMutationOperatorStatistics();

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
