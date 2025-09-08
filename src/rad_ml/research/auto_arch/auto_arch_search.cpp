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
#include <rad_ml/research/auto_arch/genetic_operators.hpp>
#include <rad_ml/research/auto_arch_search.hpp>
#include <set>
#include <tuple>

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

// Implementation moved to auto_arch/grid.cpp

// Implementation moved to auto_arch/random.cpp

// Implementation moved to auto_arch/evolutionary.cpp

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

// ExportResults moved to auto_arch/io.cpp

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
