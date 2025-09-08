/**
 * @file random.cpp
 * @brief Random search implementation for AutoArchSearch
 */

#include <iostream>
#include <rad_ml/research/auto_arch_search.hpp>

namespace rad_ml {
namespace research {

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

    size_t completed = 0;
    size_t attempts = 0;
    const size_t max_attempts = max_iterations * 5;  // guard against infinite retries
    while (completed < max_iterations && attempts < max_attempts) {
        ++attempts;
        auto config = generateRandomConfig();
        if (tested_configs_.count(config) > 0) {
            continue;  // skip duplicate without decrementing progress
        }

        auto result = testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);
        tested_configs_[config] = result;

        if (result.accuracy_preservation > best_preservation) {
            best_preservation = result.accuracy_preservation;
            best_config = config;
            best_result = result;

            std::cout << "New best configuration found (iteration " << completed
                      << "):" << std::endl;
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

        ++completed;
        if (save_interval_iterations_ > 0 && completed % save_interval_iterations_ == 0) {
            saveResultsToFile();
        }
    }

    saveResultsToFile();
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, completed,
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

}  // namespace research
}  // namespace rad_ml
