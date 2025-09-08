/**
 * @file grid.cpp
 * @brief Grid search implementation for AutoArchSearch
 */

#include <iostream>
#include <numeric>
#include <rad_ml/research/auto_arch_search.hpp>

namespace rad_ml {
namespace research {

SearchResult AutoArchSearch::findOptimalArchitecture(size_t max_epochs, bool use_monte_carlo,
                                                     size_t monte_carlo_trials)
{
    std::cout << "Starting grid search for optimal architecture..." << std::endl;
    if (use_monte_carlo) {
        std::cout << "Using Monte Carlo testing with " << monte_carlo_trials
                  << " trials per configuration" << std::endl;
    }

    auto configs = generateAllConfigs();
    std::cout << "Testing " << configs.size() << " configurations" << std::endl;

    size_t iteration = 0;
    double best_preservation = 0.0;
    NetworkConfig best_config;
    ArchitectureTestResult best_result;

    for (const auto& config : configs) {
        auto result = testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);
        tested_configs_[config] = result;

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

        ++iteration;
        if (iteration % 10 == 0) {
            saveResultsToFile();
        }
    }

    saveResultsToFile();
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, configs.size(),
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

}  // namespace research
}  // namespace rad_ml
