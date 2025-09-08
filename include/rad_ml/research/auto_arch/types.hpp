/**
 * @file types.hpp
 * @brief Common types for Auto Architecture Search
 */

#pragma once

#include <cmath>
#include <rad_ml/neural/protected_neural_network.hpp>
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

}  // namespace research
}  // namespace rad_ml
