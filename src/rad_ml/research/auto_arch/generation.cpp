/**
 * @file generation.cpp
 * @brief Configuration generation utilities for AutoArchSearch
 */

#include <random>
#include <vector>

#include <rad_ml/research/auto_arch_search.hpp>

namespace rad_ml {
namespace research {

NetworkConfig AutoArchSearch::generateRandomConfig()
{
    size_t num_hidden_layers;
    if (fixed_hidden_layers_ > 0) {
        num_hidden_layers = fixed_hidden_layers_;
    }
    else {
        std::uniform_int_distribution<size_t> layers_dist(1, 3);
        num_hidden_layers = layers_dist(random_generator_);
    }

    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(input_size_);

    std::uniform_int_distribution<size_t> width_idx_dist(0, width_options_.size() - 1);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layer_sizes.push_back(width_options_[width_idx_dist(random_generator_)]);
    }

    layer_sizes.push_back(output_size_);

    std::uniform_int_distribution<size_t> dropout_idx_dist(0, dropout_options_.size() - 1);
    double dropout_rate = dropout_options_[dropout_idx_dist(random_generator_)];

    bool use_residual = false;
    if (test_residual_connections_) {
        std::uniform_int_distribution<int> residual_dist(0, 1);
        use_residual = residual_dist(random_generator_) > 0;
    }

    std::uniform_int_distribution<size_t> protection_idx_dist(0, protection_levels_.size() - 1);
    auto protection_level = protection_levels_[protection_idx_dist(random_generator_)];

    return NetworkConfig(layer_sizes, dropout_rate, use_residual, protection_level);
}

std::vector<NetworkConfig> AutoArchSearch::generateAllConfigs()
{
    std::vector<NetworkConfig> configs;

    std::vector<std::vector<size_t>> layer_patterns;

    if (fixed_hidden_layers_ == 0) {
        for (auto& width : width_options_) {
            layer_patterns.push_back({input_size_, width, output_size_});
        }
        for (auto& width1 : width_options_) {
            for (auto& width2 : width_options_) {
                if (width1 != width2) {
                    layer_patterns.push_back({input_size_, width1, width2, output_size_});
                }
            }
        }
    }
    else {
        std::vector<size_t> base_pattern = {input_size_};
        std::vector<std::vector<size_t>> hidden_layer_combinations;
        generateLayerSizeCombinations(hidden_layer_combinations, {}, fixed_hidden_layers_);
        for (const auto& hidden_layers : hidden_layer_combinations) {
            std::vector<size_t> pattern = base_pattern;
            pattern.insert(pattern.end(), hidden_layers.begin(), hidden_layers.end());
            pattern.push_back(output_size_);
            layer_patterns.push_back(pattern);
        }
    }

    for (const auto& layer_sizes : layer_patterns) {
        for (auto& dropout : dropout_options_) {
            for (auto& protection : protection_levels_) {
                configs.push_back(NetworkConfig(layer_sizes, dropout, false, protection));
                if (test_residual_connections_ && layer_sizes.size() >= 4) {
                    configs.push_back(NetworkConfig(layer_sizes, dropout, true, protection));
                }
            }
        }
    }

    return configs;
}

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

}  // namespace research
}  // namespace rad_ml


