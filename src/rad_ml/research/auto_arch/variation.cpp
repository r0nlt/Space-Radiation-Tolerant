/**
 * @file variation.cpp
 * @brief Mutation and crossover implementations for AutoArchSearch
 */

#include <algorithm>
#include <rad_ml/research/auto_arch_search.hpp>
#include <random>

namespace rad_ml {
namespace research {

NetworkConfig AutoArchSearch::mutateConfig(const NetworkConfig& config, double mutation_rate)
{
    if (adaptive_controller_ && adaptive_mutation_enabled_) {
        return adaptive_controller_->adaptiveMutate(config, mutation_rate);
    }
    return mutateConfigBasic(config, mutation_rate);
}

NetworkConfig AutoArchSearch::mutateConfigBasic(const NetworkConfig& config, double mutation_rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    if (mutation_dist(random_generator_) < mutation_rate) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(random_generator_);
            std::uniform_int_distribution<size_t> width_idx_dist(0, width_options_.size() - 1);
            mutated.layer_sizes[layer_idx] = width_options_[width_idx_dist(random_generator_)];
        }
    }

    if (mutation_dist(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> dropout_idx_dist(0, dropout_options_.size() - 1);
        mutated.dropout_rate = dropout_options_[dropout_idx_dist(random_generator_)];
    }

    if (test_residual_connections_ && mutation_dist(random_generator_) < mutation_rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    if (mutation_dist(random_generator_) < mutation_rate) {
        std::uniform_int_distribution<size_t> protection_idx_dist(0, protection_levels_.size() - 1);
        mutated.protection_level = protection_levels_[protection_idx_dist(random_generator_)];
    }

    return mutated;
}

NetworkConfig AutoArchSearch::crossoverConfigs(const NetworkConfig& parent1,
                                               const NetworkConfig& parent2)
{
    NetworkConfig child;

    if (parent1.layer_sizes.size() != parent2.layer_sizes.size()) {
        std::uniform_int_distribution<int> parent_choice(0, 1);
        child.layer_sizes =
            parent_choice(random_generator_) == 0 ? parent1.layer_sizes : parent2.layer_sizes;
    }
    else {
        child.layer_sizes.push_back(input_size_);
        for (size_t i = 1; i < parent1.layer_sizes.size() - 1; ++i) {
            std::uniform_int_distribution<int> parent_choice(0, 1);
            child.layer_sizes.push_back(parent_choice(random_generator_) == 0
                                            ? parent1.layer_sizes[i]
                                            : parent2.layer_sizes[i]);
        }
        child.layer_sizes.push_back(output_size_);
    }

    std::uniform_int_distribution<int> parent_choice(0, 1);
    child.dropout_rate =
        parent_choice(random_generator_) == 0 ? parent1.dropout_rate : parent2.dropout_rate;
    child.has_residual_connections = parent_choice(random_generator_) == 0
                                         ? parent1.has_residual_connections
                                         : parent2.has_residual_connections;
    child.protection_level =
        parent_choice(random_generator_) == 0 ? parent1.protection_level : parent2.protection_level;

    return child;
}

}  // namespace research
}  // namespace rad_ml
