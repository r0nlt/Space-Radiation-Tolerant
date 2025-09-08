/**
 * @file evolutionary.cpp
 * @brief Evolutionary (genetic) search implementation for AutoArchSearch
 */

#include <iostream>
#include <numeric>
#include <rad_ml/research/auto_arch/genetic_operators.hpp>
#include <rad_ml/research/auto_arch_search.hpp>

namespace rad_ml {
namespace research {

SearchResult AutoArchSearch::evolutionarySearch(size_t population_size, size_t generations,
                                                double mutation_rate, size_t max_epochs,
                                                bool use_monte_carlo, size_t monte_carlo_trials)
{
    std::cout << "Starting evolutionary search for optimal architecture..." << std::endl;
    if (use_monte_carlo) {
        std::cout << "Using Monte Carlo testing with " << monte_carlo_trials
                  << " trials per configuration" << std::endl;
    }

    std::vector<NetworkConfig> population;
    std::vector<double> fitness;

    for (size_t i = 0; i < population_size; ++i) {
        population.push_back(generateRandomConfig());
    }

    double best_preservation = 0.0;
    NetworkConfig best_config;
    ArchitectureTestResult best_result;

    for (size_t gen = 0; gen < generations; ++gen) {
        std::cout << "Generation " << gen + 1 << "/" << generations << std::endl;
        fitness.clear();

        for (const auto& config : population) {
            ArchitectureTestResult result;
            auto it = tested_configs_.find(config);
            if (it != tested_configs_.end()) {
                result = it->second;
            }
            else {
                result = testConfiguration(config, max_epochs, use_monte_carlo, monte_carlo_trials);
                tested_configs_[config] = result;
            }

            double score = result.accuracy_preservation;
            fitness.push_back(score);
            if (score > best_preservation) {
                best_preservation = score;
                best_config = config;
                best_result = result;
            }
        }

        std::vector<NetworkConfig> new_population;

        // Elitism: keep best 20%
        std::vector<size_t> indices(population_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&fitness](size_t a, size_t b) { return fitness[a] > fitness[b]; });

        size_t elite_count = population_size / 5;
        for (size_t i = 0; i < elite_count; ++i) {
            new_population.push_back(population[indices[i]]);
        }

        // Adaptive mutation controller wiring
        if (adaptive_mutation_enabled_ && !adaptive_controller_) {
            adaptive_controller_ =
                std::make_unique<AdaptiveMutationController>(this, random_generator_);
        }

        // Diversity-aware adaptive mutation rate
        double current_mutation_rate = mutation_rate;
        if (adaptive_mutation_enabled_) {
            bool use_cached = false;
            if (mutation_rate_schedule_interval_ > 0) {
                size_t compute_every = mutation_rate_schedule_interval_;
                use_cached = (gen % compute_every != 0);
            }
            if (gen >= mutation_rate_freeze_after_gen_) {
                use_cached = true;
            }
            if (!use_cached || !last_computed_mutation_rate_.has_value()) {
                current_mutation_rate =
                    calculateAdaptiveMutationRate(population, fitness, gen, generations);
                last_computed_mutation_rate_ = current_mutation_rate;
            }
            else {
                current_mutation_rate = *last_computed_mutation_rate_;
            }
        }

        // Fill rest with crossover and mutation
        std::uniform_int_distribution<size_t> idx_dist(0, population_size - 1);
        while (new_population.size() < population_size) {
            const NetworkConfig& parent1 = population[idx_dist(random_generator_)];
            const NetworkConfig& parent2 = population[idx_dist(random_generator_)];
            NetworkConfig child = crossoverConfigs(parent1, parent2);

            if (adaptive_mutation_enabled_ && adaptive_controller_) {
                child = adaptive_controller_->adaptiveMutate(child, current_mutation_rate);
            }
            else {
                child = mutateConfig(child, current_mutation_rate);
            }

            new_population.push_back(child);
        }

        population = std::move(new_population);
        if (gen % 2 == 0) {
            saveResultsToFile();
        }
    }

    saveResultsToFile();
    return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                        best_result.accuracy_preservation, generations * population_size,
                        best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                        best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
}

}  // namespace research
}  // namespace rad_ml
