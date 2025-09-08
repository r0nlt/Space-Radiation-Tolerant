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

        // Optional: initialize basic QD manager lazily
        if (qd_enabled_ && !qd_manager_) {
            qd_manager_.reset(new QualityDiversityManager([&](const NetworkConfig& cfg) {
                if (tested_configs_.count(cfg) == 0) {
                    auto res =
                        testConfiguration(cfg, max_epochs, use_monte_carlo, monte_carlo_trials);
                    tested_configs_[cfg] = res;
                }
                return tested_configs_[cfg];
            }));
        }

        // Optional: initialize advanced QD manager lazily
        if (advanced_qd_enabled_ && !advanced_qd_manager_) {
            advanced_qd_manager_ = std::make_unique<AdvancedQualityDiversityManager>();
        }

        // Fill rest with crossover and mutation using tournament selection (k=3)
        auto tournament_select = [&](size_t tournament_size) -> const NetworkConfig& {
            std::uniform_int_distribution<size_t> idx_dist(0, population_size - 1);
            size_t best_idx = idx_dist(random_generator_);
            for (size_t t = 1; t < tournament_size; ++t) {
                size_t cand_idx = idx_dist(random_generator_);
                if (fitness[cand_idx] > fitness[best_idx]) best_idx = cand_idx;
            }
            return population[best_idx];
        };

        // Track operator indices and improvements for adaptive credit updates
        std::vector<size_t> used_operator_indices;
        std::vector<double> operator_improvements;

        while (new_population.size() < population_size) {
            const NetworkConfig& parent1 = tournament_select(3);
            const NetworkConfig& parent2 = tournament_select(3);
            NetworkConfig child = crossoverConfigs(parent1, parent2);

            // Record best parent fitness to compute improvement
            double parent_fitness = 0.0;
            {
                // get indices by scanning; small populations keep this cheap
                for (size_t idx = 0; idx < population.size(); ++idx) {
                    if (population[idx].layer_sizes == parent1.layer_sizes &&
                        population[idx].dropout_rate == parent1.dropout_rate &&
                        population[idx].has_residual_connections ==
                            parent1.has_residual_connections &&
                        population[idx].protection_level == parent1.protection_level) {
                        parent_fitness = std::max(parent_fitness, fitness[idx]);
                    }
                    if (population[idx].layer_sizes == parent2.layer_sizes &&
                        population[idx].dropout_rate == parent2.dropout_rate &&
                        population[idx].has_residual_connections ==
                            parent2.has_residual_connections &&
                        population[idx].protection_level == parent2.protection_level) {
                        parent_fitness = std::max(parent_fitness, fitness[idx]);
                    }
                }
            }

            if (adaptive_mutation_enabled_ && adaptive_controller_) {
                child = adaptive_controller_->adaptiveMutate(child, current_mutation_rate);
                used_operator_indices.push_back(
                    adaptive_controller_->getLastSelectedOperatorIndex());
            }
            else {
                child = mutateConfig(child, current_mutation_rate);
            }

            new_population.push_back(child);

            // Evaluate child to compute improvement (cache-aware)
            if (tested_configs_.count(child) == 0) {
                auto result =
                    testConfiguration(child, max_epochs, use_monte_carlo, monte_carlo_trials);
                tested_configs_[child] = result;
            }
            double child_fitness = tested_configs_[child].accuracy_preservation;
            operator_improvements.push_back(child_fitness - parent_fitness);

            // Optionally register with QD map (example key: full layer sizes signature)
            if (qd_enabled_ && qd_manager_) {
                std::string key;
                key.reserve(64);
                for (size_t i = 0; i < child.layer_sizes.size(); ++i) {
                    key += std::to_string(child.layer_sizes[i]);
                    if (i + 1 < child.layer_sizes.size()) key += "-";
                }
                qd_manager_->emplaceArchitecture(std::move(key),
                                                 std::make_pair(child, child_fitness));
            }

            // Optionally update advanced QD archive
            if (advanced_qd_enabled_ && advanced_qd_manager_) {
                const auto& res = tested_configs_[child];
                advanced_qd_manager_->addToArchive(child, res, gen);
            }
        }

        // If enabled, replace worst K with diverse elites and log coverage
        if (advanced_qd_enabled_ && advanced_qd_manager_) {
            auto elites =
                advanced_qd_manager_->sampleDiverseElites(std::max<size_t>(1, population_size / 5));
            size_t injected = 0;
            // Ensure final size == population_size.
            const size_t desired = population_size;
            if (!elites.empty()) {
                const size_t available = new_population.size();
                const size_t slots = available > desired ? available - desired : 0;
                const size_t to_replace = std::min(slots + elites.size(), available) > 0
                                              ? std::min(slots + elites.size(), available)
                                              : 0;
                if (to_replace > 0) {
                    new_population.erase(new_population.end() - to_replace, new_population.end());
                }
                for (const auto& e : elites) {
                    if (new_population.size() >= desired) break;
                    new_population.push_back(e);
                    ++injected;
                }
                // If still short (elites < needed), backfill with top survivors
                while (new_population.size() < desired && !indices.empty()) {
                    new_population.push_back(
                        population[indices[new_population.size() % indices.size()]]);
                }
            }
            auto analytics = advanced_qd_manager_->getAnalytics();
            std::cout << "QD coverage: " << std::fixed << std::setprecision(4)
                      << (analytics.coverage_percentage * 100.0) << "% (occupied "
                      << analytics.total_occupied_cells << ")"
                      << ", elites injected: " << injected << std::endl;
        }

        population = std::move(new_population);
        // Update adaptive credits if applicable
        if (adaptive_mutation_enabled_ && adaptive_controller_ && !used_operator_indices.empty() &&
            operator_improvements.size() == used_operator_indices.size()) {
            adaptive_controller_->updateOperatorCredits(operator_improvements,
                                                        used_operator_indices);
        }

        if (save_interval_generations_ > 0 && gen % save_interval_generations_ == 0) {
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
