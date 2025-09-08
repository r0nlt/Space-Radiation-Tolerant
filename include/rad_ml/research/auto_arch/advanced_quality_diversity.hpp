#pragma once

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <rad_ml/research/architecture_tester.hpp>
#include <rad_ml/research/auto_arch/types.hpp>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rad_ml {
namespace research {

class AdvancedQualityDiversityManager {
   private:
    struct RadiationAwareBehaviorDescriptor {
        double architectural_complexity;
        double protection_efficiency;
        double computational_cost;
        double radiation_tolerance;
        double graceful_degradation;
        double power_efficiency;

        RadiationAwareBehaviorDescriptor() = default;
        RadiationAwareBehaviorDescriptor(const RadiationAwareBehaviorDescriptor& other) = default;
        RadiationAwareBehaviorDescriptor(RadiationAwareBehaviorDescriptor&& other) noexcept =
            default;
        RadiationAwareBehaviorDescriptor& operator=(const RadiationAwareBehaviorDescriptor& other) =
            default;
        RadiationAwareBehaviorDescriptor& operator=(
            RadiationAwareBehaviorDescriptor&& other) noexcept = default;
    };

    struct ArchiveCell {
        NetworkConfig best_config;
        double fitness_score;
        RadiationAwareBehaviorDescriptor behavior;
        size_t generation_discovered;
        double novelty_score;
        std::vector<double> objective_vector;

        ArchiveCell()
            : fitness_score(-std::numeric_limits<double>::infinity()),
              generation_discovered(0),
              novelty_score(0.0)
        {
        }

        bool isEmpty() const { return fitness_score == -std::numeric_limits<double>::infinity(); }
    };

    static constexpr size_t BASE_GRID_RESOLUTION = 10;
    static constexpr size_t MAX_GRID_RESOLUTION = 50;
    static constexpr size_t BEHAVIORAL_DIMENSIONS = 6;

    std::vector<ArchiveCell> behavioral_archive_;
    size_t current_grid_resolution_;

    std::vector<RadiationAwareBehaviorDescriptor> novelty_archive_;
    static constexpr size_t K_NEAREST_NEIGHBORS = 5;

    mutable std::mutex archive_mutex_;
    mutable std::mutex novelty_mutex_;

    size_t total_evaluations_;
    size_t archive_updates_;
    double coverage_percentage_;

    mutable std::mt19937 rng_;

   public:
    AdvancedQualityDiversityManager()
        : current_grid_resolution_(BASE_GRID_RESOLUTION),
          total_evaluations_(0),
          archive_updates_(0),
          coverage_percentage_(0.0),
          rng_(std::random_device{}())
    {
        const size_t total_cells =
            static_cast<size_t>(std::pow(current_grid_resolution_, BEHAVIORAL_DIMENSIONS));
        behavioral_archive_.resize(total_cells);
    }

    bool addToArchive(const NetworkConfig& config, const ArchitectureTestResult& test_result,
                      size_t generation = 0)
    {
        auto behavior = calculateRadiationAwareBehavior(config, test_result);
        double novelty = calculateNoveltyScore(behavior);
        auto cell_index = coordsToIndex(discretizeBehavior(behavior));

        std::lock_guard<std::mutex> lock(archive_mutex_);
        ArchiveCell& cell = behavioral_archive_[cell_index];
        double fitness = calculateCombinedFitness(test_result, novelty);

        if (cell.isEmpty() || fitness > cell.fitness_score) {
            cell.best_config = config;
            cell.fitness_score = fitness;
            cell.behavior = std::move(behavior);
            cell.generation_discovered = generation;
            cell.novelty_score = novelty;
            cell.objective_vector = extractObjectiveVector(test_result);

            ++archive_updates_;
            updateNoveltyArchive(cell.behavior);
            return true;
        }
        return false;
    }

    std::future<std::vector<std::pair<NetworkConfig, bool>>> evaluatePopulationBatch(
        const std::vector<NetworkConfig>& population, size_t generation = 0,
        const std::function<ArchitectureTestResult(const NetworkConfig&)>& evaluator = {})
    {
        return std::async(std::launch::async, [this, population, generation, evaluator]() {
            const size_t concurrency = std::max(1u, std::thread::hardware_concurrency());
            const size_t batch_size =
                std::min(population.size(), static_cast<size_t>(concurrency * 2));

            std::vector<std::pair<NetworkConfig, bool>> results;
            results.reserve(population.size());

            for (size_t start = 0; start < population.size(); start += batch_size) {
                const size_t end = std::min(start + batch_size, population.size());
                std::vector<std::future<std::pair<NetworkConfig, bool>>> batch_futures;
                batch_futures.reserve(end - start);

                for (size_t i = start; i < end; ++i) {
                    batch_futures.emplace_back(std::async(
                        std::launch::async, [this, &population, i, generation, &evaluator]() {
                            return evaluateAndArchive(population[i], generation, evaluator);
                        }));
                }
                for (auto& f : batch_futures) results.emplace_back(f.get());
            }
            updateArchiveStatistics();
            return results;
        });
    }

    std::vector<NetworkConfig> sampleDiverseElites(size_t sample_size) const
    {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        std::vector<const ArchiveCell*> non_empty;
        non_empty.reserve(behavioral_archive_.size());
        for (const auto& cell : behavioral_archive_)
            if (!cell.isEmpty()) non_empty.push_back(&cell);
        std::vector<NetworkConfig> elites;
        if (non_empty.empty() || sample_size == 0) return elites;

        size_t fitness_samples = static_cast<size_t>(sample_size * 0.4);
        size_t novelty_samples = static_cast<size_t>(sample_size * 0.3);
        size_t diverse_samples = sample_size - fitness_samples - novelty_samples;

        auto fitness_sorted = non_empty;
        std::sort(fitness_sorted.begin(), fitness_sorted.end(),
                  [](const ArchiveCell* a, const ArchiveCell* b) {
                      return a->fitness_score > b->fitness_score;
                  });
        for (size_t i = 0; i < std::min(fitness_samples, fitness_sorted.size()); ++i)
            elites.push_back(fitness_sorted[i]->best_config);

        auto novelty_sorted = non_empty;
        std::sort(novelty_sorted.begin(), novelty_sorted.end(),
                  [](const ArchiveCell* a, const ArchiveCell* b) {
                      return a->novelty_score > b->novelty_score;
                  });
        for (size_t i = 0; i < std::min(novelty_samples, novelty_sorted.size()); ++i)
            elites.push_back(novelty_sorted[i]->best_config);

        auto remaining = non_empty;
        for (size_t i = 0; i < diverse_samples && !remaining.empty(); ++i) {
            std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
            size_t idx = dist(rng_);
            elites.push_back(remaining[idx]->best_config);
            remaining.erase(remaining.begin() + idx);
        }
        return elites;
    }

    struct ArchiveAnalytics {
        size_t total_occupied_cells = 0;
        double coverage_percentage = 0.0;
        double average_fitness = 0.0;
        double fitness_variance = 0.0;
        double behavioral_diversity = 0.0;
        size_t total_evaluations = 0;
    };

    ArchiveAnalytics getAnalytics() const
    {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        ArchiveAnalytics a;
        a.total_evaluations = total_evaluations_;

        std::vector<double> fitness_values;
        std::vector<RadiationAwareBehaviorDescriptor> behaviors;
        for (const auto& cell : behavioral_archive_) {
            if (!cell.isEmpty()) {
                ++a.total_occupied_cells;
                fitness_values.push_back(cell.fitness_score);
                behaviors.push_back(cell.behavior);
            }
        }
        a.coverage_percentage = behavioral_archive_.empty()
                                    ? 0.0
                                    : static_cast<double>(a.total_occupied_cells) /
                                          static_cast<double>(behavioral_archive_.size());
        if (!fitness_values.empty()) {
            a.average_fitness = std::accumulate(fitness_values.begin(), fitness_values.end(), 0.0) /
                                fitness_values.size();
            double mean = a.average_fitness;
            double var_sum = 0.0;
            for (double f : fitness_values) var_sum += (f - mean) * (f - mean);
            a.fitness_variance = var_sum / fitness_values.size();
            a.behavioral_diversity = calculateBehavioralDiversity(behaviors);
        }
        return a;
    }

   private:
    RadiationAwareBehaviorDescriptor calculateRadiationAwareBehavior(
        const NetworkConfig& config, const ArchitectureTestResult& result) const
    {
        RadiationAwareBehaviorDescriptor b{};
        size_t total_params = 0;
        for (size_t i = 1; i < config.layer_sizes.size(); ++i)
            total_params += config.layer_sizes[i - 1] * config.layer_sizes[i];
        b.architectural_complexity =
            std::log(static_cast<double>(total_params) + 1.0) / std::log(1000000.0);
        if (result.errors_detected > 0) {
            b.protection_efficiency = static_cast<double>(result.errors_corrected) /
                                      static_cast<double>(result.errors_detected);
        }
        else {
            b.protection_efficiency = 1.0;
        }
        b.computational_cost =
            std::min(1.0, (result.execution_time_ms / 1000.0) * (1.0 + b.architectural_complexity));
        b.radiation_tolerance = result.accuracy_preservation / 100.0;
        double degradation_rate = (result.baseline_accuracy - result.radiation_accuracy) /
                                  std::max(1e-9, result.baseline_accuracy);
        b.graceful_degradation = 1.0 - std::max(0.0, degradation_rate);
        double protection_overhead = getProtectionOverhead(config.protection_level);
        b.power_efficiency = 1.0 / (1.0 + protection_overhead + b.architectural_complexity);
        return b;
    }

    double calculateNoveltyScore(const RadiationAwareBehaviorDescriptor& behavior) const
    {
        std::lock_guard<std::mutex> lock(novelty_mutex_);
        if (novelty_archive_.size() < K_NEAREST_NEIGHBORS) return 1.0;
        std::vector<double> distances;
        distances.reserve(novelty_archive_.size());
        for (const auto& archived : novelty_archive_)
            distances.push_back(calculateBehavioralDistance(behavior, archived));
        std::partial_sort(distances.begin(), distances.begin() + K_NEAREST_NEIGHBORS,
                          distances.end());
        double sum = 0.0;
        for (size_t i = 0; i < K_NEAREST_NEIGHBORS; ++i) sum += distances[i];
        return sum / K_NEAREST_NEIGHBORS;
    }

    static double calculateBehavioralDistance(const RadiationAwareBehaviorDescriptor& a,
                                              const RadiationAwareBehaviorDescriptor& b)
    {
        double dx1 = a.architectural_complexity - b.architectural_complexity;
        double dx2 = a.protection_efficiency - b.protection_efficiency;
        double dx3 = a.computational_cost - b.computational_cost;
        double dx4 = a.radiation_tolerance - b.radiation_tolerance;
        double dx5 = a.graceful_degradation - b.graceful_degradation;
        double dx6 = a.power_efficiency - b.power_efficiency;
        return std::sqrt(dx1 * dx1 + dx2 * dx2 + dx3 * dx3 + dx4 * dx4 + dx5 * dx5 + dx6 * dx6);
    }

    std::vector<size_t> discretizeBehavior(const RadiationAwareBehaviorDescriptor& behavior) const
    {
        auto discretize = [this](double value) -> size_t {
            double clamped = std::max(0.0, std::min(1.0, value));
            size_t coord = static_cast<size_t>(clamped * (current_grid_resolution_ - 1));
            return std::min(coord, current_grid_resolution_ - 1);
        };
        std::vector<size_t> coords;
        coords.reserve(BEHAVIORAL_DIMENSIONS);
        coords.push_back(discretize(behavior.architectural_complexity));
        coords.push_back(discretize(behavior.protection_efficiency));
        coords.push_back(discretize(behavior.computational_cost));
        coords.push_back(discretize(behavior.radiation_tolerance));
        coords.push_back(discretize(behavior.graceful_degradation));
        coords.push_back(discretize(behavior.power_efficiency));
        return coords;
    }

    size_t coordsToIndex(const std::vector<size_t>& coords) const
    {
        size_t index = 0;
        size_t multiplier = 1;
        for (size_t i = 0; i < coords.size(); ++i) {
            index += coords[i] * multiplier;
            multiplier *= current_grid_resolution_;
        }
        return index;
    }

    static double getProtectionOverhead(neural::ProtectionLevel level)
    {
        switch (level) {
            case neural::ProtectionLevel::NONE:
                return 0.0;
            case neural::ProtectionLevel::CHECKSUM_ONLY:
                return 0.1;
            case neural::ProtectionLevel::SELECTIVE_TMR:
                return 0.5;
            case neural::ProtectionLevel::FULL_TMR:
                return 1.0;
            case neural::ProtectionLevel::ADAPTIVE_TMR:
                return 0.7;
            case neural::ProtectionLevel::SPACE_OPTIMIZED:
                return 0.3;
            default:
                return 0.0;
        }
    }

    std::pair<NetworkConfig, bool> evaluateAndArchive(
        const NetworkConfig& config, size_t generation,
        const std::function<ArchitectureTestResult(const NetworkConfig&)>& evaluator)
    {
        ArchitectureTestResult result;
        if (evaluator) result = evaluator(config);
        bool added = addToArchive(config, result, generation);
        ++total_evaluations_;
        return std::make_pair(config, added);
    }

    void updateArchiveStatistics()
    {
        size_t occupied = 0;
        for (const auto& cell : behavioral_archive_)
            if (!cell.isEmpty()) ++occupied;
        coverage_percentage_ =
            behavioral_archive_.empty()
                ? 0.0
                : static_cast<double>(occupied) / static_cast<double>(behavioral_archive_.size());
    }

    static double calculateCombinedFitness(const ArchitectureTestResult& result, double novelty)
    {
        return 0.8 * result.accuracy_preservation + 0.2 * novelty * 100.0;
    }

    static std::vector<double> extractObjectiveVector(const ArchitectureTestResult& result)
    {
        return {result.accuracy_preservation, result.baseline_accuracy, result.execution_time_ms};
    }

    void updateNoveltyArchive(const RadiationAwareBehaviorDescriptor& behavior)
    {
        std::lock_guard<std::mutex> lock(novelty_mutex_);
        novelty_archive_.push_back(behavior);
        if (novelty_archive_.size() > 1000) novelty_archive_.erase(novelty_archive_.begin());
    }

    static double calculateBehavioralDiversity(
        const std::vector<RadiationAwareBehaviorDescriptor>& behaviors)
    {
        if (behaviors.size() <= 1) return 0.0;
        double total_distance = 0.0;
        size_t pair_count = 0;
        for (size_t i = 0; i < behaviors.size(); ++i) {
            for (size_t j = i + 1; j < behaviors.size(); ++j) {
                total_distance += calculateBehavioralDistance(behaviors[i], behaviors[j]);
                ++pair_count;
            }
        }
        return pair_count > 0 ? total_distance / static_cast<double>(pair_count) : 0.0;
    }
};

}  // namespace research
}  // namespace rad_ml
