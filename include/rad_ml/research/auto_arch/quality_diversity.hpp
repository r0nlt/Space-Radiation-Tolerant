/**
 * @file quality_diversity.hpp
 * @brief Quality-Diversity manager with async evaluation (C++11-compatible)
 */

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <rad_ml/research/architecture_tester.hpp>
#include <rad_ml/research/auto_arch/types.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rad_ml {
namespace research {

class QualityDiversityManager {
   public:
    struct BehaviorDescriptor {
        double complexity;
        double efficiency;
        double cost;
        double tolerance;

        BehaviorDescriptor() : complexity(0.0), efficiency(0.0), cost(0.0), tolerance(0.0) {}

        BehaviorDescriptor(BehaviorDescriptor&& other) noexcept
            : complexity(other.complexity),
              efficiency(other.efficiency),
              cost(other.cost),
              tolerance(other.tolerance)
        {
        }

        BehaviorDescriptor(const BehaviorDescriptor& other) = default;
    };

   private:
    std::function<ArchitectureTestResult(const NetworkConfig&)> evaluation_function_;
    std::unordered_map<std::string, std::pair<NetworkConfig, double>> behavior_map_;
    mutable std::mutex map_mutex_;

   public:
    explicit QualityDiversityManager(
        std::function<ArchitectureTestResult(const NetworkConfig&)> evaluation_function)
        : evaluation_function_(std::move(evaluation_function))
    {
    }

    template <typename... Args>
    void emplaceArchitecture(Args&&... args)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        behavior_map_.emplace(std::forward<Args>(args)...);
    }

    std::future<std::vector<ArchitectureTestResult>> evaluateAsync(
        const std::vector<NetworkConfig>& population)
    {
        return std::async(std::launch::async, [this, &population]() {
            std::vector<std::future<ArchitectureTestResult>> futures;
            futures.reserve(population.size());
            for (const auto& config : population) {
                futures.emplace_back(std::async(std::launch::async, [this, &config]() {
                    return evaluation_function_(config);
                }));
            }

            std::vector<ArchitectureTestResult> results;
            results.reserve(futures.size());
            for (auto& f : futures) {
                results.emplace_back(f.get());
            }
            return results;
        });
    }
};

}  // namespace research
}  // namespace rad_ml
