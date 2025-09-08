/**
 * @file genetic_operators.hpp
 * @brief Adaptive mutation controller and genetic operator interfaces for AAS
 */

#pragma once

#include <functional>
#include <rad_ml/research/auto_arch/types.hpp>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace rad_ml {
namespace research {

class AutoArchSearch;  // forward decl

/**
 * @brief Advanced Multi-Operator Adaptive Strategy (MOAS) Controller
 */
class AdaptiveMutationController {
   private:
    AutoArchSearch* parent_;

    struct MutationOperator {
        std::function<NetworkConfig(const NetworkConfig&, double)> operator_func;
        std::string name;
        double success_rate;
        double credit_score;
        int applications;
        double total_improvement;

        MutationOperator(std::function<NetworkConfig(const NetworkConfig&, double)> func,
                         const std::string& op_name)
            : operator_func(func),
              name(op_name),
              success_rate(0.5),
              credit_score(0.0),
              applications(0),
              total_improvement(0.0)
        {
        }
    };

    std::vector<MutationOperator> mutation_operators_;
    std::vector<double> operator_probabilities_;
    std::mt19937* random_generator_;
    size_t last_selected_operator_index_ = 0;

    const double learning_rate_ = 0.1;
    const double exploration_factor_ = 0.1;
    const int min_applications_ = 5;

   public:
    AdaptiveMutationController(AutoArchSearch* parent, std::mt19937& rng);
    void addMutationOperator(std::function<NetworkConfig(const NetworkConfig&, double)> op,
                             const std::string& name);
    NetworkConfig adaptiveMutate(const NetworkConfig& config, double base_rate);
    size_t getLastSelectedOperatorIndex() const { return last_selected_operator_index_; }
    void updateOperatorCredits(const std::vector<double>& improvement_scores,
                               const std::vector<size_t>& used_operators);
    std::vector<std::tuple<std::string, int, double, double>> getOperatorStatistics() const;
    std::vector<double> getOperatorProbabilities() const { return operator_probabilities_; }
    double getExplorationFactor() const { return exploration_factor_; }
    void resetStatistics();

   private:
    void initializeDefaultOperators();
    size_t selectOperatorDynamically();
    void updateProbabilities();
    std::vector<double> softmax(const std::vector<double>& credit_scores) const;
    size_t getRandomLayerSize();
    double getRandomDropoutRate();
    neural::ProtectionLevel getRandomProtectionLevel();
    NetworkConfig mutateArchitectureFocused(const NetworkConfig& config, double rate);
    NetworkConfig mutateParameterFocused(const NetworkConfig& config, double rate);
    NetworkConfig mutateProtectionFocused(const NetworkConfig& config, double rate);
    NetworkConfig mutateBalanced(const NetworkConfig& config, double rate);
    NetworkConfig mutateAggressive(const NetworkConfig& config, double rate);
};

}  // namespace research
}  // namespace rad_ml
