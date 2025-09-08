/**
 * @file genetic_operators.cpp
 * @brief Implementation of adaptive mutation controller and operators for AAS
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <rad_ml/research/auto_arch/genetic_operators.hpp>
#include <rad_ml/research/auto_arch_search.hpp>

namespace rad_ml {
namespace research {

//==============================================================================
// ADAPTIVE MUTATION CONTROLLER IMPLEMENTATION
//==============================================================================

AdaptiveMutationController::AdaptiveMutationController(AutoArchSearch* parent, std::mt19937& rng)
    : parent_(parent), random_generator_(&rng)
{
    // Initialize with default operators
    initializeDefaultOperators();
}

void AdaptiveMutationController::initializeDefaultOperators()
{
    using MutationFunc =
        NetworkConfig (AdaptiveMutationController::*)(const NetworkConfig&, double);

    auto create_wrapper = [this](MutationFunc func) {
        return [this, func](const NetworkConfig& config, double rate) -> NetworkConfig {
            return (this->*func)(config, rate);
        };
    };

    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateArchitectureFocused),
                        "Architecture");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateParameterFocused),
                        "Parameters");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateProtectionFocused),
                        "Protection");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateBalanced), "Balanced");
    addMutationOperator(create_wrapper(&AdaptiveMutationController::mutateAggressive),
                        "Aggressive");
}

NetworkConfig AdaptiveMutationController::mutateArchitectureFocused(const NetworkConfig& config,
                                                                    double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    if (mutation_dist(*random_generator_) < rate * 1.5) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    if (mutation_dist(*random_generator_) < rate * 0.7) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateParameterFocused(const NetworkConfig& config,
                                                                 double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    if (mutation_dist(*random_generator_) < rate * 1.5) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    if (mutation_dist(*random_generator_) < rate * 0.5) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateProtectionFocused(const NetworkConfig& config,
                                                                  double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    if (mutation_dist(*random_generator_) < rate * 1.5) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    if (mutation_dist(*random_generator_) < rate * 0.5) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateBalanced(const NetworkConfig& config, double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);

    if (mutation_dist(*random_generator_) < rate) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    if (mutation_dist(*random_generator_) < rate) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    return mutated;
}

NetworkConfig AdaptiveMutationController::mutateAggressive(const NetworkConfig& config, double rate)
{
    NetworkConfig mutated = config;
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
    double aggressive_rate = rate * 2.0;

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        if (mutated.layer_sizes.size() > 2) {
            std::uniform_int_distribution<size_t> layer_idx_dist(1, mutated.layer_sizes.size() - 2);
            size_t layer_idx = layer_idx_dist(*random_generator_);
            mutated.layer_sizes[layer_idx] = getRandomLayerSize();
        }
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.dropout_rate = getRandomDropoutRate();
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.has_residual_connections = !mutated.has_residual_connections;
    }

    if (mutation_dist(*random_generator_) < aggressive_rate) {
        mutated.protection_level = getRandomProtectionLevel();
    }

    return mutated;
}

void AdaptiveMutationController::addMutationOperator(
    std::function<NetworkConfig(const NetworkConfig&, double)> op, const std::string& name)
{
    mutation_operators_.emplace_back(op, name);
    updateProbabilities();
}

NetworkConfig AdaptiveMutationController::adaptiveMutate(const NetworkConfig& config,
                                                         double base_rate)
{
    if (mutation_operators_.empty()) {
        return config;
    }

    size_t selected_operator = selectOperatorDynamically();
    last_selected_operator_index_ = selected_operator;
    NetworkConfig mutated = mutation_operators_[selected_operator].operator_func(config, base_rate);
    mutation_operators_[selected_operator].applications++;
    return mutated;
}

void AdaptiveMutationController::updateOperatorCredits(
    const std::vector<double>& improvement_scores, const std::vector<size_t>& used_operators)
{
    if (improvement_scores.size() != used_operators.size()) {
        return;
    }

    for (size_t i = 0; i < improvement_scores.size(); ++i) {
        size_t op_idx = used_operators[i];
        if (op_idx < mutation_operators_.size()) {
            double improvement = improvement_scores[i];
            mutation_operators_[op_idx].total_improvement += improvement;

            double success = (improvement > 0.0) ? 1.0 : 0.0;
            mutation_operators_[op_idx].success_rate =
                0.9 * mutation_operators_[op_idx].success_rate + 0.1 * success;

            double avg_improvement = mutation_operators_[op_idx].applications > 0
                                         ? mutation_operators_[op_idx].total_improvement /
                                               mutation_operators_[op_idx].applications
                                         : 0.0;

            mutation_operators_[op_idx].credit_score =
                learning_rate_ * avg_improvement +
                (1.0 - learning_rate_) * mutation_operators_[op_idx].credit_score;
        }
    }

    updateProbabilities();
}

std::vector<std::tuple<std::string, int, double, double>>
AdaptiveMutationController::getOperatorStatistics() const
{
    std::vector<std::tuple<std::string, int, double, double>> stats;
    for (const auto& op : mutation_operators_) {
        stats.emplace_back(op.name, op.applications, op.success_rate, op.credit_score);
    }
    return stats;
}

void AdaptiveMutationController::resetStatistics()
{
    for (auto& op : mutation_operators_) {
        op.success_rate = 0.5;
        op.credit_score = 0.0;
        op.applications = 0;
        op.total_improvement = 0.0;
    }
    updateProbabilities();
}

size_t AdaptiveMutationController::selectOperatorDynamically()
{
    if (mutation_operators_.empty()) {
        return 0;
    }

    std::uniform_real_distribution<double> explore_dist(0.0, 1.0);
    if (explore_dist(*random_generator_) < exploration_factor_) {
        std::uniform_int_distribution<size_t> op_dist(0, mutation_operators_.size() - 1);
        return op_dist(*random_generator_);
    }
    else {
        if (!operator_probabilities_.empty()) {
            std::discrete_distribution<size_t> prob_dist(operator_probabilities_.begin(),
                                                         operator_probabilities_.end());
            return prob_dist(*random_generator_);
        }
        else {
            std::uniform_int_distribution<size_t> op_dist(0, mutation_operators_.size() - 1);
            return op_dist(*random_generator_);
        }
    }
}

void AdaptiveMutationController::updateProbabilities()
{
    if (mutation_operators_.empty()) {
        operator_probabilities_.clear();
        return;
    }

    std::vector<double> credit_scores;
    credit_scores.reserve(mutation_operators_.size());
    for (const auto& op : mutation_operators_) {
        credit_scores.push_back(op.credit_score);
    }
    operator_probabilities_ = softmax(credit_scores);
}

std::vector<double> AdaptiveMutationController::softmax(
    const std::vector<double>& credit_scores) const
{
    if (credit_scores.empty()) {
        return {};
    }

    double max_val = *std::max_element(credit_scores.begin(), credit_scores.end());
    std::vector<double> exp_scores;
    exp_scores.reserve(credit_scores.size());
    double sum_exp = 0.0;

    for (double score : credit_scores) {
        double exp_score = std::exp(score - max_val);
        exp_scores.push_back(exp_score);
        sum_exp += exp_score;
    }

    std::vector<double> probabilities;
    probabilities.reserve(exp_scores.size());
    for (double exp_score : exp_scores) {
        probabilities.push_back(exp_score / sum_exp);
    }
    return probabilities;
}

//==============================================================================
// ADAPTIVE MUTATION CONTROLLER HELPER METHODS
//==============================================================================

size_t AdaptiveMutationController::getRandomLayerSize()
{
    if (parent_ && !parent_->width_options_.empty()) {
        std::uniform_int_distribution<size_t> width_idx_dist(0, parent_->width_options_.size() - 1);
        return parent_->width_options_[width_idx_dist(*random_generator_)];
    }
    std::uniform_int_distribution<size_t> size_dist(32, 256);
    return size_dist(*random_generator_);
}

double AdaptiveMutationController::getRandomDropoutRate()
{
    if (parent_ && !parent_->dropout_options_.empty()) {
        std::uniform_int_distribution<size_t> dropout_idx_dist(
            0, parent_->dropout_options_.size() - 1);
        return parent_->dropout_options_[dropout_idx_dist(*random_generator_)];
    }
    std::uniform_real_distribution<double> dropout_dist(0.1, 0.6);
    return dropout_dist(*random_generator_);
}

neural::ProtectionLevel AdaptiveMutationController::getRandomProtectionLevel()
{
    if (parent_ && !parent_->protection_levels_.empty()) {
        std::uniform_int_distribution<size_t> protection_idx_dist(
            0, parent_->protection_levels_.size() - 1);
        return parent_->protection_levels_[protection_idx_dist(*random_generator_)];
    }
    std::uniform_int_distribution<int> prot_dist(0, 4);
    return static_cast<neural::ProtectionLevel>(prot_dist(*random_generator_));
}

}  // namespace research
}  // namespace rad_ml
