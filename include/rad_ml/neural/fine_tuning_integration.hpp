#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>

#include "fine_tuning.hpp"
#include "adaptive_protection.hpp"
#include "advanced_reed_solomon.hpp"
#include "sensitivity_analysis.hpp"
#include "../radiation/environment.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Integrated fine-tuning system for radiation-tolerant neural networks
 * 
 * This class combines all the fine-tuning components into a single, easy-to-use
 * interface that follows NASA/ESA aerospace standards.
 */
template <typename NetworkType>
class FineTuningOptimizer {
public:
    /**
     * @brief Constructor with environment
     * 
     * @param environment Current radiation environment
     */
    explicit FineTuningOptimizer(
        std::shared_ptr<radiation::Environment> environment = nullptr)
        : environment_(environment),
          sensitivity_analyzer_(),
          layer_optimizer_(),
          error_learner_(environment),
          memory_optimizer_() {}
    
    /**
     * @brief Set radiation environment
     * 
     * @param environment New environment
     */
    void setEnvironment(std::shared_ptr<radiation::Environment> environment) {
        environment_ = environment;
    }
    
    /**
     * @brief Get current radiation environment
     * 
     * @return Shared pointer to environment
     */
    std::shared_ptr<radiation::Environment> getEnvironment() const {
        return environment_;
    }
    
    /**
     * @brief Optimize network with all fine-tuning techniques
     * 
     * @tparam DatasetType Type of dataset
     * @param network Network to optimize
     * @param validation_data Validation dataset
     * @param optimization_level Level of optimization (0-3, with 3 being most aggressive)
     * @return True if optimization was successful
     */
    template <typename DatasetType>
    bool optimizeNetwork(
        NetworkType& network,
        const DatasetType& validation_data,
        int optimization_level = 2) {
        
        if (!environment_) {
            return false;  // Environment required
        }
        
        // Apply optimizations based on level
        bool success = true;
        
        // Level 0: Basic weight sensitivity only
        auto sensitivities = sensitivity_analyzer_.analyzeWeightSensitivity(network, validation_data);
        sensitivity_analyzer_.applyProtectionProfile(network, sensitivities);
        
        if (optimization_level >= 1) {
            // Level 1: Add layer-specific protection
            layer_optimizer_.optimizeLayerProtection(network, *environment_);
        }
        
        if (optimization_level >= 2) {
            // Level 2: Add memory layout optimization
            memory_optimizer_.optimizeLayout(network, sensitivities);
            
            // Configure Reed-Solomon where appropriate
            configureReedSolomon(network, sensitivities);
        }
        
        if (optimization_level >= 3) {
            // Level 3: Add error pattern learning
            // This requires error history, which might not be available initially
            if (!error_history_.empty()) {
                error_learner_.learnFromObservedErrors(error_history_, *environment_);
                applyErrorPatternProtection(network, sensitivities);
            } else {
                // Warning: Level 3 requires error history
                success = false;
            }
        }
        
        return success;
    }
    
    /**
     * @brief Update error history for pattern learning
     * 
     * @param errors New error events
     */
    void updateErrorHistory(const std::vector<error::ErrorEvent>& errors) {
        error_history_.insert(error_history_.end(), errors.begin(), errors.end());
        
        // If environment is available, update error learner
        if (environment_) {
            error_learner_.learnFromObservedErrors(error_history_, *environment_);
        }
    }
    
    /**
     * @brief Validate fine-tuning effectiveness
     * 
     * @tparam DatasetType Type of dataset
     * @param network Network to validate
     * @param test_data Test dataset
     * @param environments Vector of environments to test
     * @param report_file Optional report filename
     * @return Validation results
     */
    template <typename DatasetType>
    typename FineTuningValidation::ValidationResults validateOptimizations(
        NetworkType& network,
        const DatasetType& test_data,
        const std::vector<radiation::Environment>& environments,
        const std::optional<std::string>& report_file = std::nullopt) {
        
        FineTuningValidation validator;
        auto results = validator.validateOptimizations(network, environments, test_data);
        
        if (report_file) {
            validator.generateReport(results, *report_file);
        }
        
        return results;
    }

private:
    std::shared_ptr<radiation::Environment> environment_;
    EnhancedSensitivityAnalyzer sensitivity_analyzer_;
    LayerProtectionOptimizer layer_optimizer_;
    ErrorPatternLearner error_learner_;
    MemoryLayoutOptimizer memory_optimizer_;
    std::vector<error::ErrorEvent> error_history_;
    
    /**
     * @brief Configure Reed-Solomon protection for high-sensitivity weights
     * 
     * @param network Network to configure
     * @param sensitivities Weight sensitivities
     */
    void configureReedSolomon(
        NetworkType& network,
        const std::vector<float>& sensitivities) {
        
        // Find sensitivity threshold for top 20% of weights
        std::vector<float> sorted_sensitivities = sensitivities;
        std::sort(sorted_sensitivities.begin(), sorted_sensitivities.end());
        float high_threshold = sorted_sensitivities[static_cast<size_t>(0.8 * sensitivities.size())];
        
        // Apply RS protection to high-sensitivity weights
        for (size_t i = 0; i < sensitivities.size(); ++i) {
            if (sensitivities[i] >= high_threshold) {
                network.setWeightProtectionMethod(i, ProtectionMethod::REED_SOLOMON);
            }
        }
    }
    
    /**
     * @brief Apply error pattern-based protection strategies
     * 
     * @param network Network to configure
     * @param sensitivities Weight sensitivities
     */
    void applyErrorPatternProtection(
        NetworkType& network,
        const std::vector<float>& sensitivities) {
        
        // Process weights in blocks
        constexpr size_t BLOCK_SIZE = 64;
        std::vector<float> block;
        
        for (size_t i = 0; i < sensitivities.size(); i += BLOCK_SIZE) {
            // Extract block
            block.clear();
            for (size_t j = i; j < i + BLOCK_SIZE && j < sensitivities.size(); ++j) {
                block.push_back(network.getWeight(j));
            }
            
            // Get recommended protection strategy
            auto strategy = error_learner_.recommendStrategy(block, *environment_);
            
            // Apply strategy to all weights in the block
            for (size_t j = i; j < i + BLOCK_SIZE && j < sensitivities.size(); ++j) {
                network.setWeightProtectionStrategy(j, strategy);
            }
        }
    }
};

} // namespace neural
} // namespace rad_ml 