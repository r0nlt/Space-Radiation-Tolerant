#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include "selective_hardening.hpp"
#include "sensitivity_analysis.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Space mission profiles for protection optimization
 */
enum class MissionProfile {
    EARTH_ORBIT,         ///< Standard Earth orbit (LEO/MEO)
    DEEP_SPACE,          ///< Deep space missions (beyond lunar)
    LUNAR_SURFACE,       ///< Lunar landing/surface operations
    MARS_SURFACE,        ///< Mars landing/surface operations
    JUPITER_FLYBY,       ///< Jupiter and radiation belt environments
    SOLAR_OBSERVATORY    ///< Solar observation/close proximity
};

/**
 * @brief Protection policy for a specific layer in the neural network
 */
class LayerProtectionPolicy {
public:
    /**
     * @brief Defines how a layer should be protected
     */
    struct ProtectionConfig {
        ProtectionLevel level;
        double resource_allocation;  // Percentage of total resources to allocate (0-1)
        bool dynamic_adjustment;     // Whether protection can be adjusted at runtime
        std::string strategy_name;   // Name of specific strategy variant
        MissionProfile mission_profile = MissionProfile::EARTH_ORBIT; // Default mission profile
    };
    
    /**
     * @brief Constructor with protection configuration
     * 
     * @param layer_name Name of the layer
     * @param config Protection configuration
     */
    LayerProtectionPolicy(
        const std::string& layer_name,
        const ProtectionConfig& config
    ) : layer_name_(layer_name), config_(config) {}
    
    /**
     * @brief Get the protection level for this layer
     * 
     * @return Protection level
     */
    ProtectionLevel getProtectionLevel() const {
        return config_.level;
    }
    
    /**
     * @brief Get resource allocation for this layer
     * 
     * @return Resource allocation percentage (0-1)
     */
    double getResourceAllocation() const {
        return config_.resource_allocation;
    }
    
    /**
     * @brief Check if this layer uses dynamic adjustment
     * 
     * @return True if dynamic adjustment is enabled
     */
    bool usesDynamicAdjustment() const {
        return config_.dynamic_adjustment;
    }
    
    /**
     * @brief Get the strategy variant name
     * 
     * @return Strategy name
     */
    const std::string& getStrategyName() const {
        return config_.strategy_name;
    }
    
    /**
     * @brief Get the layer name
     * 
     * @return Layer name
     */
    const std::string& getLayerName() const {
        return layer_name_;
    }
    
    /**
     * @brief Get the mission profile for this protection policy
     * 
     * @return Mission profile
     */
    MissionProfile getMissionProfile() const {
        return config_.mission_profile;
    }
    
    /**
     * @brief Set protection level
     * 
     * @param level New protection level
     */
    void setProtectionLevel(ProtectionLevel level) {
        config_.level = level;
    }
    
    /**
     * @brief Set resource allocation
     * 
     * @param allocation New resource allocation (0-1)
     */
    void setResourceAllocation(double allocation) {
        if (allocation < 0.0) allocation = 0.0;
        if (allocation > 1.0) allocation = 1.0;
        config_.resource_allocation = allocation;
    }
    
    /**
     * @brief Enable or disable dynamic adjustment
     * 
     * @param enabled Whether dynamic adjustment should be enabled
     */
    void setDynamicAdjustment(bool enabled) {
        config_.dynamic_adjustment = enabled;
    }
    
    /**
     * @brief Set the mission profile
     * 
     * @param profile New mission profile
     */
    void setMissionProfile(MissionProfile profile) {
        config_.mission_profile = profile;
        // Automatically adjust protection level based on mission profile
        adjustProtectionForMission();
    }
    
    /**
     * @brief Adjust protection level based on mission profile
     */
    void adjustProtectionForMission() {
        switch (config_.mission_profile) {
            case MissionProfile::EARTH_ORBIT:
                // Standard protection
                // No changes to default
                break;
                
            case MissionProfile::DEEP_SPACE:
                // Higher radiation, increase protection
                if (config_.level < ProtectionLevel::HEALTH_WEIGHTED_TMR) {
                    config_.level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
                }
                config_.dynamic_adjustment = true;
                break;
                
            case MissionProfile::LUNAR_SURFACE:
                // Moderate protection
                if (config_.level < ProtectionLevel::SELECTIVE_TMR) {
                    config_.level = ProtectionLevel::SELECTIVE_TMR;
                }
                break;
                
            case MissionProfile::MARS_SURFACE:
                // Moderate protection, but with different resource profile
                if (config_.level < ProtectionLevel::SELECTIVE_TMR) {
                    config_.level = ProtectionLevel::SELECTIVE_TMR;
                }
                // Adjust resource allocation based on Mars-specific considerations
                config_.resource_allocation = std::min(1.0, config_.resource_allocation * 1.2);
                break;
                
            case MissionProfile::JUPITER_FLYBY:
                // Extreme radiation, maximum protection
                config_.level = ProtectionLevel::FULL_TMR;
                config_.dynamic_adjustment = true;
                config_.resource_allocation = std::min(1.0, config_.resource_allocation * 1.5);
                break;
                
            case MissionProfile::SOLAR_OBSERVATORY:
                // Extreme radiation with thermal considerations
                config_.level = ProtectionLevel::FULL_TMR;
                config_.dynamic_adjustment = true;
                config_.strategy_name = "solar_optimized";
                break;
        }
    }

private:
    std::string layer_name_;
    ProtectionConfig config_;
};

/**
 * @brief Manager for layer-specific protection policies
 */
class LayerProtectionManager {
public:
    /**
     * @brief Create a protection policy for a layer
     * 
     * @param layer_name Name of the layer
     * @param config Protection configuration
     */
    void addProtectionPolicy(
        const std::string& layer_name,
        const LayerProtectionPolicy::ProtectionConfig& config) {
        
        policies_[layer_name] = std::make_shared<LayerProtectionPolicy>(layer_name, config);
    }
    
    /**
     * @brief Get the protection policy for a layer
     * 
     * @param layer_name Name of the layer
     * @return Shared pointer to the protection policy, or nullptr if not found
     */
    std::shared_ptr<LayerProtectionPolicy> getPolicy(const std::string& layer_name) const {
        auto it = policies_.find(layer_name);
        if (it != policies_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Check if a layer has a protection policy
     * 
     * @param layer_name Name of the layer
     * @return True if the layer has a protection policy
     */
    bool hasPolicy(const std::string& layer_name) const {
        return policies_.find(layer_name) != policies_.end();
    }
    
    /**
     * @brief Get all layer names with protection policies
     * 
     * @return Vector of layer names
     */
    std::vector<std::string> getPolicyLayers() const {
        std::vector<std::string> result;
        for (const auto& [name, _] : policies_) {
            result.push_back(name);
        }
        return result;
    }
    
    /**
     * @brief Remove a protection policy
     * 
     * @param layer_name Name of the layer
     * @return True if the policy was removed, false if it didn't exist
     */
    bool removePolicy(const std::string& layer_name) {
        auto it = policies_.find(layer_name);
        if (it != policies_.end()) {
            policies_.erase(it);
            return true;
        }
        return false;
    }
    
    /**
     * @brief Create protection policies based on sensitivity analysis
     * 
     * @param sensitivity_results Sensitivity analysis results
     * @param resource_budget Total resource budget (0-1)
     * @param base_strategy Base hardening strategy to apply
     */
    void createPoliciesFromAnalysis(
        const TopologicalAnalysisResult& sensitivity_results,
        double resource_budget = 1.0,
        HardeningStrategy base_strategy = HardeningStrategy::RESOURCE_CONSTRAINED) {
        
        // Clear existing policies
        policies_.clear();
        
        // Calculate total criticality to normalize resource allocation
        double total_criticality = 0.0;
        for (const auto& [layer_name, criticality] : sensitivity_results.layer_criticality) {
            total_criticality += criticality;
        }
        
        if (total_criticality <= 0.0) {
            return;  // No critical layers found
        }
        
        // Create policies based on criticality scores
        for (const auto& [layer_name, criticality] : sensitivity_results.layer_criticality) {
            // Allocate resources proportionally to criticality
            double allocation = (criticality / total_criticality) * resource_budget;
            
            // Determine protection level based on criticality and strategy
            ProtectionLevel level = determineProtectionLevel(criticality, base_strategy);
            
            // Configure protection
            LayerProtectionPolicy::ProtectionConfig config;
            config.level = level;
            config.resource_allocation = allocation;
            config.dynamic_adjustment = (criticality > 0.7);  // Enable dynamic adjustment for highly critical layers
            config.strategy_name = getStrategyVariant(layer_name, criticality, sensitivity_results);
            
            // Add policy
            addProtectionPolicy(layer_name, config);
        }
    }
    
    /**
     * @brief Apply the importance decay strategy
     * 
     * Adjusts resource allocation based on layer depth, focusing more on input layers.
     * 
     * @param network_depth Total depth of the network
     * @param layer_depths Map of layer names to their depths (0 = input, higher = deeper)
     */
    void applyImportanceDecayStrategy(
        int network_depth,
        const std::map<std::string, int>& layer_depths) {
        
        if (network_depth <= 0) return;
        
        // Calculate decay factors using an exponential decay function
        for (auto& [layer_name, policy] : policies_) {
            auto depth_it = layer_depths.find(layer_name);
            if (depth_it != layer_depths.end()) {
                int depth = depth_it->second;
                
                // Calculate decay factor (1.0 at input, decreasing exponentially)
                double decay_factor = std::exp(-2.0 * depth / static_cast<double>(network_depth));
                
                // Apply decay to resource allocation
                double new_allocation = policy->getResourceAllocation() * decay_factor;
                policy->setResourceAllocation(new_allocation);
            }
        }
        
        // Renormalize allocations to maintain the same total resource usage
        normalizeResourceAllocations();
    }
    
    /**
     * @brief Normalize resource allocations to sum to 1.0
     */
    void normalizeResourceAllocations() {
        double total_allocation = 0.0;
        
        // Calculate current total
        for (const auto& [_, policy] : policies_) {
            total_allocation += policy->getResourceAllocation();
        }
        
        if (total_allocation <= 0.0) return;
        
        // Normalize each allocation
        for (auto& [_, policy] : policies_) {
            double normalized = policy->getResourceAllocation() / total_allocation;
            policy->setResourceAllocation(normalized);
        }
    }
    
    /**
     * @brief Configure protection policies for a specific space mission
     * 
     * @param mission Mission profile to configure for
     * @param analysis_results Optional sensitivity analysis to incorporate
     */
    void configureForMission(
        MissionProfile mission,
        const TopologicalAnalysisResult* analysis_results = nullptr) {
        
        // Set mission profile for all existing policies
        for (auto& [layer_name, policy] : policies_) {
            policy->setMissionProfile(mission);
        }
        
        // Apply mission-specific global adjustments
        switch (mission) {
            case MissionProfile::JUPITER_FLYBY:
                // For extreme radiation environments, ensure input layers have extra protection
                ensureInputLayersProtection(ProtectionLevel::HEALTH_WEIGHTED_TMR);
                break;
                
            case MissionProfile::SOLAR_OBSERVATORY:
                // For solar observation, ensure dynamic adjustment is enabled for all layers
                enableDynamicAdjustment();
                break;
                
            case MissionProfile::DEEP_SPACE:
                // For deep space, apply importance decay to conserve resources
                if (analysis_results != nullptr) {
                    // Build layer depths map from analysis
                    std::map<std::string, int> layer_depths;
                    int max_depth = 0;
                    
                    for (const auto& layer_name : getPolicyLayers()) {
                        // Simple approximation - could be improved with actual network topology
                        int depth = 0;
                        for (const auto& path : analysis_results->critical_paths) {
                            // Count position in path (assuming format: "layer1 -> layer2 -> ...")
                            size_t pos = path.find(layer_name);
                            if (pos != std::string::npos) {
                                // Count arrows before this layer name
                                size_t arrow_count = 0;
                                for (size_t i = 0; i < pos; i++) {
                                    if (path.substr(i, 4) == " -> ") {
                                        arrow_count++;
                                    }
                                }
                                depth = std::max(depth, static_cast<int>(arrow_count));
                            }
                        }
                        layer_depths[layer_name] = depth;
                        max_depth = std::max(max_depth, depth);
                    }
                    
                    // Apply decay strategy with the calculated depths
                    applyImportanceDecayStrategy(max_depth, layer_depths);
                }
                break;
                
            default:
                // Other mission profiles use default policy configurations
                break;
        }
    }
    
    /**
     * @brief Ensure input layers have at least the specified protection level
     * 
     * @param minimum_level Minimum protection level for input layers
     */
    void ensureInputLayersProtection(ProtectionLevel minimum_level) {
        // This implementation assumes input layers have names containing "input"
        // In a real implementation, you would use network topology information
        for (auto& [layer_name, policy] : policies_) {
            if (layer_name.find("input") != std::string::npos || 
                layer_name == "input" || 
                layer_name.substr(0, 5) == "input") {
                
                if (policy->getProtectionLevel() < minimum_level) {
                    policy->setProtectionLevel(minimum_level);
                }
            }
        }
    }
    
    /**
     * @brief Enable dynamic adjustment for all protection policies
     */
    void enableDynamicAdjustment() {
        for (auto& [_, policy] : policies_) {
            policy->setDynamicAdjustment(true);
        }
    }

private:
    std::map<std::string, std::shared_ptr<LayerProtectionPolicy>> policies_;
    
    /**
     * @brief Determine protection level based on criticality and strategy
     * 
     * @param criticality Criticality score (0-1)
     * @param strategy Base hardening strategy
     * @return Appropriate protection level
     */
    ProtectionLevel determineProtectionLevel(
        double criticality,
        HardeningStrategy strategy) {
        
        if (strategy == HardeningStrategy::FIXED_THRESHOLD) {
            // Simple threshold-based decision
            if (criticality > 0.8) return ProtectionLevel::FULL_TMR;
            if (criticality > 0.6) return ProtectionLevel::APPROXIMATE_TMR;
            if (criticality > 0.4) return ProtectionLevel::CHECKSUM_WITH_RECOVERY;
            if (criticality > 0.2) return ProtectionLevel::CHECKSUM_ONLY;
            return ProtectionLevel::NONE;
        }
        else if (strategy == HardeningStrategy::RESOURCE_CONSTRAINED) {
            // More granular approach for resource constrained scenario
            if (criticality > 0.9) return ProtectionLevel::FULL_TMR;
            if (criticality > 0.75) return ProtectionLevel::APPROXIMATE_TMR;
            if (criticality > 0.6) return ProtectionLevel::SELECTIVE_TMR;
            if (criticality > 0.4) return ProtectionLevel::CHECKSUM_WITH_RECOVERY;
            if (criticality > 0.2) return ProtectionLevel::CHECKSUM_ONLY;
            return ProtectionLevel::NONE;
        }
        else {
            // Default fallback
            if (criticality > 0.5) return ProtectionLevel::CHECKSUM_WITH_RECOVERY;
            if (criticality > 0.2) return ProtectionLevel::CHECKSUM_ONLY;
            return ProtectionLevel::NONE;
        }
    }
    
    /**
     * @brief Determine the specific strategy variant based on layer characteristics
     * 
     * @param layer_name Name of the layer
     * @param criticality Criticality score
     * @param analysis Analysis results
     * @return Strategy variant name
     */
    std::string getStrategyVariant(
        const std::string& layer_name,
        double criticality,
        const TopologicalAnalysisResult& analysis) {
        
        // Check if this is an information bottleneck
        if (analysis.information_bottleneck_score.count(layer_name) && 
            analysis.information_bottleneck_score.at(layer_name) > 0.5) {
            return "bottleneck_protection";
        }
        
        // Check if this layer has high fan-out (distributes errors widely)
        if (analysis.fan_out_count.count(layer_name) && 
            analysis.fan_out_count.at(layer_name) > 2) {
            return "fanout_isolation";
        }
        
        // Check if this is on a critical path
        for (const auto& path : analysis.critical_paths) {
            if (path.find(layer_name) != std::string::npos) {
                return "critical_path_protection";
            }
        }
        
        // Default variant based on criticality
        if (criticality > 0.8) return "high_criticality";
        if (criticality > 0.5) return "medium_criticality";
        return "low_criticality";
    }
};

} // namespace neural
} // namespace rad_ml 