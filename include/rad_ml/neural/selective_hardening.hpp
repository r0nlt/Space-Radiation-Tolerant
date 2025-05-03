#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>

#include "../tmr/approximate_tmr.hpp"
#include "../tmr/health_weighted_tmr.hpp"
#include "../tmr/enhanced_tmr.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Protection level for neural network components
 */
enum class ProtectionLevel {
    NONE,               ///< No protection
    CHECKSUM_ONLY,      ///< Only use CRC checksum
    APPROXIMATE_TMR,    ///< Use approximate TMR with reduced precision
    HEALTH_WEIGHTED_TMR,///< Use health-weighted TMR
    FULL_TMR            ///< Use full TMR with CRC checksums
};

/**
 * @brief Criticality analysis metrics for neural network components
 */
struct CriticalityMetrics {
    double sensitivity;          ///< Sensitivity to bit flips (0-1)
    double activation_frequency; ///< How often component is active (0-1)
    double output_influence;     ///< Influence on final output (0-1)
    double complexity;           ///< Implementation complexity cost (0-1)
    double memory_usage;         ///< Memory usage factor (0-1)
    
    /**
     * @brief Calculate overall criticality score
     * 
     * @param weights Weights for different metrics
     * @return Overall criticality score (0-1)
     */
    double calculateScore(const std::map<std::string, double>& weights = {}) const {
        // Default weights
        double w_sens = weights.count("sensitivity") ? weights.at("sensitivity") : 0.35;
        double w_freq = weights.count("activation_frequency") ? weights.at("activation_frequency") : 0.2;
        double w_infl = weights.count("output_influence") ? weights.at("output_influence") : 0.3;
        double w_comp = weights.count("complexity") ? weights.at("complexity") : 0.1;
        double w_mem = weights.count("memory_usage") ? weights.at("memory_usage") : 0.05;
        
        // Normalize weights
        double sum = w_sens + w_freq + w_infl + w_comp + w_mem;
        w_sens /= sum;
        w_freq /= sum;
        w_infl /= sum;
        w_comp /= sum;
        w_mem /= sum;
        
        // Calculate weighted score
        return (sensitivity * w_sens) +
               (activation_frequency * w_freq) +
               (output_influence * w_infl) +
               (complexity * w_comp) +
               (memory_usage * w_mem);
    }
};

/**
 * @brief Component of a neural network that can be protected
 */
struct NetworkComponent {
    std::string id;                ///< Unique identifier
    std::string type;              ///< Component type (weight, bias, activation)
    std::string layer_name;        ///< Layer name
    size_t layer_index;            ///< Layer index
    size_t index;                  ///< Component index within layer
    double value;                  ///< Component value
    CriticalityMetrics criticality; ///< Criticality metrics
    ProtectionLevel protection;    ///< Current protection level
};

/**
 * @brief Strategy for selective hardening
 */
enum class HardeningStrategy {
    FIXED_THRESHOLD,           ///< Protect components above a fixed criticality threshold
    RESOURCE_CONSTRAINED,      ///< Protect most critical components within a resource budget
    ADAPTIVE_RUNTIME,          ///< Adapt protection at runtime based on error rates
    LAYERWISE_IMPORTANCE,      ///< Protect based on layer importance (e.g., output layers more critical)
    GRADIENT_BASED             ///< Protect based on gradient magnitude during training
};

/**
 * @brief Result of a sensitivity analysis
 */
struct SensitivityAnalysisResult {
    std::vector<NetworkComponent> ranked_components;  ///< Components ranked by criticality
    std::map<std::string, double> layer_criticality;  ///< Layer-level criticality scores
    std::map<std::string, ProtectionLevel> protection_map; ///< Recommended protection levels
    double resource_usage;        ///< Estimated resource usage (0-1)
    double expected_error_rate;   ///< Expected error rate with protection
    double baseline_error_rate;   ///< Baseline error rate without protection
};

/**
 * @brief Configuration for the selective hardening analysis
 */
struct HardeningConfig {
    HardeningStrategy strategy;    ///< Strategy to use
    double criticality_threshold;  ///< Threshold for FIXED_THRESHOLD strategy (0-1)
    double resource_budget;        ///< Resource budget for RESOURCE_CONSTRAINED strategy (0-1)
    double protection_overhead;    ///< Overhead for each protection level (0-1)
    std::map<std::string, double> metric_weights; ///< Weights for criticality metrics
    
    /**
     * @brief Create default configuration
     */
    static HardeningConfig defaultConfig() {
        HardeningConfig config;
        config.strategy = HardeningStrategy::RESOURCE_CONSTRAINED;
        config.criticality_threshold = 0.7;
        config.resource_budget = 0.3;
        config.protection_overhead = 0.2;
        
        // Default metric weights
        config.metric_weights["sensitivity"] = 0.35;
        config.metric_weights["activation_frequency"] = 0.2;
        config.metric_weights["output_influence"] = 0.3;
        config.metric_weights["complexity"] = 0.1;
        config.metric_weights["memory_usage"] = 0.05;
        
        return config;
    }
};

/**
 * @brief Selective hardening for neural networks
 */
class SelectiveHardening {
public:
    /**
     * @brief Constructor with hardening configuration
     * 
     * @param config Hardening configuration
     */
    explicit SelectiveHardening(const HardeningConfig& config = HardeningConfig::defaultConfig())
        : config_(config) {}
    
    /**
     * @brief Update hardening configuration
     * 
     * @param config New configuration
     */
    void updateConfig(const HardeningConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief Get current hardening configuration
     * 
     * @return Current configuration
     */
    HardeningConfig getConfig() const {
        return config_;
    }
    
    /**
     * @brief Analyze network components and determine protection levels
     * 
     * @param components Network components to analyze
     * @return Analysis results including protection recommendations
     */
    SensitivityAnalysisResult analyzeAndProtect(
        const std::vector<NetworkComponent>& components) {
        
        SensitivityAnalysisResult result;
        
        // Copy components and sort by criticality score
        result.ranked_components = components;
        std::sort(result.ranked_components.begin(), result.ranked_components.end(), 
            [this](const NetworkComponent& a, const NetworkComponent& b) {
                return a.criticality.calculateScore(config_.metric_weights) > 
                       b.criticality.calculateScore(config_.metric_weights);
            });
        
        // Calculate layer-level criticality
        calculateLayerCriticality(components, result.layer_criticality);
        
        // Apply hardening strategy
        switch (config_.strategy) {
            case HardeningStrategy::FIXED_THRESHOLD:
                applyFixedThresholdStrategy(result);
                break;
            case HardeningStrategy::RESOURCE_CONSTRAINED:
                applyResourceConstrainedStrategy(result);
                break;
            case HardeningStrategy::LAYERWISE_IMPORTANCE:
                applyLayerwiseImportanceStrategy(result);
                break;
            case HardeningStrategy::GRADIENT_BASED:
                // This would require gradient information, use fallback
                applyResourceConstrainedStrategy(result);
                break;
            case HardeningStrategy::ADAPTIVE_RUNTIME:
                // Initial protection is same as resource constrained
                applyResourceConstrainedStrategy(result);
                break;
        }
        
        // Calculate resource usage and expected error rates
        calculateResourceUsage(result);
        calculateExpectedErrorRates(result);
        
        return result;
    }
    
    /**
     * @brief Apply protection to a component based on analysis results
     * 
     * @param value Value to protect
     * @param component_id Component identifier
     * @param analysis_results Analysis results with protection map
     * @return Protected value (template varies based on protection level)
     */
    template <typename T>
    auto applyProtection(
        T value, 
        const std::string& component_id, 
        const SensitivityAnalysisResult& analysis_results) {
        
        if (!analysis_results.protection_map.count(component_id)) {
            // No protection specified, return unprotected value
            return value;
        }
        
        ProtectionLevel level = analysis_results.protection_map.at(component_id);
        
        switch (level) {
            case ProtectionLevel::NONE:
                return value;
                
            case ProtectionLevel::CHECKSUM_ONLY: {
                // Simple wrapper with CRC
                struct ChecksumProtected {
                    T value;
                    uint32_t checksum;
                    
                    ChecksumProtected(T v) : value(v) {
                        // Simple CRC calculation
                        const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
                        size_t size = sizeof(T);
                        
                        uint32_t crc = 0xFFFFFFFF;
                        for (size_t i = 0; i < size; i++) {
                            crc ^= data[i];
                            for (int j = 0; j < 8; j++) {
                                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                            }
                        }
                        checksum = ~crc;
                    }
                    
                    bool verify() const {
                        // Recalculate and check
                        const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
                        size_t size = sizeof(T);
                        
                        uint32_t crc = 0xFFFFFFFF;
                        for (size_t i = 0; i < size; i++) {
                            crc ^= data[i];
                            for (int j = 0; j < 8; j++) {
                                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                            }
                        }
                        crc = ~crc;
                        
                        return crc == checksum;
                    }
                    
                    T getValue() const {
                        return value;
                    }
                };
                
                return ChecksumProtected(value);
            }
                
            case ProtectionLevel::APPROXIMATE_TMR: {
                // Use Approximate TMR with reduced precision
                return rad_ml::tmr::ApproximateTMR<T>(
                    value, 
                    {rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
                     rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
                     rad_ml::tmr::ApproximationType::EXACT}
                );
            }
                
            case ProtectionLevel::HEALTH_WEIGHTED_TMR: {
                // Use Health-Weighted TMR
                return rad_ml::tmr::HealthWeightedTMR<T>(value);
            }
                
            case ProtectionLevel::FULL_TMR: {
                // Use Enhanced TMR with CRC checksums
                return rad_ml::tmr::EnhancedTMR<T>(value);
            }
                
            default:
                return value;
        }
    }
    
    /**
     * @brief Update adaptive protection based on runtime error statistics
     * 
     * This is used with the ADAPTIVE_RUNTIME strategy to adjust protection
     * levels based on observed error rates during mission execution.
     * 
     * @param analysis_results Previous analysis results
     * @param error_counts Error counts per component
     * @param correction_counts Successful correction counts per component
     * @return Updated analysis results with adjusted protection levels
     */
    SensitivityAnalysisResult updateAdaptiveProtection(
        const SensitivityAnalysisResult& analysis_results,
        const std::map<std::string, int>& error_counts,
        const std::map<std::string, int>& correction_counts) {
        
        if (config_.strategy != HardeningStrategy::ADAPTIVE_RUNTIME) {
            // Not in adaptive mode, return unchanged
            return analysis_results;
        }
        
        SensitivityAnalysisResult updated_results = analysis_results;
        
        // For each component with error statistics
        for (const auto& [component_id, error_count] : error_counts) {
            if (error_count == 0) continue;
            
            // Get correction count (default to 0 if not present)
            int corrections = 0;
            if (correction_counts.count(component_id)) {
                corrections = correction_counts.at(component_id);
            }
            
            // Calculate correction rate
            double correction_rate = (error_count > 0) 
                ? static_cast<double>(corrections) / error_count 
                : 1.0;
            
            // Current protection level
            ProtectionLevel current_level = ProtectionLevel::NONE;
            if (analysis_results.protection_map.count(component_id)) {
                current_level = analysis_results.protection_map.at(component_id);
            }
            
            // Determine if we need to increase protection
            if (correction_rate < 0.8 && current_level != ProtectionLevel::FULL_TMR) {
                // Increase protection level
                ProtectionLevel new_level;
                
                switch (current_level) {
                    case ProtectionLevel::NONE:
                        new_level = ProtectionLevel::CHECKSUM_ONLY;
                        break;
                    case ProtectionLevel::CHECKSUM_ONLY:
                        new_level = ProtectionLevel::APPROXIMATE_TMR;
                        break;
                    case ProtectionLevel::APPROXIMATE_TMR:
                        new_level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
                        break;
                    case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                        new_level = ProtectionLevel::FULL_TMR;
                        break;
                    default:
                        new_level = current_level;
                }
                
                updated_results.protection_map[component_id] = new_level;
            }
            else if (correction_rate > 0.95 && error_count < 3 && 
                    current_level != ProtectionLevel::NONE) {
                // Consider decreasing protection if correction rate is high
                // and error count is low
                
                ProtectionLevel new_level;
                
                switch (current_level) {
                    case ProtectionLevel::FULL_TMR:
                        new_level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
                        break;
                    case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                        new_level = ProtectionLevel::APPROXIMATE_TMR;
                        break;
                    case ProtectionLevel::APPROXIMATE_TMR:
                        new_level = ProtectionLevel::CHECKSUM_ONLY;
                        break;
                    case ProtectionLevel::CHECKSUM_ONLY:
                        new_level = ProtectionLevel::NONE;
                        break;
                    default:
                        new_level = current_level;
                }
                
                updated_results.protection_map[component_id] = new_level;
            }
        }
        
        // Recalculate resource usage and expected error rates
        calculateResourceUsage(updated_results);
        calculateExpectedErrorRates(updated_results);
        
        return updated_results;
    }
    
    /**
     * @brief Get a human-readable report about the protection scheme
     * 
     * @param analysis_results Analysis results
     * @return Formatted report string
     */
    std::string getProtectionReport(const SensitivityAnalysisResult& analysis_results) const {
        std::string report = "Selective Hardening Protection Report\n";
        report += "----------------------------------\n\n";
        
        // Strategy information
        report += "Strategy: " + getStrategyName(config_.strategy) + "\n";
        report += "Resource Usage: " + std::to_string(analysis_results.resource_usage * 100.0) + 
                 "% of available budget (" + std::to_string(config_.resource_budget * 100.0) + "%)\n";
        report += "Expected Error Rate: " + std::to_string(analysis_results.expected_error_rate) + 
                 " (Baseline: " + std::to_string(analysis_results.baseline_error_rate) + ")\n\n";
        
        // Protection level counts
        std::map<ProtectionLevel, int> protection_counts;
        for (const auto& [id, level] : analysis_results.protection_map) {
            protection_counts[level]++;
        }
        
        report += "Protection Level Distribution:\n";
        for (const auto& [level, count] : protection_counts) {
            report += "  " + getProtectionLevelName(level) + ": " + 
                     std::to_string(count) + " components\n";
        }
        report += "\n";
        
        // Layer criticality
        report += "Layer Criticality Scores:\n";
        for (const auto& [layer, score] : analysis_results.layer_criticality) {
            report += "  " + layer + ": " + std::to_string(score) + "\n";
        }
        report += "\n";
        
        // Top 10 most critical components
        report += "Top 10 Most Critical Components:\n";
        int count = 0;
        for (const auto& comp : analysis_results.ranked_components) {
            if (count++ >= 10) break;
            
            ProtectionLevel level = ProtectionLevel::NONE;
            if (analysis_results.protection_map.count(comp.id)) {
                level = analysis_results.protection_map.at(comp.id);
            }
            
            report += "  " + comp.id + " (Layer: " + comp.layer_name + 
                     ", Type: " + comp.type + 
                     ", Criticality: " + std::to_string(comp.criticality.calculateScore(config_.metric_weights)) + 
                     ", Protection: " + getProtectionLevelName(level) + ")\n";
        }
        
        return report;
    }
    
private:
    HardeningConfig config_;
    
    /**
     * @brief Calculate criticality scores for each layer
     * 
     * @param components Network components
     * @param layer_criticality Output map of layer name to criticality score
     */
    void calculateLayerCriticality(
        const std::vector<NetworkComponent>& components,
        std::map<std::string, double>& layer_criticality) {
        
        // Group components by layer
        std::map<std::string, std::vector<NetworkComponent>> layer_components;
        for (const auto& comp : components) {
            layer_components[comp.layer_name].push_back(comp);
        }
        
        // Calculate average criticality for each layer
        for (const auto& [layer_name, comps] : layer_components) {
            double total_score = 0.0;
            for (const auto& comp : comps) {
                total_score += comp.criticality.calculateScore(config_.metric_weights);
            }
            layer_criticality[layer_name] = total_score / comps.size();
        }
    }
    
    /**
     * @brief Apply fixed threshold hardening strategy
     * 
     * @param result Analysis results to update
     */
    void applyFixedThresholdStrategy(SensitivityAnalysisResult& result) {
        for (const auto& comp : result.ranked_components) {
            double score = comp.criticality.calculateScore(config_.metric_weights);
            
            if (score >= config_.criticality_threshold) {
                result.protection_map[comp.id] = ProtectionLevel::FULL_TMR;
            }
            else if (score >= config_.criticality_threshold * 0.8) {
                result.protection_map[comp.id] = ProtectionLevel::HEALTH_WEIGHTED_TMR;
            }
            else if (score >= config_.criticality_threshold * 0.6) {
                result.protection_map[comp.id] = ProtectionLevel::APPROXIMATE_TMR;
            }
            else if (score >= config_.criticality_threshold * 0.4) {
                result.protection_map[comp.id] = ProtectionLevel::CHECKSUM_ONLY;
            }
            else {
                result.protection_map[comp.id] = ProtectionLevel::NONE;
            }
        }
    }
    
    /**
     * @brief Apply resource constrained hardening strategy
     * 
     * @param result Analysis results to update
     */
    void applyResourceConstrainedStrategy(SensitivityAnalysisResult& result) {
        double usage = 0.0;
        double budget = config_.resource_budget;
        
        // Calculate protection level costs
        std::map<ProtectionLevel, double> level_costs = {
            {ProtectionLevel::NONE, 0.0},
            {ProtectionLevel::CHECKSUM_ONLY, 0.05},
            {ProtectionLevel::APPROXIMATE_TMR, 0.15},
            {ProtectionLevel::HEALTH_WEIGHTED_TMR, 0.25},
            {ProtectionLevel::FULL_TMR, 0.33}
        };
        
        // Start with no protection
        for (const auto& comp : result.ranked_components) {
            result.protection_map[comp.id] = ProtectionLevel::NONE;
        }
        
        // Allocate protection to most critical components first
        for (const auto& comp : result.ranked_components) {
            // Try to apply highest level of protection possible within budget
            for (ProtectionLevel level : {
                ProtectionLevel::FULL_TMR,
                ProtectionLevel::HEALTH_WEIGHTED_TMR,
                ProtectionLevel::APPROXIMATE_TMR,
                ProtectionLevel::CHECKSUM_ONLY
            }) {
                double cost = level_costs[level] * (1.0 + comp.criticality.complexity);
                
                if (usage + cost <= budget) {
                    result.protection_map[comp.id] = level;
                    usage += cost;
                    break;
                }
            }
            
            // If we've used up the budget, stop
            if (usage >= budget) {
                break;
            }
        }
    }
    
    /**
     * @brief Apply layerwise importance hardening strategy
     * 
     * @param result Analysis results to update
     */
    void applyLayerwiseImportanceStrategy(SensitivityAnalysisResult& result) {
        // Sort layers by criticality
        std::vector<std::pair<std::string, double>> sorted_layers;
        for (const auto& [layer, score] : result.layer_criticality) {
            sorted_layers.push_back({layer, score});
        }
        
        std::sort(sorted_layers.begin(), sorted_layers.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
        
        // Assign protection levels based on layer criticality
        double total_layers = sorted_layers.size();
        double usage = 0.0;
        double budget = config_.resource_budget;
        
        for (const auto& comp : result.ranked_components) {
            // Find layer position in sorted list
            auto it = std::find_if(sorted_layers.begin(), sorted_layers.end(),
                [&](const auto& layer_pair) {
                    return layer_pair.first == comp.layer_name;
                });
            
            if (it == sorted_layers.end()) continue;
            
            size_t layer_pos = std::distance(sorted_layers.begin(), it);
            double layer_percentile = 1.0 - (layer_pos / total_layers);
            
            // Assign protection level based on layer criticality
            ProtectionLevel level;
            double component_cost = 0.0;
            
            if (layer_percentile >= 0.8) {
                level = ProtectionLevel::FULL_TMR;
                component_cost = 0.33;
            }
            else if (layer_percentile >= 0.6) {
                level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
                component_cost = 0.25;
            }
            else if (layer_percentile >= 0.4) {
                level = ProtectionLevel::APPROXIMATE_TMR;
                component_cost = 0.15;
            }
            else if (layer_percentile >= 0.2) {
                level = ProtectionLevel::CHECKSUM_ONLY;
                component_cost = 0.05;
            }
            else {
                level = ProtectionLevel::NONE;
                component_cost = 0.0;
            }
            
            // Adjust based on individual component criticality
            double score = comp.criticality.calculateScore(config_.metric_weights);
            if (score > 0.8 && level != ProtectionLevel::FULL_TMR) {
                // Upgrade protection if highly critical
                if (level == ProtectionLevel::HEALTH_WEIGHTED_TMR) {
                    level = ProtectionLevel::FULL_TMR;
                    component_cost = 0.33;
                }
                else if (level == ProtectionLevel::APPROXIMATE_TMR) {
                    level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
                    component_cost = 0.25;
                }
                else if (level == ProtectionLevel::CHECKSUM_ONLY) {
                    level = ProtectionLevel::APPROXIMATE_TMR;
                    component_cost = 0.15;
                }
                else if (level == ProtectionLevel::NONE) {
                    level = ProtectionLevel::CHECKSUM_ONLY;
                    component_cost = 0.05;
                }
            }
            
            // Check budget
            if (usage + component_cost <= budget) {
                result.protection_map[comp.id] = level;
                usage += component_cost;
            }
            else {
                result.protection_map[comp.id] = ProtectionLevel::NONE;
            }
        }
    }
    
    /**
     * @brief Calculate resource usage for protection scheme
     * 
     * @param result Analysis results to update
     */
    void calculateResourceUsage(SensitivityAnalysisResult& result) {
        double usage = 0.0;
        
        // Protection level costs
        std::map<ProtectionLevel, double> level_costs = {
            {ProtectionLevel::NONE, 0.0},
            {ProtectionLevel::CHECKSUM_ONLY, 0.05},
            {ProtectionLevel::APPROXIMATE_TMR, 0.15},
            {ProtectionLevel::HEALTH_WEIGHTED_TMR, 0.25},
            {ProtectionLevel::FULL_TMR, 0.33}
        };
        
        for (const auto& [id, level] : result.protection_map) {
            // Find component to get complexity
            auto it = std::find_if(result.ranked_components.begin(), result.ranked_components.end(),
                [&](const NetworkComponent& comp) {
                    return comp.id == id;
                });
            
            if (it == result.ranked_components.end()) continue;
            
            // Add component cost adjusted for complexity
            usage += level_costs[level] * (1.0 + it->criticality.complexity);
        }
        
        result.resource_usage = usage;
    }
    
    /**
     * @brief Calculate expected error rates
     * 
     * @param result Analysis results to update
     */
    void calculateExpectedErrorRates(SensitivityAnalysisResult& result) {
        // Simple model for error rates
        double baseline_rate = 0.0;
        double protected_rate = 0.0;
        
        // Protection level error reduction factors
        std::map<ProtectionLevel, double> error_reduction = {
            {ProtectionLevel::NONE, 0.0},
            {ProtectionLevel::CHECKSUM_ONLY, 0.3},
            {ProtectionLevel::APPROXIMATE_TMR, 0.7},
            {ProtectionLevel::HEALTH_WEIGHTED_TMR, 0.9},
            {ProtectionLevel::FULL_TMR, 0.99}
        };
        
        for (const auto& comp : result.ranked_components) {
            double comp_error_rate = comp.criticality.sensitivity * 0.01;
            baseline_rate += comp_error_rate;
            
            // Apply protection
            ProtectionLevel level = ProtectionLevel::NONE;
            if (result.protection_map.count(comp.id)) {
                level = result.protection_map.at(comp.id);
            }
            
            double reduction = error_reduction[level];
            protected_rate += comp_error_rate * (1.0 - reduction);
        }
        
        result.baseline_error_rate = baseline_rate;
        result.expected_error_rate = protected_rate;
    }
    
    /**
     * @brief Get human-readable name for a protection level
     * 
     * @param level Protection level
     * @return Name string
     */
    std::string getProtectionLevelName(ProtectionLevel level) const {
        switch (level) {
            case ProtectionLevel::NONE:
                return "No Protection";
            case ProtectionLevel::CHECKSUM_ONLY:
                return "Checksum Only";
            case ProtectionLevel::APPROXIMATE_TMR:
                return "Approximate TMR";
            case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                return "Health-Weighted TMR";
            case ProtectionLevel::FULL_TMR:
                return "Full TMR";
            default:
                return "Unknown";
        }
    }
    
    /**
     * @brief Get human-readable name for a hardening strategy
     * 
     * @param strategy Hardening strategy
     * @return Name string
     */
    std::string getStrategyName(HardeningStrategy strategy) const {
        switch (strategy) {
            case HardeningStrategy::FIXED_THRESHOLD:
                return "Fixed Threshold";
            case HardeningStrategy::RESOURCE_CONSTRAINED:
                return "Resource Constrained";
            case HardeningStrategy::ADAPTIVE_RUNTIME:
                return "Adaptive Runtime";
            case HardeningStrategy::LAYERWISE_IMPORTANCE:
                return "Layerwise Importance";
            case HardeningStrategy::GRADIENT_BASED:
                return "Gradient Based";
            default:
                return "Unknown";
        }
    }
};

} // namespace neural
} // namespace rad_ml 