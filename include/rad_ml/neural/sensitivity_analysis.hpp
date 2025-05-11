/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "../error/error_handling.hpp"
#include "../radiation/environment.hpp"
#include "selective_hardening.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Layer information for network analysis
 */
struct LayerInfo {
    std::string name;
    std::string type;  // conv, dense, pool, etc.
    size_t index;
    size_t input_size;
    size_t output_size;
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
    bool is_critical;  // Manually marked as critical
};

/**
 * @brief Parameter information for network analysis
 */
struct ParameterInfo {
    std::string id;
    std::string layer_name;
    std::string param_type;  // weight, bias, etc.
    size_t index;
    double value;
};

/**
 * @brief Activation sample for analysis
 */
struct ActivationSample {
    std::string layer_name;
    std::vector<double> values;
    double sparsity;  // Percentage of zero values
};

/**
 * @brief Sample for sensitivity analysis
 */
struct SampleType {
    std::vector<double> input;
    std::vector<double> expected_output;
};

/**
 * @brief Result of topological network analysis
 */
struct TopologicalAnalysisResult {
    std::map<std::string, double> layer_criticality;
    std::map<std::string, double> propagation_factors;
    std::vector<std::string> critical_paths;
    std::map<std::string, size_t> fan_in_count;  // Input connections per layer
    std::map<std::string, size_t> fan_out_count; // Output connections per layer
    std::map<std::string, double> information_bottleneck_score;
};

/**
 * @brief Extended topological analysis result with space environment considerations
 */
struct SpaceAwareTopologicalResult : public TopologicalAnalysisResult {
    // Radiation vulnerability factors per layer
    std::map<std::string, double> radiation_vulnerability;
    
    // Thermal sensitivity factors per layer
    std::map<std::string, double> thermal_sensitivity;
    
    // Power consumption impact per layer
    std::map<std::string, double> power_impact;
    
    // Overall space environment impact score (0-1)
    double space_environment_impact = 0.0;
    
    // Reference to radiation environment
    std::shared_ptr<radiation::Environment> environment;
};

/**
 * @brief Analyzer for neural network topology
 */
class TopologicalAnalyzer {
public:
    /**
     * @brief Analyze network structure to determine criticality
     * 
     * @tparam NetworkType Type of neural network
     * @param network Network to analyze
     * @return Result of topological analysis
     */
    template <typename NetworkType>
    TopologicalAnalysisResult analyzeNetwork(const NetworkType& network) {
        TopologicalAnalysisResult result;
        
        // Extract network topology
        auto layers = extractNetworkLayers(network);
        
        // Calculate connectivity metrics
        calculateConnectivityMetrics(layers, result);
        
        // Calculate input-to-output influence paths
        calculateCriticalPaths(layers, result.critical_paths);
        
        // Determine propagation factors (how errors multiply through the network)
        for (const auto& layer : layers) {
            result.propagation_factors[layer.name] = 
                calculatePropagationFactor(layer, layers);
        }
        
        // Calculate information bottleneck scores
        calculateInformationBottlenecks(layers, result);
        
        // Assign criticality scores based on topology
        for (const auto& layer : layers) {
            result.layer_criticality[layer.name] = 
                calculateTopologicalCriticality(layer, result);
        }
        
        return result;
    }

private:
    /**
     * @brief Extract layer information from a network
     * 
     * @tparam NetworkType Type of neural network
     * @param network Network to analyze
     * @return Vector of layer information
     */
    template <typename NetworkType>
    std::vector<LayerInfo> extractNetworkLayers(const NetworkType& network) {
        // This would need to be specialized for different network types
        // Here's a generic implementation that uses the network's getLayerInfo if available
        if constexpr (has_get_layer_info<NetworkType>::value) {
            return network.getLayerInfo();
        } 
        else {
            // Fallback implementation for networks without direct layer info access
            std::vector<LayerInfo> layers;
            // Try to infer layers based on network structure
            // This is a placeholder and would need to be implemented for specific network types
            return layers;
        }
    }
    
    /**
     * @brief Calculate connectivity metrics for layers
     * 
     * @param layers Network layers
     * @param result Analysis result to update
     */
    void calculateConnectivityMetrics(
        const std::vector<LayerInfo>& layers,
        TopologicalAnalysisResult& result) {
        
        // Initialize counters
        for (const auto& layer : layers) {
            result.fan_in_count[layer.name] = 0;
            result.fan_out_count[layer.name] = 0;
        }
        
        // Count connections
        for (const auto& layer : layers) {
            result.fan_in_count[layer.name] += layer.input_layers.size();
            
            for (const auto& output_layer : layer.output_layers) {
                result.fan_out_count[layer.name]++;
            }
        }
    }
    
    /**
     * @brief Find critical paths from input to output
     * 
     * @param layers Network layers
     * @param critical_paths Vector to store critical paths
     */
    void calculateCriticalPaths(
        const std::vector<LayerInfo>& layers,
        std::vector<std::string>& critical_paths) {
        
        // Find input and output layers
        std::vector<std::string> input_layers;
        std::vector<std::string> output_layers;
        
        for (const auto& layer : layers) {
            if (layer.input_layers.empty()) {
                input_layers.push_back(layer.name);
            }
            
            if (layer.output_layers.empty()) {
                output_layers.push_back(layer.name);
            }
        }
        
        // Build layer map for quick lookups
        std::unordered_map<std::string, LayerInfo> layer_map;
        for (const auto& layer : layers) {
            layer_map[layer.name] = layer;
        }
        
        // Find all paths from input to output layers
        for (const auto& input_layer : input_layers) {
            for (const auto& output_layer : output_layers) {
                findPaths(input_layer, output_layer, layer_map, critical_paths);
            }
        }
    }
    
    /**
     * @brief Helper to find all paths between two layers
     */
    void findPaths(
        const std::string& start,
        const std::string& end,
        const std::unordered_map<std::string, LayerInfo>& layer_map,
        std::vector<std::string>& critical_paths,
        std::unordered_set<std::string> visited = {},
        std::string current_path = "") {
        
        visited.insert(start);
        
        if (current_path.empty()) {
            current_path = start;
        } else {
            current_path += " -> " + start;
        }
        
        if (start == end) {
            critical_paths.push_back(current_path);
        } else {
            auto it = layer_map.find(start);
            if (it != layer_map.end()) {
                for (const auto& next_layer : it->second.output_layers) {
                    if (visited.find(next_layer) == visited.end()) {
                        findPaths(next_layer, end, layer_map, critical_paths, visited, current_path);
                    }
                }
            }
        }
        
        // Remove from visited for backtracking
        visited.erase(start);
    }
    
    /**
     * @brief Calculate propagation factor for a layer
     * 
     * @param layer Layer to analyze
     * @param layers All layers in the network
     * @return Propagation factor
     */
    double calculatePropagationFactor(
        const LayerInfo& layer,
        const std::vector<LayerInfo>& layers) {
        
        // Calculate how much errors can propagate and multiply
        double propagation_factor = 1.0;
        
        // Different layer types have different propagation characteristics
        if (layer.type == "dense") {
            // Dense layers can propagate errors widely
            propagation_factor = std::sqrt(layer.output_size / static_cast<double>(layer.input_size));
        } 
        else if (layer.type == "conv") {
            // Convolutional layers propagate locally
            propagation_factor = 0.7;
        } 
        else if (layer.type == "pool") {
            // Pooling can reduce propagation but concentrate errors
            propagation_factor = 0.5;
        } 
        else if (layer.type == "dropout") {
            // Dropout adds robustness
            propagation_factor = 0.3;
        } 
        else if (layer.type == "batch_norm") {
            // Batch normalization can amplify errors
            propagation_factor = 1.2;
        } 
        else {
            // Default for unknown layer types
            propagation_factor = 1.0;
        }
        
        // Adjust for fan-out - more outputs means more propagation
        size_t fan_out = 0;
        for (const auto& out_layer : layer.output_layers) {
            fan_out++;
        }
        
        if (fan_out > 1) {
            propagation_factor *= (1.0 + 0.1 * std::log(fan_out));
        }
        
        return propagation_factor;
    }
    
    /**
     * @brief Calculate information bottlenecks in the network
     */
    void calculateInformationBottlenecks(
        const std::vector<LayerInfo>& layers,
        TopologicalAnalysisResult& result) {
        
        for (const auto& layer : layers) {
            double bottleneck_score = 0.0;
            
            // Get fan-in and fan-out
            size_t fan_in = result.fan_in_count[layer.name];
            size_t fan_out = result.fan_out_count[layer.name];
            
            // Check input/output size ratio
            double size_ratio = layer.output_size / static_cast<double>(layer.input_size);
            
            // Calculate bottleneck score
            // Higher score = more of a bottleneck = more critical
            if (fan_in > 1 && size_ratio < 1.0) {
                // Converging information from multiple sources and compressing
                bottleneck_score = (fan_in * 0.3) * (1.0 - size_ratio);
            }
            else if (fan_out > 1 && size_ratio < 1.0) {
                // Information compressed and then distributed widely
                bottleneck_score = (fan_out * 0.3) * (1.0 - size_ratio);
            }
            else {
                // Not a significant bottleneck
                bottleneck_score = 0.1;
            }
            
            // Bottlenecks in the middle of the network are most critical
            if (!layer.input_layers.empty() && !layer.output_layers.empty()) {
                bottleneck_score *= 1.5;
            }
            
            result.information_bottleneck_score[layer.name] = bottleneck_score;
        }
    }
    
    /**
     * @brief Calculate topological criticality for a layer
     */
    double calculateTopologicalCriticality(
        const LayerInfo& layer,
        const TopologicalAnalysisResult& analysis) {
        
        double criticality = 0.0;
        
        // Consider propagation factor (25%)
        double propagation_contrib = 0.0;
        if (analysis.propagation_factors.count(layer.name)) {
            propagation_contrib = analysis.propagation_factors.at(layer.name) / 2.0; // Normalize to 0-0.5 range
        }
        
        // Consider information bottleneck (25%)
        double bottleneck_contrib = 0.0;
        if (analysis.information_bottleneck_score.count(layer.name)) {
            bottleneck_contrib = std::min(0.5, analysis.information_bottleneck_score.at(layer.name));
        }
        
        // Consider connectivity (25%)
        double connectivity_factor = 0.0;
        if (analysis.fan_in_count.count(layer.name) && analysis.fan_out_count.count(layer.name)) {
            size_t fan_in = analysis.fan_in_count.at(layer.name);
            size_t fan_out = analysis.fan_out_count.at(layer.name);
            
            // Normalize to 0-0.5 range
            connectivity_factor = std::min(0.5, (fan_in + fan_out) * 0.1);
        }
        
        // Consider critical paths (25%)
        double critical_path_contrib = 0.0;
        for (const auto& path : analysis.critical_paths) {
            if (path.find(layer.name) != std::string::npos) {
                critical_path_contrib += 0.1;
            }
        }
        critical_path_contrib = std::min(0.5, critical_path_contrib);
        
        // Sum all contributions
        criticality = propagation_contrib + bottleneck_contrib + 
                      connectivity_factor + critical_path_contrib;
        
        // Add bonus for manually marked critical layers
        if (layer.is_critical) {
            criticality += 0.2;
        }
        
        // Normalize to 0-1 range
        return std::min(1.0, criticality);
    }
    
    // Helper to check if a network type has getLayerInfo method
    template<typename T, typename = void>
    struct has_get_layer_info : std::false_type {};
    
    template<typename T>
    struct has_get_layer_info<T, 
        std::void_t<decltype(std::declval<T>().getLayerInfo())>> 
        : std::true_type {};
};

/**
 * @brief Gradient-based importance mapping for neural networks
 */
class GradientImportanceMapper {
public:
    /**
     * @brief Calculate parameter importance using gradient-based sensitivity
     * 
     * @tparam NetworkType Type of neural network
     * @tparam DatasetType Type of dataset container
     * @param network Network to analyze
     * @param calibration_dataset Dataset for calibration
     * @return Importance scores for network parameters
     */
    template <typename NetworkType, typename DatasetType>
    std::map<std::string, double> calculateParameterImportance(
        const NetworkType& network,
        const DatasetType& calibration_dataset) {
        
        std::map<std::string, double> importance_map;
        
        // Get network parameters
        auto parameters = getNetworkParameters(network);
        
        // For each parameter in the network
        for (const auto& param : parameters) {
            // Compute sensitivity by measuring output delta over parameter delta
            double sensitivity = 0.0;
            
            // For each sample in calibration dataset (limit to first 50 samples for efficiency)
            const size_t max_samples = std::min(calibration_dataset.size(), size_t(50));
            for (size_t i = 0; i < max_samples; ++i) {
                const auto& sample = calibration_dataset[i];
                
                // Original output
                auto original_output = network.forward(sample.input);
                
                // Perturb parameter slightly
                auto perturbed_network = network;
                perturbParameter(perturbed_network, param.id, param.value * 1.001);
                
                // Get new output
                auto perturbed_output = perturbed_network.forward(sample.input);
                
                // Measure output difference
                double output_delta = calculateOutputDifference(
                    original_output, perturbed_output);
                    
                // Accumulate sensitivity
                sensitivity += output_delta / (param.value * 0.001);
            }
            
            // Average sensitivity across dataset
            sensitivity /= max_samples;
            
            // Store absolute sensitivity in importance map
            importance_map[param.id] = std::abs(sensitivity);
        }
        
        // Normalize importance values
        normalizeImportanceMap(importance_map);
        
        return importance_map;
    }
    
    /**
     * @brief Calculate activation-based importance for layers
     * 
     * @tparam NetworkType Type of neural network
     * @tparam DatasetType Type of dataset container
     * @param network Network to analyze
     * @param calibration_dataset Dataset for calibration
     * @return Activation-based importance for each layer
     */
    template <typename NetworkType, typename DatasetType>
    std::map<std::string, double> calculateActivationImportance(
        const NetworkType& network,
        const DatasetType& calibration_dataset) {
        
        std::map<std::string, double> activation_importance;
        std::map<std::string, double> sparsity_scores;
        std::map<std::string, double> magnitude_scores;
        
        // For each sample in calibration dataset (limit to first 50 samples for efficiency)
        const size_t max_samples = std::min(calibration_dataset.size(), size_t(50));
        for (size_t i = 0; i < max_samples; ++i) {
            const auto& sample = calibration_dataset[i];
            
            // Get layer activations
            auto activations = network.getLayerActivations(sample.input);
            
            // Process each layer's activation
            for (const auto& activation : activations) {
                // Calculate sparsity (% of zeros or near-zeros)
                double sparsity = calculateSparsity(activation.values);
                
                // Calculate magnitude (L2 norm)
                double magnitude = calculateMagnitude(activation.values);
                
                // Accumulate scores
                sparsity_scores[activation.layer_name] += sparsity;
                magnitude_scores[activation.layer_name] += magnitude;
            }
        }
        
        // Average and combine scores
        for (auto& [layer_name, sparsity] : sparsity_scores) {
            sparsity /= max_samples;
            double magnitude = magnitude_scores[layer_name] / max_samples;
            
            // Sparse activations with high magnitude are more critical
            // (they represent concentrated information flow)
            activation_importance[layer_name] = (1.0 - sparsity) * magnitude;
        }
        
        // Normalize importance values
        normalizeImportanceMap(activation_importance);
        
        return activation_importance;
    }

private:
    /**
     * @brief Extract parameters from a network
     * 
     * @tparam NetworkType Type of neural network
     * @param network Network to extract parameters from
     * @return Vector of parameter information
     */
    template <typename NetworkType>
    std::vector<ParameterInfo> getNetworkParameters(const NetworkType& network) {
        // This would need to be specialized for different network types
        // Here's a generic implementation that uses the network's getParameters if available
        if constexpr (has_get_parameters<NetworkType>::value) {
            return network.getParameters();
        } 
        else {
            // Fallback implementation for networks without direct parameter access
            std::vector<ParameterInfo> parameters;
            // Try to infer parameters based on network structure
            // This is a placeholder and would need to be implemented for specific network types
            return parameters;
        }
    }
    
    /**
     * @brief Perturb a specific parameter in the network
     * 
     * @tparam NetworkType Type of neural network
     * @param network Network to modify
     * @param param_id Parameter identifier
     * @param new_value New parameter value
     */
    template <typename NetworkType>
    void perturbParameter(NetworkType& network, const std::string& param_id, double new_value) {
        // This would need to be specialized for different network types
        // Here's a generic implementation that uses the network's setParameterValue if available
        if constexpr (has_set_parameter<NetworkType>::value) {
            network.setParameterValue(param_id, new_value);
        }
        else {
            // Fallback implementation would depend on specific network type
            // This is a placeholder and would need to be implemented for specific network types
            error::ErrorHandler::logError(
                error::ErrorInfo(
                    error::ErrorCode::OPERATION_FAILED,
                    error::ErrorCategory::VALIDATION,
                    error::ErrorSeverity::WARNING,
                    "Parameter perturbation not implemented for this network type"
                )
            );
        }
    }
    
    /**
     * @brief Calculate difference between two network outputs
     * 
     * @tparam OutputType Type of network output
     * @param output1 First output
     * @param output2 Second output
     * @return Difference measure
     */
    template <typename OutputType>
    double calculateOutputDifference(const OutputType& output1, const OutputType& output2) {
        // For vector outputs, use L2 norm of difference
        if constexpr (is_vector<OutputType>::value) {
            double sum_squared_diff = 0.0;
            for (size_t i = 0; i < output1.size(); ++i) {
                double diff = output1[i] - output2[i];
                sum_squared_diff += diff * diff;
            }
            return std::sqrt(sum_squared_diff);
        }
        // For scalar outputs, use absolute difference
        else {
            return std::abs(output1 - output2);
        }
    }
    
    /**
     * @brief Calculate sparsity of activation values
     * 
     * @param values Activation values
     * @return Sparsity measure (0-1, higher means more sparse)
     */
    double calculateSparsity(const std::vector<double>& values) {
        if (values.empty()) {
            return 0.0;
        }
        
        const double threshold = 1e-6;
        size_t zero_count = 0;
        
        for (double value : values) {
            if (std::abs(value) < threshold) {
                zero_count++;
            }
        }
        
        return static_cast<double>(zero_count) / values.size();
    }
    
    /**
     * @brief Calculate magnitude of activation values
     * 
     * @param values Activation values
     * @return Magnitude (L2 norm)
     */
    double calculateMagnitude(const std::vector<double>& values) {
        double sum_squared = 0.0;
        for (double value : values) {
            sum_squared += value * value;
        }
        return std::sqrt(sum_squared);
    }
    
    /**
     * @brief Normalize importance values to 0-1 range
     * 
     * @param importance_map Map of importance values to normalize
     */
    void normalizeImportanceMap(std::map<std::string, double>& importance_map) {
        // Find maximum importance value
        double max_importance = 0.0;
        for (const auto& [id, importance] : importance_map) {
            max_importance = std::max(max_importance, importance);
        }
        
        // Normalize all values
        if (max_importance > 0.0) {
            for (auto& [id, importance] : importance_map) {
                importance /= max_importance;
            }
        }
    }
    
    // Helper to check if a network type has getParameters method
    template<typename T, typename = void>
    struct has_get_parameters : std::false_type {};
    
    template<typename T>
    struct has_get_parameters<T, 
        std::void_t<decltype(std::declval<T>().getParameters())>> 
        : std::true_type {};
        
    // Helper to check if a network type has setParameterValue method
    template<typename T, typename = void>
    struct has_set_parameter : std::false_type {};
    
    template<typename T>
    struct has_set_parameter<T, 
        std::void_t<decltype(std::declval<T>().setParameterValue(std::string(), double()))>> 
        : std::true_type {};
        
    // Helper to check if a type is a vector-like container
    template<typename T, typename = void>
    struct is_vector : std::false_type {};
    
    template<typename T>
    struct is_vector<T, 
        std::void_t<decltype(std::declval<T>().size()),
                   decltype(std::declval<T>()[0])>> 
        : std::true_type {};
};

/**
 * @brief Enhanced analyzer for space mission radiation effects
 */
class SpaceEnvironmentAnalyzer {
public:
    /**
     * @brief Constructor with radiation environment
     * 
     * @param environment Space radiation environment
     */
    explicit SpaceEnvironmentAnalyzer(
        std::shared_ptr<radiation::Environment> environment = nullptr)
        : environment_(environment) {}
    
    /**
     * @brief Set the radiation environment
     * 
     * @param environment Environment to analyze for
     */
    void setEnvironment(std::shared_ptr<radiation::Environment> environment) {
        environment_ = environment;
    }
    
    /**
     * @brief Get the current environment
     * 
     * @return Current environment
     */
    std::shared_ptr<radiation::Environment> getEnvironment() const {
        return environment_;
    }
    
    /**
     * @brief Analyze network for space radiation vulnerability
     * 
     * @tparam NetworkType Type of neural network
     * @param network Network to analyze
     * @param base_analysis Base topological analysis result
     * @return Space-aware analysis result
     */
    template <typename NetworkType>
    SpaceAwareTopologicalResult analyzeSpaceRadiationEffects(
        const NetworkType& network,
        const TopologicalAnalysisResult& base_analysis) {
        
        SpaceAwareTopologicalResult result;
        
        // Copy base analysis results
        result.layer_criticality = base_analysis.layer_criticality;
        result.propagation_factors = base_analysis.propagation_factors;
        result.critical_paths = base_analysis.critical_paths;
        result.fan_in_count = base_analysis.fan_in_count;
        result.fan_out_count = base_analysis.fan_out_count;
        result.information_bottleneck_score = base_analysis.information_bottleneck_score;
        
        // Store environment reference
        result.environment = environment_;
        
        // If no environment specified, use LEO as default
        if (!environment_) {
            result.environment = radiation::Environment::createEnvironment(
                radiation::EnvironmentType::LOW_EARTH_ORBIT);
        }
        
        // Extract network layers
        auto layers = extractLayers(network);
        
        // Calculate radiation vulnerability factors
        calculateRadiationVulnerability(layers, result);
        
        // Calculate thermal sensitivity (important for space)
        calculateThermalSensitivity(layers, result);
        
        // Calculate power impact (critical for space missions)
        calculatePowerImpact(layers, result);
        
        // Calculate overall space environment impact
        calculateSpaceEnvironmentImpact(result);
        
        return result;
    }
    
    /**
     * @brief Calculate radiation-adjusted criticality scores
     * 
     * @param base_result Base analysis result
     * @param space_result Space-aware analysis result
     * @return Map of layers to radiation-adjusted criticality
     */
    std::map<std::string, double> calculateRadiationAdjustedCriticality(
        const TopologicalAnalysisResult& base_result,
        const SpaceAwareTopologicalResult& space_result) {
        
        std::map<std::string, double> adjusted_criticality;
        
        for (const auto& [layer_name, criticality] : base_result.layer_criticality) {
            double rad_factor = 1.0;
            if (space_result.radiation_vulnerability.count(layer_name)) {
                rad_factor = 1.0 + space_result.radiation_vulnerability.at(layer_name);
            }
            
            adjusted_criticality[layer_name] = criticality * rad_factor;
        }
        
        // Normalize values to 0-1 range
        double max_value = 0.0;
        for (const auto& [_, value] : adjusted_criticality) {
            max_value = std::max(max_value, value);
        }
        
        if (max_value > 0.0) {
            for (auto& [_, value] : adjusted_criticality) {
                value /= max_value;
            }
        }
        
        return adjusted_criticality;
    }
    
    /**
     * @brief Get recommended protection levels based on space environment
     * 
     * @param space_result Space-aware analysis result
     * @return Map of layers to recommended protection levels
     */
    std::map<std::string, ProtectionLevel> getRecommendedProtectionLevels(
        const SpaceAwareTopologicalResult& space_result) {
        
        std::map<std::string, ProtectionLevel> protection_levels;
        
        for (const auto& [layer_name, criticality] : space_result.layer_criticality) {
            // Base on radiation-adjusted criticality
            double rad_factor = 1.0;
            if (space_result.radiation_vulnerability.count(layer_name)) {
                rad_factor = space_result.radiation_vulnerability.at(layer_name);
            }
            
            double adjusted_criticality = criticality * (1.0 + rad_factor);
            
            // Environment flux factor (normalized to 0-1 for common environments)
            double flux_factor = 0.5;  // Default mid-range
            if (space_result.environment) {
                double flux = space_result.environment->getSEUFlux();
                // Normalize to 0-1 scale (adjust constants based on expected range)
                flux_factor = std::min(1.0, std::max(0.0, std::log10(flux + 1e-15) + 15) / 10.0);
            }
            
            // Determine protection level based on criticality and environment
            if (adjusted_criticality > 0.8 || flux_factor > 0.8) {
                protection_levels[layer_name] = ProtectionLevel::FULL_TMR;
            }
            else if (adjusted_criticality > 0.6 || flux_factor > 0.6) {
                protection_levels[layer_name] = ProtectionLevel::HEALTH_WEIGHTED_TMR;
            }
            else if (adjusted_criticality > 0.4 || flux_factor > 0.4) {
                protection_levels[layer_name] = ProtectionLevel::SELECTIVE_TMR;
            }
            else if (adjusted_criticality > 0.2 || flux_factor > 0.2) {
                protection_levels[layer_name] = ProtectionLevel::CHECKSUM_WITH_RECOVERY;
            }
            else {
                protection_levels[layer_name] = ProtectionLevel::CHECKSUM_ONLY;
            }
        }
        
        return protection_levels;
    }

private:
    std::shared_ptr<radiation::Environment> environment_;
    
    /**
     * @brief Extract layer information from network
     */
    template <typename NetworkType>
    std::vector<LayerInfo> extractLayers(const NetworkType& network) {
        // Similar to TopologicalAnalyzer::extractNetworkLayers
        if constexpr (has_get_layer_info<NetworkType>::value) {
            return network.getLayerInfo();
        } 
        else {
            // Fallback implementation
            std::vector<LayerInfo> layers;
            return layers;
        }
    }
    
    /**
     * @brief Calculate radiation vulnerability for each layer
     */
    void calculateRadiationVulnerability(
        const std::vector<LayerInfo>& layers,
        SpaceAwareTopologicalResult& result) {
        
        // Get environment SEU flux
        float flux = 1e-7f; // Default LEO value
        if (environment_) {
            flux = environment_->getSEUFlux();
        }
        
        for (const auto& layer : layers) {
            // Base vulnerability on layer type and size
            double vulnerability = 0.5; // Default mid-range
            
            // Consider layer type
            if (layer.type == "dense") {
                // Dense layers have more parameters, higher vulnerability
                vulnerability = 0.7;
            }
            else if (layer.type == "conv") {
                // Convolutional layers have parameter sharing, lower vulnerability
                vulnerability = 0.5;
            }
            else if (layer.type == "pool" || layer.type == "activation") {
                // Pooling and activation layers have few/no parameters
                vulnerability = 0.3;
            }
            else if (layer.type == "batch_norm") {
                // Batch normalization has critical statistical parameters
                vulnerability = 0.8;
            }
            
            // Adjust for input/output size (more memory = more vulnerability)
            double size_factor = std::min(1.0, std::log10(layer.input_size * layer.output_size) / 10.0);
            vulnerability = 0.5 * vulnerability + 0.5 * size_factor;
            
            // Scale by environment flux (normalized, higher flux = higher vulnerability)
            double flux_factor = std::min(1.0, std::log10(flux + 1e-15) + 15);
            vulnerability *= (0.5 + 0.5 * flux_factor / 10.0);
            
            // Adjust for criticality (critical layers are more important to protect)
            if (layer.is_critical) {
                vulnerability *= 1.2;
            }
            
            // Clamp to 0-1 range
            vulnerability = std::min(1.0, std::max(0.0, vulnerability));
            
            result.radiation_vulnerability[layer.name] = vulnerability;
        }
    }
    
    /**
     * @brief Calculate thermal sensitivity for each layer
     */
    void calculateThermalSensitivity(
        const std::vector<LayerInfo>& layers,
        SpaceAwareTopologicalResult& result) {
        
        for (const auto& layer : layers) {
            // Base thermal sensitivity on layer type and computational requirements
            double thermal_sensitivity = 0.5; // Default mid-range
            
            // Consider layer type
            if (layer.type == "dense") {
                // Dense layers have high computation, higher thermal output
                thermal_sensitivity = 0.7;
            }
            else if (layer.type == "conv") {
                // Convolutional layers are computationally intensive
                thermal_sensitivity = 0.8;
            }
            else if (layer.type == "pool") {
                // Pooling layers are simpler
                thermal_sensitivity = 0.3;
            }
            else if (layer.type == "activation") {
                // Activation functions are relatively simple
                thermal_sensitivity = 0.2;
            }
            
            // Adjust for layer size (more computation = more heat)
            double size_factor = std::min(1.0, std::log10(layer.input_size * layer.output_size) / 10.0);
            thermal_sensitivity = 0.6 * thermal_sensitivity + 0.4 * size_factor;
            
            // Clamp to 0-1 range
            thermal_sensitivity = std::min(1.0, std::max(0.0, thermal_sensitivity));
            
            result.thermal_sensitivity[layer.name] = thermal_sensitivity;
        }
    }
    
    /**
     * @brief Calculate power impact for each layer
     */
    void calculatePowerImpact(
        const std::vector<LayerInfo>& layers,
        SpaceAwareTopologicalResult& result) {
        
        for (const auto& layer : layers) {
            // Base power impact on layer type and computational requirements
            double power_impact = 0.5; // Default mid-range
            
            // Consider layer type (similar to thermal, but with different scaling)
            if (layer.type == "dense") {
                power_impact = 0.6;
            }
            else if (layer.type == "conv") {
                power_impact = 0.8;
            }
            else if (layer.type == "pool") {
                power_impact = 0.3;
            }
            else if (layer.type == "activation") {
                power_impact = 0.2;
            }
            
            // Adjust for size and complexity
            double size_factor = std::min(1.0, std::log10(layer.input_size * layer.output_size) / 10.0);
            power_impact = 0.7 * power_impact + 0.3 * size_factor;
            
            // Protection increases power requirements
            if (result.layer_criticality.count(layer.name)) {
                double criticality = result.layer_criticality.at(layer.name);
                // Higher criticality = more protection = more power
                power_impact *= (1.0 + 0.5 * criticality);
            }
            
            // Clamp to 0-1 range
            power_impact = std::min(1.0, std::max(0.0, power_impact));
            
            result.power_impact[layer.name] = power_impact;
        }
    }
    
    /**
     * @brief Calculate overall space environment impact score
     */
    void calculateSpaceEnvironmentImpact(SpaceAwareTopologicalResult& result) {
        double total_rad_vulnerability = 0.0;
        double total_thermal_sensitivity = 0.0;
        double total_power_impact = 0.0;
        double count = 0.0;
        
        for (const auto& [layer_name, _] : result.layer_criticality) {
            if (result.radiation_vulnerability.count(layer_name)) {
                total_rad_vulnerability += result.radiation_vulnerability.at(layer_name);
            }
            
            if (result.thermal_sensitivity.count(layer_name)) {
                total_thermal_sensitivity += result.thermal_sensitivity.at(layer_name);
            }
            
            if (result.power_impact.count(layer_name)) {
                total_power_impact += result.power_impact.at(layer_name);
            }
            
            count++;
        }
        
        if (count > 0) {
            // Weighted impact calculation
            double avg_rad = total_rad_vulnerability / count;
            double avg_thermal = total_thermal_sensitivity / count;
            double avg_power = total_power_impact / count;
            
            // Weights based on space mission priorities
            result.space_environment_impact = 0.5 * avg_rad + 0.3 * avg_thermal + 0.2 * avg_power;
        }
        else {
            result.space_environment_impact = 0.0;
        }
    }
    
    // Helper template for checking if a network has getLayerInfo
    template<typename T, typename = void>
    struct has_get_layer_info : std::false_type {};
    
    template<typename T>
    struct has_get_layer_info<T, 
        std::void_t<decltype(std::declval<T>().getLayerInfo())>> 
        : std::true_type {};
};

} // namespace neural
} // namespace rad_ml 
