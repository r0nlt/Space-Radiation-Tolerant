#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>

#include "../../include/rad_ml/neural/sensitivity_analysis.hpp"
#include "../../include/rad_ml/neural/layer_protection_policy.hpp"
#include "../../include/rad_ml/neural/selective_hardening.hpp"
#include "../../include/rad_ml/neural/network_model.hpp"
#include "../../include/rad_ml/radiation/seu_simulator.hpp"
#include "../../include/rad_ml/radiation/environment.hpp"

using namespace rad_ml;
using namespace rad_ml::neural;

// Mock neural network implementation for testing
class MockNeuralNetwork {
public:
    MockNeuralNetwork() {
        // Create a simple feedforward network structure
        // Input -> Hidden1 -> Hidden2 -> Output
        setupLayers();
        setupWeights();
    }
    
    // Forward pass through the network
    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> h1_out(h1_size, 0.0);
        std::vector<double> h2_out(h2_size, 0.0);
        std::vector<double> output(output_size, 0.0);
        
        // Input to Hidden1
        for (size_t i = 0; i < h1_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < input_size; ++j) {
                sum += input[j] * weights_ih1[j][i];
            }
            h1_out[i] = activation(sum + biases_h1[i]);
        }
        
        // Hidden1 to Hidden2
        for (size_t i = 0; i < h2_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < h1_size; ++j) {
                sum += h1_out[j] * weights_h1h2[j][i];
            }
            h2_out[i] = activation(sum + biases_h2[i]);
        }
        
        // Hidden2 to Output
        for (size_t i = 0; i < output_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < h2_size; ++j) {
                sum += h2_out[j] * weights_h2o[j][i];
            }
            output[i] = activation(sum + biases_o[i]);
        }
        
        return output;
    }
    
    // Get layer activations for a given input
    std::vector<ActivationSample> getLayerActivations(const std::vector<double>& input) const {
        std::vector<ActivationSample> activations;
        
        // Calculate activations for each layer
        std::vector<double> h1_out(h1_size, 0.0);
        std::vector<double> h2_out(h2_size, 0.0);
        std::vector<double> output(output_size, 0.0);
        
        // Input to Hidden1
        for (size_t i = 0; i < h1_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < input_size; ++j) {
                sum += input[j] * weights_ih1[j][i];
            }
            h1_out[i] = activation(sum + biases_h1[i]);
        }
        
        // Hidden1 to Hidden2
        for (size_t i = 0; i < h2_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < h1_size; ++j) {
                sum += h1_out[j] * weights_h1h2[j][i];
            }
            h2_out[i] = activation(sum + biases_h2[i]);
        }
        
        // Hidden2 to Output
        for (size_t i = 0; i < output_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < h2_size; ++j) {
                sum += h2_out[j] * weights_h2o[j][i];
            }
            output[i] = activation(sum + biases_o[i]);
        }
        
        // Create activation samples
        ActivationSample input_act;
        input_act.layer_name = "input";
        input_act.values = input;
        
        ActivationSample h1_act;
        h1_act.layer_name = "hidden1";
        h1_act.values = h1_out;
        
        ActivationSample h2_act;
        h2_act.layer_name = "hidden2";
        h2_act.values = h2_out;
        
        ActivationSample out_act;
        out_act.layer_name = "output";
        out_act.values = output;
        
        activations.push_back(input_act);
        activations.push_back(h1_act);
        activations.push_back(h2_act);
        activations.push_back(out_act);
        
        return activations;
    }
    
    // Get layer information for network analysis
    std::vector<LayerInfo> getLayerInfo() const {
        std::vector<LayerInfo> layers;
        
        // Input layer
        LayerInfo input_layer;
        input_layer.name = "input";
        input_layer.type = "input";
        input_layer.index = 0;
        input_layer.input_size = input_size;
        input_layer.output_size = input_size;
        input_layer.output_layers = {"hidden1"};
        input_layer.is_critical = false;
        
        // Hidden 1 layer
        LayerInfo h1_layer;
        h1_layer.name = "hidden1";
        h1_layer.type = "dense";
        h1_layer.index = 1;
        h1_layer.input_size = input_size;
        h1_layer.output_size = h1_size;
        h1_layer.input_layers = {"input"};
        h1_layer.output_layers = {"hidden2"};
        h1_layer.is_critical = false;
        
        // Hidden 2 layer
        LayerInfo h2_layer;
        h2_layer.name = "hidden2";
        h2_layer.type = "dense";
        h2_layer.index = 2;
        h2_layer.input_size = h1_size;
        h2_layer.output_size = h2_size;
        h2_layer.input_layers = {"hidden1"};
        h2_layer.output_layers = {"output"};
        h2_layer.is_critical = true;  // Marking as critical for testing
        
        // Output layer
        LayerInfo output_layer;
        output_layer.name = "output";
        output_layer.type = "dense";
        output_layer.index = 3;
        output_layer.input_size = h2_size;
        output_layer.output_size = output_size;
        output_layer.input_layers = {"hidden2"};
        output_layer.is_critical = true;
        
        layers.push_back(input_layer);
        layers.push_back(h1_layer);
        layers.push_back(h2_layer);
        layers.push_back(output_layer);
        
        return layers;
    }
    
    // Get all network parameters
    std::vector<ParameterInfo> getParameters() const {
        std::vector<ParameterInfo> params;
        
        // Input to Hidden1 weights
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < h1_size; ++j) {
                ParameterInfo param;
                param.id = "w_ih1_" + std::to_string(i) + "_" + std::to_string(j);
                param.layer_name = "hidden1";
                param.param_type = "weight";
                param.index = i * h1_size + j;
                param.value = weights_ih1[i][j];
                params.push_back(param);
            }
        }
        
        // Hidden1 biases
        for (size_t i = 0; i < h1_size; ++i) {
            ParameterInfo param;
            param.id = "b_h1_" + std::to_string(i);
            param.layer_name = "hidden1";
            param.param_type = "bias";
            param.index = i;
            param.value = biases_h1[i];
            params.push_back(param);
        }
        
        // Hidden1 to Hidden2 weights
        for (size_t i = 0; i < h1_size; ++i) {
            for (size_t j = 0; j < h2_size; ++j) {
                ParameterInfo param;
                param.id = "w_h1h2_" + std::to_string(i) + "_" + std::to_string(j);
                param.layer_name = "hidden2";
                param.param_type = "weight";
                param.index = i * h2_size + j;
                param.value = weights_h1h2[i][j];
                params.push_back(param);
            }
        }
        
        // Hidden2 biases
        for (size_t i = 0; i < h2_size; ++i) {
            ParameterInfo param;
            param.id = "b_h2_" + std::to_string(i);
            param.layer_name = "hidden2";
            param.param_type = "bias";
            param.index = i;
            param.value = biases_h2[i];
            params.push_back(param);
        }
        
        // Hidden2 to Output weights
        for (size_t i = 0; i < h2_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                ParameterInfo param;
                param.id = "w_h2o_" + std::to_string(i) + "_" + std::to_string(j);
                param.layer_name = "output";
                param.param_type = "weight";
                param.index = i * output_size + j;
                param.value = weights_h2o[i][j];
                params.push_back(param);
            }
        }
        
        // Output biases
        for (size_t i = 0; i < output_size; ++i) {
            ParameterInfo param;
            param.id = "b_o_" + std::to_string(i);
            param.layer_name = "output";
            param.param_type = "bias";
            param.index = i;
            param.value = biases_o[i];
            params.push_back(param);
        }
        
        return params;
    }
    
    // Set parameter value
    void setParameterValue(const std::string& param_id, double new_value) {
        // Parse parameter ID to determine what to modify
        if (param_id.find("w_ih1_") == 0) {
            size_t pos1 = param_id.find_first_of('_', 6);
            size_t pos2 = param_id.find_last_of('_');
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos1 + 1, pos2 - pos1 - 1));
                size_t j = std::stoi(param_id.substr(pos2 + 1));
                if (i < input_size && j < h1_size) {
                    weights_ih1[i][j] = new_value;
                }
            }
        }
        else if (param_id.find("b_h1_") == 0) {
            size_t pos = param_id.find_last_of('_');
            if (pos != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos + 1));
                if (i < h1_size) {
                    biases_h1[i] = new_value;
                }
            }
        }
        else if (param_id.find("w_h1h2_") == 0) {
            size_t pos1 = param_id.find_first_of('_', 7);
            size_t pos2 = param_id.find_last_of('_');
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos1 + 1, pos2 - pos1 - 1));
                size_t j = std::stoi(param_id.substr(pos2 + 1));
                if (i < h1_size && j < h2_size) {
                    weights_h1h2[i][j] = new_value;
                }
            }
        }
        else if (param_id.find("b_h2_") == 0) {
            size_t pos = param_id.find_last_of('_');
            if (pos != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos + 1));
                if (i < h2_size) {
                    biases_h2[i] = new_value;
                }
            }
        }
        else if (param_id.find("w_h2o_") == 0) {
            size_t pos1 = param_id.find_first_of('_', 6);
            size_t pos2 = param_id.find_last_of('_');
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos1 + 1, pos2 - pos1 - 1));
                size_t j = std::stoi(param_id.substr(pos2 + 1));
                if (i < h2_size && j < output_size) {
                    weights_h2o[i][j] = new_value;
                }
            }
        }
        else if (param_id.find("b_o_") == 0) {
            size_t pos = param_id.find_last_of('_');
            if (pos != std::string::npos) {
                size_t i = std::stoi(param_id.substr(pos + 1));
                if (i < output_size) {
                    biases_o[i] = new_value;
                }
            }
        }
    }
    
private:
    // Network architecture
    static constexpr size_t input_size = 4;
    static constexpr size_t h1_size = 8;
    static constexpr size_t h2_size = 6;
    static constexpr size_t output_size = 2;
    
    // Weights and biases
    std::vector<std::vector<double>> weights_ih1;  // Input to Hidden1
    std::vector<double> biases_h1;                 // Hidden1 biases
    std::vector<std::vector<double>> weights_h1h2; // Hidden1 to Hidden2
    std::vector<double> biases_h2;                 // Hidden2 biases
    std::vector<std::vector<double>> weights_h2o;  // Hidden2 to Output
    std::vector<double> biases_o;                  // Output biases
    
    // Activation function (ReLU for simplicity)
    double activation(double x) const {
        return x > 0 ? x : 0.1 * x;  // Leaky ReLU
    }
    
    // Initialize network layers
    void setupLayers() {
        // Initialize with proper sizes
        weights_ih1.resize(input_size, std::vector<double>(h1_size, 0.0));
        biases_h1.resize(h1_size, 0.0);
        
        weights_h1h2.resize(h1_size, std::vector<double>(h2_size, 0.0));
        biases_h2.resize(h2_size, 0.0);
        
        weights_h2o.resize(h2_size, std::vector<double>(output_size, 0.0));
        biases_o.resize(output_size, 0.0);
    }
    
    // Initialize weights with random values
    void setupWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        
        // Input to Hidden1
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < h1_size; ++j) {
                weights_ih1[i][j] = dist(gen);
            }
        }
        
        // Hidden1 biases
        for (size_t i = 0; i < h1_size; ++i) {
            biases_h1[i] = dist(gen);
        }
        
        // Hidden1 to Hidden2
        for (size_t i = 0; i < h1_size; ++i) {
            for (size_t j = 0; j < h2_size; ++j) {
                weights_h1h2[i][j] = dist(gen);
            }
        }
        
        // Hidden2 biases
        for (size_t i = 0; i < h2_size; ++i) {
            biases_h2[i] = dist(gen);
        }
        
        // Hidden2 to Output
        for (size_t i = 0; i < h2_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                weights_h2o[i][j] = dist(gen);
            }
        }
        
        // Output biases
        for (size_t i = 0; i < output_size; ++i) {
            biases_o[i] = dist(gen);
        }
    }
};

// Generate a synthetic dataset for testing
std::vector<SampleType> generateDataset(size_t num_samples) {
    std::vector<SampleType> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (size_t i = 0; i < num_samples; ++i) {
        SampleType sample;
        
        // Generate input
        sample.input.resize(4);
        for (size_t j = 0; j < 4; ++j) {
            sample.input[j] = dist(gen);
        }
        
        // Generate expected output (simple function for testing)
        sample.expected_output.resize(2);
        sample.expected_output[0] = 0.5 * sample.input[0] + 0.3 * sample.input[1] - 0.2 * sample.input[2] + 0.1 * sample.input[3];
        sample.expected_output[1] = -0.1 * sample.input[0] + 0.2 * sample.input[1] + 0.4 * sample.input[2] - 0.3 * sample.input[3];
        
        dataset.push_back(sample);
    }
    
    return dataset;
}

// Convert network components to a format compatible with SelectiveHardening
std::vector<NetworkComponent> prepareNetworkComponents(const MockNeuralNetwork& network) {
    std::vector<NetworkComponent> components;
    
    // Get layer information
    auto layers = network.getLayerInfo();
    
    // Get parameters
    auto params = network.getParameters();
    
    // Create components for each parameter
    for (const auto& param : params) {
        NetworkComponent component;
        component.id = param.id;
        component.type = param.param_type;
        component.parent = param.layer_name;
        component.size_bytes = sizeof(double);  // Assuming double precision parameters
        
        // Assign criticality metrics (would be calculated in real implementation)
        component.metrics.sensitivity = 0.5;      // Default 
        component.metrics.activation_freq = 0.5;  // Default
        component.metrics.output_influence = 0.5; // Default
        component.metrics.custom_metric = 0.0;    // No custom metric for test
        
        components.push_back(component);
    }
    
    return components;
}

// Main function for testing
int main() {
    std::cout << "Neural Network Sensitivity-Based Protection Test" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Create a mock neural network
    MockNeuralNetwork network;
    
    // Generate synthetic dataset
    auto dataset = generateDataset(100);
    
    // Initialize analyzers
    TopologicalAnalyzer topo_analyzer;
    GradientImportanceMapper gradient_mapper;
    
    std::cout << "Running topological analysis..." << std::endl;
    auto topo_result = topo_analyzer.analyzeNetwork(network);
    
    std::cout << "Topological analysis results:" << std::endl;
    std::cout << "  Layer criticality:" << std::endl;
    for (const auto& [layer_name, criticality] : topo_result.layer_criticality) {
        std::cout << "    " << layer_name << ": " << std::fixed << std::setprecision(4) << criticality << std::endl;
    }
    
    std::cout << "\nRunning gradient-based importance mapping..." << std::endl;
    auto param_importance = gradient_mapper.calculateParameterImportance(network, dataset);
    
    // Get layer-based activation importance
    auto activation_importance = gradient_mapper.calculateActivationImportance(network, dataset);
    
    std::cout << "Activation importance results:" << std::endl;
    for (const auto& [layer_name, importance] : activation_importance) {
        std::cout << "    " << layer_name << ": " << std::fixed << std::setprecision(4) << importance << std::endl;
    }
    
    // Create layer protection policies
    std::cout << "\nCreating layer protection policies..." << std::endl;
    LayerProtectionManager policy_manager;
    policy_manager.createPoliciesFromAnalysis(topo_result, 0.8, HardeningStrategy::RESOURCE_CONSTRAINED);
    
    std::cout << "Layer protection policies:" << std::endl;
    for (const auto& layer_name : policy_manager.getPolicyLayers()) {
        auto policy = policy_manager.getPolicy(layer_name);
        if (policy) {
            std::cout << "    " << layer_name << ": " 
                      << "Level=" << static_cast<int>(policy->getProtectionLevel()) << ", "
                      << "Resources=" << std::fixed << std::setprecision(4) << policy->getResourceAllocation() << ", "
                      << "Strategy=" << policy->getStrategyName() << std::endl;
        }
    }
    
    // Apply importance decay strategy
    std::cout << "\nApplying importance decay strategy..." << std::endl;
    
    // Set up layer depths (0 = input, higher = deeper)
    std::map<std::string, int> layer_depths;
    layer_depths["input"] = 0;
    layer_depths["hidden1"] = 1;
    layer_depths["hidden2"] = 2;
    layer_depths["output"] = 3;
    
    policy_manager.applyImportanceDecayStrategy(3, layer_depths);
    
    std::cout << "Updated layer protection policies after decay:" << std::endl;
    for (const auto& layer_name : policy_manager.getPolicyLayers()) {
        auto policy = policy_manager.getPolicy(layer_name);
        if (policy) {
            std::cout << "    " << layer_name << ": " 
                      << "Level=" << static_cast<int>(policy->getProtectionLevel()) << ", "
                      << "Resources=" << std::fixed << std::setprecision(4) << policy->getResourceAllocation() << ", "
                      << "Strategy=" << policy->getStrategyName() << std::endl;
        }
    }
    
    // Set up selective hardening with network components
    std::cout << "\nSetting up selective hardening..." << std::endl;
    auto components = prepareNetworkComponents(network);
    
    // Run analysis with different strategies
    std::vector<HardeningStrategy> strategies = {
        HardeningStrategy::FIXED_THRESHOLD,
        HardeningStrategy::RESOURCE_CONSTRAINED,
        HardeningStrategy::TOPOLOGICAL,
        HardeningStrategy::GRADIENT_BASED,
        HardeningStrategy::LAYER_SPECIFIC,
        HardeningStrategy::IMPORTANCE_DECAY
    };
    
    for (auto strategy : strategies) {
        SelectiveHardening hardening(strategy, 0.8);
        
        // Set network-aware strategy data if needed
        if (strategy == HardeningStrategy::TOPOLOGICAL) {
            hardening.setTopologicalAnalysis(std::make_shared<TopologicalAnalysisResult>(topo_result));
        }
        else if (strategy == HardeningStrategy::GRADIENT_BASED) {
            hardening.setGradientImportanceMapping(param_importance);
        }
        else if (strategy == HardeningStrategy::LAYER_SPECIFIC) {
            hardening.setLayerProtectionManager(std::make_shared<LayerProtectionManager>(policy_manager));
        }
        else if (strategy == HardeningStrategy::IMPORTANCE_DECAY) {
            hardening.setLayerDepths(layer_depths, 3);
        }
        
        // Run analysis
        std::cout << "\nAnalyzing with strategy: " << static_cast<int>(strategy) << std::endl;
        auto result = hardening.analyzeNetworkComponents(components);
        
        // Report results
        std::cout << "  Total resource usage: " << std::fixed << std::setprecision(4) << result.total_resource_usage << std::endl;
        std::cout << "  Estimated error resilience: " << std::fixed << std::setprecision(4) << result.estimated_error_resilience << std::endl;
        
        // Count protection levels
        std::map<ProtectionLevel, int> protection_counts;
        for (const auto& [id, level] : result.protection_assignments) {
            protection_counts[level]++;
        }
        
        std::cout << "  Protection level assignments:" << std::endl;
        for (const auto& [level, count] : protection_counts) {
            std::cout << "    Level " << static_cast<int>(level) << ": " << count << " components" << std::endl;
        }
    }
    
    std::cout << "\nTest completed successfully." << std::endl;
    return 0;
} 