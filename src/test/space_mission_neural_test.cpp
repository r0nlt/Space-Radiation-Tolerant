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
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>

#include "../../include/rad_ml/neural/sensitivity_analysis.hpp"
#include "../../include/rad_ml/neural/layer_protection_policy.hpp"
#include "../../include/rad_ml/neural/selective_hardening.hpp"
#include "../../include/rad_ml/radiation/space_mission.hpp"
#include "../../include/rad_ml/radiation/seu_simulator.hpp"

using namespace rad_ml;
using namespace rad_ml::neural;
using namespace rad_ml::radiation;

// Simple mock neural network for testing the space mission protection
class SpaceMissionNN {
public:
    SpaceMissionNN() {
        // Create layer info
        setupLayers();
    }
    
    // Get layer information for topology analysis
    std::vector<LayerInfo> getLayerInfo() const {
        return layers_;
    }
    
    // Test forward pass with error simulation
    std::vector<double> forward(const std::vector<double>& input, 
                                double error_probability = 0.0) const {
        // Simple feed-forward network simulation
        std::vector<double> result(4, 0.0);
        
        // Inject errors based on probability
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Apply simple transformation and possible error injection
        for (size_t i = 0; i < result.size(); ++i) {
            // Simple weighted combination
            for (size_t j = 0; j < input.size(); ++j) {
                result[i] += input[j] * 0.5;
            }
            
            // Inject error?
            if (dis(gen) < error_probability) {
                result[i] = -result[i]; // Sign flip error
            }
        }
        
        return result;
    }
    
private:
    std::vector<LayerInfo> layers_;
    
    void setupLayers() {
        // Input layer
        LayerInfo input;
        input.name = "input";
        input.type = "input";
        input.index = 0;
        input.input_size = 6;
        input.output_size = 6;
        input.output_layers = {"hidden1"};
        input.is_critical = false;
        
        // Hidden layer 1
        LayerInfo hidden1;
        hidden1.name = "hidden1";
        hidden1.type = "dense";
        hidden1.index = 1;
        hidden1.input_size = 6;
        hidden1.output_size = 10;
        hidden1.input_layers = {"input"};
        hidden1.output_layers = {"hidden2"};
        hidden1.is_critical = false;
        
        // Hidden layer 2
        LayerInfo hidden2;
        hidden2.name = "hidden2";
        hidden2.type = "dense";
        hidden2.index = 2;
        hidden2.input_size = 10;
        hidden2.output_size = 8;
        hidden2.input_layers = {"hidden1"};
        hidden2.output_layers = {"output"};
        hidden2.is_critical = true;
        
        // Output layer
        LayerInfo output;
        output.name = "output";
        output.type = "dense";
        output.index = 3;
        output.input_size = 8;
        output.output_size = 4;
        output.input_layers = {"hidden2"};
        output.is_critical = true;
        
        // Store layers
        layers_ = {input, hidden1, hidden2, output};
    }
};

// Helper to print environment info
void printEnvironmentInfo(const std::shared_ptr<Environment>& env) {
    if (!env) {
        std::cout << "  Environment: (null)" << std::endl;
        return;
    }
    
    std::cout << "  Environment: " << env->getName() << std::endl;
    std::cout << "  SEU Flux: " << std::scientific << std::setprecision(3) 
              << env->getSEUFlux() << " SEUs/cm²/s" << std::endl;
    std::cout << "  SEU Cross-section: " << std::scientific << std::setprecision(3)
              << env->getSEUCrossSection() << " cm²/bit" << std::endl;
}

// Helper to print protection info
void printProtectionLevels(const std::map<std::string, ProtectionLevel>& levels) {
    std::cout << "Protection Levels:" << std::endl;
    
    for (const auto& [layer, level] : levels) {
        std::cout << "  " << std::left << std::setw(10) << layer << ": ";
        
        switch (level) {
            case ProtectionLevel::NONE:
                std::cout << "No Protection";
                break;
                
            case ProtectionLevel::CHECKSUM_ONLY:
                std::cout << "Checksum Only";
                break;
                
            case ProtectionLevel::CHECKSUM_WITH_RECOVERY:
                std::cout << "Checksum with Recovery";
                break;
                
            case ProtectionLevel::SELECTIVE_TMR:
                std::cout << "Selective TMR";
                break;
                
            case ProtectionLevel::APPROXIMATE_TMR:
                std::cout << "Approximate TMR";
                break;
                
            case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                std::cout << "Health-Weighted TMR";
                break;
                
            case ProtectionLevel::FULL_TMR:
                std::cout << "Full TMR";
                break;
                
            default:
                std::cout << "Unknown";
        }
        
        std::cout << std::endl;
    }
}

// Helper to run a mission simulation
void runMissionSimulation(const std::shared_ptr<SpaceMission>& mission, 
                         const SpaceMissionNN& network) {
    std::cout << "\nRunning Mission Simulation: " << mission->getName() << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Analyze network topology
    TopologicalAnalyzer analyzer;
    auto topo_result = analyzer.analyzeNetwork(network);
    
    // Create space environment analyzer
    SpaceEnvironmentAnalyzer space_analyzer;
    
    // Set up simulation parameters
    const int num_phases_to_sample = 3;  // Sample a subset of phases
    const int samples_per_phase = 10;    // Number of forward passes per phase
    
    // Get phases
    const auto& phases = mission->getPhases();
    if (phases.empty()) {
        std::cout << "No mission phases defined." << std::endl;
        return;
    }
    
    // Sample phases
    int phase_step = std::max(1, static_cast<int>(phases.size()) / num_phases_to_sample);
    
    // Track error rates
    std::map<std::string, double> phase_error_rates;
    std::map<std::string, std::map<std::string, double>> layer_error_rates;
    
    std::cout << "\nAnalyzing " << num_phases_to_sample 
              << " phases of the mission..." << std::endl;
    
    // Sample phases
    for (size_t i = 0; i < phases.size(); i += phase_step) {
        const auto& phase = phases[i];
        
        std::cout << "\nPhase: " << phase.name << std::endl;
        printEnvironmentInfo(phase.environment);
        
        // Set environment for analysis
        space_analyzer.setEnvironment(phase.environment);
        
        // Perform space-aware analysis
        auto space_result = space_analyzer.analyzeSpaceRadiationEffects(network, topo_result);
        
        // Get recommended protection levels
        auto protection_levels = space_analyzer.getRecommendedProtectionLevels(space_result);
        
        std::cout << "Space Environment Impact: " << std::fixed << std::setprecision(3) 
                  << space_result.space_environment_impact << std::endl;
        
        printProtectionLevels(protection_levels);
        
        // Simulate network performance
        double error_probability = space_result.space_environment_impact * 0.1;  // Scale for demonstration
        int error_count = 0;
        
        std::map<std::string, int> layer_errors;
        for (const auto& layer : network.getLayerInfo()) {
            layer_errors[layer.name] = 0;
        }
        
        // Run forward passes
        std::cout << "\nRunning " << samples_per_phase << " neural network inferences..." << std::endl;
        
        std::vector<double> sample_input = {0.5, 0.2, 0.7, -0.1, 0.3, 0.6};
        
        for (int j = 0; j < samples_per_phase; ++j) {
            // Inject errors based on space environment
            auto result = network.forward(sample_input, error_probability);
            
            // Check for errors (simplified for demo)
            bool has_error = false;
            for (size_t k = 0; k < result.size(); ++k) {
                if (result[k] < 0) {  // Our simplified error is a sign flip
                    has_error = true;
                    break;
                }
            }
            
            if (has_error) {
                error_count++;
                
                // Randomly assign error to a layer (simplified)
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_int_distribution<> layer_dist(0, network.getLayerInfo().size() - 1);
                int layer_idx = layer_dist(gen);
                
                layer_errors[network.getLayerInfo()[layer_idx].name]++;
            }
        }
        
        // Calculate error rates
        double phase_error_rate = static_cast<double>(error_count) / samples_per_phase;
        phase_error_rates[phase.name] = phase_error_rate;
        
        std::cout << "Phase error rate: " << std::fixed << std::setprecision(2) 
                  << (phase_error_rate * 100.0) << "%" << std::endl;
        
        // Store layer-specific error rates
        for (const auto& [layer_name, errors] : layer_errors) {
            double layer_error_rate = static_cast<double>(errors) / samples_per_phase;
            layer_error_rates[phase.name][layer_name] = layer_error_rate;
            
            if (errors > 0) {
                std::cout << "  Layer " << layer_name << " error rate: " 
                          << std::fixed << std::setprecision(2)
                          << (layer_error_rate * 100.0) << "%" << std::endl;
            }
        }
        
        // Apply mission-specific protection
        LayerProtectionManager protection_manager;
        
        // Convert mission target to mission profile
        neural::MissionProfile profile;
        switch (mission->getTarget()) {
            case MissionTarget::EARTH_LEO:
            case MissionTarget::EARTH_MEO:
            case MissionTarget::EARTH_GEO:
                profile = neural::MissionProfile::EARTH_ORBIT;
                break;
                
            case MissionTarget::MOON:
                profile = neural::MissionProfile::LUNAR_SURFACE;
                break;
                
            case MissionTarget::MARS:
                profile = neural::MissionProfile::MARS_SURFACE;
                break;
                
            case MissionTarget::JUPITER:
                profile = neural::MissionProfile::JUPITER_FLYBY;
                break;
                
            case MissionTarget::SOLAR_PROBE:
                profile = neural::MissionProfile::SOLAR_OBSERVATORY;
                break;
                
            default:
                profile = neural::MissionProfile::DEEP_SPACE;
                break;
        }
        
        // Create protection policies from analysis
        protection_manager.createPoliciesFromAnalysis(topo_result);
        
        // Configure for the specific mission
        protection_manager.configureForMission(profile, &topo_result);
        
        // Display configured protection policies
        std::cout << "\nMission-specific protection policies:" << std::endl;
        for (const auto& layer_name : protection_manager.getPolicyLayers()) {
            auto policy = protection_manager.getPolicy(layer_name);
            if (!policy) continue;
            
            std::cout << "  Layer: " << std::left << std::setw(10) << layer_name;
            std::cout << " | Protection: " << std::setw(10) 
                     << static_cast<int>(policy->getProtectionLevel());
            std::cout << " | Resources: " << std::fixed << std::setprecision(2) 
                     << (policy->getResourceAllocation() * 100.0) << "%";
            std::cout << " | Dynamic: " << (policy->usesDynamicAdjustment() ? "Yes" : "No");
            std::cout << " | Strategy: " << policy->getStrategyName();
            std::cout << std::endl;
        }
    }
    
    // Summarize overall mission
    std::cout << "\nMission Summary" << std::endl;
    std::cout << "==============" << std::endl;
    std::cout << "Total mission radiation exposure: " << std::scientific 
              << mission->calculateTotalRadiationExposure() << std::endl;
    
    auto worst_env = mission->getWorstCaseEnvironment();
    std::cout << "Worst-case environment: ";
    if (worst_env) {
        std::cout << worst_env->getName() << " (Flux: " << std::scientific 
                  << worst_env->getSEUFlux() << " SEUs/cm²/s)" << std::endl;
    } else {
        std::cout << "None" << std::endl;
    }
    
    // Most vulnerable phase
    std::string most_vulnerable_phase;
    double highest_error_rate = 0.0;
    for (const auto& [phase, rate] : phase_error_rates) {
        if (rate > highest_error_rate) {
            highest_error_rate = rate;
            most_vulnerable_phase = phase;
        }
    }
    
    std::cout << "Most vulnerable phase: " << most_vulnerable_phase
              << " (Error rate: " << std::fixed << std::setprecision(2)
              << (highest_error_rate * 100.0) << "%)" << std::endl;
    
    // Most vulnerable layer
    std::string most_vulnerable_layer;
    highest_error_rate = 0.0;
    
    for (const auto& [phase, layer_rates] : layer_error_rates) {
        for (const auto& [layer, rate] : layer_rates) {
            if (rate > highest_error_rate) {
                highest_error_rate = rate;
                most_vulnerable_layer = layer;
            }
        }
    }
    
    std::cout << "Most vulnerable layer: " << most_vulnerable_layer
              << " (Error rate: " << std::fixed << std::setprecision(2)
              << (highest_error_rate * 100.0) << "%)" << std::endl;
}

int main() {
    std::cout << "Space Mission Neural Network Protection Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Create a mock neural network
    SpaceMissionNN network;
    
    // Create missions
    auto leo_mission = SpaceMission::createStandardMission(MissionTarget::EARTH_LEO);
    auto lunar_mission = SpaceMission::createStandardMission(MissionTarget::MOON);
    auto mars_mission = SpaceMission::createStandardMission(MissionTarget::MARS);
    auto jupiter_mission = SpaceMission::createStandardMission(MissionTarget::JUPITER);
    auto solar_mission = SpaceMission::createStandardMission(MissionTarget::SOLAR_PROBE);
    
    // Run simulations for different missions
    runMissionSimulation(leo_mission, network);
    runMissionSimulation(lunar_mission, network);
    runMissionSimulation(mars_mission, network);
    runMissionSimulation(jupiter_mission, network);
    runMissionSimulation(solar_mission, network);
    
    std::cout << "\nTest completed successfully." << std::endl;
    return 0;
} 
