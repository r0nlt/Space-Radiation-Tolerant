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
#include "rad_ml/tmr/enhanced_tmr.hpp"
#include "rad_ml/testing/radiation_simulator.hpp"
#include "rad_ml/testing/mission_simulator.hpp"
#include <array>
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <cstring>

using namespace rad_ml::tmr;
using namespace rad_ml::testing;

// Simple mock neural network for demonstration
class SimpleNeuralNetwork {
private:
    // Protected weights using Enhanced TMR
    std::array<EnhancedTMR<float>, 10> weights_;
    
    // Configuration
    std::chrono::milliseconds scrub_interval_{1000};
    bool recovery_enabled_ = false;
    uint32_t checkpoint_interval_s_ = 0;
    
public:
    SimpleNeuralNetwork() {
        // Initialize weights with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (int i = 0; i < 10; ++i) {
            weights_[i] = EnhancedTMR<float>(dist(gen));
        }
    }
    
    // Getters and setters for configuration
    void setScrubInterval(std::chrono::milliseconds interval) {
        scrub_interval_ = interval;
        
        // Set verification interval for all weights
        for (auto& weight : weights_) {
            weight.setVerificationInterval(interval);
        }
        
        std::cout << "Scrub interval set to " << interval.count() << " ms" << std::endl;
    }
    
    void enableRecovery(uint32_t checkpoint_interval_s) {
        recovery_enabled_ = true;
        checkpoint_interval_s_ = checkpoint_interval_s;
        std::cout << "Recovery enabled with checkpoint interval of " 
                  << checkpoint_interval_s << " seconds" << std::endl;
    }
    
    // Perform a simple inference (just for demonstration)
    float runInference(float input) {
        float output = 0.0f;
        
        // Simple weighted sum
        for (int i = 0; i < 10; ++i) {
            output += weights_[i].get() * input;
        }
        
        return output;
    }
    
    // Repair all weights
    void repair() {
        for (auto& weight : weights_) {
            weight.regenerateCopies();
        }
    }
    
    // Get error statistics
    struct ErrorStats {
        size_t crc_validation_failures = 0;
        size_t voting_disagreements = 0;
    };
    
    ErrorStats getErrorStats() const {
        ErrorStats total;
        
        // In a real implementation, we would collect stats from all weights
        // For this demo we just use the first weight's stats
        std::string stats_str = weights_[0].getErrorStats();
        
        // Parse the error stats string to extract the values
        // In a real implementation, we would have a better API for this
        if (stats_str.find("CRC validation failures: ") != std::string::npos) {
            size_t pos = stats_str.find("CRC validation failures: ") + 
                         std::string("CRC validation failures: ").length();
            total.crc_validation_failures = std::stoul(stats_str.substr(pos));
        }
        
        if (stats_str.find("Voting disagreements: ") != std::string::npos) {
            size_t pos = stats_str.find("Voting disagreements: ") + 
                         std::string("Voting disagreements: ").length();
            total.voting_disagreements = std::stoul(stats_str.substr(pos));
        }
        
        return total;
    }
    
    // Reset error statistics
    void resetErrorStats() {
        for (auto& weight : weights_) {
            weight.resetErrorStats();
        }
    }
};

// Print a horizontal line
void printSeparator() {
    std::cout << "\n" << std::setfill('-') << std::setw(80) << "" << "\n" << std::setfill(' ') << std::endl;
}

/**
 * Boost radiation rates for demonstration purposes
 * 
 * @param simulator The simulator to modify
 * @param boost_factor How much to boost radiation rates
 */
void boostRadiationRates(RadiationSimulator& simulator, float boost_factor) {
    // Get current environment params
    auto params = simulator.getSimulationEnvironment();
    
    // Artificially boost radiation by increasing solar activity
    params.solar_activity *= boost_factor;
    
    // Decrease shielding for even more radiation
    params.shielding_thickness_mm = std::max(0.5, params.shielding_thickness_mm / boost_factor);
    
    // Set South Atlantic Anomaly to true for boosted radiation
    params.inside_saa = true;
    
    // Update the simulator
    simulator.updateEnvironment(params);
    
    std::cout << "BOOSTED RADIATION RATES FOR DEMONSTRATION:" << std::endl;
    std::cout << simulator.getEnvironmentDescription() << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Radiation-Tolerant Machine Learning Mission Simulator\n" 
              << "=====================================================" << std::endl;
    
    // Parse command line arguments for mission
    std::string mission_name = "LEO";
    if (argc > 1) {
        mission_name = argv[1];
    }
    
    // Create mission profile and adaptive protection configuration
    MissionProfile profile = MissionProfile::createStandard(mission_name);
    AdaptiveProtectionConfig protection_config = AdaptiveProtectionConfig::createStandard(mission_name);
    
    std::cout << "Creating mission profile for: " << mission_name << std::endl;
    
    // Create mission simulator
    MissionSimulator simulator(profile, protection_config);
    
    // Boost radiation rates for demonstration purposes
    boostRadiationRates(simulator.getRadiationSimulator(), 50.0f);
    
    // Print mission details
    printSeparator();
    std::cout << "Mission Environment Description:" << std::endl;
    std::cout << simulator.getRadiationSimulator().getEnvironmentDescription() << std::endl;
    printSeparator();
    
    // Create neural network
    std::cout << "Initializing neural network..." << std::endl;
    SimpleNeuralNetwork network;
    
    // Configure network based on mission profile
    std::string protection_level = simulator.getProtectionConfig().enable_tmr_medium ?
                           "MEDIUM" : (simulator.getProtectionConfig().enable_tmr_high ? "HIGH" : "LOW");
    
    std::cout << "Configuring network with protection level: " << protection_level << std::endl;
    
    // Set network protection based on mission profile
    network.setScrubInterval(std::chrono::milliseconds(5000));
    if (protection_level == "MEDIUM" || protection_level == "HIGH") {
        network.enableRecovery(300);
    }
    
    // Register memory regions with the simulator
    SimpleNeuralNetwork* network_ptr = &network;
    simulator.registerMemoryRegion(network_ptr, sizeof(SimpleNeuralNetwork), true);
    
    // Run simulated mission
    printSeparator();
    std::cout << "Beginning mission simulation...\n" << std::endl;
    
    // Run the simulation with a callback for environment changes
    auto stats = simulator.runSimulation(
        std::chrono::seconds(30),  // Simulate 30 seconds of mission time
        std::chrono::seconds(3),   // Time step of 3 seconds
        [&network](const RadiationSimulator::EnvironmentParams& env) {
            std::cout << "\nEnvironment changed to: " << env.mission_name << std::endl;
            
            // In a real system, we would adapt protection based on environment
            if (env.inside_saa || env.solar_activity > 5.0) {
                std::cout << "High radiation detected - increasing protection" << std::endl;
                network.setScrubInterval(std::chrono::milliseconds(500));
                network.enableRecovery(60);
            } else {
                std::cout << "Normal radiation levels - standard protection" << std::endl;
                network.setScrubInterval(std::chrono::milliseconds(5000));
            }
        }
    );
    
    // Print mission statistics
    printSeparator();
    std::cout << "Mission Complete - Final Statistics" << std::endl;
    printSeparator();
    std::cout << stats.getReport() << std::endl;
    
    // Test the neural network after the mission
    std::cout << "Testing neural network after mission..." << std::endl;
    float input = 0.5f;
    float output = network.runInference(input);
    std::cout << "Inference result: " << input << " → " << output << std::endl;
    
    // Print network error statistics
    auto network_stats = network.getErrorStats();
    std::cout << "Neural network error statistics:" << std::endl;
    std::cout << "  CRC validation failures: " << network_stats.crc_validation_failures << std::endl;
    std::cout << "  Voting disagreements: " << network_stats.voting_disagreements << std::endl;
    
    return 0;
} 
