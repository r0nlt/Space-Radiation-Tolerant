#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;

// Structure to hold visualization data
struct VisualizationPoint {
    double time;
    std::string environment;
    int protection_level;
    std::string protection_name;
    double checkpoint_interval;
    double temperature_factor;
    double radiation_factor;
    double synergy_factor;
    int corrections;
    int errors_detected;
};

// Create realistic radiation environments
sim::RadiationEnvironment createEnvironment(const std::string& env_name) {
    sim::RadiationEnvironment env;
    
    if (env_name == "LEO") {
        // Low Earth Orbit (International Space Station-like)
        env.trapped_proton_flux = 1.0e7;  // particles/cm²/s
        env.trapped_electron_flux = 5.0e6;
        env.temperature.min = 270.0;  // Kelvin
        env.temperature.max = 290.0;
        env.solar_activity = 0.2;      // Scale 0-1
        env.saa_region = false;
    } 
    else if (env_name == "SAA") {
        // South Atlantic Anomaly (high radiation region of LEO)
        env.trapped_proton_flux = 5.0e9;
        env.trapped_electron_flux = 1.0e9;
        env.temperature.min = 280.0;
        env.temperature.max = 310.0;
        env.solar_activity = 0.3;
        env.saa_region = true;
    }
    else if (env_name == "GEO") {
        // Geostationary Orbit
        env.trapped_proton_flux = 5.0e7;
        env.trapped_electron_flux = 2.0e7;
        env.temperature.min = 250.0;
        env.temperature.max = 320.0;
        env.solar_activity = 0.4;
        env.saa_region = false;
    }
    else if (env_name == "LUNAR") {
        // Lunar orbit/surface
        env.trapped_proton_flux = 8.0e7;
        env.trapped_electron_flux = 3.0e7;
        env.temperature.min = 100.0;
        env.temperature.max = 390.0;
        env.solar_activity = 0.5;
        env.saa_region = false;
    }
    else if (env_name == "MARS") {
        // Mars orbit/surface
        env.trapped_proton_flux = 2.0e8;
        env.trapped_electron_flux = 8.0e7;
        env.temperature.min = 150.0;
        env.temperature.max = 300.0;
        env.solar_activity = 0.4;
        env.saa_region = false;
    }
    else if (env_name == "SOLAR_STORM") {
        // Solar storm
        env.trapped_proton_flux = 1.0e10;
        env.trapped_electron_flux = 5.0e9;
        env.temperature.min = 300.0;
        env.temperature.max = 350.0;
        env.solar_activity = 0.9;
        env.saa_region = false;
    }
    else if (env_name == "JUPITER") {
        // Jupiter environment (extremely high radiation)
        env.trapped_proton_flux = 1.0e12;
        env.trapped_electron_flux = 5.0e11;
        env.temperature.min = 120.0;
        env.temperature.max = 400.0;
        env.solar_activity = 1.0;
        env.saa_region = false;
    }
    else {
        // Default - minimal radiation (clean room on Earth)
        env.trapped_proton_flux = 1.0e5;
        env.trapped_electron_flux = 1.0e4;
        env.temperature.min = 270.0;
        env.temperature.max = 280.0;
        env.solar_activity = 0.1;
        env.saa_region = false;
    }
    
    return env;
}

// Convert protection level enum to string
std::string protectionLevelToString(tmr::ProtectionLevel level) {
    switch (level) {
        case tmr::ProtectionLevel::NONE: return "None";
        case tmr::ProtectionLevel::BASIC_TMR: return "Basic TMR";
        case tmr::ProtectionLevel::ENHANCED_TMR: return "Enhanced TMR";
        case tmr::ProtectionLevel::STUCK_BIT_TMR: return "Stuck-Bit TMR";
        case tmr::ProtectionLevel::HEALTH_WEIGHTED_TMR: return "Health-Weighted TMR";
        case tmr::ProtectionLevel::HYBRID_REDUNDANCY: return "Hybrid Redundancy";
        default: return "Unknown";
    }
}

// Create aerospace-grade aluminum material properties
core::MaterialProperties createAluminumProperties() {
    core::MaterialProperties aluminum;
    
    // Standard aerospace-grade aluminum properties
    aluminum.name = "Aerospace-Grade Aluminum";
    aluminum.density = 2.7;  // g/cm³
    aluminum.yield_strength = 270.0;  // MPa
    aluminum.radiation_tolerance = 50.0;  // Relative scale 0-100
    
    return aluminum;
}

// Calculate error rate based on environment
double calculateErrorRate(const sim::RadiationEnvironment& env) {
    // Base rate from proton and electron flux
    double base_rate = (env.trapped_proton_flux * 2.0e-12) + 
                       (env.trapped_electron_flux * 5.0e-13);
    
    // Scale with temperature
    double avg_temp = (env.temperature.min + env.temperature.max) / 2.0;
    double temp_factor = 1.0 + std::max(0.0, (avg_temp - 273.0) / 100.0);
    
    // Solar activity multiplier
    double solar_factor = 1.0 + (env.solar_activity * 0.5);
    
    // SAA region multiplier
    double saa_factor = env.saa_region ? 1.5 : 1.0;
    
    // Calculate final error rate
    double error_rate = base_rate * temp_factor * solar_factor * saa_factor;
    
    // Cap at reasonable range
    return std::min(std::max(error_rate, 0.0), 0.5);
}

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " Radiation-Tolerant ML Protection Visualizer\n";
    std::cout << "====================================================\n\n";
    
    std::vector<VisualizationPoint> data_points;
    
    // Create protection system
    core::MaterialProperties aluminum = createAluminumProperties();
    tmr::PhysicsDrivenProtection protection(aluminum, 1);
    
    // Define test environments
    std::vector<std::string> environments = {
        "NONE", "LEO", "GEO", "SAA", "MARS", "SOLAR_STORM", "JUPITER"
    };
    
    // Create output file
    std::string filename = "protection_visualization.csv";
    std::ofstream csv_file(filename);
    
    // Write CSV header
    csv_file << "Time,Environment,ProtectionLevel,ProtectionName,CheckpointInterval,"
             << "TemperatureFactor,RadiationFactor,SynergyFactor,Corrections,ErrorsDetected,"
             << "ErrorRate,SuccessRate\n";
    
    std::cout << "Simulating protection across different environments...\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Time" 
              << std::setw(12) << "Environment"
              << std::setw(20) << "Protection"
              << std::setw(15) << "Checkpoint(s)"
              << std::setw(10) << "Errors"
              << std::setw(10) << "Fixed"
              << "Success(%)\n";
    std::cout << "-----------------------------------------------------------------\n";
    
    // Simulation time
    double total_time = 0.0;
    
    // Error and correction counters
    int total_operations = 0;
    int total_corrections = 0;
    int total_errors = 0;
    
    // Run simulation for each environment
    for (const auto& env_name : environments) {
        // Create environment
        sim::RadiationEnvironment env = createEnvironment(env_name);
        
        // Calculate error rate for this environment
        double error_rate = calculateErrorRate(env);
        
        // Update protection
        protection.updateEnvironment(env);
        
        // For SAA and SOLAR_STORM, set mission phase
        if (env_name == "SAA") {
            protection.enterMissionPhase("SAA_CROSSING");
        } else if (env_name == "SOLAR_STORM") {
            protection.enterMissionPhase("SOLAR_STORM");
        } else {
            protection.enterMissionPhase("NOMINAL");
        }
        
        // Get current protection level
        tmr::ProtectionLevel level = protection.getCurrentGlobalProtection();
        std::string protection_name = protectionLevelToString(level);
        
        // Get checkpoint interval
        double checkpoint_interval = protection.getCheckpointInterval();
        
        // Get physics factors
        double temp_factor, rad_factor, synergy_factor;
        protection.getCurrentFactors(temp_factor, rad_factor, synergy_factor);
        
        // Random generator for error injection
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Simulate operations in this environment
        const int operations_per_env = 200;
        int environment_corrections = 0;
        int environment_errors = 0;
        
        // For each simulated operation in this environment
        for (int i = 0; i < operations_per_env; i++) {
            // Increment simulation time
            total_time += 1.0;
            total_operations++;
            
            // Create operation with error injection
            auto error_prone_op = [&dist, &gen, error_rate]() -> int {
                if (dist(gen) < error_rate) {
                    return -999;  // Simulated error
                }
                return 42;  // Correct value
            };
            
            // Execute with protection
            tmr::TMRResult<int> result = protection.executeProtected<int>(error_prone_op);
            
            // Track results
            if (result.error_detected) {
                environment_errors++;
                total_errors++;
            }
            
            if (result.value == 42) {
                environment_corrections++;
                total_corrections++;
            }
            
            // Record data point for visualization (every 20 operations)
            if (i % 20 == 0 || i == operations_per_env - 1) {
                VisualizationPoint point;
                point.time = total_time;
                point.environment = env_name;
                point.protection_level = static_cast<int>(level);
                point.protection_name = protection_name;
                point.checkpoint_interval = checkpoint_interval;
                point.temperature_factor = temp_factor;
                point.radiation_factor = rad_factor;
                point.synergy_factor = synergy_factor;
                point.corrections = environment_corrections;
                point.errors_detected = environment_errors;
                
                data_points.push_back(point);
                
                // Calculate success rate for this point
                double success_rate = static_cast<double>(environment_corrections) / (i + 1) * 100.0;
                
                // Write to CSV
                csv_file << point.time << ","
                         << point.environment << ","
                         << point.protection_level << ","
                         << point.protection_name << ","
                         << point.checkpoint_interval << ","
                         << point.temperature_factor << ","
                         << point.radiation_factor << ","
                         << point.synergy_factor << ","
                         << point.corrections << ","
                         << point.errors_detected << ","
                         << error_rate << ","
                         << success_rate << "\n";
                
                // Print progress
                if (i == operations_per_env - 1) {
                    std::cout << std::left << std::setw(10) << std::fixed << std::setprecision(1) << total_time
                              << std::setw(12) << env_name
                              << std::setw(20) << protection_name
                              << std::setw(15) << std::fixed << std::setprecision(2) << checkpoint_interval
                              << std::setw(10) << environment_errors
                              << std::setw(10) << environment_corrections
                              << std::fixed << std::setprecision(2) << success_rate << "%\n";
                }
            }
        }
    }
    
    csv_file.close();
    
    // Calculate overall success rate
    double overall_success_rate = static_cast<double>(total_corrections) / total_operations * 100.0;
    
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Simulation complete!\n";
    std::cout << "Total operations: " << total_operations << "\n";
    std::cout << "Total errors detected: " << total_errors << "\n";
    std::cout << "Total corrections applied: " << total_corrections << "\n";
    std::cout << "Overall success rate: " << std::fixed << std::setprecision(2) 
              << overall_success_rate << "%\n";
    std::cout << "Visualization data saved to " << filename << "\n";
    std::cout << "====================================================\n";
    
    return 0;
} 