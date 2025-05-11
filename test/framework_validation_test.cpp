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
#include <iomanip>
#include <random>
#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/tmr/basic_tmr.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;

// Create realistic radiation environments based on space missions
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

// Test basic TMR functionality with deliberate error injection
bool testBasicTMR() {
    std::cout << "Testing Basic TMR functionality...\n";
    
    // Create TMR with initial value
    tmr::BasicTMR<int> basic_tmr(42);
    
    // Get initial value to verify constructor worked
    int initial = basic_tmr.get();
    std::cout << "  Initial value: " << initial << "\n";
    bool constructor_check = (initial == 42);
    
    // Test repair functionality (should be a no-op with no errors)
    basic_tmr.repair();
    int after_repair = basic_tmr.get();
    std::cout << "  Value after repair (no errors): " << after_repair << "\n";
    bool repair_check = (after_repair == 42);
    
    // Directly access copies for manual corruption (using implementation detail)
    int* copies = const_cast<int*>(basic_tmr.getCopies());
    if (copies) {
        copies[0] = 100;  // Corrupt first copy
        std::cout << "  Manually corrupted first copy to 100\n";
        
        // TMR should correct this through voting
        int corrected = basic_tmr.get();
        std::cout << "  Value after corruption: " << corrected << "\n";
        bool correction_check = (corrected == 42);
        
        // Corrupt majority (this should cause failure without repair)
        copies[1] = 100;  // Now 2 copies have wrong value
        int corrupted_majority = basic_tmr.get();
        std::cout << "  Value after majority corruption: " << corrupted_majority << "\n";
        bool corruption_check = (corrupted_majority == 100);  // Should return corrupted value
        
        // Repair should restore correct value
        basic_tmr.repair();
        int after_full_repair = basic_tmr.get();
        std::cout << "  Value after full repair: " << after_full_repair << "\n";
        bool full_repair_check = (after_full_repair == 42);
        
        bool overall = constructor_check && repair_check && correction_check && 
                      corruption_check && full_repair_check;
        
        std::cout << "  Basic TMR test: " << (overall ? "PASSED" : "FAILED") << "\n\n";
        return overall;
    } else {
        std::cout << "  ERROR: Could not access TMR copies for testing\n";
        return false;
    }
}

// Test physics-driven protection with different environments
bool testPhysicsDrivenProtection() {
    std::cout << "Testing Physics-Driven Protection...\n";
    
    // Create material properties
    auto aluminum = createAluminumProperties();
    
    // Create protection system with 3 layers
    tmr::PhysicsDrivenProtection protection(aluminum, 3);
    
    // Test environments
    std::vector<std::string> environments = {
        "NONE", "LEO", "GEO", "SAA", "SOLAR_STORM", "JUPITER"
    };
    
    std::cout << "  Environment  | Protection Level | Checkpoint Int | Temp Factor | Rad Factor\n";
    std::cout << "  -------------|------------------|----------------|-------------|----------\n";
    
    bool environments_correct = true;
    
    for (const auto& env_name : environments) {
        // Create environment
        sim::RadiationEnvironment env = createEnvironment(env_name);
        
        // Configure protection for this environment
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
        
        // Get checkpoint interval
        double interval = protection.getCheckpointInterval();
        
        // Get physics factors
        double temp_factor, rad_factor, synergy_factor;
        protection.getCurrentFactors(temp_factor, rad_factor, synergy_factor);
        
        // Check if protection level is appropriate for environment
        bool level_correct = true;
        
        if (env_name == "NONE" || env_name == "LEO") {
            // Should use basic protection in low radiation
            level_correct = (level == tmr::ProtectionLevel::BASIC_TMR || 
                           level == tmr::ProtectionLevel::ENHANCED_TMR);
        } else if (env_name == "GEO") {
            // Should use enhanced protection in medium radiation
            level_correct = (level == tmr::ProtectionLevel::ENHANCED_TMR || 
                           level == tmr::ProtectionLevel::HEALTH_WEIGHTED_TMR);
        } else if (env_name == "SAA" || env_name == "SOLAR_STORM" || env_name == "JUPITER") {
            // Should use maximum protection in high radiation
            level_correct = (level == tmr::ProtectionLevel::HYBRID_REDUNDANCY || 
                           level == tmr::ProtectionLevel::HEALTH_WEIGHTED_TMR);
        }
        
        if (!level_correct) {
            environments_correct = false;
        }
        
        // Output results in table format
        std::cout << "  " << std::left << std::setw(13) << env_name 
                  << "| " << std::setw(17) << protectionLevelToString(level)
                  << "| " << std::setw(15) << std::fixed << std::setprecision(2) << interval
                  << "| " << std::setw(12) << std::fixed << std::setprecision(2) << temp_factor
                  << "| " << std::fixed << std::setprecision(2) << rad_factor << "\n";
    }
    
    std::cout << "\n  Physics-driven protection test: " 
              << (environments_correct ? "PASSED" : "FAILED") << "\n\n";
    
    return environments_correct;
}

// Test error injection and recovery
bool testErrorInjectionAndRecovery() {
    std::cout << "Testing Error Injection and Recovery...\n";
    
    // Create material properties
    auto aluminum = createAluminumProperties();
    
    // Create protection system
    tmr::PhysicsDrivenProtection protection(aluminum, 1);
    
    // Set high radiation environment (Jupiter)
    protection.updateEnvironment(createEnvironment("JUPITER"));
    
    // Number of test iterations
    const int iterations = 1000;
    int corrected_count = 0;
    int detected_count = 0;
    
    std::cout << "  Running " << iterations << " operations with simulated errors...\n";
    
    // Set fixed error rate
    const double error_rate = 0.3;  // 30% error rate
    
    // Create a random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (int i = 0; i < iterations; i++) {
        // Lambda to simulate operation with errors
        auto error_prone_op = [&dist, &gen, error_rate]() -> int {
            // Simulate bit flips based on error rate
            if (dist(gen) < error_rate) {
                return -999;  // Error value
            }
            return 42;  // Correct value
        };
        
        // Execute operation with protection
        tmr::TMRResult<int> result = protection.executeProtected<int>(error_prone_op);
        
        // Track detections and corrections
        if (result.error_detected) {
            detected_count++;
        }
        
        if (result.value == 42) {
            corrected_count++;
        }
    }
    
    // Calculate success metrics
    double detection_rate = static_cast<double>(detected_count) / iterations;
    double correction_rate = static_cast<double>(corrected_count) / iterations;
    
    std::cout << "  Error injection rate: " << (error_rate * 100.0) << "%\n";
    std::cout << "  Error detection rate: " << (detection_rate * 100.0) << "%\n";
    std::cout << "  Successful operations: " << corrected_count << "/" << iterations 
              << " (" << (correction_rate * 100.0) << "%)\n";
    
    // Check if success rate is better than without protection
    // Without protection, success should be approximately (1 - error_rate)
    double expected_unprotected = 1.0 - error_rate;
    bool improved = (correction_rate > expected_unprotected);
    
    // Check if we achieved at least 90% success even with 30% error rate
    bool high_success = (correction_rate >= 0.9);
    
    std::cout << "  Protection improved success rate: " << (improved ? "Yes" : "No") << "\n";
    std::cout << "  Achieved >90% success rate: " << (high_success ? "Yes" : "No") << "\n";
    std::cout << "  Error injection and recovery test: " 
              << ((improved && high_success) ? "PASSED" : "FAILED") << "\n\n";
    
    return (improved && high_success);
}

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " Radiation-Tolerant ML Framework Validation Test\n";
    std::cout << "====================================================\n\n";
    
    bool passed = true;
    
    // Test basic TMR functionality
    passed &= testBasicTMR();
    
    // Test physics-driven protection
    passed &= testPhysicsDrivenProtection();
    
    // Test error injection and recovery
    passed &= testErrorInjectionAndRecovery();
    
    // Overall test result
    std::cout << "====================================================\n";
    std::cout << "Overall validation test result: " << (passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "====================================================\n";
    
    return passed ? 0 : 1;
} 
