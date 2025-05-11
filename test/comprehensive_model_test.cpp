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
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cmath>

// Framework core components
#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;
using namespace rad_ml::tmr;

// Simple binary classifier that can be protected with TMR
class RadiationProtectedClassifier {
private:
    // Network parameters (simple 3-feature binary classifier)
    std::vector<float> weights; // 3 weights
    float bias;
    
    // Physics-driven protection system
    PhysicsDrivenProtection protection;
    
public:
    RadiationProtectedClassifier(const core::MaterialProperties& material)
        : protection(material, 1) // 1 layer (simplified model)
    {
        // Initialize with fixed weights for testing
        weights = {0.5f, -0.3f, 0.8f};
        bias = -0.1f;
        
        // Configure protection
        protection.setLayerSensitivity(0, 1.0); // Single layer, full sensitivity
    }
    
    // Configure protection based on environment
    void configureProtection(const sim::RadiationEnvironment& env, const std::string& mission_phase) {
        protection.updateEnvironment(env);
        protection.enterMissionPhase(mission_phase);
        
        // Get temperature and radiation factors used in protection
        double temperature_factor, radiation_factor, synergy_factor;
        protection.getCurrentFactors(temperature_factor, radiation_factor, synergy_factor);
        
        std::cout << "Protection configuration:"
                  << "\n  - Current protection level: " << static_cast<int>(protection.getCurrentGlobalProtection())
                  << "\n  - Temperature factor: " << temperature_factor
                  << "\n  - Radiation factor: " << radiation_factor
                  << "\n  - Synergy factor: " << synergy_factor << std::endl;
    }
    
    // Get current checkpoint interval
    double getCheckpointInterval() const {
        return protection.getCheckpointInterval();
    }
    
    // Logistic function
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    // Forward pass with TMR protection
    float classify(const std::vector<float>& features) {
        // Define protected operation
        auto protected_op = [this, &features]() -> float {
            // Simple dot product + bias
            float logit = bias;
            for (size_t i = 0; i < std::min(weights.size(), features.size()); i++) {
                logit += weights[i] * features[i];
            }
            return sigmoid(logit);
        };
        
        // Apply TMR protection from framework
        TMRResult<float> result = protection.executeProtected<float>(
            protected_op, 0, 1.0);
        
        return result.value;
    }
    
    // Predict binary class (0 or 1)
    int predict(const std::vector<float>& features) {
        float probability = classify(features);
        return (probability >= 0.5f) ? 1 : 0;
    }
    
    // Inject errors into weights (to simulate SEUs)
    void injectErrors(float error_rate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> bit_pos_dist(0, 31); // For 32-bit float
        
        // Inject bit flips in weights
        for (auto& weight : weights) {
            if (dist(gen) < error_rate) {
                // Simulate bit flip
                uint32_t* bits = reinterpret_cast<uint32_t*>(&weight);
                int bit_pos = bit_pos_dist(gen);
                *bits ^= (1u << bit_pos);
            }
        }
        
        // Maybe flip a bit in bias
        if (dist(gen) < error_rate) {
            uint32_t* bits = reinterpret_cast<uint32_t*>(&bias);
            int bit_pos = bit_pos_dist(gen);
            *bits ^= (1u << bit_pos);
        }
    }
    
    // Get a copy of the current weights
    std::vector<float> getWeights() const {
        return weights;
    }
    
    // Get the current bias
    float getBias() const {
        return bias;
    }
    
    // Set weights and bias
    void setParameters(const std::vector<float>& new_weights, float new_bias) {
        // Copy weights
        for (size_t i = 0; i < std::min(weights.size(), new_weights.size()); i++) {
            weights[i] = new_weights[i];
        }
        bias = new_bias;
    }
    
    // Get current protection level
    ProtectionLevel getCurrentProtection() const {
        return protection.getCurrentGlobalProtection();
    }
};

// Generate synthetic binary classification data
void generateTestData(std::vector<std::vector<float>>& features, 
                    std::vector<int>& labels,
                    int num_samples) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    for (int i = 0; i < num_samples; i++) {
        // Generate 3 random features
        std::vector<float> feature = {dist(gen), dist(gen), dist(gen)};
        
        // Calculate ground truth (similar to the model but without errors)
        // We use a fixed decision boundary: 0.5*x1 - 0.3*x2 + 0.8*x3 - 0.1 > 0
        float decision = 0.5f * feature[0] - 0.3f * feature[1] + 0.8f * feature[2] - 0.1f;
        int label = (decision > 0) ? 1 : 0;
        
        features.push_back(feature);
        labels.push_back(label);
    }
}

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

// Convert environment to approximate bit-flip error rate for injection testing
double estimateErrorRate(const sim::RadiationEnvironment& env) {
    // This is a simplified model to convert radiation levels to bit-flip probability
    // In a real system, this would be based on device physics and empirical data
    
    // Baseline from proton and electron flux
    double base_rate = (env.trapped_proton_flux * 2.0e-12) + 
                        (env.trapped_electron_flux * 5.0e-13);
    
    // Scale with temperature (higher temp = more errors)
    double avg_temp = (env.temperature.min + env.temperature.max) / 2.0;
    double temp_factor = 1.0 + std::max(0.0, (avg_temp - 273.0) / 100.0);
    
    // Solar activity multiplier
    double solar_factor = 1.0 + (env.solar_activity * 0.5);
    
    // SAA region multiplier
    double saa_factor = env.saa_region ? 1.5 : 1.0;
    
    // Apply all factors
    double error_rate = base_rate * temp_factor * solar_factor * saa_factor;
    
    // Clamp to reasonable range (0-1)
    return std::min(std::max(error_rate, 0.0), 0.5); // Cap at 50% for usable results
}

// Helper function to convert protection level to string
std::string protectionLevelToString(ProtectionLevel level) {
    switch (level) {
        case ProtectionLevel::NONE: return "None";
        case ProtectionLevel::BASIC_TMR: return "Basic TMR";
        case ProtectionLevel::ENHANCED_TMR: return "Enhanced TMR";
        case ProtectionLevel::STUCK_BIT_TMR: return "Stuck-Bit TMR";
        case ProtectionLevel::HEALTH_WEIGHTED_TMR: return "Health-Weighted TMR";
        case ProtectionLevel::HYBRID_REDUNDANCY: return "Hybrid Redundancy";
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

// Evaluate model accuracy with and without protection
double evaluateAccuracy(
    RadiationProtectedClassifier& model,
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels,
    float error_rate,
    bool show_progress = false
) {
    // Save original parameters
    std::vector<float> original_weights = model.getWeights();
    float original_bias = model.getBias();
    
    int correct = 0;
    size_t total = features.size();
    
    // Determine step size for progress updates
    size_t progress_step = std::max(size_t(1), total / 10);
    
    for (size_t i = 0; i < features.size(); i++) {
        // Show progress indicator if requested
        if (show_progress && (i % progress_step == 0)) {
            std::cout << "\r  Progress: " << std::fixed << std::setprecision(1) 
                      << (100.0 * i / total) << "%" << std::flush;
        }
        
        // Inject errors if required
        if (error_rate > 0) {
            // Reset to original parameters
            model.setParameters(original_weights, original_bias);
            
            // Inject errors based on error rate
            model.injectErrors(error_rate);
        }
        
        // Make prediction
        int prediction = model.predict(features[i]);
        
        // Check accuracy
        if (prediction == labels[i]) {
            correct++;
        }
    }
    
    // Clear progress indicator
    if (show_progress) {
        std::cout << "\r  Progress: 100.0%" << std::endl;
    }
    
    // Restore original parameters
    model.setParameters(original_weights, original_bias);
    
    // Return accuracy
    return static_cast<double>(correct) / features.size();
}

// Main test function
void comprehensiveModelTest() {
    std::cout << "===================================================\n";
    std::cout << " Comprehensive Physics-Driven Protection Test\n";
    std::cout << "===================================================\n";
    
    // Create aluminum material properties
    auto aluminum = createAluminumProperties();
    
    // Initialize classifier with physics protection
    RadiationProtectedClassifier model(aluminum);
    
    // Generate synthetic test data
    const int num_samples = 1000;
    std::vector<std::vector<float>> test_features;
    std::vector<int> test_labels;
    
    std::cout << "Generating synthetic test data...\n";
    generateTestData(test_features, test_labels, num_samples);
    std::cout << "Generated " << test_features.size() << " test samples\n\n";
    
    // Define environments to test
    std::vector<std::string> environments = {
        "NONE", "LEO", "SAA", "GEO", "LUNAR", "MARS", "SOLAR_STORM", "JUPITER"
    };
    
    // Results table headers
    std::cout << "================================================================\n";
    std::cout << std::left << std::setw(15) << "Environment" 
              << std::setw(10) << "Error Rate" 
              << std::setw(20) << "Protection Level" 
              << std::setw(15) << "Accuracy (%)" 
              << std::setw(15) << "Checkpoint (s)" << "\n";
    std::cout << "================================================================\n";
    
    // Open results CSV file
    std::ofstream results_file("comprehensive_protection_results.csv");
    results_file << "Environment,Error Rate,Protection Level,Accuracy,Checkpoint Interval\n";
    
    // Store results for analysis
    std::vector<std::pair<double, double>> environment_results; // Pairs of (error_rate, accuracy)
    
    // Evaluate baseline accuracy (no errors)
    std::cout << "Evaluating baseline accuracy (no errors)..." << std::endl;
    double baseline_accuracy = evaluateAccuracy(model, test_features, test_labels, 0.0, true);
    std::cout << "Baseline accuracy (no errors): " 
              << std::fixed << std::setprecision(2) << (baseline_accuracy * 100.0) << "%\n\n";
    
    // Run tests for each environment
    for (const auto& env_name : environments) {
        std::cout << "Testing environment: " << env_name << std::endl;
        
        // Create physics-based environment
        sim::RadiationEnvironment env = createEnvironment(env_name);
        
        // Estimate error rate for this environment
        double error_rate = estimateErrorRate(env);
        
        // Set mission phase based on environment
        std::string phase = "NOMINAL";
        if (env_name == "SAA") {
            phase = "SAA_CROSSING";
        } else if (env_name == "SOLAR_STORM") {
            phase = "SOLAR_STORM";
        }
        
        // Configure model for this environment
        model.configureProtection(env, phase);
        
        // Get protection level from physics model
        ProtectionLevel protection_level = model.getCurrentProtection();
        
        // Get checkpoint interval
        double checkpoint_interval = model.getCheckpointInterval();
        
        // For high radiation environments, use a sample limiter to speed up testing
        size_t sample_limit = test_features.size();
        bool use_limited_samples = false;
        
        if (env_name == "SOLAR_STORM" || env_name == "JUPITER") {
            sample_limit = std::min(sample_limit, size_t(200)); // Limit to 200 samples
            use_limited_samples = true;
            std::cout << "  Using reduced sample set (" << sample_limit << " samples) for high radiation environment" << std::endl;
        }
        
        // Create limited samples if needed
        std::vector<std::vector<float>> limited_features;
        std::vector<int> limited_labels;
        
        if (use_limited_samples) {
            limited_features.insert(limited_features.end(), test_features.begin(), test_features.begin() + sample_limit);
            limited_labels.insert(limited_labels.end(), test_labels.begin(), test_labels.begin() + sample_limit);
        }
        
        // Evaluate accuracy with current configuration
        double accuracy;
        if (use_limited_samples) {
            accuracy = evaluateAccuracy(model, limited_features, limited_labels, error_rate, true);
        } else {
            accuracy = evaluateAccuracy(model, test_features, test_labels, error_rate, true);
        }
        
        // Store results
        environment_results.push_back(std::make_pair(error_rate, accuracy));
        
        // Output results
        std::cout << std::left << std::setw(15) << env_name 
                  << std::setw(10) << std::fixed << std::setprecision(6) << error_rate
                  << std::setw(20) << protectionLevelToString(protection_level)
                  << std::setw(15) << std::fixed << std::setprecision(2) << (accuracy * 100.0)
                  << std::setw(15) << std::fixed << std::setprecision(2) << checkpoint_interval << "\n";
        
        // Write to CSV
        results_file << env_name << "," << error_rate << "," << protectionLevelToString(protection_level) << "," 
                     << accuracy << "," << checkpoint_interval << "\n";
        
        // Flush results file to ensure data is written in case of crash
        results_file.flush();
    }
    
    std::cout << "----------------------------------------------------------------\n";
    
    // Additional test: Fixed environments with multiple error rates
    std::cout << "\nProtection effectiveness vs. error rate:\n";
    std::cout << "================================================================\n";
    std::cout << std::left << std::setw(15) << "Error Rate" 
              << std::setw(15) << "Accuracy (%)" 
              << std::setw(20) << "Protection Level" << "\n";
    std::cout << "================================================================\n";
    
    // Create fixed high-radiation environment (Jupiter-like)
    sim::RadiationEnvironment high_rad_env = createEnvironment("JUPITER");
    model.configureProtection(high_rad_env, "NOMINAL");
    
    // Use a smaller sample set for error rate tests
    size_t error_test_samples = 200;
    std::vector<std::vector<float>> error_test_features(test_features.begin(), 
                                                      test_features.begin() + error_test_samples);
    std::vector<int> error_test_labels(test_labels.begin(), 
                                      test_labels.begin() + error_test_samples);
    
    std::cout << "Using " << error_test_samples << " samples for error rate testing..." << std::endl;
    
    // Test with fixed error rates directly
    std::vector<double> test_error_rates = {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4};
    for (double test_error_rate : test_error_rates) {
        std::cout << "Testing error rate: " << test_error_rate << std::endl;
        
        double accuracy = evaluateAccuracy(model, error_test_features, error_test_labels, test_error_rate, true);
        
        // Output results
        std::cout << std::left << std::setw(15) << std::fixed << std::setprecision(2) << test_error_rate
                  << std::setw(15) << std::fixed << std::setprecision(2) << (accuracy * 100.0)
                  << std::setw(20) << protectionLevelToString(model.getCurrentProtection()) << "\n";
        
        // Write to CSV
        results_file << "ERROR_TEST," << test_error_rate << "," 
                     << protectionLevelToString(model.getCurrentProtection()) << "," 
                     << accuracy << ",0\n";
        
        // Flush results file
        results_file.flush();
    }
    
    results_file.close();
    std::cout << "\nTest completed. Results saved to comprehensive_protection_results.csv\n";
}

int main() {
    comprehensiveModelTest();
    return 0;
} 
