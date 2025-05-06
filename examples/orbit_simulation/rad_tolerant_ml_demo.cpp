#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;
using namespace rad_ml::tmr;

// Simple neural network with protected layers
class RadiationTolerantNN {
private:
    PhysicsDrivenProtection protection;
    std::vector<ProtectedNeuralLayer<float>> layers;
    std::vector<int> layer_sizes;
    std::string model_name;
    int error_count = 0;
    int total_inferences = 0;

public:
    RadiationTolerantNN(const std::string& name, 
                        const core::MaterialProperties& material,
                        const std::vector<int>& sizes) 
        : protection(material, sizes.size()-1),
          layer_sizes(sizes),
          model_name(name)
    {
        std::cout << "Creating radiation-tolerant neural network: " << name << std::endl;
        std::cout << "Architecture: ";
        for (size_t i = 0; i < sizes.size(); i++) {
            std::cout << sizes[i];
            if (i < sizes.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        
        // Create protected layers
        for (size_t i = 0; i < sizes.size()-1; i++) {
            // More sensitive/critical for early layers
            double criticality = 1.0 - 0.5 * static_cast<double>(i) / (sizes.size()-1);
            
            // Set layer sensitivity
            protection.setLayerSensitivity(i, criticality);
            
            // Create the protected layer
            layers.emplace_back(i, protection, criticality);
            
            // Initialize with some random weights for demonstration
            std::vector<float> weights(sizes[i] * sizes[i+1]);
            initializeWeights(weights, sizes[i], sizes[i+1]);
            layers[i].setWeights(weights);
        }
    }
    
    void updateEnvironment(const sim::RadiationEnvironment& env, double stress_MPa = 0.0) {
        protection.updateEnvironment(env, stress_MPa);
    }
    
    void enterMissionPhase(const std::string& phase) {
        protection.enterMissionPhase(phase);
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> current = input;
        
        for (auto& layer : layers) {
            current = layer.forward(current);
        }
        
        total_inferences++;
        return current;
    }
    
    void recordError() {
        error_count++;
    }
    
    double getErrorRate() const {
        if (total_inferences == 0) return 0.0;
        return static_cast<double>(error_count) / total_inferences;
    }
    
    ProtectionLevel getCurrentProtection() const {
        return protection.getCurrentGlobalProtection();
    }
    
    double getCheckpointInterval() const {
        return protection.getCheckpointInterval();
    }
    
    void printProtectionFactors() const {
        double temp_factor, stress_factor, synergy_factor;
        protection.getCurrentFactors(temp_factor, stress_factor, synergy_factor);
        
        std::cout << "Protection factors for " << model_name << ":" << std::endl;
        std::cout << "  Temperature factor: " << std::fixed << std::setprecision(3) << temp_factor << std::endl;
        std::cout << "  Mechanical stress factor: " << stress_factor << std::endl;
        std::cout << "  Synergy factor: " << synergy_factor << std::endl;
    }
    
private:
    void initializeWeights(std::vector<float>& weights, int inputs, int outputs) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-0.5, 0.5);
        
        for (int i = 0; i < inputs * outputs; i++) {
            weights[i] = static_cast<float>(dist(gen));
        }
    }
};

// Simulated spacecraft environment manager
class SpacecraftEnvironment {
private:
    // Current orbit position (0-1 normalized)
    double orbit_position = 0.0;
    
    // Orbit parameters
    double orbit_period_minutes;
    bool is_geo_orbit;
    bool has_saa_crossing;
    
    // Current environment
    sim::RadiationEnvironment current_env;
    
    // Mechanical stress
    double base_mechanical_stress_MPa = 5.0;
    double current_mechanical_stress_MPa = 5.0;
    
    // Random generators for realistic variations
    std::mt19937 rng;
    std::normal_distribution<double> temp_variation;
    std::normal_distribution<double> flux_variation;
    
    // Mission phase
    std::string current_phase = "NOMINAL";
    
public:
    SpacecraftEnvironment(double period_min, bool geo, bool saa_crossing)
        : orbit_period_minutes(period_min),
          is_geo_orbit(geo),
          has_saa_crossing(saa_crossing),
          rng(std::random_device()()),
          temp_variation(0.0, 5.0),
          flux_variation(1.0, 0.2)
    {
        // Initialize environment
        setupInitialEnvironment();
    }
    
    // Update environment based on orbit position
    void update(double minutes_elapsed) {
        // Update orbit position
        orbit_position += (minutes_elapsed / orbit_period_minutes);
        if (orbit_position >= 1.0) {
            orbit_position -= 1.0;
        }
        
        // Update environment based on position
        updateRadiationEnvironment();
        
        // Update mechanical stress (occasional variations)
        if (std::uniform_real_distribution<>(0.0, 1.0)(rng) < 0.05) {
            current_mechanical_stress_MPa = base_mechanical_stress_MPa * 
                std::uniform_real_distribution<>(0.8, 3.0)(rng);
        } else {
            // Gradually return to base level
            current_mechanical_stress_MPa = 0.9 * current_mechanical_stress_MPa + 
                                          0.1 * base_mechanical_stress_MPa;
        }
    }
    
    // Check for special radiation events
    void checkForRadiationEvents(double minutes_elapsed) {
        // Random solar activity events
        if (std::uniform_real_distribution<>(0.0, 1.0)(rng) < 0.001 * minutes_elapsed) {
            std::cout << "\n*** ALERT: Solar flare detected! ***\n" << std::endl;
            current_env.solar_activity = std::uniform_real_distribution<>(0.7, 0.95)(rng);
            current_phase = "SOLAR_STORM";
        }
        
        // Return to nominal after some time if in solar storm
        if (current_phase == "SOLAR_STORM" && 
            std::uniform_real_distribution<>(0.0, 1.0)(rng) < 0.05 * minutes_elapsed) {
            std::cout << "\n*** Solar activity returning to normal levels ***\n" << std::endl;
            current_env.solar_activity = std::uniform_real_distribution<>(0.1, 0.3)(rng);
            current_phase = "NOMINAL";
        }
    }
    
    // Get current environment
    const sim::RadiationEnvironment& getEnvironment() const {
        return current_env;
    }
    
    // Get current mechanical stress
    double getMechanicalStress() const {
        return current_mechanical_stress_MPa;
    }
    
    // Get current mission phase
    const std::string& getMissionPhase() const {
        return current_phase;
    }
    
    // Print current status
    void printStatus() const {
        std::cout << "Orbit position: " << std::fixed << std::setprecision(2) 
                  << (orbit_position * 100.0) << "% ";
        
        // Location description
        if (is_geo_orbit) {
            std::cout << "GEO ";
        } else {
            std::cout << "LEO ";
            if (has_saa_crossing && isSAARegion()) {
                std::cout << "- SAA CROSSING ";
            }
        }
        
        std::cout << "- Phase: " << current_phase << std::endl;
        
        // Radiation environment
        std::cout << "Temperature: " << std::fixed << std::setprecision(1) 
                  << (current_env.temperature.min + current_env.temperature.max) / 2.0 << "K, ";
        
        std::cout << "Proton flux: " << std::scientific << std::setprecision(2) 
                  << current_env.trapped_proton_flux << ", ";
        
        std::cout << "Electron flux: " << current_env.trapped_electron_flux << ", ";
        
        std::cout << "Solar activity: " << std::fixed << std::setprecision(2) 
                  << current_env.solar_activity << std::endl;
        
        std::cout << "Mechanical stress: " << std::fixed << std::setprecision(1) 
                  << current_mechanical_stress_MPa << " MPa" << std::endl;
    }
    
private:
    // Initialize environment based on orbit type
    void setupInitialEnvironment() {
        if (is_geo_orbit) {
            // Geostationary orbit environment
            current_env.trapped_proton_flux = 5.0e7;
            current_env.trapped_electron_flux = 2.0e7;
            current_env.temperature.min = 250.0; // K
            current_env.temperature.max = 320.0; // K
            current_env.solar_activity = 0.3;
            current_env.saa_region = false;
        } else {
            // Low Earth orbit environment
            current_env.trapped_proton_flux = 1.0e7;
            current_env.trapped_electron_flux = 5.0e6;
            current_env.temperature.min = 270.0; // K
            current_env.temperature.max = 290.0; // K
            current_env.solar_activity = 0.2;
            current_env.saa_region = false;
        }
    }
    
    // Update environment based on current position
    void updateRadiationEnvironment() {
        // Check if in South Atlantic Anomaly for LEO
        if (!is_geo_orbit && has_saa_crossing) {
            current_env.saa_region = isSAARegion();
            
            if (current_env.saa_region) {
                // In SAA region - higher radiation
                current_env.trapped_proton_flux = 5.0e9 * flux_variation(rng);
                current_env.trapped_electron_flux = 1.0e9 * flux_variation(rng);
                current_phase = "SAA_CROSSING";
            } else if (current_phase == "SAA_CROSSING") {
                // Just exited SAA
                current_env.trapped_proton_flux = 1.0e7 * flux_variation(rng);
                current_env.trapped_electron_flux = 5.0e6 * flux_variation(rng);
                current_phase = "NOMINAL";
            } else {
                // Normal LEO variation
                current_env.trapped_proton_flux = 1.0e7 * flux_variation(rng);
                current_env.trapped_electron_flux = 5.0e6 * flux_variation(rng);
            }
        } else if (is_geo_orbit) {
            // GEO orbit variations
            current_env.trapped_proton_flux = 5.0e7 * flux_variation(rng);
            current_env.trapped_electron_flux = 2.0e7 * flux_variation(rng);
        }
        
        // Temperature variations based on sun exposure
        double sun_exposure = calculateSunExposure();
        current_env.temperature.min = (is_geo_orbit ? 230.0 : 260.0) + 20.0 * sun_exposure + temp_variation(rng);
        current_env.temperature.max = (is_geo_orbit ? 300.0 : 280.0) + 40.0 * sun_exposure + temp_variation(rng);
    }
    
    // Check if current position is in the South Atlantic Anomaly
    bool isSAARegion() const {
        // SAA crossing occurs roughly between 30-45% of orbit for a typical LEO
        return (orbit_position > 0.30 && orbit_position < 0.45);
    }
    
    // Calculate sun exposure (0-1) based on position
    double calculateSunExposure() const {
        // Simple sinusoidal model of sun exposure during orbit
        return 0.5 + 0.5 * sin(orbit_position * 2.0 * M_PI);
    }
};

// Image classification task simulator
class ImageClassifier {
private:
    RadiationTolerantNN model;
    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;
    std::mt19937 rng;
    
public:
    ImageClassifier(const core::MaterialProperties& material)
        : model("ImageClassifier", material, {784, 128, 64, 10}),
          rng(std::random_device()())
    {
        // Generate simulated test data (simplified MNIST-like)
        generateTestData(100);
    }
    
    void updateEnvironment(const SpacecraftEnvironment& spacecraft) {
        model.updateEnvironment(
            spacecraft.getEnvironment(),
            spacecraft.getMechanicalStress()
        );
        model.enterMissionPhase(spacecraft.getMissionPhase());
    }
    
    void runInferenceWorkload(int num_images) {
        std::cout << "\nRunning image classification workload (" << num_images << " images)..." << std::endl;
        
        int correct = 0;
        std::uniform_int_distribution<> dist(0, test_images.size() - 1);
        
        for (int i = 0; i < num_images; i++) {
            int img_idx = dist(rng);
            std::vector<float> result = model.forward(test_images[img_idx]);
            
            // Find predicted class (max value)
            int predicted_class = 0;
            float max_val = result[0];
            for (size_t j = 1; j < result.size(); j++) {
                if (result[j] > max_val) {
                    max_val = result[j];
                    predicted_class = j;
                }
            }
            
            // Check if correct
            if (predicted_class == test_labels[img_idx]) {
                correct++;
            } else {
                model.recordError();
            }
        }
        
        double accuracy = static_cast<double>(correct) / num_images;
        std::cout << "Classification results:" << std::endl;
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) 
                  << (accuracy * 100.0) << "%" << std::endl;
        std::cout << "  Overall error rate: " << std::fixed << std::setprecision(4) 
                  << (model.getErrorRate() * 100.0) << "%" << std::endl;
        
        // Output protection information
        std::cout << "Current protection level: " 
                  << protectionLevelToString(model.getCurrentProtection()) << std::endl;
        std::cout << "Checkpoint interval: " << std::fixed << std::setprecision(1)
                  << model.getCheckpointInterval() << " seconds" << std::endl;
        
        model.printProtectionFactors();
    }
    
private:
    void generateTestData(int num_samples) {
        std::cout << "Generating simulated test data..." << std::endl;
        
        std::uniform_real_distribution<> pixel_dist(0.0, 1.0);
        std::uniform_int_distribution<> label_dist(0, 9);
        
        for (int i = 0; i < num_samples; i++) {
            // Generate random image (784 = 28x28 dimensions for MNIST)
            std::vector<float> image(784);
            for (auto& pixel : image) {
                pixel = static_cast<float>(pixel_dist(rng));
            }
            test_images.push_back(image);
            
            // Generate random label
            test_labels.push_back(label_dist(rng));
        }
        
        std::cout << "Generated " << num_samples << " test samples" << std::endl;
    }
    
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
};

// Main simulation
int main() {
    std::cout << "===============================================" << std::endl;
    std::cout << "  Radiation-Tolerant ML Framework Simulation" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Setup material properties
    core::MaterialProperties aluminum;
    aluminum.yield_strength = 270.0; // MPa
    aluminum.radiation_tolerance = 50.0; // Relative scale
    
    // Create spacecraft environment for LEO with SAA crossing
    SpacecraftEnvironment spacecraft(90.0, false, true); // 90-minute LEO orbit with SAA crossing
    
    // Create ML workload
    ImageClassifier classifier(aluminum);
    
    // Output file for logging
    std::ofstream log_file("radiation_ml_simulation.log");
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file" << std::endl;
        return 1;
    }
    
    // Run simulation
    const int simulation_minutes = 180; // 3 hours
    const double time_step_minutes = 5.0; // 5 minutes per step
    
    std::cout << "\nStarting " << simulation_minutes << " minute orbital simulation..." << std::endl;
    
    for (double elapsed_time = 0; elapsed_time < simulation_minutes; elapsed_time += time_step_minutes) {
        // Print simulation time
        int hours = static_cast<int>(elapsed_time) / 60;
        int mins = static_cast<int>(elapsed_time) % 60;
        std::cout << "\n===============================================" << std::endl;
        std::cout << "Mission Time: " << std::setfill('0') << std::setw(2) << hours 
                  << ":" << std::setfill('0') << std::setw(2) << mins << std::endl;
        
        // Update spacecraft environment
        spacecraft.update(time_step_minutes);
        spacecraft.checkForRadiationEvents(time_step_minutes);
        spacecraft.printStatus();
        
        // Update ML protection
        classifier.updateEnvironment(spacecraft);
        
        // Run ML workload
        classifier.runInferenceWorkload(20); // Process 20 images each step
        
        // Log to file
        log_file << elapsed_time << "," 
                 << spacecraft.getEnvironment().trapped_proton_flux << "," 
                 << spacecraft.getEnvironment().saa_region << "," 
                 << static_cast<int>(classifier.getCurrentProtection()) << std::endl;
        
        // Slight pause for readability in the demo
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    log_file.close();
    std::cout << "\nSimulation complete. Results logged to radiation_ml_simulation.log" << std::endl;
    
    return 0;
} 