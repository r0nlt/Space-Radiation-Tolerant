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
#include <vector>
#include <string>
#include <map>
#include <rad_ml/physics/quantum_integration.hpp>

// Define required material properties for the example
struct Material {
    std::string name;
    CrystalLattice crystal;
    
    Material(const std::string& n, const CrystalLattice& c) : name(n), crystal(c) {}
};

// Define a simplified radiation simulation
class RadiationSimulator {
private:
    // Simulation parameters
    double temperature;
    double feature_size;
    double radiation_intensity;
    rad_ml::physics::QuantumCorrectionConfig quantum_config;
    bool verbose;
    
public:
    RadiationSimulator(
        double temp = 300.0,
        double size = 45.0,
        double intensity = 1e5,
        bool verbose_output = true) 
        : temperature(temp), 
          feature_size(size),
          radiation_intensity(intensity),
          verbose(verbose_output) {
        
        // Initialize quantum configuration with defaults
        quantum_config = rad_ml::physics::getDefaultQuantumConfig();
    }
    
    // Set configuration options
    void setQuantumConfig(const rad_ml::physics::QuantumCorrectionConfig& config) {
        quantum_config = config;
    }
    
    // Enable or disable quantum corrections
    void enableQuantumCorrections(bool enable) {
        quantum_config.enable_quantum_corrections = enable;
    }
    
    // Force quantum corrections regardless of thresholds
    void forceQuantumCorrections(bool force) {
        quantum_config.force_quantum_corrections = force;
    }
    
    // Set environment parameters
    void setTemperature(double temp) {
        temperature = temp;
    }
    
    void setFeatureSize(double size) {
        feature_size = size;
    }
    
    void setRadiationIntensity(double intensity) {
        radiation_intensity = intensity;
    }
    
    // Run radiation simulation for a given material
    DefectDistribution simulateRadiationDamage(const Material& material) {
        if (verbose) {
            std::cout << "Simulating radiation damage for " << material.name << std::endl;
            std::cout << "  Temperature: " << temperature << " K" << std::endl;
            std::cout << "  Feature size: " << feature_size << " nm" << std::endl;
            std::cout << "  Radiation intensity: " << radiation_intensity << " rad/s" << std::endl;
        }
        
        // Create a classical defect distribution based on the material and conditions
        DefectDistribution classical_defects = createClassicalDefects(material);
        
        if (verbose) {
            std::cout << "Classical defect simulation completed." << std::endl;
            printDefectDistribution("Classical defects", classical_defects);
        }
        
        // Check if quantum corrections should be applied
        bool apply_quantum = rad_ml::physics::shouldApplyQuantumCorrections(
            temperature, feature_size, radiation_intensity, quantum_config);
        
        if (verbose) {
            std::cout << "Quantum corrections " << (apply_quantum ? "will" : "will not") 
                      << " be applied based on current conditions." << std::endl;
        }
        
        // Apply quantum corrections if appropriate
        DefectDistribution final_defects;
        
        if (apply_quantum) {
            if (verbose) {
                std::cout << "Applying quantum field corrections..." << std::endl;
            }
            
            final_defects = rad_ml::physics::applyQuantumCorrectionsToSimulation(
                classical_defects, 
                material.crystal, 
                temperature, 
                feature_size, 
                radiation_intensity, 
                quantum_config);
                
            if (verbose) {
                printDefectDistribution("Quantum-corrected defects", final_defects);
                compareDistributions(classical_defects, final_defects);
            }
        } else {
            final_defects = classical_defects;
            
            if (verbose) {
                std::cout << "Using classical defect model (no quantum corrections)." << std::endl;
            }
        }
        
        return final_defects;
    }
    
private:
    // Create a classical defect distribution based on material properties and radiation
    DefectDistribution createClassicalDefects(const Material& material) {
        // This is a simplified model for demonstration purposes
        // A real implementation would use proper radiation physics calculations
        
        DefectDistribution defects;
        
        // Base values depend on the lattice type
        double base_vacancy = 100.0;
        double base_interstitial = 80.0;
        double base_complex = 20.0;
        double base_dislocation = 15.0;
        
        // Scale based on radiation intensity (simple linear relationship)
        double intensity_factor = radiation_intensity / 1e5;
        
        // Temperature effects (defects anneal at higher temperatures)
        double temp_factor = std::exp(-temperature / 600.0);
        
        // Calculate defect counts
        defects["vacancy"] = base_vacancy * intensity_factor * (1.0 + 0.5 * temp_factor);
        defects["interstitial"] = base_interstitial * intensity_factor * (1.0 + 0.3 * temp_factor);
        defects["complex"] = base_complex * intensity_factor * (1.0 + 0.7 * temp_factor);
        defects["dislocation"] = base_dislocation * intensity_factor * (1.0 + 0.1 * temp_factor);
        
        return defects;
    }
    
    // Print a defect distribution
    void printDefectDistribution(const std::string& title, const DefectDistribution& defects) {
        std::cout << title << ":" << std::endl;
        
        double total = 0.0;
        for (const auto& defect_pair : defects) {
            std::cout << "  " << std::left << std::setw(15) << defect_pair.first 
                      << std::fixed << std::setprecision(2) << defect_pair.second << std::endl;
            total += defect_pair.second;
        }
        
        std::cout << "  " << std::left << std::setw(15) << "TOTAL" 
                  << std::fixed << std::setprecision(2) << total << std::endl;
        std::cout << std::endl;
    }
    
    // Compare two defect distributions
    void compareDistributions(
        const DefectDistribution& classical,
        const DefectDistribution& quantum) {
        
        std::cout << "Comparison of classical vs. quantum-corrected defects:" << std::endl;
        
        double total_classical = 0.0;
        double total_quantum = 0.0;
        
        for (const auto& defect_pair : classical) {
            const std::string& defect_type = defect_pair.first;
            double classical_count = defect_pair.second;
            double quantum_count = quantum.at(defect_type);
            
            double percent_diff = ((quantum_count - classical_count) / classical_count) * 100.0;
            
            std::cout << "  " << std::left << std::setw(15) << defect_type 
                      << std::fixed << std::setprecision(2) << classical_count << " -> " 
                      << std::setprecision(2) << quantum_count 
                      << " (" << std::showpos << std::setprecision(2) << percent_diff << "%) " 
                      << std::noshowpos << std::endl;
                      
            total_classical += classical_count;
            total_quantum += quantum_count;
        }
        
        double total_percent_diff = ((total_quantum - total_classical) / total_classical) * 100.0;
        std::cout << "  " << std::left << std::setw(15) << "TOTAL" 
                  << std::fixed << std::setprecision(2) << total_classical << " -> " 
                  << std::setprecision(2) << total_quantum 
                  << " (" << std::showpos << std::setprecision(2) << total_percent_diff << "%) " 
                  << std::noshowpos << std::endl;
                  
        std::cout << std::endl;
    }
};

// Create test materials
Material createSilicon() {
    return Material("Silicon", CrystalLattice(CrystalLattice::DIAMOND, 5.431, 1.1));
}

Material createGermanium() {
    return Material("Germanium", CrystalLattice(CrystalLattice::DIAMOND, 5.658, 0.67));
}

Material createGaAs() {
    return Material("GaAs", CrystalLattice(CrystalLattice::DIAMOND, 5.653, 0.84));
}

// Main example function
int main() {
    std::cout << "=== Quantum Field Theory Integration Example ===" << std::endl;
    std::cout << std::string(50, '=') << std::endl << std::endl;
    
    // Create materials
    Material silicon = createSilicon();
    Material germanium = createGermanium();
    Material gaas = createGaAs();
    
    // Create simulator with default settings (room temperature, 45nm feature size)
    RadiationSimulator simulator;
    
    std::cout << "Example 1: Room Temperature Silicon (300K, 45nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    simulator.simulateRadiationDamage(silicon);
    std::cout << std::endl;
    
    // Change to low temperature to trigger quantum effects
    std::cout << "Example 2: Low Temperature Silicon (77K, 45nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    simulator.setTemperature(77.0);
    simulator.simulateRadiationDamage(silicon);
    std::cout << std::endl;
    
    // Change to nanoscale device to trigger quantum effects
    std::cout << "Example 3: Nanoscale Silicon Device (300K, 10nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    simulator.setTemperature(300.0);
    simulator.setFeatureSize(10.0);
    simulator.simulateRadiationDamage(silicon);
    std::cout << std::endl;
    
    // Test other materials
    std::cout << "Example 4: Cryogenic GaAs Nano-Device (4.2K, 5nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    simulator.setTemperature(4.2);
    simulator.setFeatureSize(5.0);
    simulator.simulateRadiationDamage(gaas);
    std::cout << std::endl;
    
    // Compare with quantum corrections disabled
    std::cout << "Example 5: Quantum vs. Classical Comparison" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Run with quantum corrections
    simulator.enableQuantumCorrections(true);
    simulator.setTemperature(77.0);
    simulator.setFeatureSize(10.0);
    std::cout << "WITH quantum corrections:" << std::endl;
    DefectDistribution quantum_result = simulator.simulateRadiationDamage(germanium);
    
    // Run the same scenario without quantum corrections
    simulator.enableQuantumCorrections(false);
    std::cout << std::endl << "WITHOUT quantum corrections:" << std::endl;
    DefectDistribution classical_result = simulator.simulateRadiationDamage(germanium);
    
    // Enable quantum for the last example
    simulator.enableQuantumCorrections(true);
    
    std::cout << "Integration example completed successfully." << std::endl;
    std::cout << "Quantum field theory corrections have been seamlessly integrated into the radiation simulation." << std::endl;
    
    return 0;
} 
