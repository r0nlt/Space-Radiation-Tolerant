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
#include <fstream>
#include <vector>
#include <string>
#include <rad_ml/physics/quantum_integration.hpp>

// Main test function
int main() {
    std::cout << "=== Quantum Integration Test ===" << std::endl;
    std::cout << std::string(50, '=') << std::endl << std::endl;
    
    rad_ml::physics::CrystalLattice silicon(rad_ml::physics::CrystalLattice::DIAMOND, 5.431, 1.1);
    rad_ml::physics::CrystalLattice germanium(rad_ml::physics::CrystalLattice::DIAMOND, 5.658, 0.67);
    rad_ml::physics::CrystalLattice gaas(rad_ml::physics::CrystalLattice::DIAMOND, 5.653, 0.84);
    
    // Create basic defect distribution for testing
    rad_ml::physics::DefectDistribution defects;
    defects["vacancy"] = 100.0;
    defects["interstitial"] = 80.0;
    defects["complex"] = 20.0;
    defects["dislocation"] = 15.0;
    
    // Test different temperature conditions
    std::vector<double> temperatures = {300.0, 150.0, 77.0, 4.2};
    
    std::cout << "Testing quantum integration with different temperatures:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (double temp : temperatures) {
        // Get default config
        rad_ml::physics::QuantumCorrectionConfig config = rad_ml::physics::getDefaultQuantumConfig();
        
        // Apply quantum corrections
        rad_ml::physics::DefectDistribution corrected = rad_ml::physics::applyQuantumCorrectionsToSimulation(
            defects, silicon, temp, 45.0, 1e5, config);
        
        // Calculate total defects
        double total_original = 0.0;
        double total_corrected = 0.0;
        
        for (const auto& defect_pair : defects) {
            total_original += defect_pair.second;
            total_corrected += corrected[defect_pair.first];
        }
        
        double percent_diff = ((total_corrected - total_original) / total_original) * 100.0;
        
        std::cout << "Temperature " << temp << "K: "
                 << std::fixed << std::setprecision(2) << total_original << " -> " 
                 << std::fixed << std::setprecision(2) << total_corrected
                 << " (" << std::showpos << std::setprecision(2) << percent_diff << "%)" 
                 << std::noshowpos << std::endl;
    }
    
    std::cout << std::endl;
    
    // Test different feature sizes
    std::vector<double> feature_sizes = {45.0, 22.0, 10.0, 5.0, 3.0};
    
    std::cout << "Testing quantum integration with different feature sizes:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (double size : feature_sizes) {
        // Get default config
        rad_ml::physics::QuantumCorrectionConfig config = rad_ml::physics::getDefaultQuantumConfig();
        
        // Apply quantum corrections
        rad_ml::physics::DefectDistribution corrected = rad_ml::physics::applyQuantumCorrectionsToSimulation(
            defects, silicon, 300.0, size, 1e5, config);
        
        // Calculate total defects
        double total_original = 0.0;
        double total_corrected = 0.0;
        
        for (const auto& defect_pair : defects) {
            total_original += defect_pair.second;
            total_corrected += corrected[defect_pair.first];
        }
        
        double percent_diff = ((total_corrected - total_original) / total_original) * 100.0;
        
        std::cout << "Feature size " << size << "nm: "
                 << std::fixed << std::setprecision(2) << total_original << " -> " 
                 << std::fixed << std::setprecision(2) << total_corrected
                 << " (" << std::showpos << std::setprecision(2) << percent_diff << "%)" 
                 << std::noshowpos << std::endl;
    }
    
    std::cout << std::endl;
    
    // Test different materials
    std::cout << "Testing quantum integration with different materials:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    std::vector<std::pair<std::string, rad_ml::physics::CrystalLattice>> materials = {
        {"Silicon", silicon},
        {"Germanium", germanium},
        {"GaAs", gaas}
    };
    
    for (const auto& material_pair : materials) {
        // Get default config
        rad_ml::physics::QuantumCorrectionConfig config = rad_ml::physics::getDefaultQuantumConfig();
        
        // Test at low temperature to see more quantum effects
        rad_ml::physics::DefectDistribution corrected = rad_ml::physics::applyQuantumCorrectionsToSimulation(
            defects, material_pair.second, 77.0, 10.0, 1e5, config);
        
        // Calculate total defects
        double total_original = 0.0;
        double total_corrected = 0.0;
        
        for (const auto& defect_pair : defects) {
            total_original += defect_pair.second;
            total_corrected += corrected[defect_pair.first];
        }
        
        double percent_diff = ((total_corrected - total_original) / total_original) * 100.0;
        
        std::cout << material_pair.first << " (77K, 10nm): "
                 << std::fixed << std::setprecision(2) << total_original << " -> " 
                 << std::fixed << std::setprecision(2) << total_corrected
                 << " (" << std::showpos << std::setprecision(2) << percent_diff << "%)" 
                 << std::noshowpos << std::endl;
    }
    
    std::cout << std::endl;
    
    // Export test results to CSV
    std::ofstream results_file("quantum_integration_test_results.csv");
    results_file << "Material,Temperature,Feature Size,Classical Total,Quantum Total,Difference (%)\n";
    
    for (const auto& material_pair : materials) {
        for (double temp : temperatures) {
            for (double size : feature_sizes) {
                rad_ml::physics::QuantumCorrectionConfig config = rad_ml::physics::getDefaultQuantumConfig();
                rad_ml::physics::DefectDistribution quantum = rad_ml::physics::applyQuantumCorrectionsToSimulation(
                    defects, material_pair.second, temp, size, 1e5, config);
                
                double total_classical = 0.0;
                double total_quantum = 0.0;
                
                for (const auto& defect : defects) {
                    total_classical += defect.second;
                    total_quantum += quantum[defect.first];
                }
                
                double diff_percent = ((total_quantum - total_classical) / total_classical) * 100.0;
                
                results_file << material_pair.first << ","
                            << temp << ","
                            << size << ","
                            << total_classical << ","
                            << total_quantum << ","
                            << diff_percent << "\n";
            }
        }
    }
    
    results_file.close();
    std::cout << "Test results saved to quantum_integration_test_results.csv" << std::endl;
    
    return 0;
} 
