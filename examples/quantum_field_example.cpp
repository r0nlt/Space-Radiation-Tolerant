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
#include <vector>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>
#include <rad_ml/physics/field_theory.hpp>

using namespace rad_ml::physics;

int main() {
    // Set up quantum field theory parameters
    QFTParameters qft_params;
    qft_params.hbar = 6.582119569e-16;  // Reduced Planck constant in eV·s
    qft_params.mass = 1.0e-30;          // Particle mass in kg
    qft_params.coupling_constant = 0.1;  // Dimensionless coupling constant
    qft_params.potential_coefficient = 0.5; // Coefficient in potential term
    qft_params.lattice_spacing = 0.1;    // Spatial lattice spacing in nm
    qft_params.time_step = 1.0e-18;      // Time step in seconds
    qft_params.dimensions = 3;           // 3D simulation

    // Create a crystal lattice for silicon
    CrystalLattice silicon = CrystalLatticeFactory::FCC(5.431); // Silicon lattice constant in Angstroms
    
    // Calculate displacement energy
    double displacement_energy = calculateDisplacementEnergy(silicon, qft_params);
    std::cout << "Displacement energy: " << displacement_energy << " eV" << std::endl;
    
    // Simulate a displacement cascade
    double pka_energy = 1000.0; // 1 keV primary knock-on atom
    DefectDistribution defects = simulateDisplacementCascade(
        silicon, pka_energy, qft_params, displacement_energy);
    
    // Apply quantum field corrections
    double temperature = 300.0; // K
    DefectDistribution corrected_defects = applyQuantumFieldCorrections(
        defects, silicon, qft_params, temperature);
    
    // Calculate and print the differences
    std::cout << "Classical vs. Quantum-Corrected Defect Counts:" << std::endl;
    
    // Print interstitials
    std::cout << "Interstitials:" << std::endl;
    for (size_t i = 0; i < defects.interstitials.size(); i++) {
        std::cout << "  Region " << i << ": " 
                  << defects.interstitials[i] << " vs. " 
                  << corrected_defects.interstitials[i] << " ("
                  << (corrected_defects.interstitials[i] / defects.interstitials[i] - 1.0) * 100.0
                  << "% change)" << std::endl;
    }
    
    // Initialize a quantum field for Klein-Gordon equation
    std::vector<int> grid_dimensions = {32, 32, 32};
    QuantumField<3> scalar_field(grid_dimensions, qft_params.lattice_spacing);
    scalar_field.initializeGaussian(0.0, 0.1);
    
    // Create a Klein-Gordon equation solver
    KleinGordonEquation kg_equation(qft_params);
    
    // Evolve the field for 100 steps
    std::cout << "\nEvolving Klein-Gordon field..." << std::endl;
    for (int step = 0; step < 100; step++) {
        kg_equation.evolveField(scalar_field);
        
        // Calculate and print the total energy every 10 steps
        if (step % 10 == 0) {
            double energy = scalar_field.calculateTotalEnergy(qft_params);
            std::cout << "Step " << step << ": Total energy = " << energy << std::endl;
        }
    }
    
    // Initialize quantum fields for electromagnetic simulation
    QuantumField<3> electric_field(grid_dimensions, qft_params.lattice_spacing);
    QuantumField<3> magnetic_field(grid_dimensions, qft_params.lattice_spacing);
    
    // Initialize with a plane wave
    electric_field.initializeCoherentState(1.0, 0.0);
    magnetic_field.initializeCoherentState(1.0, 1.57); // π/2 phase shift
    
    // Create Maxwell equations solver
    MaxwellEquations maxwell_equations(qft_params);
    
    // Evolve the electromagnetic field for 100 steps
    std::cout << "\nEvolving electromagnetic field..." << std::endl;
    for (int step = 0; step < 100; step++) {
        maxwell_equations.evolveField(electric_field, magnetic_field);
        
        // Calculate and print field correlation every 20 steps
        if (step % 20 == 0) {
            auto correlation = electric_field.calculateCorrelationFunction(10);
            std::cout << "Step " << step << ": Correlation at distance 1 = " 
                      << correlation(1, 0) << std::endl;
        }
    }
    
    return 0;
} 
