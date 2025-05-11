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
#include <rad_ml/physics/quantum_integration.hpp>
#include <cmath>
#include <iostream>
#include <algorithm> // For std::max, std::min

namespace rad_ml {
namespace physics {

QFTParameters createQFTParameters(const CrystalLattice& crystal, double feature_size_nm) {
    QFTParameters params;
    
    // Physics constants
    params.hbar = 6.582119569e-16;  // reduced Planck constant (eV·s)
    
    // Material-specific parameters
    // Mass depends on the material type
    switch (crystal.type) {
        case CrystalLattice::FCC_TYPE:
            params.mass = 1.0e-30; // Default FCC mass value
            break;
        case CrystalLattice::BCC:
            params.mass = 1.1e-30; // BCC materials have slightly different effective mass
            break;
        case CrystalLattice::DIAMOND:
            params.mass = 0.9e-30; // Diamond lattice materials
            break;
        default:
            params.mass = 1.0e-30; // Default value
    }
    
    // Scaling parameters based on material properties
    params.coupling_constant = 0.1 * (crystal.lattice_constant / 5.0); // Scale with lattice constant
    params.potential_coefficient = 0.5;
    
    // Feature size impacts lattice spacing parameter
    params.lattice_spacing = feature_size_nm / 100.0; // Convert to appropriate scale
    
    // Simulation parameters
    params.time_step = 1.0e-18; // fs time scale
    params.dimensions = 3;
    
    return params;
}

bool shouldApplyQuantumCorrections(
    double temperature,
    double feature_size,
    double radiation_intensity,
    const QuantumCorrectionConfig& config) {
    
    if (!config.enable_quantum_corrections) {
        return false;
    }
    
    if (config.force_quantum_corrections) {
        return true;
    }
    
    // Apply based on thresholds
    bool temperature_criterion = temperature < config.temperature_threshold;
    bool feature_size_criterion = feature_size < config.feature_size_threshold;
    bool radiation_criterion = radiation_intensity > config.radiation_intensity_threshold;
    
    // Apply corrections if any criterion is met
    return temperature_criterion || feature_size_criterion || radiation_criterion;
}

DefectDistribution applyQuantumCorrectionsToSimulation(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    double temperature,
    double feature_size_nm,
    double radiation_intensity,
    const QuantumCorrectionConfig& config) {
    
    // Check if we should apply quantum corrections
    if (!shouldApplyQuantumCorrections(temperature, feature_size_nm, radiation_intensity, config)) {
        return defects; // Return original defects without quantum corrections
    }
    
    // Create QFT parameters based on material properties
    QFTParameters qft_params = createQFTParameters(crystal, feature_size_nm);
    
    // Apply quantum field corrections
    DefectDistribution corrected_defects = applyQuantumFieldCorrections(
        defects, crystal, qft_params, temperature);
    
    return corrected_defects;
}

double calculateQuantumEnhancementFactor(double temperature, double feature_size) {
    // Base enhancement (no enhancement = 1.0)
    double enhancement = 1.0;
    
    // Temperature effect: More pronounced at low temperatures
    // Adjusted to be more conservative and physically realistic
    if (temperature < 150.0) {
        // Use a gradual, physically based scaling that approaches asymptotic limit
        // Avoid exponential growth that could be too aggressive
        double temp_factor = std::min(5.0, 150.0 / std::max(temperature, 10.0));
        
        // Cap the maximum enhancement from temperature alone to 4% (down from 5%)
        enhancement *= (1.0 + temp_factor * 0.04); 
    }
    
    // Size effect: More pronounced at small feature sizes
    // More conservative scaling based on physical principles
    if (feature_size < 20.0) {
        // Linear factor with saturation to avoid unrealistic scaling
        double size_factor = std::min(4.0, 20.0 / std::max(feature_size, 2.0));
        
        // Cap the maximum enhancement from size alone to 8% (down from 10%)
        enhancement *= (1.0 + size_factor * 0.08);
    }
    
    return enhancement;
}

// Implementation of functions defined in quantum_field_theory.hpp
// These would normally be in quantum_field_theory.cpp, but for testing we'll define them here

DefectDistribution applyQuantumFieldCorrections(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    const QFTParameters& qft_params,
    double temperature) {
    
    // Start with a copy of the original defects
    DefectDistribution corrected_defects = defects;
    
    // Calculate quantum corrections
    double tunneling_probability = calculateQuantumTunnelingProbability(
        crystal.barrier_height,
        qft_params.mass,
        qft_params.hbar,
        temperature
    );
    
    // Apply Klein-Gordon correction factor
    double kg_correction = solveKleinGordonEquation(
        qft_params.hbar,
        qft_params.mass,
        qft_params.potential_coefficient,
        qft_params.coupling_constant,
        qft_params.lattice_spacing,
        qft_params.time_step
    );
    
    // Calculate zero-point energy contribution
    double zpe_contribution = calculateZeroPointEnergyContribution(
        qft_params.hbar,
        qft_params.mass,
        crystal.lattice_constant,
        temperature
    );
    
    // Apply corrections to each defect type
    for (auto& defect_pair : corrected_defects) {
        std::string defect_type = defect_pair.first;
        double& defect_count = defect_pair.second;
        
        // Apply different correction factors based on defect type
        // More conservative scaling factors based on physical principles
        if (defect_type == "vacancy") {
            // Vacancies are less affected by tunneling
            defect_count *= (1.0 + 0.4 * tunneling_probability + 0.6 * kg_correction);
        } else if (defect_type == "interstitial") {
            // Interstitials are strongly affected by tunneling
            defect_count *= (1.0 + 1.2 * tunneling_probability + 0.8 * kg_correction);
        } else if (defect_type == "complex") {
            // Complex defects show intermediate behavior
            defect_count *= (1.0 + 0.8 * tunneling_probability + 0.8 * kg_correction);
        } else {
            // Default correction
            defect_count *= (1.0 + 0.6 * tunneling_probability + 0.7 * kg_correction);
        }
        
        // Add zero-point energy contribution, less aggressive scaling
        defect_count += zpe_contribution * defect_count * 0.008; // Apply as a smaller percentage (0.8% vs 1%)
    }
    
    // Log the correction factors
    std::cout << "Applied quantum corrections with factors: " << std::endl;
    std::cout << "  - Tunneling probability: " << tunneling_probability << std::endl;
    std::cout << "  - Klein-Gordon correction: " << kg_correction << std::endl;
    std::cout << "  - Zero-point energy contribution: " << zpe_contribution << std::endl;
    
    return corrected_defects;
}

double calculateQuantumTunnelingProbability(
    double barrier_height,
    double mass,
    double hbar,
    double temperature) {
    
    // Improved numerical stability for WKB approximation
    const double barrier_width = 1.0; // nm
    const double kb = 8.617333262e-5; // Boltzmann constant in eV/K
    
    // Prevent division by zero or negative temperatures
    double safe_temp = std::max(temperature, 1.0); // Minimum 1K to avoid div by zero
    
    // Calculate thermal energy with bounds check
    double thermal_energy = kb * safe_temp;
    
    // More numerically stable calculation with bounds checking
    // Prevent potential overflow in sqrt and exp operations
    
    // Safety check for barrier height
    double safe_barrier = std::max(barrier_height, 0.01); // Minimum 0.01 eV
    
    // Capped exponent calculation for numerical stability
    double exponent_term = -2.0 * barrier_width * std::sqrt(2.0 * mass * safe_barrier) / hbar;
    exponent_term = std::max(-30.0, exponent_term); // Prevent extreme underflow
    
    double base_probability = std::exp(exponent_term);
    
    // Bound base probability to physically reasonable values
    base_probability = std::min(0.1, base_probability); // Cap at 10% max
    
    // Temperature correction with improved stability
    double temp_ratio = thermal_energy / (2.0 * safe_barrier);
    temp_ratio = std::min(10.0, temp_ratio); // Prevent extreme values
    
    double temp_factor = std::exp(-temp_ratio);
    
    // Final bounded probability
    double result = base_probability * temp_factor;
    
    // Additional sanity check for final result
    return std::min(0.05, std::max(0.0, result)); // Keep between 0% and 5%
}

double solveKleinGordonEquation(
    double hbar,
    double mass,
    double potential_coeff,
    double coupling_constant,
    double lattice_spacing,
    double time_step) {
    
    // Simplified Klein-Gordon equation solution
    // In a full implementation, this would involve solving the differential equation
    
    // Simplified model: correction factor based on quantum parameters
    // Added bounds checking for numerical stability
    double safe_lattice_spacing = std::max(lattice_spacing, 0.001); // Avoid division by very small values
    double safe_mass = std::max(mass, 1.0e-32); // Avoid division by very small mass
    
    double wave_factor = hbar / (safe_mass * safe_lattice_spacing * safe_lattice_spacing);
    double potential_factor = potential_coeff * safe_lattice_spacing;
    double coupling_factor = coupling_constant * time_step;
    
    // Combine factors (this is a simplified model)
    double correction = wave_factor * (1.0 + potential_factor + coupling_factor);
    
    // Scale to a reasonable correction range
    // Reduced from 1% to 0.8% base correction for more conservative scaling
    correction = 0.008 * correction;
    
    // Enforce reasonable bounds on correction factor
    return std::min(0.04, std::max(0.0, correction)); // Keep between 0 and 4%
}

double calculateZeroPointEnergyContribution(
    double hbar,
    double mass,
    double lattice_constant,
    double temperature) {
    
    // Simple harmonic oscillator zero-point energy: E₀ = ħω/2
    // where ω = √(k/m) and k is the spring constant
    
    // Estimate spring constant from lattice properties
    double spring_constant = 10.0; // eV/Å² (typical for covalent bonds)
    
    // Safety check for mass - avoid division by zero
    double safe_mass = std::max(mass, 1.0e-32);
    
    // Calculate angular frequency with bounds checking
    double omega = std::sqrt(spring_constant / safe_mass);
    
    // Calculate zero-point energy
    double zpe = 0.5 * hbar * omega;
    
    // Temperature scaling (ZPE becomes more significant at lower temperatures)
    const double kb = 8.617333262e-5; // Boltzmann constant in eV/K
    
    // Safety check for temperature - avoid division by zero
    double safe_temp = std::max(temperature, 1.0); // Minimum 1K
    
    double thermal_energy = kb * safe_temp;
    
    // Calculate ratio of ZPE to thermal energy with bounded result
    double zpe_significance = zpe / (thermal_energy + zpe);
    
    // Ensure result is within physically reasonable bounds
    return std::min(0.1, std::max(0.0, zpe_significance)); // Cap at 10%
}

} // namespace physics
} // namespace rad_ml 
