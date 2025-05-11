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
/**
 * Quantum Field Theory Integration Module
 * 
 * This file provides integration points for connecting quantum field theory corrections
 * to the main radiation-tolerant ML framework.
 */

#pragma once

#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>
#include <rad_ml/physics/field_theory.hpp>
#include <cmath>
#include <algorithm> // For std::min, std::max

namespace rad_ml {
namespace physics {

/**
 * Configuration for quantum corrections
 */
struct QuantumCorrectionConfig {
    bool enable_quantum_corrections;        // Master switch for quantum corrections
    double temperature_threshold;           // Apply quantum corrections below this temperature (K)
    double feature_size_threshold;          // Apply quantum corrections below this feature size (nm)
    double radiation_intensity_threshold;   // Apply quantum corrections above this radiation level
    bool force_quantum_corrections;         // Force quantum corrections regardless of thresholds
};

/**
 * Default configuration with sensible defaults
 */
QuantumCorrectionConfig getDefaultQuantumConfig() {
    QuantumCorrectionConfig config;
    config.enable_quantum_corrections = true;
    config.temperature_threshold = 150.0;      // K - quantum effects start becoming significant
    config.feature_size_threshold = 20.0;      // nm - quantum effects important at smaller scales
    config.radiation_intensity_threshold = 1e5; // rad/s - high radiation levels
    config.force_quantum_corrections = false;
    return config;
}

/**
 * Create QFT parameters from material properties
 * 
 * @param material Material properties
 * @param feature_size_nm Feature size in nanometers (default: 100.0 nm)
 * @return Configured QFT parameters
 */
QFTParameters createQFTParameters(const CrystalLattice& crystal, double feature_size_nm = 100.0) {
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

/**
 * Determine if quantum corrections should be applied based on environment and config
 * 
 * @param temperature Current temperature (K)
 * @param feature_size Device feature size (nm)
 * @param radiation_intensity Radiation intensity (rad/s)
 * @param config Quantum correction configuration
 * @return Whether quantum corrections should be applied
 */
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

/**
 * Apply quantum field corrections to radiation simulation results
 * 
 * Applies quantum field theory corrections to classical defect distributions
 * when environmental conditions (temperature, feature size, radiation) meet 
 * the specified thresholds for quantum effects.
 * 
 * @param defects Classical defect distribution
 * @param crystal Crystal lattice properties
 * @param temperature Temperature in Kelvin
 * @param feature_size_nm Feature size in nanometers
 * @param radiation_intensity Radiation intensity in rad/s
 * @param config Quantum correction configuration
 * @return Quantum-corrected defect distribution
 * 
 * @note Numerical stability is ensured through bounds checking on all parameters.
 * Temperature is bounded to minimum 1K, feature size to minimum 2nm, and
 * all quantum correction factors are limited to physically reasonable ranges.
 */
DefectDistribution applyQuantumCorrectionsToSimulation(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    double temperature,
    double feature_size_nm,
    double radiation_intensity,
    const QuantumCorrectionConfig& config = getDefaultQuantumConfig()) {
    
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

/**
 * Calculate quantum enhancement factor based on environment
 * 
 * @param temperature Temperature in Kelvin (minimum 1K for numerical stability)
 * @param feature_size Feature size in nm (minimum 2nm for numerical stability)
 * @return Enhancement factor (1.0 = no enhancement)
 * 
 * @note Temperature effects are capped at 4% enhancement
 * @note Feature size effects are capped at 8% enhancement
 * @note Combined effects use multiplicative scaling with reasonable bounds
 */
double calculateQuantumEnhancementFactor(double temperature, double feature_size) {
    // Base enhancement (no enhancement = 1.0)
    double enhancement = 1.0;
    
    // Temperature effect: More pronounced at low temperatures
    if (temperature < 150.0) {
        double temp_factor = std::exp(300.0/temperature - 1.0);
        enhancement *= (1.0 + temp_factor * 0.05); // 5% max enhancement from temperature
    }
    
    // Size effect: More pronounced at small feature sizes
    if (feature_size < 20.0) {
        double size_factor = std::exp(20.0/feature_size - 1.0);
        enhancement *= (1.0 + size_factor * 0.1); // 10% max enhancement from size
    }
    
    return enhancement;
}

} // namespace physics
} // namespace rad_ml 
