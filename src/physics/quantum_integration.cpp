#include <algorithm>  // For std::max, std::min
#include <cmath>
#include <iostream>
#include <rad_ml/physics/quantum_integration.hpp>

namespace rad_ml {
namespace physics {

QFTParameters createQFTParameters(const CrystalLattice& crystal, double feature_size_nm)
{
    QFTParameters params;

    // Physics constants
    params.hbar = 6.582119569e-16;  // reduced Planck constant (eV·s)

    // Material-specific parameters
    // Mass depends on the material type
    switch (crystal.type) {
        case CrystalLattice::Type::FCC:
            params.masses[ParticleType::Proton] = 1.0e-30;  // Default FCC mass value
            break;
        case CrystalLattice::Type::BCC:
            params.masses[ParticleType::Proton] =
                1.1e-30;  // BCC materials have slightly different effective mass
            break;
        case CrystalLattice::Type::DIAMOND:
            params.masses[ParticleType::Proton] = 0.9e-30;  // Diamond lattice materials
            break;
        default:
            params.masses[ParticleType::Proton] = 1.0e-30;  // Default value
    }

    // Set values for other common particles
    params.masses[ParticleType::Electron] =
        params.masses[ParticleType::Proton] * 0.0005;  // Approx electron/proton mass ratio
    params.masses[ParticleType::Neutron] =
        params.masses[ParticleType::Proton] * 1.001;  // Slightly heavier than proton
    params.masses[ParticleType::Photon] = 1.0e-32;    // Very small but non-zero for calculations

    // Scaling parameters based on material properties
    double coupling_constant_base =
        0.1 * (crystal.lattice_constant / 5.0);  // Scale with lattice constant

    // Set coupling constants for all particles
    params.coupling_constants[ParticleType::Proton] = coupling_constant_base;
    params.coupling_constants[ParticleType::Electron] =
        coupling_constant_base * 1.2;  // Electrons couple more strongly
    params.coupling_constants[ParticleType::Neutron] =
        coupling_constant_base * 0.8;  // Neutrons couple more weakly
    params.coupling_constants[ParticleType::Photon] =
        coupling_constant_base * 1.5;  // Photons have different coupling

    params.potential_coefficient = 0.5;

    // Feature size impacts lattice spacing parameter
    params.lattice_spacing = feature_size_nm / 100.0;  // Convert to appropriate scale

    // Simulation parameters
    params.time_step = 1.0e-18;  // fs time scale
    params.dimensions = 3;

    return params;
}

bool shouldApplyQuantumCorrections(double temperature, double feature_size,
                                   double radiation_intensity,
                                   const QuantumCorrectionConfig& config)
{
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

DefectDistribution applyQuantumCorrectionsToSimulation(const DefectDistribution& defects,
                                                       const CrystalLattice& crystal,
                                                       double temperature, double feature_size_nm,
                                                       double radiation_intensity,
                                                       const QuantumCorrectionConfig& config)
{
    // Check if we should apply quantum corrections
    if (!shouldApplyQuantumCorrections(temperature, feature_size_nm, radiation_intensity, config)) {
        return defects;  // Return original defects without quantum corrections
    }

    // Create QFT parameters based on material properties
    QFTParameters qft_params = createQFTParameters(crystal, feature_size_nm);

    // Apply quantum field corrections
    DefectDistribution corrected_defects =
        applyQuantumFieldCorrections(defects, crystal, qft_params, temperature);

    return corrected_defects;
}

double calculateQuantumEnhancementFactor(double temperature, double feature_size)
{
    // Base enhancement (no enhancement = 1.0)
    double enhancement = 1.0;

    // Temperature effect: More pronounced at low temperatures (but conservative)
    if (temperature < 150.0) {
        // Use proper Arrhenius behavior instead of exponential scaling
        double activation_energy = 0.05;  // eV (small activation energy for quantum effects)
        double kB = 8.617333262e-5;       // eV/K
        double reference_temp = 150.0;    // Reference temperature

        // Arrhenius temperature dependence (conservative)
        double temp_factor =
            std::exp(activation_energy / kB * (1.0 / reference_temp - 1.0 / temperature));

        // Cap the maximum enhancement from temperature alone to 2% (more conservative)
        enhancement *= (1.0 + std::min(0.02, (temp_factor - 1.0) * 0.02));
    }

    // Size effect: More pronounced at small feature sizes (quantum confinement)
    if (feature_size < 50.0) {
        // Quantum confinement energy scaling
        double confinement_factor = 50.0 / std::max(feature_size, 5.0);  // Avoid division by zero

        // Physical scaling based on 1/L² quantum confinement
        double size_enhancement = std::pow(confinement_factor, 0.5);  // Square root scaling

        // Cap the maximum enhancement from size alone to 3% (conservative)
        enhancement *= (1.0 + std::min(0.03, (size_enhancement - 1.0) * 0.03));
    }

    // Ensure total enhancement doesn't exceed 5%
    enhancement = std::min(enhancement, 1.05);

    return enhancement;
}

// Implementation of functions defined in quantum_field_theory.hpp
// These would normally be in quantum_field_theory.cpp, but for testing we'll define them here

DefectDistribution applyQuantumFieldCorrections(const DefectDistribution& defects,
                                                const CrystalLattice& crystal,
                                                const QFTParameters& qft_params, double temperature)
{
    // Start with a copy of the original defects
    DefectDistribution corrected_defects = defects;

    // Get the mass for proton (default particle)
    double mass = qft_params.getMass(ParticleType::Proton);

    // Calculate quantum corrections
    double tunneling_probability = calculateQuantumTunnelingProbability(
        crystal.barrier_height, mass, qft_params.hbar, temperature);

    // Apply Klein-Gordon correction factor
    double kg_correction =
        solveKleinGordonEquation(qft_params.hbar, mass, qft_params.potential_coefficient,
                                 qft_params.getCouplingConstant(ParticleType::Proton),
                                 qft_params.lattice_spacing, qft_params.time_step);

    // Calculate zero-point energy contribution
    double zpe_contribution = calculateZeroPointEnergyContribution(
        qft_params.hbar, mass, crystal.lattice_constant, temperature);

    // Process each particle type
    std::vector<ParticleType> particle_types = {ParticleType::Proton, ParticleType::Electron,
                                                ParticleType::Neutron, ParticleType::Photon};

    for (const auto& particle_type : particle_types) {
        // Skip if no defects for this particle type
        if (corrected_defects.interstitials.find(particle_type) ==
                corrected_defects.interstitials.end() &&
            corrected_defects.vacancies.find(particle_type) == corrected_defects.vacancies.end() &&
            corrected_defects.clusters.find(particle_type) == corrected_defects.clusters.end()) {
            continue;
        }

        // Apply corrections to interstitials
        auto it_interstitials = corrected_defects.interstitials.find(particle_type);
        if (it_interstitials != corrected_defects.interstitials.end()) {
            for (auto& value : it_interstitials->second) {
                // Interstitials are strongly affected by tunneling
                value *= (1.0 + 1.2 * tunneling_probability + 0.8 * kg_correction);
                // Add zero-point energy contribution
                value += zpe_contribution * value * 0.008;
            }
        }

        // Apply corrections to vacancies
        auto it_vacancies = corrected_defects.vacancies.find(particle_type);
        if (it_vacancies != corrected_defects.vacancies.end()) {
            for (auto& value : it_vacancies->second) {
                // Vacancies are less affected by tunneling
                value *= (1.0 + 0.4 * tunneling_probability + 0.6 * kg_correction);
                // Add zero-point energy contribution
                value += zpe_contribution * value * 0.008;
            }
        }

        // Apply corrections to clusters
        auto it_clusters = corrected_defects.clusters.find(particle_type);
        if (it_clusters != corrected_defects.clusters.end()) {
            for (auto& value : it_clusters->second) {
                // Complex defects show intermediate behavior
                value *= (1.0 + 0.8 * tunneling_probability + 0.8 * kg_correction);
                // Add zero-point energy contribution
                value += zpe_contribution * value * 0.008;
            }
        }
    }

    // Log the correction factors
    std::cout << "Applied quantum corrections with factors: " << std::endl;
    std::cout << "  - Tunneling probability: " << tunneling_probability << std::endl;
    std::cout << "  - Klein-Gordon correction: " << kg_correction << std::endl;
    std::cout << "  - Zero-point energy contribution: " << zpe_contribution << std::endl;

    return corrected_defects;
}

double calculateQuantumTunnelingProbability(double barrier_height, double mass, double hbar,
                                            double temperature)
{
    // Improved numerical stability for WKB approximation
    const double barrier_width = 1.0;  // nm
    const double kb = 8.617333262e-5;  // Boltzmann constant in eV/K

    // Prevent division by zero or negative temperatures
    double safe_temp = std::max(temperature, 1.0);  // Minimum 1K to avoid div by zero

    // Calculate thermal energy with bounds check
    double thermal_energy = kb * safe_temp;

    // More numerically stable calculation with bounds checking
    // Prevent potential overflow in sqrt and exp operations

    // Safety check for barrier height
    double safe_barrier = std::max(barrier_height, 0.01);  // Minimum 0.01 eV

    // Capped exponent calculation for numerical stability
    double exponent_term = -2.0 * barrier_width * std::sqrt(2.0 * mass * safe_barrier) / hbar;
    exponent_term = std::max(-30.0, exponent_term);  // Prevent extreme underflow

    double base_probability = std::exp(exponent_term);

    // Bound base probability to physically reasonable values
    base_probability = std::min(0.1, base_probability);  // Cap at 10% max

    // Temperature correction with improved stability
    double temp_ratio = thermal_energy / (2.0 * safe_barrier);
    temp_ratio = std::min(10.0, temp_ratio);  // Prevent extreme values

    double temp_factor = std::exp(-temp_ratio);

    // Final bounded probability
    double result = base_probability * temp_factor;

    // Additional sanity check for final result
    return std::min(0.05, std::max(0.0, result));  // Keep between 0% and 5%
}

double solveKleinGordonEquation(double hbar, double mass, double potential_coeff,
                                double coupling_constant, double lattice_spacing, double time_step)
{
    // Simplified Klein-Gordon equation solution
    // In a full implementation, this would involve solving the differential equation

    // Simplified model: correction factor based on quantum parameters
    // Added bounds checking for numerical stability
    double safe_lattice_spacing =
        std::max(lattice_spacing, 0.001);        // Avoid division by very small values
    double safe_mass = std::max(mass, 1.0e-32);  // Avoid division by very small mass

    // Physics-based approximation with improved stability
    double dispersion_term =
        (hbar * hbar) / (2.0 * safe_mass * safe_lattice_spacing * safe_lattice_spacing);
    dispersion_term = std::min(0.1, dispersion_term);  // Avoid extreme values

    double interaction_term = coupling_constant * potential_coeff;
    interaction_term = std::min(0.1, interaction_term);  // Avoid extreme values

    // Calculate stability parameter (CFL condition)
    double stability = dispersion_term * time_step / (safe_lattice_spacing * safe_lattice_spacing);

    // Calculate correction factor
    // More conservative model based on physics principles
    double correction_factor = stability * (1.0 + interaction_term);

    // Bound correction to physically reasonable values
    return std::min(0.05, std::max(0.001, correction_factor));
}

double calculateZeroPointEnergyContribution(double hbar, double mass, double lattice_constant,
                                            double temperature)
{
    // Zero-point energy calculation (E = hbar*omega/2)
    // More numerically stable implementation with bounds checking

    // Physics constants
    const double kb = 8.617333262e-5;  // Boltzmann constant in eV/K

    // Bound parameters for numerical stability
    double safe_temp = std::max(temperature, 1.0);          // Minimum 1K
    double safe_mass = std::max(mass, 1.0e-32);             // Avoid division by very small mass
    double safe_lattice = std::max(lattice_constant, 0.1);  // Avoid very small lattice constants

    // Calculate harmonic frequency (simplified model)
    // For silicon-like materials, typical phonon frequency ~10-20 THz
    double spring_constant =
        5.0 / (safe_lattice * safe_lattice);                // Simple model for spring constant
    double omega = std::sqrt(spring_constant / safe_mass);  // Harmonic oscillator frequency

    // Bound omega to reasonable range (avoid instability)
    omega = std::min(1.0e15, std::max(1.0e12, omega));

    // Calculate zero-point energy
    double zpe = 0.5 * hbar * omega;  // Zero-point energy in eV

    // Temperature suppression factor (ZPE effects decrease with temperature)
    double thermal_energy = kb * safe_temp;
    double suppression_factor = 1.0 / (1.0 + thermal_energy / (hbar * omega));

    // Return bounded result
    return std::min(0.05, zpe * suppression_factor);  // Cap at 0.05 eV
}

}  // namespace physics
}  // namespace rad_ml
