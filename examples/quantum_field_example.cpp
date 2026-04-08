#include <cmath>
#include <iostream>
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>
#include <vector>

using namespace rad_ml::physics;

int main()
{
    // Set up quantum field theory parameters
    QFTParameters qft_params;
    qft_params.hbar = 6.582119569e-16;  // Reduced Planck constant in eV·s
    // Set particle-specific masses with correct physical values
    qft_params.masses[ParticleType::Proton] =
        1.67262192369e-27;  // Proton mass in kg (corrected value)
    qft_params.masses[ParticleType::Electron] = 9.1093837015e-31;  // Electron mass in kg
    qft_params.masses[ParticleType::Neutron] = 1.67492749804e-27;  // Neutron mass in kg
    qft_params.masses[ParticleType::Photon] = 1.0e-36;             // Near zero for photons
    // Set particle-specific coupling constants
    qft_params.coupling_constants[ParticleType::Proton] = 0.1;  // Dimensionless coupling constant
    qft_params.potential_coefficient = 0.5;                     // Coefficient in potential term
    qft_params.lattice_spacing = 0.1;                           // Spatial lattice spacing in nm
    qft_params.time_step = 1.0e-18;                             // Time step in seconds
    qft_params.dimensions = 3;                                  // 3D simulation

    // Default particle type for this example
    ParticleType particle_type = ParticleType::Proton;

    // Create a crystal lattice for silicon
    CrystalLattice silicon =
        CrystalLatticeFactory::FCC(5.431);  // Silicon lattice constant in Angstroms

    // Calculate displacement energy
    double displacement_energy = calculateDisplacementEnergy(silicon, qft_params, particle_type);
    std::cout << "Displacement energy: " << displacement_energy << " eV" << std::endl;

    // Simulate a displacement cascade
    double pka_energy = 1000.0;  // 1 keV primary knock-on atom
    DefectDistribution defects = simulateDisplacementCascade(silicon, pka_energy, qft_params,
                                                             displacement_energy, particle_type);

    // Apply quantum field corrections
    double temperature = 300.0;  // K
    DefectDistribution corrected_defects =
        applyQuantumFieldCorrections(defects, silicon, qft_params, temperature);

    // Calculate and print the differences
    std::cout << "Classical vs. Quantum-Corrected Defect Counts:" << std::endl;

    // Define particle types for display
    std::vector<ParticleType> display_particles = {particle_type};

    // Print interstitials
    std::cout << "Interstitials:" << std::endl;
    for (const auto& type : display_particles) {
        if (defects.interstitials.find(type) != defects.interstitials.end() &&
            corrected_defects.interstitials.find(type) != corrected_defects.interstitials.end()) {
            // Get defect count vectors for this particle type
            const auto& defect_values = defects.interstitials.at(type);
            const auto& corrected_values = corrected_defects.interstitials.at(type);

            // Print data for each region
            for (size_t i = 0; i < defect_values.size() && i < corrected_values.size(); i++) {
                std::cout << "  Region " << i << " (Particle: " << static_cast<int>(type)
                          << "): " << defect_values[i] << " vs. " << corrected_values[i] << " ("
                          << (defect_values[i] > 0
                                  ? (corrected_values[i] / defect_values[i] - 1.0) * 100.0
                                  : 0.0)
                          << "% change)" << std::endl;
            }
        }
    }

    // --- Klein-Gordon lattice simulation ---
    // Use dimensionless lattice units for a well-posed PDE simulation:
    //   dx = 1.0 (lattice units), dt = 0.1 (CFL-stable for leapfrog)
    //   m = 0.1  (dimensionless mass, gives Compton wavelength ~ 10 sites)
    QFTParameters kg_params;
    kg_params.lattice_spacing = 1.0;
    kg_params.time_step = 0.1;
    kg_params.masses[ParticleType::Proton] = 0.1;
    kg_params.hbar = 1.0;
    kg_params.dimensions = 3;

    std::vector<int> grid_dimensions = {16, 16, 16};
    QuantumField<3> scalar_field(grid_dimensions, kg_params.lattice_spacing, particle_type);
    scalar_field.initializeGaussian(0.0, 0.1);

    KleinGordonEquation kg_equation(kg_params, particle_type);

    // First evolve step initializes pi, then we can compute full Hamiltonian
    kg_equation.evolveField(scalar_field);
    double initial_H = kg_equation.computeHamiltonian(scalar_field);

    std::cout << "\n=== Klein-Gordon Field Simulation ===" << std::endl;
    std::cout << "Grid: 16x16x16, dx=1.0, dt=0.1, m=0.1" << std::endl;
    std::cout << "Initial Hamiltonian: " << initial_H << std::endl;

    const int kg_steps = 200;
    std::cout << "\nEvolving Klein-Gordon field for " << kg_steps << " steps..." << std::endl;
    for (int step = 1; step <= kg_steps; step++) {
        kg_equation.evolveField(scalar_field);

        if (step % 20 == 0) {
            double H = kg_equation.computeHamiltonian(scalar_field);
            double drift = (initial_H > 0)
                ? (H - initial_H) / initial_H * 100.0
                : 0.0;
            std::cout << "Step " << step << ": H = " << H
                      << " (drift: " << drift << "%)" << std::endl;
        }
    }

    double final_H = kg_equation.computeHamiltonian(scalar_field);
    double kg_drift = (initial_H > 0)
        ? std::abs(final_H - initial_H) / initial_H * 100.0
        : 0.0;
    std::cout << "\nKlein-Gordon energy conservation: " << kg_drift << "% drift over "
              << kg_steps << " steps" << std::endl;
    std::cout << (kg_drift < 1.0 ? "PASS" : (kg_drift < 10.0 ? "MARGINAL" : "FAIL"))
              << ": energy conservation" << std::endl;

    // --- Maxwell FDTD simulation ---
    // Same lattice units; CFL for Maxwell: dt < dx/sqrt(3) ≈ 0.577
    QFTParameters em_params;
    em_params.lattice_spacing = 1.0;
    em_params.time_step = 0.1;
    em_params.masses[ParticleType::Photon] = 1.0e-36;
    em_params.coupling_constants[ParticleType::Photon] = 0.1;
    em_params.hbar = 1.0;
    em_params.dimensions = 3;

    QuantumField<3> electric_field(grid_dimensions, em_params.lattice_spacing,
                                   ParticleType::Photon);
    QuantumField<3> magnetic_field(grid_dimensions, em_params.lattice_spacing,
                                   ParticleType::Photon);

    electric_field.initializeGaussian(0.0, 0.1);
    magnetic_field.initializeGaussian(0.0, 0.0);

    MaxwellEquations maxwell_equations(em_params);

    // First step initializes velocities, then compute full Hamiltonian
    maxwell_equations.evolveField(electric_field, magnetic_field);
    double initial_em_H = maxwell_equations.computeHamiltonian(electric_field, magnetic_field);

    const int em_steps = 200;
    std::cout << "\n=== Maxwell Electromagnetic Simulation ===" << std::endl;
    std::cout << "Grid: 16x16x16, dx=1.0, dt=0.1" << std::endl;
    std::cout << "Initial EM Hamiltonian: " << initial_em_H << std::endl;

    std::cout << "\nEvolving electromagnetic field for " << em_steps << " steps..." << std::endl;
    for (int step = 1; step <= em_steps; step++) {
        maxwell_equations.evolveField(electric_field, magnetic_field);

        if (step % 20 == 0) {
            double H = maxwell_equations.computeHamiltonian(electric_field, magnetic_field);
            double drift = (initial_em_H > 0)
                ? (H - initial_em_H) / initial_em_H * 100.0
                : 0.0;
            auto correlation = electric_field.calculateCorrelationFunction(10);
            std::cout << "Step " << step << ": EM H = " << H
                      << " (drift: " << drift << "%)"
                      << ", correlation(1) = " << correlation(1, 0) << std::endl;
        }
    }

    double final_em_H = maxwell_equations.computeHamiltonian(electric_field, magnetic_field);
    double em_drift = (initial_em_H > 0)
        ? std::abs(final_em_H - initial_em_H) / initial_em_H * 100.0
        : 0.0;
    std::cout << "\nMaxwell energy conservation: " << em_drift << "% drift over "
              << em_steps << " steps" << std::endl;
    std::cout << (em_drift < 1.0 ? "PASS" : (em_drift < 10.0 ? "MARGINAL" : "FAIL"))
              << ": energy conservation" << std::endl;

    return 0;
}
