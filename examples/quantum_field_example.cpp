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

    // Initialize a quantum field for Klein-Gordon equation
    std::vector<int> grid_dimensions = {32, 32, 32};
    QuantumField<3> scalar_field(grid_dimensions, qft_params.lattice_spacing, particle_type);
    scalar_field.initializeGaussian(0.0, 0.1);

    // Record initial energy for conservation check
    double initial_energy = scalar_field.calculateTotalEnergy(qft_params);
    std::cout << "\nInitial field energy: " << initial_energy << std::endl;

    // Create a Klein-Gordon equation solver
    KleinGordonEquation kg_equation(qft_params, particle_type);

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

    // Check energy conservation
    double final_energy = scalar_field.calculateTotalEnergy(qft_params);
    std::cout << "Energy conservation check: "
              << (std::abs(final_energy - initial_energy) /
                  (initial_energy > 0 ? initial_energy : 1.0)) *
                     100.0
              << "% change" << std::endl;

    // Initialize quantum fields for electromagnetic simulation with photon particle type
    QuantumField<3> electric_field(grid_dimensions, qft_params.lattice_spacing,
                                   ParticleType::Photon);
    QuantumField<3> magnetic_field(grid_dimensions, qft_params.lattice_spacing,
                                   ParticleType::Photon);

    // Initialize with a plane wave
    electric_field.initializeCoherentState(1.0, 0.0);
    magnetic_field.initializeCoherentState(1.0, 1.57);  // π/2 phase shift

    // Record initial EM field energy
    double initial_em_energy = electric_field.calculateTotalEnergy(qft_params) +
                               magnetic_field.calculateTotalEnergy(qft_params);

    // Create Maxwell equations solver
    MaxwellEquations maxwell_equations(qft_params);

    // Evolve the electromagnetic field for 100 steps
    std::cout << "\nEvolving electromagnetic field..." << std::endl;
    for (int step = 0; step < 100; step++) {
        maxwell_equations.evolveField(electric_field, magnetic_field);

        // Calculate and print field correlation every 20 steps
        if (step % 20 == 0) {
            auto correlation = electric_field.calculateCorrelationFunction(10);
            std::cout << "Step " << step << ": Correlation at distance 1 = " << correlation(1, 0)
                      << std::endl;
        }
    }

    // Check EM field energy conservation
    double final_em_energy = electric_field.calculateTotalEnergy(qft_params) +
                             magnetic_field.calculateTotalEnergy(qft_params);
    std::cout << "EM field energy conservation check: "
              << (std::abs(final_em_energy - initial_em_energy) /
                  (initial_em_energy > 0 ? initial_em_energy : 1.0)) *
                     100.0
              << "% change" << std::endl;

    return 0;
}
