#include <cmath>
#include <iomanip>
#include <iostream>
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>
#include <vector>

using namespace rad_ml::physics;

int main()
{
    std::cout << "Multi-Particle Simulation Example" << std::endl;
    std::cout << "=================================" << std::endl;

    // Initialize QFT parameters with multiple particle types
    QFTParameters qft_params;
    qft_params.hbar = 6.582119569e-16;  // reduced Planck constant (eV·s)

    // Set parameters for different particle types with correct physical values
    qft_params.masses[ParticleType::Proton] = 1.67262192369e-27;   // Proton mass (kg)
    qft_params.masses[ParticleType::Electron] = 9.1093837015e-31;  // Electron mass (kg)
    qft_params.masses[ParticleType::Neutron] = 1.67492749804e-27;  // Neutron mass (kg)
    qft_params.masses[ParticleType::Photon] = 1.0e-36;  // Photon "mass" (effectively zero)

    qft_params.coupling_constants[ParticleType::Proton] = 0.15;
    qft_params.coupling_constants[ParticleType::Electron] = 0.20;
    qft_params.coupling_constants[ParticleType::Neutron] = 0.10;
    qft_params.coupling_constants[ParticleType::Photon] = 0.05;

    qft_params.potential_coefficient = 0.5;
    qft_params.lattice_spacing = 0.1;  // nm
    qft_params.time_step = 1.0e-18;    // seconds
    qft_params.dimensions = 3;

    // Create crystal lattice
    CrystalLattice silicon = CrystalLatticeFactory::FCC(5.431);  // Silicon

    // Define a vector of particle types to simulate
    std::vector<ParticleType> particles = {ParticleType::Proton, ParticleType::Electron,
                                           ParticleType::Neutron};

    // Calculate displacement energy for each particle type
    std::cout << "\nDisplacement Energies by Particle Type:" << std::endl;
    for (const auto& particle : particles) {
        double energy = calculateDisplacementEnergy(silicon, qft_params, particle);
        std::cout << "  - Particle Type " << static_cast<int>(particle) << ": " << std::fixed
                  << std::setprecision(2) << energy << " eV" << std::endl;
    }

    // Simulate displacement cascades for different particles
    double pka_energy = 1000.0;  // 1 keV
    double temperature = 300.0;  // Kelvin

    std::cout << "\nSimulating Displacement Cascades:" << std::endl;
    for (const auto& particle : particles) {
        double displacement_energy = calculateDisplacementEnergy(silicon, qft_params, particle);

        // Simulate cascade
        DefectDistribution defects = simulateDisplacementCascade(silicon, pka_energy, qft_params,
                                                                 displacement_energy, particle);

        // Apply quantum corrections for this specific particle type
        DefectDistribution corrected_defects =
            applyQuantumFieldCorrections(defects, silicon, qft_params, temperature, {particle});

        // Count defects
        double total_defects = 0.0;
        double total_corrected = 0.0;

        // Count interstitials
        if (defects.interstitials.find(particle) != defects.interstitials.end()) {
            for (const auto& val : defects.interstitials.at(particle)) {
                total_defects += val;
            }
        }

        // Count vacancies
        if (defects.vacancies.find(particle) != defects.vacancies.end()) {
            for (const auto& val : defects.vacancies.at(particle)) {
                total_defects += val;
            }
        }

        // Count corrected interstitials
        if (corrected_defects.interstitials.find(particle) !=
            corrected_defects.interstitials.end()) {
            for (const auto& val : corrected_defects.interstitials.at(particle)) {
                total_corrected += val;
            }
        }

        // Count corrected vacancies
        if (corrected_defects.vacancies.find(particle) != corrected_defects.vacancies.end()) {
            for (const auto& val : corrected_defects.vacancies.at(particle)) {
                total_corrected += val;
            }
        }

        // Print results with safe division
        std::cout << "  - Particle Type " << static_cast<int>(particle) << ":" << std::endl;
        std::cout << "    * Total Defects (Classical): " << total_defects << std::endl;
        std::cout << "    * Total Defects (Quantum): " << total_corrected << std::endl;
        std::cout << "    * Quantum Enhancement: "
                  << (total_defects > 0 ? ((total_corrected / total_defects) - 1.0) * 100.0 : 0.0)
                  << "%" << std::endl;
    }

    // Demonstrate multi-particle interaction
    std::cout << "\nSimulating Multi-Particle Interactions:" << std::endl;

    // Grid dimensions for quantum fields
    std::vector<int> grid_dimensions = {16, 16, 16};

    // Create quantum fields for different particles
    QuantumField<3> proton_field(grid_dimensions, qft_params.lattice_spacing, ParticleType::Proton);
    QuantumField<3> electron_field(grid_dimensions, qft_params.lattice_spacing,
                                   ParticleType::Electron);

    // Initialize fields with different patterns
    proton_field.initializeGaussian(0.0, 0.2);
    electron_field.initializeCoherentState(0.5, 0.0);

    // Record initial energies for conservation check
    double initial_proton_energy =
        proton_field.calculateTotalEnergy(qft_params, ParticleType::Proton);
    double initial_electron_energy =
        electron_field.calculateTotalEnergy(qft_params, ParticleType::Electron);

    std::cout << "  Initial proton field energy: " << initial_proton_energy << std::endl;
    std::cout << "  Initial electron field energy: " << initial_electron_energy << std::endl;

    // Create equation solvers
    KleinGordonEquation proton_kg(qft_params, ParticleType::Proton);
    KleinGordonEquation electron_kg(qft_params, ParticleType::Electron);

    // Evolve both fields for 10 steps
    std::cout << "  Evolving multi-particle quantum fields..." << std::endl;
    for (int step = 0; step < 10; step++) {
        // Evolve each field
        proton_kg.evolveField(proton_field);
        electron_kg.evolveField(electron_field);

        // Calculate total energies
        double proton_energy = proton_field.calculateTotalEnergy(qft_params, ParticleType::Proton);
        double electron_energy =
            electron_field.calculateTotalEnergy(qft_params, ParticleType::Electron);

        // Print energies every few steps
        if (step % 2 == 0) {
            std::cout << "  - Step " << step << " | Proton field energy: " << proton_energy
                      << " | Electron field energy: " << electron_energy << std::endl;
        }
    }

    // Check energy conservation
    double final_proton_energy =
        proton_field.calculateTotalEnergy(qft_params, ParticleType::Proton);
    double final_electron_energy =
        electron_field.calculateTotalEnergy(qft_params, ParticleType::Electron);

    std::cout << "\nEnergy Conservation Analysis:" << std::endl;
    std::cout << "  Proton field energy change: "
              << (std::abs(final_proton_energy - initial_proton_energy) /
                  (initial_proton_energy > 0 ? initial_proton_energy : 1.0)) *
                     100.0
              << "%" << std::endl;
    std::cout << "  Electron field energy change: "
              << (std::abs(final_electron_energy - initial_electron_energy) /
                  (initial_electron_energy > 0 ? initial_electron_energy : 1.0)) *
                     100.0
              << "%" << std::endl;

    std::cout << "\nMulti-particle simulation completed." << std::endl;
    return 0;
}
