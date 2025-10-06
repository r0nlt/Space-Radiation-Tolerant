/**
 * Advanced Quantum Models Validation Test
 *
 * This test validates the three theoretical extensions to quantum field theory:
 * 1. Dirac equation for relativistic electron effects in semiconductors
 * 2. Bethe-Salpeter equation for bound state formation in defect clusters
 * 3. Green's function methods for sophisticated propagation modeling
 */

#include <cmath>
#include <iostream>
#include <rad_ml/physics/advanced_quantum_models.hpp>
#include <vector>

using namespace rad_ml::physics;

/**
 * Test Dirac equation implementation for relativistic electron effects
 */
void testDiracEquation()
{
    std::cout << "\n=== Testing Dirac Equation Implementation ===\n";

    DiracEquationSolver dirac_solver;

    // Test case: high-energy electron in semiconductor
    Eigen::Vector3d momentum(1000.0, 0.0, 0.0);  // 1000 eV/c momentum
    double energy = 511000.0;                    // 511 keV (rest mass + kinetic energy)
    double mass = 510998.95;                     // Electron rest mass in eV/c²

    auto spinor = dirac_solver.solveDiracEquation(momentum, energy, mass);

    std::cout << "Dirac spinor components:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "ψ[" << i << "] = " << spinor(i) << "\n";
    }

    // Test relativistic cross-section calculation
    double incident_energy = 10000.0;      // 10 keV
    double scattering_angle = M_PI / 4.0;  // 45 degrees

    MaterialProperties silicon;
    silicon.band_gap = 1.12;                 // eV
    silicon.electron_effective_mass = 0.26;  // m*/m₀
    silicon.dielectric_constant = 11.7;

    double cross_section =
        dirac_solver.calculateRelativisticCrossSection(incident_energy, scattering_angle, silicon);

    std::cout << "Relativistic scattering cross-section: " << cross_section << " m²/sr\n";

    // Test relativistic displacement energy correction
    double non_rel_energy = 25.0;      // eV (typical for silicon)
    double electron_energy = 10000.0;  // 10 keV

    double relativistic_energy =
        dirac_solver.calculateRelativisticDisplacementEnergy(non_rel_energy, electron_energy);

    std::cout << "Non-relativistic displacement energy: " << non_rel_energy << " eV\n";
    std::cout << "Relativistic displacement energy: " << relativistic_energy << " eV\n";
    std::cout << "Relativistic correction factor: " << relativistic_energy / non_rel_energy << "\n";
}

/**
 * Test Bethe-Salpeter equation for bound state formation
 */
void testBetheSalpeterEquation()
{
    std::cout << "\n=== Testing Bethe-Salpeter Equation Implementation ===\n";

    BetheSalpeterSolver bse_solver;

    // Create a simple 2x2 interaction kernel for electron-hole pair
    BetheSalpeterSolver::BSEMatrix kernel(2, 2);
    kernel << std::complex<double>(-1.0, 0.0), std::complex<double>(-0.5, 0.0),
        std::complex<double>(-0.5, 0.0), std::complex<double>(-1.0, 0.0);

    double binding_energy = 0.1;  // 0.1 eV binding energy guess
    Eigen::Vector3d total_momentum(0.0, 0.0, 0.0);

    auto wavefunction = bse_solver.solveBetheSalpeter(kernel, binding_energy, total_momentum);

    std::cout << "BSE bound state wavefunction:\n";
    for (int i = 0; i < wavefunction.size(); ++i) {
        std::cout << "Γ[" << i << "] = " << wavefunction(i) << "\n";
    }

    // Test exciton binding calculation
    CrystalLattice silicon_lattice(CrystalLattice::Type::DIAMOND, 0.543, 1.0);
    Eigen::Vector3d electron_pos(0.0, 0.0, 0.0);
    Eigen::Vector3d hole_pos(0.2715, 0.2715, 0.2715);  // Half lattice constant

    auto [exciton_binding, exciton_wavefunction] = bse_solver.calculateExcitonBinding(
        electron_pos, hole_pos, silicon_lattice, 300.0);  // Room temperature

    std::cout << "Exciton binding energy: " << exciton_binding << " eV\n";
    std::cout << "Exciton wavefunction norm: " << exciton_wavefunction.norm() << "\n";

    // Test defect cluster formation
    std::vector<Eigen::Vector3d> defect_positions = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                                     Eigen::Vector3d(0.3, 0.0, 0.0),
                                                     Eigen::Vector3d(0.0, 0.3, 0.0)};

    std::vector<DefectType> defect_types = {DefectType::VACANCY, DefectType::INTERSTITIAL,
                                            DefectType::VACANCY};

    double cluster_binding = bse_solver.calculateClusterBindingEnergy(
        defect_positions, defect_types, silicon_lattice, QFTParameters{});

    std::cout << "Defect cluster binding energy: " << cluster_binding << " eV\n";
}

/**
 * Test Green's function methods for propagation modeling
 */
void testGreensFunctionPropagation()
{
    std::cout << "\n=== Testing Green's Function Propagation ===\n";

    GreensFunctionPropagator greens_propagator;

    // Test Helmholtz equation solution
    std::vector<Eigen::Vector3d> source_positions = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                                     Eigen::Vector3d(1.0, 0.0, 0.0)};

    Eigen::VectorXcd source_amplitudes(2);
    source_amplitudes << std::complex<double>(1.0, 0.0), std::complex<double>(0.5, 0.0);

    double wave_number = 2.0 * M_PI / 0.5;  // k = 2π/λ for λ = 0.5 nm

    std::vector<Eigen::Vector3d> field_points = {Eigen::Vector3d(0.5, 0.5, 0.0),
                                                 Eigen::Vector3d(1.5, 0.5, 0.0),
                                                 Eigen::Vector3d(2.0, 1.0, 0.0)};

    auto field_values = greens_propagator.solveHelmholtzEquation(
        source_positions, source_amplitudes, wave_number, field_points);

    std::cout << "Helmholtz equation solutions at field points:\n";
    for (int i = 0; i < field_values.size(); ++i) {
        std::cout << "Point " << i << ": " << field_values(i) << "\n";
    }

    // Test radiation field calculation
    std::vector<Eigen::Vector3d> particle_trajectory = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                                        Eigen::Vector3d(0.1, 0.0, 0.0),
                                                        Eigen::Vector3d(0.2, 0.0, 0.0)};

    Eigen::Vector3d particle_velocity(1.0, 0.0, 0.0);  // Moving along x-axis

    std::vector<Eigen::Vector3d> observation_points = {Eigen::Vector3d(0.5, 1.0, 0.0),
                                                       Eigen::Vector3d(1.0, 1.0, 0.0)};

    double frequency = 1e15;  // 1 PHz

    auto radiation_field = greens_propagator.calculateRadiationField(
        particle_trajectory, particle_velocity, observation_points, frequency);

    std::cout << "Radiation field at observation points:\n";
    for (int i = 0; i < radiation_field.size(); ++i) {
        std::cout << "Point " << i << ": " << radiation_field(i) << "\n";
    }

    // Test defect propagation
    DefectDistribution initial_defects;
    initial_defects.interstitials[ParticleType::Electron] = {1.0, 0.5, 0.2};
    initial_defects.interstitials[ParticleType::Proton] = {0.8, 0.3};

    GreensFunctionPropagator::GreensMatrix propagation_kernel(3, 3);
    propagation_kernel << 0.9, 0.1, 0.0, 0.1, 0.8, 0.1, 0.0, 0.1, 0.9;

    std::vector<double> time_points = {1e-15, 2e-15, 5e-15};

    auto propagated_defects =
        greens_propagator.propagateDefects(initial_defects, propagation_kernel, time_points);

    std::cout << "Defect propagation over time:\n";
    for (size_t t = 0; t < propagated_defects.size(); ++t) {
        std::cout << "Time " << time_points[t] << " s:\n";
        for (const auto& [particle_type, interstitials] : propagated_defects[t].interstitials) {
            std::cout << "  " << static_cast<int>(particle_type) << ": ";
            for (double val : interstitials) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }
}

/**
 * Test integration of all three theoretical approaches
 */
void testAdvancedQuantumIntegration()
{
    std::cout << "\n=== Testing Advanced Quantum Integration ===\n";

    AdvancedQuantumRadiationModel advanced_model;

    // Create test scenario: high-energy electron hitting silicon
    Particle incident_electron = Particle::createElectron();
    // Simulate high-energy electron (10 keV kinetic energy)
    incident_electron = Particle(ParticleType::Electron, incident_electron.mass(), -1.0, 0.5);

    MaterialProperties silicon;
    silicon.band_gap = 1.12;  // eV
    silicon.electron_effective_mass = 0.26;
    silicon.hole_effective_mass = 0.37;
    silicon.dielectric_constant = 11.7;
    silicon.phonon_frequency = 15.3;               // THz for silicon
    silicon.deformation_potential = 9.0;           // eV
    silicon.lattice_thermal_conductivity = 148.0;  // W/m·K

    CrystalLattice silicon_lattice = CrystalLatticeFactory::Diamond(0.543, 1.0);

    // Calculate comprehensive radiation effects
    DefectDistribution final_defects = advanced_model.calculateAdvancedRadiationEffects(
        incident_electron, silicon, silicon_lattice, RadiationEnvironment::LEO);

    std::cout << "Final defect distribution after advanced quantum modeling:\n";
    for (const auto& [particle_type, interstitials] : final_defects.interstitials) {
        std::cout << "Particle type " << static_cast<int>(particle_type) << ":\n";
        for (size_t i = 0; i < interstitials.size(); ++i) {
            std::cout << "  Position " << i << ": " << interstitials[i] << "\n";
        }
    }

    // Test relativistic electron cascade
    std::vector<Particle> electron_cascade = advanced_model.simulateRelativisticElectronCascade(
        incident_electron, silicon_lattice, QFTParameters{});

    std::cout << "Relativistic electron cascade generated " << electron_cascade.size()
              << " secondary electrons\n";

    // Test defect cluster formation
    std::vector<DefectCluster> defect_clusters =
        advanced_model.calculateDefectClusterFormation(final_defects, silicon_lattice, 300.0);

    std::cout << "Formed " << defect_clusters.size() << " defect clusters:\n";
    for (const auto& cluster : defect_clusters) {
        std::cout << "  Cluster with " << cluster.defect_positions.size() << " defects, "
                  << "binding energy: " << cluster.binding_energy << " eV\n";
    }

    // Test time evolution using Green's functions
    std::vector<DefectDistribution> time_evolution = advanced_model.propagateRadiationEffects(
        final_defects, silicon_lattice, {1e-15, 1e-14, 1e-13});

    std::cout << "Time evolution completed for " << time_evolution.size() << " time steps\n";
}

/**
 * Main test function
 */
int main()
{
    std::cout << "Advanced Quantum Models Validation Test\n";
    std::cout << "=====================================\n";

    try {
        testDiracEquation();
        testBetheSalpeterEquation();
        testGreensFunctionPropagation();
        testAdvancedQuantumIntegration();

        std::cout << "\n=== All Tests Completed Successfully ===\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }
}
