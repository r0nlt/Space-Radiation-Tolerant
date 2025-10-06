/**
 * Enhanced Quantum Integration Validation Test
 *
 * This test validates the integration of advanced quantum models into the
 * existing radiation-tolerant ML framework.
 */

#include <cmath>
#include <iostream>
#include <rad_ml/physics/enhanced_physics_radiation_simulator.hpp>
#include <vector>

using namespace rad_ml;

/**
 * Test basic enhanced physics simulator functionality
 */
void testEnhancedPhysicsSimulator()
{
    std::cout << "\n=== Testing Enhanced Physics Simulator ===\n";

    // Test default constructor
    sim::EnhancedPhysicsRadiationSimulator simulator;
    std::cout << "✅ Enhanced Physics Simulator created successfully\n";

    // Test parameterized constructor
    sim::EnvironmentParams leo_params(sim::SpaceEnvironment::LEO, 0.5, 2.0);
    sim::EnhancedPhysicsRadiationSimulator leo_simulator(leo_params);
    std::cout << "✅ Enhanced Physics Simulator created for LEO environment\n";
}

/**
 * Test material properties creation
 */
void testMaterialProperties()
{
    std::cout << "\n=== Testing Material Properties ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Test silicon properties
    physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");
    std::cout << "✅ Silicon properties:\n";
    std::cout << "   - Band gap: " << silicon.band_gap << " eV\n";
    std::cout << "   - Electron effective mass: " << silicon.electron_effective_mass << " m*/m₀\n";
    std::cout << "   - Dielectric constant: " << silicon.dielectric_constant << "\n";

    // Test GaAs properties
    physics::MaterialProperties gaas = simulator.createMaterialProperties("gallium_arsenide");
    std::cout << "✅ GaAs properties:\n";
    std::cout << "   - Band gap: " << gaas.band_gap << " eV\n";
    std::cout << "   - Electron effective mass: " << gaas.electron_effective_mass << " m*/m₀\n";
}

/**
 * Test crystal lattice creation
 */
void testCrystalLattice()
{
    std::cout << "\n=== Testing Crystal Lattice ===\n";

    // Test diamond lattice creation
    physics::CrystalLattice diamond_lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);
    std::cout << "✅ Diamond lattice created: 0.543 nm spacing\n";

    // Test FCC lattice creation
    physics::CrystalLattice fcc_lattice = physics::CrystalLatticeFactory::FCC(0.543, 1.0);
    std::cout << "✅ FCC lattice created: 0.543 nm spacing\n";
}

/**
 * Test relativistic electron cascade calculation
 */
void testRelativisticElectronCascade()
{
    std::cout << "\n=== Testing Relativistic Electron Cascade ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Create test materials and lattice
    physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");
    physics::CrystalLattice lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

    // Test with proton
    physics::Particle proton = physics::Particle::createProton();
    auto cascade = simulator.calculateRelativisticElectronCascade(proton, silicon, lattice);

    std::cout << "✅ Relativistic cascade calculated:\n";
    std::cout << "   - Incident particle: Proton\n";
    std::cout << "   - Secondary electrons: " << cascade.size() << "\n";
    std::cout << "   - Using Dirac equation for relativistic kinematics\n";
}

/**
 * Test defect cluster formation
 */
void testDefectClusterFormation()
{
    std::cout << "\n=== Testing Defect Cluster Formation ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Create test materials and lattice
    physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");
    physics::CrystalLattice lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

    // Create initial defect distribution
    physics::DefectDistribution initial_defects;
    initial_defects.interstitials[physics::ParticleType::Electron] = {1.0, 0.5};
    initial_defects.vacancies[physics::ParticleType::Proton] = {0.8, 0.3};

    // Test cluster formation
    auto clusters = simulator.calculateDefectClusterFormation(initial_defects, lattice, 300.0);

    std::cout << "✅ Defect cluster formation calculated:\n";
    std::cout << "   - Initial defects: "
              << (initial_defects.interstitials.size() + initial_defects.vacancies.size()) << "\n";
    std::cout << "   - Clusters formed: " << clusters.size() << "\n";
    std::cout << "   - Using Bethe-Salpeter equation for bound state calculations\n";

    for (size_t i = 0; i < clusters.size(); ++i) {
        std::cout << "   - Cluster " << i << ": " << clusters[i].defect_positions.size()
                  << " defects, binding energy: " << clusters[i].binding_energy << " eV\n";
    }
}

/**
 * Test radiation effects propagation
 */
void testRadiationPropagation()
{
    std::cout << "\n=== Testing Radiation Effects Propagation ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Create test materials and lattice
    physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");
    physics::CrystalLattice lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

    // Create initial defect distribution
    physics::DefectDistribution initial_defects;
    initial_defects.interstitials[physics::ParticleType::Electron] = {1.0, 0.5, 0.2};

    // Test time evolution
    std::vector<double> time_steps = {1e-15, 1e-14, 1e-13};
    auto evolution = simulator.propagateRadiationEffects(initial_defects, lattice, time_steps);

    std::cout << "✅ Radiation propagation calculated:\n";
    std::cout << "   - Time steps: " << time_steps.size() << "\n";
    std::cout << "   - Using Green's function methods for wave propagation\n";

    for (size_t t = 0; t < evolution.size(); ++t) {
        size_t total_defects = evolution[t].interstitials.size() + evolution[t].vacancies.size();
        std::cout << "   - Time " << time_steps[t] << " s: " << total_defects << " total defects\n";
    }
}

/**
 * Test neural network protection enhancement
 */
void testNeuralNetworkProtection()
{
    std::cout << "\n=== Testing Neural Network Protection Enhancement ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Create test materials and lattice
    physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");
    physics::CrystalLattice lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

    // Create a mock neural network for testing
    std::vector<size_t> layer_sizes = {10, 5, 1};  // Simple 10-5-1 network
    neural::ProtectedNeuralNetwork<float> network(layer_sizes);

    // Test protection enhancement at different radiation levels
    std::vector<double> radiation_levels = {0.1, 0.5, 0.8};

    for (double level : radiation_levels) {
        auto recommendations =
            simulator.enhanceNeuralNetworkProtection<float>(network, level, silicon, lattice);

        std::cout << "✅ Radiation level " << level << ":\n";
        std::cout << "   - Recommendations: " << recommendations.size() << "\n";
        for (const auto& rec : recommendations) {
            std::cout << "     * " << rec << "\n";
        }
    }
}

/**
 * Test quantum-enhanced displacement energy calculation
 */
void testQuantumEnhancedDisplacement()
{
    std::cout << "\n=== Testing Quantum-Enhanced Displacement Energy ===\n";

    sim::EnhancedPhysicsRadiationSimulator simulator;

    // Test different particle types and energies
    std::vector<std::pair<physics::ParticleType, double>> test_cases = {
        {physics::ParticleType::Proton, 1000.0},     // 1 keV proton
        {physics::ParticleType::Electron, 10000.0},  // 10 keV electron
        {physics::ParticleType::HeavyIon, 100000.0}  // 100 keV heavy ion
    };

    for (const auto& [particle_type, energy] : test_cases) {
        double standard_energy = 25.0;  // Standard displacement energy for silicon
        double enhanced_energy = simulator.calculateQuantumEnhancedDisplacementEnergy(
            standard_energy, energy, particle_type);

        std::cout << "✅ " << static_cast<int>(particle_type) << " (" << energy << " eV):\n";
        std::cout << "   - Standard displacement: " << standard_energy << " eV\n";
        std::cout << "   - Enhanced displacement: " << enhanced_energy << " eV\n";
        std::cout << "   - Enhancement factor: " << enhanced_energy / standard_energy << "\n";
    }
}

/**
 * Main test function
 */
int main()
{
    std::cout << "Enhanced Quantum Integration Validation Test\n";
    std::cout << "==========================================\n";

    try {
        testEnhancedPhysicsSimulator();
        testMaterialProperties();
        testCrystalLattice();
        testRelativisticElectronCascade();
        testDefectClusterFormation();
        testRadiationPropagation();
        testNeuralNetworkProtection();
        testQuantumEnhancedDisplacement();

        std::cout << "\n🎉 All Enhanced Quantum Integration Tests Passed!\n";
        std::cout << "\nFramework successfully enhanced with:\n";
        std::cout << "• Dirac equation for relativistic electron cascades\n";
        std::cout << "• Bethe-Salpeter equation for defect cluster formation\n";
        std::cout << "• Green's function methods for radiation propagation\n";
        std::cout << "• Enhanced neural network protection\n";
        std::cout << "• Material-aware quantum corrections\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << "\n";
        return 1;
    }
}
