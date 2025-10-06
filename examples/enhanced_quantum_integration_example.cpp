/**
 * Enhanced Quantum Integration Example
 *
 * This example demonstrates how to use the enhanced physics radiation simulator
 * that integrates advanced quantum models (Dirac equation, Bethe-Salpeter equation,
 * and Green's functions) into the existing radiation-tolerant ML framework.
 */

#include <iostream>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/physics/enhanced_physics_radiation_simulator.hpp>
#include <rad_ml/sim/physics_radiation_simulator.hpp>
#include <vector>

using namespace rad_ml;

int main()
{
    std::cout << "Enhanced Quantum Integration Example\n";
    std::cout << "==================================\n";

    try {
        // 1. Initialize enhanced physics simulator with advanced quantum models
        sim::EnhancedPhysicsRadiationSimulator enhanced_simulator;

        std::cout << "\n1. Enhanced Physics Simulator initialized with:\n";
        std::cout << "   - Dirac equation for relativistic electron effects\n";
        std::cout << "   - Bethe-Salpeter equation for defect bound states\n";
        std::cout << "   - Green's function methods for propagation modeling\n";

        // 2. Create material properties for silicon semiconductor
        physics::MaterialProperties silicon_properties =
            enhanced_simulator.createMaterialProperties("silicon");

        std::cout << "\n2. Silicon material properties:\n";
        std::cout << "   - Band gap: " << silicon_properties.band_gap << " eV\n";
        std::cout << "   - Electron effective mass: " << silicon_properties.electron_effective_mass
                  << " m*/m₀\n";
        std::cout << "   - Dielectric constant: " << silicon_properties.dielectric_constant << "\n";

        // 3. Create crystal lattice structure
        physics::CrystalLattice silicon_lattice =
            physics::CrystalLatticeFactory::Diamond(0.543, 1.0);  // 0.543 nm lattice constant

        std::cout << "\n3. Silicon crystal lattice: Diamond structure, 0.543 nm lattice constant\n";

        // 4. Calculate relativistic electron cascade for high-energy proton
        Particle incident_proton = Particle::createProton();
        std::vector<Particle> electron_cascade =
            enhanced_simulator.calculateRelativisticElectronCascade(
                incident_proton, silicon_properties, silicon_lattice);

        std::cout << "\n4. Relativistic electron cascade analysis:\n";
        std::cout << "   - Incident particle: Proton\n";
        std::cout << "   - Secondary electrons generated: " << electron_cascade.size() << "\n";
        std::cout << "   - Using Dirac equation for relativistic kinematics\n";

        // 5. Calculate defect cluster formation using Bethe-Salpeter equation
        physics::DefectDistribution initial_defects;
        initial_defects.interstitials[ParticleType::ELECTRON] = {1.0, 0.5};
        initial_defects.vacancies[ParticleType::PROTON] = {0.8, 0.3};

        std::vector<physics::DefectCluster> defect_clusters =
            enhanced_simulator.calculateDefectClusterFormation(initial_defects, silicon_lattice,
                                                               300.0);  // Room temperature

        std::cout << "\n5. Defect cluster formation analysis:\n";
        std::cout << "   - Initial defects: "
                  << initial_defects.interstitials.size() + initial_defects.vacancies.size()
                  << "\n";
        std::cout << "   - Defect clusters formed: " << defect_clusters.size() << "\n";
        std::cout << "   - Using Bethe-Salpeter equation for bound state calculations\n";

        for (size_t i = 0; i < defect_clusters.size(); ++i) {
            std::cout << "   - Cluster " << i << ": " << defect_clusters[i].defect_positions.size()
                      << " defects, binding energy: " << defect_clusters[i].binding_energy
                      << " eV\n";
        }

        // 6. Propagate radiation effects using Green's function methods
        std::vector<double> time_steps = {1e-15, 1e-14,
                                          1e-13};  // femtoseconds to hundreds of femtoseconds
        std::vector<physics::DefectDistribution> time_evolution =
            enhanced_simulator.propagateRadiationEffects(initial_defects, silicon_lattice,
                                                         time_steps);

        std::cout << "\n6. Radiation effect propagation:\n";
        std::cout << "   - Time steps: " << time_steps.size() << "\n";
        std::cout << "   - Using Green's function methods for wave propagation\n";
        std::cout << "   - Time evolution calculated over " << time_steps.back() << " seconds\n";

        for (size_t t = 0; t < time_evolution.size(); ++t) {
            size_t total_defects =
                time_evolution[t].interstitials.size() + time_evolution[t].vacancies.size();
            std::cout << "   - Time " << time_steps[t] << " s: " << total_defects
                      << " total defects\n";
        }

        // 7. Calculate comprehensive radiation effects using all three methods
        physics::DefectDistribution enhanced_defects =
            enhanced_simulator.calculateEnhancedRadiationEffects(
                incident_proton, silicon_properties, silicon_lattice,
                sim::RadiationEnvironment::LEO);

        std::cout << "\n7. Comprehensive radiation effects (LEO environment):\n";
        std::cout << "   - Using Dirac + BSE + Green's function integration\n";
        std::cout << "   - Total interstitials: " << enhanced_defects.interstitials.size() << "\n";
        std::cout << "   - Total vacancies: " << enhanced_defects.vacancies.size() << "\n";
        std::cout << "   - Total clusters: " << enhanced_defects.clusters.size() << "\n";

        // 8. Neural network protection enhancement
        std::cout << "\n8. Neural network protection enhancement:\n";

        // Create a simple neural network for demonstration
        neural::ProtectedNeuralNetwork<float> network;
        std::vector<size_t> layer_sizes = {10, 5, 1};  // Simple 10-5-1 network

        // Apply quantum-enhanced protection
        std::vector<std::string> protection_recommendations =
            enhanced_simulator.enhanceNeuralNetworkProtection(
                network, 0.5, silicon_properties, silicon_lattice);  // Moderate radiation

        std::cout << "   - Current radiation level: Moderate (0.5)\n";
        std::cout << "   - Protection recommendations:\n";
        for (const auto& recommendation : protection_recommendations) {
            std::cout << "     * " << recommendation << "\n";
        }

        // 9. Calculate phonon-mediated interactions
        std::vector<Eigen::Vector3d> defect_positions = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                                         Eigen::Vector3d(0.5, 0.0, 0.0),
                                                         Eigen::Vector3d(0.0, 0.5, 0.0)};

        Eigen::MatrixXd phonon_interactions =
            enhanced_simulator.calculatePhononMediatedInteractions(defect_positions,
                                                                   silicon_lattice, 300.0);

        std::cout << "\n9. Phonon-mediated defect interactions:\n";
        std::cout << "   - Defect positions analyzed: " << defect_positions.size() << "\n";
        std::cout << "   - Interaction matrix dimensions: " << phonon_interactions.rows() << "x"
                  << phonon_interactions.cols() << "\n";
        std::cout << "   - Temperature: 300 K (room temperature)\n";
        std::cout << "   - Phonon dispersion: Linear approximation\n";

        // 10. Quantum-enhanced displacement energy calculation
        double standard_displacement = 25.0;  // eV for silicon
        double particle_energy = 10000.0;     // 10 keV
        double enhanced_displacement =
            enhanced_simulator.calculateQuantumEnhancedDisplacementEnergy(
                standard_displacement, particle_energy, ParticleType::PROTON);

        std::cout << "\n10. Quantum-enhanced displacement energy:\n";
        std::cout << "    - Standard displacement energy: " << standard_displacement << " eV\n";
        std::cout << "    - Particle energy: " << particle_energy << " eV\n";
        std::cout << "    - Relativistic enhancement factor: "
                  << enhanced_displacement / standard_displacement << "\n";
        std::cout << "    - Enhanced displacement energy: " << enhanced_displacement << " eV\n";

        std::cout << "\n=== Enhanced Quantum Integration Example Completed Successfully ===\n";

        // Summary of enhancements
        std::cout << "\nFramework Enhancements Summary:\n";
        std::cout << "1. ✅ Dirac equation for relativistic electron cascade modeling\n";
        std::cout << "2. ✅ Bethe-Salpeter equation for defect cluster formation\n";
        std::cout << "3. ✅ Green's function methods for radiation propagation\n";
        std::cout << "4. ✅ Integration with existing neural network protection\n";
        std::cout << "5. ✅ Material-specific quantum corrections\n";
        std::cout << "6. ✅ Phonon-mediated interaction modeling\n";
        std::cout << "7. ✅ Temperature-dependent quantum effects\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Example failed with error: " << e.what() << "\n";
        return 1;
    }
}
