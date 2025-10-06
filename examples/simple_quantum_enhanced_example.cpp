/**
 * Simple Quantum Enhanced Example
 *
 * Simplified example showing how to use the enhanced quantum models
 * without complex setup - demonstrates the key functionality in a few lines.
 */

#include <iostream>
#include <rad_ml/physics/enhanced_physics_radiation_simulator.hpp>

using namespace rad_ml;

int main()
{
    std::cout << "Simple Quantum Enhanced Example\n";
    std::cout << "==============================\n";

    try {
        // 1. Initialize enhanced simulator (one line!)
        sim::EnhancedPhysicsRadiationSimulator simulator;

        std::cout << "✅ Enhanced Physics Simulator initialized\n";

        // 2. Create material properties (simple call)
        physics::MaterialProperties silicon = simulator.createMaterialProperties("silicon");

        std::cout << "✅ Silicon material: " << silicon.band_gap << " eV band gap\n";

        // 3. Create crystal lattice (simple call)
        physics::CrystalLattice lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

        std::cout << "✅ Crystal lattice: Diamond structure, 0.543 nm spacing\n";

        // 4. Calculate relativistic electron cascade (one call!)
        Particle proton = Particle::createProton();
        auto cascade = simulator.calculateRelativisticElectronCascade(proton, silicon, lattice);

        std::cout << "✅ Relativistic cascade: " << cascade.size() << " secondary electrons\n";

        // 5. Calculate defect clusters (one call!)
        physics::DefectDistribution initial_defects;
        initial_defects.interstitials[ParticleType::ELECTRON] = {1.0};

        auto clusters = simulator.calculateDefectClusterFormation(initial_defects, lattice, 300.0);

        std::cout << "✅ Defect clusters: " << clusters.size() << " clusters formed\n";

        // 6. Propagate effects (one call!)
        std::vector<double> time_steps = {1e-15, 1e-14};
        auto evolution = simulator.propagateRadiationEffects(initial_defects, lattice, time_steps);

        std::cout << "✅ Time evolution: " << evolution.size() << " time steps\n";

        // 7. Neural network protection (one call!)
        neural::ProtectedNeuralNetwork<float> network;
        auto recommendations =
            simulator.enhanceNeuralNetworkProtection(network, 0.5, silicon, lattice);

        std::cout << "✅ Neural protection: " << recommendations.size() << " recommendations\n";
        for (const auto& rec : recommendations) {
            std::cout << "   - " << rec << "\n";
        }

        std::cout << "\n🎉 All quantum enhancements working!\n";
        std::cout << "\nSummary of what you just used:\n";
        std::cout << "• Dirac equation for relativistic electron cascades\n";
        std::cout << "• Bethe-Salpeter equation for defect cluster formation\n";
        std::cout << "• Green's function methods for radiation propagation\n";
        std::cout << "• Enhanced neural network protection\n";
        std::cout << "• Material-aware quantum corrections\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
}
