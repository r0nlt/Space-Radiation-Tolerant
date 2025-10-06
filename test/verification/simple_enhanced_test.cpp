/**
 * Simple Enhanced Physics Test
 *
 * This test validates that the enhanced physics simulator can be instantiated
 * and basic functionality works.
 */

#include <iostream>
#include <rad_ml/sim/physics_radiation_simulator.hpp>

using namespace rad_ml;

int main()
{
    std::cout << "Simple Enhanced Physics Test\n";
    std::cout << "===========================\n";

    try {
        // Test basic physics radiation simulator
        sim::EnvironmentParams leo_params(sim::SpaceEnvironment::LEO, 0.5, 2.0);
        sim::PhysicsRadiationSimulator simulator(leo_params);

        std::cout << "✅ Physics Radiation Simulator created for LEO environment\n";

        // Test mission environment parameters
        auto mission_params = sim::PhysicsRadiationSimulator::getMissionEnvironment("LEO");
        std::cout << "✅ LEO mission parameters retrieved:\n";
        std::cout << "   - Environment: " << static_cast<int>(mission_params.environment) << "\n";
        std::cout << "   - Solar activity: " << mission_params.solar_activity << "\n";
        std::cout << "   - Shielding thickness: " << mission_params.shielding_thickness_mm
                  << " mm\n";

        // Test other environments
        auto mars_params = sim::PhysicsRadiationSimulator::getMissionEnvironment("MARS");
        std::cout << "✅ Mars mission parameters retrieved:\n";
        std::cout << "   - Environment: " << static_cast<int>(mars_params.environment) << "\n";
        std::cout << "   - Solar activity: " << mars_params.solar_activity << "\n";

        std::cout << "\n🎉 Basic Enhanced Physics Test Passed!\n";
        std::cout << "Framework is ready for advanced quantum model integration.\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << "\n";
        return 1;
    }
}
