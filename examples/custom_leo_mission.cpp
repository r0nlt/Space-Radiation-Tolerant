// Example: Custom Two-Week LEO Mission Simulation
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <rad_ml/radiation/environment.hpp>
#include <rad_ml/radiation/space_mission.hpp>

using namespace rad_ml::radiation;

int main()
{
    // Define a two-week LEO mission phase
    auto leo_env = Environment::createEnvironment(EnvironmentType::LOW_EARTH_ORBIT);
    MissionPhase leo_phase("LEO Phase", MissionPhaseType::EARTH_ORBIT, leo_env,
                           std::chrono::hours(24 * 14),  // 2 weeks
                           1.0,                          // AU from Sun
                           2.0                           // mm Al shielding
    );

    // Create the mission and add the phase
    SpaceMission mission("Two-Week LEO Mission", MissionTarget::EARTH_LEO);
    mission.addPhase(leo_phase);

    // Print SEU flux and duration for verification
    double flux = leo_env->getSEUFlux();
    // Calculate duration in seconds for 2 weeks
    long long duration = 24LL * 14 * 3600;  // 1,209,600 seconds
    std::cout << "LEO SEU flux: " << flux << " particles/cm²/s" << std::endl;
    std::cout << "Duration: " << duration << " seconds" << std::endl;

    // Simulate: Calculate total radiation exposure
    double total_exposure = mission.calculateTotalRadiationExposure();
    std::cout << "Total radiation exposure over 2 weeks in LEO: " << total_exposure
              << " (flux-time product)" << std::endl;

    // Verify the exposure calculation
    double expected_exposure = flux * duration;
    std::cout << "Expected exposure: " << expected_exposure << std::endl;
    assert(std::abs(total_exposure - expected_exposure) < 1e-8 && "Exposure calculation mismatch!");

    // Optionally, get the environment at day 7
    auto env_at_day7 = mission.getEnvironmentAtTime(std::chrono::hours(24 * 7));
    std::cout << "Environment at day 7: " << env_at_day7->getName() << std::endl;

    return 0;
}
