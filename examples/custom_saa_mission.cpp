// Example: Custom Two-Week SAA Mission Simulation
#include <chrono>
#include <iostream>
#include <memory>
#include <rad_ml/radiation/environment.hpp>
#include <rad_ml/radiation/space_mission.hpp>

using namespace rad_ml::radiation;

int main()
{
    // Define a two-week SAA mission phase
    auto saa_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "South Atlantic Anomaly");
    saa_env->setSEUFlux(1e-6f);  // Example: higher flux for SAA
    saa_env->setSEUCrossSection(1e-14f);

    MissionPhase saa_phase("SAA Phase", MissionPhaseType::EARTH_ORBIT, saa_env,
                           std::chrono::hours(24 * 14),  // 2 weeks
                           1.0,                          // AU from Sun
                           2.0                           // mm Al shielding
    );

    // Create the mission and add the phase
    SpaceMission mission("Two-Week SAA Mission", MissionTarget::EARTH_LEO);
    mission.addPhase(saa_phase);

    // Simulate: Calculate total radiation exposure
    double total_exposure = mission.calculateTotalRadiationExposure();
    std::cout << "Total radiation exposure over 2 weeks in SAA: " << total_exposure
              << " (flux-time product)" << std::endl;

    // Optionally, get the environment at day 7
    auto env_at_day7 = mission.getEnvironmentAtTime(std::chrono::hours(24 * 7));
    std::cout << "Environment at day 7: " << env_at_day7->getName() << std::endl;

    return 0;
}
