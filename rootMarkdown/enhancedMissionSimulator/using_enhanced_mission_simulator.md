# Using the Enhanced Mission Simulator (v0.9.6)

```cpp
#include "rad_ml/testing/mission_simulator.hpp"
#include "rad_ml/tmr/enhanced_tmr.hpp"

using namespace rad_ml::testing;
using namespace rad_ml::tmr;

int main() {
    // Create a mission profile for Low Earth Orbit
    MissionProfile profile = MissionProfile::createStandard("LEO");

    // Configure adaptive protection
    AdaptiveProtectionConfig protection_config;
    protection_config.enable_tmr_medium = true;
    protection_config.memory_scrubbing_interval_ms = 5000;

    // Create mission simulator
    MissionSimulator simulator(profile, protection_config);

    // Create your neural network
    YourNeuralNetwork network;

    // Register important memory regions for radiation simulation
    simulator.registerMemoryRegion(network.getWeightsPtr(),
                                 network.getWeightsSize(),
                                 true);  // Enable protection

    // Run the simulation for 30 mission seconds
    auto stats = simulator.runSimulation(
        std::chrono::seconds(30),
        std::chrono::seconds(3),
        [&network](const RadiationEnvironment& env) {
            // Adapt protection based on environment
            if (env.inside_saa || env.solar_activity > 5.0) {
                network.increaseProtectionLevel();
            } else {
                network.useStandardProtection();
            }
        }
    );

    // Print mission statistics
    std::cout << stats.getReport() << std::endl;

    // Test neural network after the mission
    network.runInference(test_data);

    return 0;
}
```
