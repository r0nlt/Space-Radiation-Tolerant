# Quick Start Guide

Here's how to use the framework to protect a simple ML inference operation:

```cpp
#include "rad_ml/api/protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"

using namespace rad_ml;

int main() {
    // 1. Initialize protection with material properties
    core::MaterialProperties aluminum;
    aluminum.radiation_tolerance = 50.0; // Standard aluminum
    tmr::PhysicsDrivenProtection protection(aluminum);

    // 2. Configure for your target environment
    sim::RadiationEnvironment env = sim::createEnvironment(sim::Environment::LEO);
    protection.updateEnvironment(env);

    // 3. Define your ML inference operation
    auto my_ml_operation = []() {
        // Your ML model inference code here
        float result = 0.0f; // Replace with actual inference
        return result;
    };

    // 4. Execute with radiation protection
    auto result = protection.executeProtected<float>(my_ml_operation);

    // 5. Check for detected errors
    if (result.error_detected) {
        std::cout << "Error detected and "
                  << (result.error_corrected ? "corrected!" : "not corrected")
                  << std::endl;
    }

    return 0;
}
```
