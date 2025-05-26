# Modeling Mission-Phase-Dependent Environments in `rad_ml`

---

## Overview

The `rad_ml` framework enables you to model **realistic, time-varying radiation environments** by defining complex mission profiles with multiple phases. Each phase can have its own environment, duration, and shielding, allowing you to simulate the changing conditions experienced during space missions, satellite operations, or other mission-critical scenarios.

---

## Core Concepts

### **SpaceMission**
- Represents an entire mission, composed of multiple phases.
- Each mission has a name, target (e.g., Mars, Moon, GEO), and a sequence of phases.

### **MissionPhase**
- Represents a single phase of a mission (e.g., launch, Earth orbit, lunar surface).
- Each phase has:
  - A name and type
  - A radiation environment (predefined or custom)
  - A duration (in seconds)
  - Shielding and distance parameters

---

## How to Define a Mission with Phases

```cpp
#include <rad_ml/radiation/space_mission.hpp>
#include <rad_ml/radiation/environment.hpp>

using namespace rad_ml::radiation;

// Create mission phases
MissionPhase launch_phase(
    "Launch", MissionPhaseType::LAUNCH,
    Environment::createEnvironment(EnvironmentType::GROUND_LEVEL),
    std::chrono::seconds(600), // 10 minutes
    1.0, // AU from Sun
    5.0  // mm Al shielding
);

MissionPhase leo_phase(
    "LEO", MissionPhaseType::EARTH_ORBIT,
    Environment::createEnvironment(EnvironmentType::LOW_EARTH_ORBIT),
    std::chrono::hours(24 * 7), // 1 week
    1.0,
    2.0
);

MissionPhase lunar_surface(
    "Lunar Surface", MissionPhaseType::PLANETARY_SURFACE,
    Environment::createEnvironment(EnvironmentType::LUNAR),
    std::chrono::hours(24 * 14), // 2 weeks
    1.0,
    10.0
);

// Compose the mission
SpaceMission mission("Lunar Mission", MissionTarget::MOON);
mission.addPhase(launch_phase)
       .addPhase(leo_phase)
       .addPhase(lunar_surface);
```

---

## Using Mission Phases in Simulation

- The framework can:
  - Retrieve the environment at any mission time (`getEnvironmentAtTime()`)
  - Calculate total radiation exposure across all phases
  - Simulate SEUs, TID, and other effects phase-by-phase

```cpp
// Get environment at a specific mission time
auto env = mission.getEnvironmentAtTime(std::chrono::hours(100));

// Calculate total exposure
double total_exposure = mission.calculateTotalRadiationExposure();
```

---

## Best Practices

- **Define realistic durations and shielding** for each phase.
- **Use custom environments** for unique mission scenarios (e.g., solar storms).
- **Leverage mission profiles** for validation, testing, and adaptive protection.
- **Reference mission phases** in your simulation and error injection routines for accurate, time-dependent modeling.

---

## Where is this used?

- Mission simulation examples (`examples/`)
- Validation and reporting modules
- Adaptive protection and error injection routines

---

## See Also

- [FAQ: Can I model time-varying or mission-phase-dependent environments?](../../../FAQ.md)
- [Radiation Environment Module](RADIATION.md)
