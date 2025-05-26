# Advanced Simulation & Modeling in `rad_ml`

---

## Can I simulate radiation effects for custom or rare environments (e.g., deep space, asteroid belt, solar storms)?

**Yes!** The `rad_ml` framework is highly flexible and allows you to model any environment by using `EnvironmentType::CUSTOM` and setting the relevant parameters.

### **How to Model a Custom Environment**

```cpp
#include <rad_ml/radiation/environment.hpp>
using namespace rad_ml::radiation;

// Create a custom environment (e.g., deep space)
auto custom_env = std::make_shared<Environment>(EnvironmentType::CUSTOM);
custom_env->setSEUFlux(2e-8f);           // Set custom flux (particles/cm²/s)
custom_env->setSEUCrossSection(1e-13f); // Set custom cross-section (cm²/bit)
custom_env->setProperty("LET", 45.0f); // Set custom LET (MeV·cm²/mg)
custom_env->setProperty("TID", 100.0f); // Set custom TID (krad)
```

- You can use published data, test results, or simulation outputs (e.g., from GEANT4) to set these parameters.
- Custom environments can be used in mission phases, SEU simulation, and validation routines.

---

## How do I simulate a solar particle event or sudden radiation spike?

**Create a mission phase with a high-flux environment!**

### **Example: Solar Particle Event Phase**

```cpp
#include <rad_ml/radiation/space_mission.hpp>
#include <rad_ml/radiation/environment.hpp>
using namespace rad_ml::radiation;

// Solar flare environment
auto solar_flare_env = Environment::createEnvironment(EnvironmentType::SOLAR_FLARE);

// Define a short, high-flux mission phase
MissionPhase solar_event(
    "Solar Particle Event", MissionPhaseType::SOLAR_ENCOUNTER,
    solar_flare_env,
    std::chrono::hours(12), // 12-hour event
    0.9, // AU from Sun
    2.0  // mm Al shielding
);

// Add to your mission profile as needed
```

- You can also use a custom environment for more precise control over flux and spectrum.
- Set the phase duration to match the expected event length.

---

## Best Practices

- **Use EnvironmentType::CUSTOM** for any scenario not covered by predefined types.
- **Reference published data** or simulation results for realistic parameters.
- **Combine multiple phases** in your mission profile to capture time-varying and event-driven scenarios.
- **Document your environment parameters** for reproducibility and validation.

---

## Cross-Reference: How the Framework Supports Advanced Simulation

### Custom or Rare Environments
- `EnvironmentType::CUSTOM` in [`environment.hpp`](environment.hpp)
- Methods: `setSEUFlux`, `setSEUCrossSection`, `setProperty` in `class Environment`
- Used in:
  - Mission phases: `MissionPhase` in [`space_mission.hpp`](space_mission.hpp)
  - SEU simulation: `SEUSimulator` in [`seu_simulator.hpp`](seu_simulator.hpp)
  - Validation routines: see `src/validation/`

### Solar Particle Events & Radiation Spikes
- High-flux phases: create `MissionPhase` with `EnvironmentType::SOLAR_FLARE` or custom
- Mission profiles: `SpaceMission` in [`space_mission.hpp`](space_mission.hpp)
- Methods: `addPhase`, `getEnvironmentAtTime`, `calculateTotalRadiationExposure`
- Used in simulation and validation modules

### Example Locations
- Example usage in [`examples/`](../../../../examples/)
- Documentation: [`MISSION_PHASES.md`](MISSION_PHASES.md), [`PARTICLE_TYPES.md`](PARTICLE_TYPES.md)

---

## Technical Feature Reference

For technical engineers seeking direct code references:

- **Custom Environment Implementation:**
  - `include/rad_ml/radiation/environment.hpp`
    - `enum class EnvironmentType { ... CUSTOM ... }`
    - `class Environment` (`setSEUFlux`, `setSEUCrossSection`, `setProperty`)
- **Mission Phases & Profiles:**
  - `include/rad_ml/radiation/space_mission.hpp`
    - `class SpaceMission`, `struct MissionPhase`
    - Methods: `addPhase`, `getEnvironmentAtTime`, `calculateTotalRadiationExposure`
- **SEU Simulation:**
  - `include/rad_ml/radiation/seu_simulator.hpp`
    - `class SEUSimulator` (uses environment parameters)
- **Validation & Testing:**
  - `src/validation/` (accepts custom environments and mission profiles)
- **Examples:**
  - `examples/` directory (look for custom environments and mission phases)
- **Documentation:**
  - [`MISSION_PHASES.md`](MISSION_PHASES.md)
  - [`PARTICLE_TYPES.md`](PARTICLE_TYPES.md)
  - [FAQ](../../FAQ/FAQ.md)

---

## See Also

- [Modeling Mission-Phase-Dependent Environments](MISSION_PHASES.md)
- [Simulating Specific Particle Types](PARTICLE_TYPES.md)
- [FAQ](../../FAQ/FAQ.md)
