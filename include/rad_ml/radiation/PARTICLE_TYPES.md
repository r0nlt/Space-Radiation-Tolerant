# Simulating Specific Particle Types in `rad_ml`

---

## Overview

The `rad_ml` framework is designed to flexibly simulate the effects of **specific particle types** (such as protons, heavy ions, neutrons, etc.) on digital electronics and neural networks. While the framework does not have explicit classes for each particle type, it enables accurate modeling by allowing users to set the **particle flux** and **SEU cross-section** for any scenario.

> **Note:** While the primary effect modeled at the bit level is the Single Event Upset (SEU), the framework also provides tools and algorithms for evaluating **Total Ionizing Dose (TID)**, **Single Event Latchup (SEL)**, and **LET-based effects** in its validation and mission assessment modules. Users can further extend modeling via custom environment properties.

---

## How It Works

- **Particle Type Abstraction:**
  - The framework abstracts over particle type, focusing on the *effect* (e.g., SEU/bit flip rate) rather than the physical identity of the particle.
  - Users can simulate any particle type by providing the correct flux (particles/cm²/s) and cross-section (cm²/bit) for their device and scenario.

- **Custom Environments:**
  - Use the `EnvironmentType::CUSTOM` option to define a scenario for a specific particle type.
  - Set the flux and cross-section to match published data or test results for protons, heavy ions, etc.

---

## Example: Simulating Protons vs. Heavy Ions

```cpp
// Proton scenario
auto proton_env = std::make_shared<rad_ml::radiation::Environment>(rad_ml::radiation::EnvironmentType::CUSTOM);
proton_env->setSEUFlux(1e5f);           // Proton flux (protons/cm²/s)
proton_env->setSEUCrossSection(1e-13f); // Proton SEU cross-section (cm²/bit)

// Heavy ion scenario
auto ion_env = std::make_shared<rad_ml::radiation::Environment>(rad_ml::radiation::EnvironmentType::CUSTOM);
ion_env->setSEUFlux(1e2f);              // Heavy ion flux (ions/cm²/s)
ion_env->setSEUCrossSection(5e-12f);    // Heavy ion SEU cross-section (cm²/bit)
```

- These environments can be used with the SEU simulator or in mission profiles to model the effects of different particle types.

---

## Advanced: LET, Energy, and Custom Properties

- Users can add custom properties (e.g., LET, energy spectrum, TID, SEL probability) to environments for more detailed modeling:

```cpp
env->setProperty("LET", 37.5f); // Linear Energy Transfer in MeV·cm²/mg
env->setProperty("energy", 100.0f); // Particle energy in MeV
env->setProperty("TID", 50.0f); // Total Ionizing Dose in krad
env->setProperty("SEL_probability", 1e-5f); // SEL probability per device per day
```

---

## Dual Approach: Physics Modeling vs. System-Level Simulation

The framework supports two complementary approaches to particle and radiation effect modeling:

| Use Case/Module         | Particle Type Handling                | Example/Algorithmic Use                |
|------------------------ |-------------------------------------- |----------------------------------------|
| Physics/QFT             | Explicit (proton, neutron, etc.)      | Mass, coupling, quantum effects        |
| SEU/Bit Flip Simulation | Abstracted (via flux/cross-section)   | User sets values for scenario          |
| Validation/Testing      | LET, TID, SEL, cross-section metrics  | NASA/ESA protocols, Weibull modeling   |
| Custom Properties       | User-defined (LET, energy, TID, etc.) | `env->setProperty("LET", ...)`       |

- **Physics/Quantum Modeling:**
  - Use explicit particle types for fundamental calculations (mass, charge, quantum effects).
- **System/Radiation Effect Simulation:**
  - Abstract over particle type, letting the user set the scenario via parameters (flux, cross-section, etc.).
- **Validation and Compliance:**
  - Use LET, TID, SEL, and cross-section as the main metrics, referencing standards and test data.

---

## Key Points

- The framework does **not** internally distinguish between particle types in system-level simulation; it relies on user-supplied parameters for accuracy.
- In addition to SEUs, the framework provides tools for evaluating TID, SEL, and LET-based effects in its validation and mission assessment modules.
- This approach allows simulation of any scenario for which you have flux and cross-section data.
- For reference values, see radiation test reports, NASA/ESA standards, or GEANT4 simulations.

---

## Conclusion

This dual approach gives the `rad_ml` framework both **flexibility** and **scientific rigor**:
- Advanced users can perform detailed physics modeling with explicit particle types.
- System and mission-level users can efficiently simulate radiation effects using parameter abstraction, custom environments, and standards-based validation.

---

## See Also

- [FAQ: What types of radiation effects can I simulate?](../../../FAQ.md)
