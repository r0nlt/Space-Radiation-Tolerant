# RADIATION.md

## Radiation Core Module — rad_ml Framework

This directory provides the core logic for adaptive and physics-driven radiation protection in the rad_ml framework. It enables robust, resource-efficient operation of machine learning and critical software in space and other radiation-prone environments.

---

### Key Components

- **AdaptiveProtection (adaptive_protection.hpp):**
  - Dynamically adjusts protection strategies (redundancy, scrubbing, checkpointing) based on real-time error rates and environmental conditions.
  - Supports multiple protection levels: MINIMAL, STANDARD, ENHANCED, MAXIMUM.
  - Tracks radiation environment (particle flux, bit flips, computation errors) and uses an exponential moving average for robust assessment.
  - Notifies other system components of protection level changes via a callback system.
  - Thread-safe and suitable for both simulation and deployment.

---

### How This Module is Used in the Framework

- **Mission Simulation:**
  Integrated into the `MissionSimulator` and `AdaptiveProtectionConfig` to provide realistic, environment-driven adaptation of protection strategies during simulated space missions. Protection levels and mechanisms are automatically tuned as the simulated environment changes (e.g., LEO, Mars, Jupiter, SAA, solar storms).

- **Neural Network and Memory Protection:**
  Used to adaptively protect neural network weights, activations, and memory regions. The protection level can be escalated during high-radiation events and relaxed to save resources in benign conditions.

- **Testing and Verification:**
  Extensively tested in the framework's verification and Monte Carlo simulation code, ensuring that adaptive protection responds correctly to error spikes and environmental changes, and that callbacks are triggered as expected.

- **Python and C++ API Integration:**
  Exposed as a configurable strategy in both C++ and Python APIs, allowing users to set adaptation intervals, sensitivity thresholds, and resource optimization parameters for their specific mission or application.

---

### Typical Usage

- Automatically escalate protection during solar storms or high-radiation events, and relax it to save resources in safe conditions.
- Integrate with simulation modules or real-time telemetry for closed-loop, adaptive protection.
- Used by higher-level controllers or neural network wrappers to optimize the trade-off between reliability and performance.

---

### Example Scenario

1. The system detects a spike in bit flips and computation errors.
2. `AdaptiveProtection` increases the protection level from STANDARD to ENHANCED or MAXIMUM.
3. When the error rate drops, it automatically reduces the protection level to save computational resources.

---

### Extensibility

- Easily extendable to support new protection levels or more detailed environmental models.
- Thresholds and configuration parameters can be adapted for different mission profiles or hardware.
- Callbacks and hooks allow other framework components to respond to protection level changes in real time.

---

### Summary

The `rad_ml/core/radiation` directory is essential for enabling adaptive, closed-loop radiation protection in the rad_ml framework, ensuring both reliability and efficiency in challenging environments. It is deeply integrated into mission simulation, neural network protection, and both C++ and Python APIs.
