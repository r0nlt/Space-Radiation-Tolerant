# Radiation Environment Module — `rad_ml/radiation/`

---

## Overview

The `rad_ml/radiation/` directory provides the core abstractions and simulation tools for modeling, configuring, and simulating radiation environments and their effects on electronic systems. It is foundational for enabling realistic, mission-specific, and scientifically accurate studies of radiation-induced errors in the rad_ml framework.

> This module enables the definition of space environments, simulation of single event upsets (SEUs), and construction of complex mission profiles for robust, radiation-aware AI and system design.

---

## File-by-File Analysis

### [`environment.hpp`](environment.hpp)
- **Purpose:**
  Defines the `Environment` class and `EnvironmentType` enum for representing and configuring radiation environments (e.g., LEO, GEO, Mars, Jupiter, Solar Flare, Custom).
- **Key Features:**
  - Predefined and custom environments with default SEU flux and cross-section values.
  - Methods to set/get SEU flux, cross-section, and arbitrary properties.
  - Factory method for creating standard environments.
- **Role:**
  Provides a flexible, extensible abstraction for radiation conditions, used throughout the framework for simulation, protection, and mission modeling.

---

### [`seu_simulator.hpp`](seu_simulator.hpp)
- **Purpose:**
  Implements the `SEUSimulator` class for simulating single event upsets (bit flips) in memory regions due to radiation.
- **Key Features:**
  - Configurable by environment and random seed.
  - Injects bit flips into memory based on SEU flux, cross-section, memory size, and exposure duration.
  - Uses Poisson statistics for realistic SEU event modeling.
  - Supports batch simulation across multiple memory regions with callback support.
- **Role:**
  Enables realistic, statistically accurate simulation of radiation-induced memory errors for testing, validation, and research.

---

### [`space_mission.hpp`](space_mission.hpp)
- **Purpose:**
  Provides abstractions for modeling complex space missions, including mission phases, targets, and time-varying environments.
- **Key Features:**
  - `MissionPhaseType` and `MissionTarget` enums for standardizing mission scenarios.
  - `MissionPhase` struct for associating environment, duration, and shielding with each phase.
  - `SpaceMission` class for building mission profiles, querying environment at a given time, and calculating total radiation exposure.
  - Factory methods for standard missions (LEO, GEO, Mars, Jupiter, etc.).
- **Role:**
  Supports end-to-end mission simulation, allowing the framework to model how radiation exposure changes over time and mission context.

---

## Technical Capabilities

- **Environment Modeling:**
  Predefined and user-configurable environments for a wide range of space and terrestrial scenarios.
- **SEU Simulation:**
  Realistic, probabilistic modeling of bit flips in memory, supporting both single and batch region simulation.
- **Mission Profiling:**
  Construction of multi-phase missions with time-varying environments and shielding, supporting integrated exposure calculations.
- **Integration:**
  Used by protection, training, and testing modules to drive realistic error injection and adaptive protection strategies.

---

## Typical Use Cases

- **Radiation-Aware Training and Testing:**
  Simulate realistic error rates for neural network training, inference, and validation.
- **Mission-Critical System Design:**
  Model and analyze the impact of radiation on electronics for specific mission profiles (e.g., Mars rover, Jupiter orbiter).
- **Research and Validation:**
  Enable reproducible, scientifically rigorous studies of radiation effects and mitigation strategies.

---

## Summary

> The `rad_ml/radiation/` directory is the foundation for all radiation environment modeling and simulation in the rad_ml framework. It enables accurate, mission-specific, and extensible studies of radiation effects, supporting robust AI and system design for space and other harsh environments.
