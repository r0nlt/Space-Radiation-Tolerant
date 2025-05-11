<!--
 Copyright (C) 2025 Rishab Nuguru

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
-->
# Radiation-Tolerant Machine Learning Framework Testing

This directory contains tools for testing and demonstrating the radiation-tolerant ML framework using simulated space radiation environments.

## Components

1. **Unit Tests**: Comprehensive test suite in `test/tmr/rad_ml_framework_test.cpp`
2. **Spacecraft Orbit Simulation**: Interactive demo application in `examples/orbit_simulation/rad_tolerant_ml_demo.cpp`

## Running the Tests

To build and run the test suite:

```bash
# Build the tests
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make rad_ml_framework_test

# Run the tests
./test/rad_ml_framework_test
```

The test suite includes:

- **Physics Model Tests**: Validates the NASA physics models for temperature, mechanical stress, and synergistic effects
- **Protection Level Tests**: Confirms that protection levels adapt correctly to different radiation environments
- **Resource Allocation Tests**: Verifies that the sensitivity-based allocation algorithm prioritizes critical layers
- **Error Correction Tests**: Measures the error correction capabilities of TMR strategies
- **Performance Tests**: Quantifies the computational overhead of different protection levels
- **Mission Scenario Tests**: End-to-end tests simulating complete mission scenarios with changing radiation environments

## Running the Demo Application

To build and run the interactive orbit simulation:

```bash
# Build the demo
mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=ON
make rad_tolerant_ml_demo

# Run the demo
./examples/orbit_simulation/rad_tolerant_ml_demo
```

The demo application:

1. Simulates a spacecraft in orbit (either LEO or GEO) with realistic radiation environments
2. Models the South Atlantic Anomaly, solar storms, and other radiation events
3. Runs a neural network-based image classifier protected by our framework
4. Dynamically adapts protection levels based on the changing radiation environment
5. Logs protection levels and performance metrics throughout the mission

## Simulation Features

- **Orbit Simulation**: Models Low Earth Orbit (LEO) with South Atlantic Anomaly crossings, or Geostationary Orbit (GEO)
- **Physics-Driven Protection**: Demonstrates how NASA physics models drive the protection strategies
- **Realistic Environment Factors**:
  - Trapped proton and electron fluxes
  - Temperature variations based on sun exposure
  - Mechanical stress fluctuations
  - Solar activity events
- **Protection Adaptation**: Shows how protection strategies adapt at multiple time scales:
  - Microsecond: Per-computation TMR
  - Second: Layer-level protection
  - Minute: Mission phase adaptation
  - Hour: System health monitoring

## Analyzing Results

The demo generates a log file (`radiation_ml_simulation.log`) that can be used to analyze the behavior of the protection system. The log includes:

- Simulation time
- Radiation levels
- South Atlantic Anomaly region flag
- Current protection level

This data can be visualized using plotting tools to show how protection levels adapt to the radiation environment over time.

## Extending the Framework

To test your own neural network models with the radiation-tolerant framework:

1. Create a `PhysicsDrivenProtection` instance with appropriate material properties
2. Wrap your neural network layers with `ProtectedNeuralLayer` instances
3. Feed radiation environment data from your specific mission profile
4. Monitor protection levels and error rates as your model executes

See the `RadiationTolerantNN` class in the demo for an example implementation. 
