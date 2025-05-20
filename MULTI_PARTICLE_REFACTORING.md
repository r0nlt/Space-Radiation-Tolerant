# Multi-Particle Support for rad_ml Physics Framework

## Overview

This document details the refactoring work done to support multiple particle types in the rad_ml quantum/physics simulation framework. The refactoring introduces a type-safe particle system and modernizes the codebase with C++17/20 features, making it more robust, extensible, and maintainable.

## Key Enhancements

### 1. Type-Safe Particle Type Representation

- Replaced legacy `enum ParticleType` with a type-safe `enum class ParticleType` which prevents implicit conversions and provides proper scoping.
- Added support for various particle types: protons, electrons, neutrons, photons, heavy ions, positrons, muons, and neutrinos.
- Created a `Particle` class that encapsulates physical properties such as mass, charge, and spin quantum number.
- Added factory methods for creating common particle types with their physical constants.

### 2. Multi-Particle Data Structures

- Refactored `DefectDistribution` to use `std::unordered_map<ParticleType, std::vector<double>>` instead of direct vectors, allowing particle-specific defect tracking.
- Updated `QFTParameters` to store particle-specific properties in maps:
  - Replaced single `mass` with `std::unordered_map<ParticleType, double> masses`
  - Replaced single `coupling_constant` with `std::unordered_map<ParticleType, double> coupling_constants`
- Added backward compatibility methods to ensure existing code continues to work.
- Similar updates to `ExtendedQFTParameters` for particle-specific decoherence rates and dissipation coefficients.

### 3. Quantum Field Theory Updates

- Added particle type tracking to the `QuantumField` class.
- Modified `KleinGordonEquation`, `DiracEquation`, and `MaxwellEquations` to be particle-type aware.
- Updated field evolution logic to check for compatible particle types and handle multi-particle scenarios.
- Added references to the QFT parameters in each equation class for better data access.

### 4. Modern C++17/20 Features

- Added `std::optional` parameters to allow selective overriding of particle types in method calls.
- Used `std::unique_ptr` for memory-safe ownership in factory functions.
- Employed `std::reference_wrapper` for collections of non-owning references.
- Used `std::clamp` instead of min/max combinations for bound checking.
- Leveraged structured bindings in range-based for loops for clearer code.
- Improved parameter passing with reference qualifiers throughout the codebase.

### 5. Enhanced APIs for Multi-Particle Support

- Created the `createParticleField` factory function to simplify creation of particle-specific quantum fields.
- Added the `simulateMultiParticleInteraction` function to simulate interactions between different particle fields.
- Updated `applyQuantumFieldCorrections` to handle multi-particle defect distributions.
- Added particle type parameters to all relevant physics functions.

### 6. Documentation Improvements

- Updated function documentation with Doxygen-style markup.
- Added clear parameter descriptions to document the multi-particle behavior.
- Enhanced code comments to explain particle-specific logic and algorithms.

## Example Code

A new example file `multi_particle_simulation_example.cpp` has been created to demonstrate the multi-particle capabilities:

- Shows how to create and initialize fields for different particle types
- Demonstrates how to simulate cascades with multiple particle types
- Provides examples of combining and processing defects from different particles
- Illustrates quantum field interactions between different particle types

## Usage Guidelines

### Creating Particle-Specific Parameters

```cpp
QFTParameters params;
params.masses[ParticleType::Electron] = 9.1093837e-31;
params.coupling_constants[ParticleType::Proton] = 0.15;
```

### Simulating Different Particle Types

```cpp
// Create fields for different particles
auto electron_field = createParticleField(grid_dims, spacing, ParticleType::Electron, params);
auto proton_field = createParticleField(grid_dims, spacing, ParticleType::Proton, params);

// Simulate particle-specific cascades
DefectDistribution electron_defects = simulateDisplacementCascade(
    crystal, energy, params, threshold, ParticleType::Electron);
```

### Processing Multi-Particle Data

```cpp
// Apply quantum corrections to specific particle types
std::vector<ParticleType> particles_to_correct = {
    ParticleType::Electron,
    ParticleType::Proton
};
DefectDistribution corrected = applyQuantumFieldCorrections(
    defects, crystal, params, temperature, particles_to_correct);
```

## Backward Compatibility

Care has been taken to maintain backward compatibility with existing code:
- Default parameter values match the previous implementation's behavior
- Helper methods like `getMass()` and `getCouplingConstant()` provide access to the first particle type if not specified
- Implicit conversions are prevented through the use of `enum class`, reducing bugs

## Future Work

Potential areas for future enhancement include:
- Adding more particle types and their specific physics
- Implementing quantum entanglement between different particle fields
- Introducing particle interaction channels and decay processes
- Optimizing multi-particle simulations for performance
- Supporting relativistic effects for high-energy particles
