# Radiation-Tolerant ML API: Directory Overview and Usage Guide

## Directory Purpose

This directory (`include/rad_ml/api/`) provides unified and extensible APIs for the Radiation-Tolerant Machine Learning (rad_ml) framework. It enables users to:
- Access all major TMR and protection features through a single interface.
- Define and register custom defense strategies.
- Integrate advanced protection into neural networks and value computations.
- Leverage both built-in and user-defined protection mechanisms in a modular way.

---

## File-by-File API Summary

### `rad_ml.hpp` — Unified High-Level API
- **Purpose:** Central entry point for the rad_ml framework, aggregating all major components and providing convenience functions, type aliases, and factory helpers.
- **Key Features:**
  - Versioning and initialization/shutdown helpers.
  - Unified TMR type aliases and factory functions (e.g., `make_tmr::standard`, `make_tmr::enhanced`).
  - High-level memory management and error handling utilities.
  - Helpers for neural network protection, simulation, and testing.
- **Design Logic:**
  - Simplifies usage by exposing a single header for most framework features.
  - Encourages best practices (e.g., memory protection, error logging) by default.

#### Example: Creating and Using a TMR Instance
```cpp
#include <rad_ml/api/rad_ml.hpp>

// Create a standard TMR-protected integer
auto tmr_int = rad_ml::make_tmr::standard<int>(42);
int value = tmr_int->get();
tmr_int->set(100);
```

#### Example: Protecting a Neural Network
```cpp
#include <rad_ml/api/rad_ml.hpp>

// Create a protected neural network with selective hardening
// Replace /* network constructor args */ with the arguments required by your network's constructor.
// For example, if MyNetwork(int input_dim, int hidden_dim, int output_dim):
// auto protected_nn = rad_ml::neural::createProtectedNetwork<MyNetwork>(
//     rad_ml::neural::HardeningStrategy::RESOURCE_CONSTRAINED,
//     rad_ml::neural::ProtectionLevel::FULL_TMR,
//     10, 20, 2 // Example: input=10, hidden=20, output=2
// );
auto protected_nn = rad_ml::neural::createProtectedNetwork<MyNetwork>(
    rad_ml::neural::HardeningStrategy::RESOURCE_CONSTRAINED,
    rad_ml::neural::ProtectionLevel::FULL_TMR,
    /* network constructor args */
);
```

---

### `custom_defense_api.hpp` / `custom_defense_api.cpp` — Custom Defense Strategies
- **Purpose:** Enable users to define, register, and use custom protection strategies beyond the built-in TMR variants.
- **Key Features:**
  - `DefenseStrategy` enum and `DefenseConfig` struct for flexible configuration.
  - `DefenseFactory` and `UnifiedDefenseSystem` for managing and applying strategies.
  - Support for custom user strategies via registration and factory pattern.
  - Integration with neural networks and value protection.
- **Design Logic:**
  - Decouples protection logic from application logic, allowing plug-and-play strategies.
  - Supports both built-in and user-defined (custom) strategies, including hardware-accelerated and physics-driven options.
  - Provides a unified interface for protecting both values and neural networks.

#### Example: Registering and Using a Custom Strategy
```cpp
// Define a custom strategy (see custom_defense_api.cpp for full example)
class CustomBitInterleaving : public tmr::ProtectionStrategy { /* ... */ };

// Register the strategy
UnifiedDefenseSystem::registerCustomStrategy(
    "bit_interleaving",
    [](const DefenseConfig& config) {
        return std::make_unique<CustomBitInterleaving>(config);
    }
);

// Use the custom strategy in a config
DefenseConfig config;
config.strategy = DefenseStrategy::CUSTOM;
config.custom_params["name"] = "bit_interleaving";
config.custom_params["interleave_factor"] = "16";

// Create a defense system and protect a value
UnifiedDefenseSystem defense(config);
auto protected_value = defense.protectValue(3.14159f);
```

#### Example: Protecting a Neural Network with a Custom Strategy
```cpp
// Create a neural network
auto neural_network = std::make_unique<SimpleNN>(10, 20, 2);

// Wrap with protection
auto protected_nn = wrapExistingNN(std::move(neural_network), config);

// Use the protected network
std::vector<float> input(10, 0.5f);
auto output = protected_nn->forward(input);
```

---

## Main Classes and Concepts

### `DefenseStrategy` and `DefenseConfig`
- **`DefenseStrategy`:** Enum listing all available protection strategies (standard, enhanced, physics-driven, custom, etc).
- **`DefenseConfig`:** Configuration struct specifying strategy, environment, neural network options, hardware acceleration, and custom parameters.

### `DefenseFactory`
- **Purpose:** Registry and factory for creating protection strategies by name.
- **Usage:**
  - Register a new strategy: `DefenseFactory::instance().registerStrategy("name", creator_fn);`
  - Create a strategy: `DefenseFactory::instance().createStrategy("name", config);`

### `UnifiedDefenseSystem`
- **Purpose:** Main interface for applying protection to values and neural networks.
- **Usage:**
  - Construct with a `DefenseConfig`.
  - Use `protectValue<T>(value)` to protect a value.
  - Use `executeProtected<T>(fn)` to run a function with protection.
  - Use `protectNeuralNetwork<T>(arch)` to create a protected neural network.
  - Update environment dynamically with `updateEnvironment()`.

### `RadiationHardenedNN`
- **Purpose:** Template wrapper for neural networks, integrating protection and defense system.
- **Usage:**
  - Construct with a neural network and config, or use helper functions.
  - Use `forward(input)` to run protected inference.

---

## Extending and Customizing the API

- **Registering Custom Strategies:**
  - Implement a subclass of `tmr::ProtectionStrategy`.
  - Register it with `UnifiedDefenseSystem::registerCustomStrategy("name", creator_fn);`
  - Use `DefenseConfig` with `strategy = DefenseStrategy::CUSTOM` and set `custom_params["name"]` to your strategy name.

- **Integrating with New Neural Network Types:**
  - Use the `RadiationHardenedNN` template or wrap your network with `wrapExistingNN`.
  - Ensure your network exposes a `forward()` method compatible with the API.

---

## Example: Full Custom API Workflow

```cpp
#include <rad_ml/api/custom_defense_api.hpp>

// 1. Register a custom strategy
UnifiedDefenseSystem::registerCustomStrategy(
    "bit_interleaving",
    [](const DefenseConfig& config) {
        return std::make_unique<CustomBitInterleaving>(config);
    }
);

// 2. Configure for Jupiter environment with custom strategy
DefenseConfig config = DefenseConfig::forEnvironment(sim::Environment::JUPITER);
config.strategy = DefenseStrategy::CUSTOM;
config.custom_params["name"] = "bit_interleaving";
config.custom_params["interleave_factor"] = "16";

// 3. Protect a neural network
auto nn = std::make_unique<SimpleNN>(10, 20, 2);
auto protected_nn = wrapExistingNN(std::move(nn), config);

// 4. Use the protected network
std::vector<float> input(10, 0.5f);
auto output = protected_nn->forward(input);

// 5. Protect and compute a value
UnifiedDefenseSystem defense(config);
auto protected_value = defense.protectValue(3.14159f);
auto result = defense.executeProtected<float>([]() { return std::sqrt(2.0f); });
```

---

## Best Practices and Notes
- Use the unified API (`rad_ml.hpp`) for most applications; drop down to custom APIs for advanced or experimental strategies.
- Always register custom strategies before using them in configs.
- Update the environment dynamically if your application changes mission phase or radiation conditions.
- For neural networks, ensure your architecture and data types are compatible with the protection wrappers.

---

## Further Reading
- See the code in `custom_defense_api.cpp` for more advanced examples and logic.
- Refer to the main framework documentation for details on TMR, neural protection, and simulation modules.

---

## Test Suite and Practical Examples

The rad_ml framework is accompanied by a comprehensive suite of unit and integration tests that exercise all major API features and extension points. These tests serve as both validation and practical usage examples for developers.

- **API Feature Tests:**
  - See [`test/api_test/custom_defense_api_test.cpp`](../../../test/api_test/custom_defense_api_test.cpp) for end-to-end tests of the custom defense API, including:
    - Registration and use of custom protection strategies
    - Value and neural network protection
    - Environment adaptation and error simulation
    - Helper functions and extension patterns
- **Framework and Model Validation:**
  - See [`test/framework_validation_test.cpp`](../../../test/framework_validation_test.cpp) and [`test/comprehensive_model_test.cpp`](../../../test/comprehensive_model_test.cpp) for realistic mission scenarios, error injection, and recovery validation.

**Tip:**
- Consult these tests for real-world usage patterns, troubleshooting, and advanced examples not covered in this manual.
- If you encounter unexpected behavior, reviewing the test cases can help clarify correct API usage and expected outcomes.

---

## Python End-to-End Test and Example

For users of the Python API and bindings, the primary end-to-end test and demonstration is:

- [`examples/advanced_radiation_comparison.py`](../../../examples/advanced_radiation_comparison.py)

This script exercises the full Python API, including:
- Advanced protection strategies (TMR, enhanced TMR, Reed-Solomon, selective hardening, resource optimization)
- Neural network integration and protection (with PyTorch models)
- Environment adaptation and dynamic protection configuration
- Monte Carlo simulation and evaluation under various radiation scenarios
- Gradient protection, error injection, and resource efficiency analysis


**Strengths:**
- Comprehensive coverage of protection strategies, neural network integration, and Monte Carlo simulation.
- Realistic and reproducible experiments using MNIST, fixed seeds, and device-agnostic code.
- Extensible and modular configuration for protection strategies and custom parameters.
- Detailed reporting, progress tracking, and visualization of results.

**Suggestions for Improvement:**
- Consider splitting the script into smaller modules for maintainability.
- Add more docstrings and inline documentation for helper functions and classes.
- Add assertions or checks to ensure protection is applied and edge cases are tested.
- Provide command-line options for batch size, epochs, and output directories.
- Consider parallelizing Monte Carlo trials for performance.
- Print a summary table of results at the end for quick comparison.

**What It Does Exceptionally Well:**
- Provides an end-to-end, realistic, and robust validation pipeline.
- Demonstrates advanced features and fallback logic.
- Serves as an educational reference for both users and developers.
