# Radiation-Tolerant Machine Learning: TMR and Advanced Protection Manual

## Directory Overview

The `include/rad_ml/tmr` directory implements a suite of advanced, radiation-hardened redundancy and protection mechanisms for machine learning and control systems. These are designed for use in extreme environments (e.g., space, high-radiation industrial settings) and are deeply integrated with the rest of the `rad_ml` framework.

### Key Features

- **Multiple TMR Variants:** Standard, Enhanced, Health-Weighted, Stuck-Bit, Approximate, and Hybrid Redundancy.
- **Physics-Driven and Adaptive Protection:** Dynamic adjustment based on real-time environment and mission phase.
- **Temporal Redundancy:** Time-based error detection and correction.
- **Integration with Machine Learning:** Selective hardening, criticality analysis, and neural network protection.
- **Resource Optimization:** Dynamic allocation and power-aware protection.

---

## File-by-File Technical Summary

### `tmr.hpp` (Standard TMR)
- Implements classic Triple Modular Redundancy: three copies, majority voting.
- Lightweight, effective for single-event upsets (SEUs).
- Used as the base for more advanced TMR types.

### `enhanced_tmr.hpp` (Enhanced TMR)
- Adds CRC32 checksums to each copy for silent data corruption detection.
- Supports advanced voting strategies for complex fault patterns (burst, word, byte errors).
- Used for higher reliability in clustered or multi-bit error environments.

### `enhanced_stuck_bit_tmr.hpp` (Enhanced Stuck Bit TMR)
- Detects and mitigates stuck bits (common in flash/space memory).
- Tracks bit-level error consistency, applies mask-aware voting and repair.
- Health scores for each copy, with diagnostics for stuck bits.

### `health_weighted_tmr.hpp` (Health-Weighted TMR)
- Tracks the reliability ("health") of each copy over time.
- Voting is weighted by health scores, allowing the system to prefer more reliable copies.
- Automatically penalizes or rewards copies based on error history.

### `approximate_tmr.hpp` (Approximate TMR)
- Allows resource savings by using reduced-precision or range-limited copies for non-critical data.
- Each copy can use a different approximation strategy (exact, reduced precision, range-limited, or custom).
- Still uses majority voting, but with tolerance for small differences.

### `hybrid_redundancy.hpp` (Hybrid Redundancy)
- Combines spatial (TMR) and temporal redundancy (repeated execution with delays).
- Includes checkpoint/rollback for recovery.
- Dynamically adjusts redundancy parameters based on radiation level.

### `temporal_redundancy.hpp` (Temporal Redundancy)
- Executes operations multiple times, with delays, to detect/correct transient faults.
- Majority voting on results; configurable number of executions and delay.

### `adaptive_protection.hpp` & `adaptive_protection_impl.hpp`
- Implements adaptive protection strategies that adjust TMR level based on:
  - Environmental factors (radiation, temperature, mission phase).
  - Component criticality (from sensitivity analysis).
- Includes `LayerProtectionPolicy` for per-layer neural network protection.
- Factory for selecting optimal TMR strategy at runtime.

### `physics_driven_protection.hpp`
- Integrates NASA/ESA physics models for radiation effects.
- Dynamically adjusts protection based on real-time environment, material properties, and mission phase.
- Multi-scale time protection and resource allocation.
- Example: `PhysicsDrivenProtection` class for neural network layers, with API for updating environment and mission phase.

---

## Integrated Advanced Features (Cross-Validated)

### Machine Learning Integration
- **Selective Hardening:** `include/rad_ml/neural/selective_hardening.hpp` provides criticality analysis and protection assignment for neural network components.
- **Adaptive Protection:** `include/rad_ml/neural/adaptive_protection.hpp` and related files implement runtime adaptation for neural weights, with support for parity, Hamming, Reed-Solomon, and TMR.
- **Layer Protection Policies:** `include/rad_ml/neural/layer_protection_policy.hpp` enables per-layer protection strategies, mission profiles, and resource budgeting.

### Advanced Physics Models
- **Quantum Effects & Material Models:** `include/rad_ml/physics/quantum_field_theory.hpp` and `include/rad_ml/core/material_database.hpp` provide advanced modeling for radiation/material interactions.
- **Synergistic Effect Prediction:** Physics-driven protection classes use these models to adjust redundancy and checkpointing.

### Resource Optimization
- **Dynamic Allocation:** `include/rad_ml/core/redundancy/space_enhanced_tmr.hpp` and `include/rad_ml/core/adaptive/adaptive_framework.hpp` support resource-aware protection, power-aware strategies, and hardware integration.
- **Hardware Acceleration:** `include/rad_ml/hw/hardware_acceleration.hpp` enables hybrid software/hardware TMR, with automatic fallback.

---

## Usage Patterns

### C++ Example: Physics-Driven Protection

```cpp
#include "rad_ml/tmr/physics_driven_protection.hpp"
rad_ml::tmr::PhysicsDrivenProtection protection(material_properties);
protection.updateEnvironment(radiation_environment);
auto result = protection.executeProtected([]{ return compute(); });
```

### C++ Example: Health-Weighted TMR

```cpp
#include "rad_ml/tmr/health_weighted_tmr.hpp"
rad_ml::tmr::HealthWeightedTMR<int> tmr(42);
int value = tmr.get();
tmr.set(43);
tmr.repair();
```

### C++ Example: Approximate TMR

```cpp
#include "rad_ml/tmr/approximate_tmr.hpp"
rad_ml::tmr::ApproximateTMR<float> atmr(
    3.14159f,
    {rad_ml::tmr::ApproximationType::EXACT,
     rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
     rad_ml::tmr::ApproximationType::RANGE_LIMITED}
);
float value = atmr.get();
atmr.set(2.71828f);
atmr.repair();
```

### C++ Example: Hybrid Redundancy

```cpp
#include "rad_ml/tmr/hybrid_redundancy.hpp"
rad_ml::tmr::HybridRedundancy<int> hybrid(42);
int value = hybrid.get();
hybrid.set(43);
hybrid.repair();
```

### C++ Example: Selective Hardening for Neural Networks

```cpp
#include "rad_ml/neural/selective_hardening.hpp"
rad_ml::neural::SelectiveHardening hardening(config);
auto analysis = hardening.analyzeAndProtect(components);
```

---

## Testing and Validation: Best Test Cases

This section highlights real-world validation and demonstration of the TMR and advanced protection mechanisms, using the best and most illustrative tests from the codebase. These tests ensure the reliability, adaptability, and effectiveness of the framework in a variety of challenging scenarios.

### Physics-Driven Protection: Environment Adaptation and Error Recovery

**Test: Environment Adaptation and Error Injection**

```cpp
// Adapted from test/framework_validation_test.cpp
auto aluminum = createAluminumProperties();
tmr::PhysicsDrivenProtection protection(aluminum, 3);

// Test environments
std::vector<std::string> environments = {"NONE", "LEO", "GEO", "SAA", "SOLAR_STORM", "JUPITER"};
for (const auto& env_name : environments) {
    sim::RadiationEnvironment env = createEnvironment(env_name);
    protection.updateEnvironment(env);
    // ... check protection level, checkpoint interval, and factors ...
}

// Error injection and recovery
tmr::PhysicsDrivenProtection protection(aluminum, 1);
protection.updateEnvironment(createEnvironment("JUPITER"));
const int iterations = 1000;
int corrected_count = 0;
for (int i = 0; i < iterations; i++) {
    auto error_prone_op = [&]() -> int {
        if (dist(gen) < 0.3) return -999; // Simulate error
        return 42;
    };
    tmr::TMRResult<int> result = protection.executeProtected<int>(error_prone_op);
    if (result.value == 42) corrected_count++;
}
// Output: High correction rate even with 30% error injection
```

### Health-Weighted TMR: Error Detection and Self-Healing

**Test: Health-Weighted TMR Corruption and Repair**

```cpp
// Adapted from examples/enhanced_features_test/enhanced_features_test.cpp
rad_ml::tmr::HealthWeightedTMR<float> hwt(3.14159f);
std::cout << "Initial value: " << hwt.get() << std::endl;
// Corrupt one copy
*reinterpret_cast<float*>(&hwt) = 2.71828f;
std::cout << "Value after corruption: " << hwt.get() << std::endl;
hwt.repair();
std::cout << "Value after repair: " << hwt.get() << std::endl;
// Output: Detects corruption, repairs to original value
```

### Approximate TMR: Resource-Efficient Protection

**Test: Approximate TMR with Mixed Precision**

```cpp
// Adapted from examples/enhanced_features_test/enhanced_features_test.cpp
rad_ml::tmr::ApproximateTMR<float> atmr(
    3.14159f,
    {rad_ml::tmr::ApproximationType::EXACT,
     rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
     rad_ml::tmr::ApproximationType::RANGE_LIMITED}
);
std::cout << "Initial value: " << atmr.get() << std::endl;
*reinterpret_cast<float*>(&atmr) = 2.71828f;
std::cout << "Value after corruption: " << atmr.get() << std::endl;
atmr.repair();
std::cout << "Value after repair: " << atmr.get() << std::endl;
// Output: Detects and repairs errors, even with mixed-precision copies
```

### Enhanced Stuck Bit TMR: Stuck Bit Detection and Recovery

**Test: Stuck Bit Simulation and Mask-Aware Repair**

```cpp
// Adapted from src/test/enhanced_features_test.cpp
rad_ml::tmr::EnhancedStuckBitTMR<uint32_t> tmr(0x12345678);
const uint32_t stuck_bit_mask = 0x00010001; // Bit 0 and 16 stuck at 1
for (int i = 0; i < 5; i++) {
    uint32_t corrupted = tmr.getCopies()[0] | stuck_bit_mask;
    tmr.corruptCopy(0, corrupted);
    std::cout << "TMR value after corruption: 0x" << std::hex << tmr.get() << std::dec << std::endl;
    tmr.repair();
    std::cout << "TMR value after repair: 0x" << std::hex << tmr.get() << std::dec << std::endl;
}
// Output: Stuck bits are detected and handled, value is restored
```

### Selective Hardening: Neural Network Component Protection

**Test: Selective Hardening with Criticality Analysis**

```cpp
// Adapted from examples/enhanced_features_test/enhanced_features_test.cpp
ProtectedNeuralNetwork nn(4, {8, 6}, 2);
rad_ml::neural::HardeningConfig config = rad_ml::neural::HardeningConfig::defaultConfig();
config.strategy = rad_ml::neural::HardeningStrategy::RESOURCE_CONSTRAINED;
config.resource_budget = 0.3;
nn.applySelectiveHardening(config);
// Output: Prints protection levels assigned to each weight/bias based on criticality
```

---

These tests, along with comprehensive unit and integration tests in the codebase, validate the robustness and adaptability of the TMR framework under a wide range of real-world and extreme conditions.

---

## References

1. NASA Radiation Effects Models
2. ESA JUICE Mission Radiation Mitigation Strategies
3. Triple Modular Redundancy in Space Applications
4. Radiation-Hardened Computing Systems Design

---

**This manual is based on the actual implementation in your `include/rad_ml` library. All features described are present and cross-validated in your codebase. For further expansion, more code samples, or deeper technical details, see the referenced modules or contact the maintainers.**
