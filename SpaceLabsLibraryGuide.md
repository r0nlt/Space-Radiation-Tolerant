# SpaceLabs AI Library Guide

This document provides detailed explanations of each library in the Radiation-Tolerant Machine Learning Framework (version 0.9.6). It serves as a technical reference for developers who need to work with specific components of the framework.

## Table of Contents

1. [Core Libraries](#core-libraries)
2. [Triple Modular Redundancy (TMR)](#triple-modular-redundancy-tmr)
3. [Quantum Field Theory (QFT)](#quantum-field-theory-qft)
4. [Neural Network Protection](#neural-network-protection)
5. [Simulation Environments](#simulation-environments)
6. [Memory Protection](#memory-protection)
7. [Error Handling](#error-handling)
8. [Hardware Abstraction Layer](#hardware-abstraction-layer)
9. [Advanced Features](#advanced-features)
10. [Power-Aware Protection](#power-aware-protection)
11. [Mission Simulation](#mission-simulation)
12. [Testing Infrastructure](#testing-infrastructure)
13. [Python Bindings](#python-bindings)

## Core Libraries

### `rad_ml_core`

The core library provides fundamental functionality for the framework.

**Key Components:**
- Framework initialization and shutdown
- Memory management
- Logging and diagnostics
- Configuration management

**Important Classes:**
- `rad_ml::Version`: Framework version information
- `rad_ml::memory::UnifiedMemoryManager`: Central memory management
- `rad_ml::error::ErrorHandler`: Error reporting and handling

**Usage Example:**
```cpp
#include "rad_ml/api/rad_ml.hpp"

using namespace rad_ml;

int main() {
    // Initialize the framework
    initialize(true, memory::MemoryProtectionLevel::BASIC);
    
    // Use framework components
    // ...
    
    // Shutdown the framework
    shutdown();
    
    return 0;
}
```

### `rad_ml_api`

Provides the public interfaces for the framework.

**Key Components:**
- Protection API
- Environment API
- Configuration API

**Important Classes:**
- `rad_ml::api::ProtectionInterface`: Base interface for protection strategies
- `rad_ml::api::EnvironmentInterface`: Interface for radiation environments
- `rad_ml::api::ConfigurationManager`: Framework configuration

## Triple Modular Redundancy (TMR)

### `rad_ml_tmr`

Implements various TMR strategies for protecting data and computations.

**Key Components:**
- Basic TMR implementation
- Enhanced TMR with error detection
- Specialized TMR variants for different types of data

**Important Classes:**
- `rad_ml::tmr::TMR<T>`: Basic TMR implementation
- `rad_ml::tmr::EnhancedTMR<T>`: Advanced TMR with enhanced error detection
- `rad_ml::tmr::HealthWeightedTMR<T>`: TMR with component health tracking
- `rad_ml::tmr::HybridRedundancy<T>`: Combined spatial and temporal redundancy

**Usage Example:**
```cpp
#include "rad_ml/tmr/enhanced_tmr.hpp"

// Create an EnhancedTMR instance to protect a floating-point value
auto protected_value = rad_ml::tmr::TMRFactory::createEnhancedTMR<float>(1.0f);

// Read the protected value
float value = protected_value->read();

// Write a new value
protected_value->write(2.0f);

// Get error statistics
auto stats = protected_value->getErrorStats();
std::cout << "Errors detected: " << stats.detected << std::endl;
std::cout << "Errors corrected: " << stats.corrected << std::endl;
```

### `rad_ml::tmr::PhysicsDrivenProtection`

Physics-based TMR that adapts to the radiation environment.

**Key Features:**
- Environment-aware protection
- Physics-driven adaptation
- Material-specific protection

**Important Methods:**
- `setRadiationEnvironment()`: Configure the radiation environment
- `executeProtected<T>()`: Execute a function with TMR protection
- `updateMaterialProperties()`: Update the material properties for protection

## Quantum Field Theory (QFT)

### `rad_ml_qft`

Implements quantum field theory models for radiation effects.

**Key Components:**
- Quantum models for particle interactions
- Radiation effects simulation at quantum scale
- Enhanced error prediction

**Important Classes:**
- `rad_ml::physics::QuantumFieldModel`: QFT-based radiation model
- `rad_ml::physics::ParticleInteraction`: Particle interaction simulation
- `rad_ml::physics::DefectDistribution`: Quantum defect modeling

**Usage Example:**
```cpp
#include "rad_ml/physics/quantum_field_theory.hpp"
#include "rad_ml/physics/particle_simulation.hpp"

// Create a simulation environment
auto env = rad_ml::simulation::createRadiationEnvironment(
    rad_ml::sim::RadiationEnvironment::MARS, 0.7);

// Configure quantum model
rad_ml::physics::QuantumFieldModel qft_model;
qft_model.setEnvironment(env);

// Run quantum simulation
auto results = qft_model.runMonteCarloSimulation(
    1000000,  // Number of particles
    rad_ml::physics::ParticleType::PROTON
);

// Get probability distributions
auto seu_distribution = results.getSEUDistribution();
```

### `rad_ml::physics::QuantumFieldTheory`

Base class for quantum field calculations.

**Key Features:**
- Field theory simulation
- Quantum state evolution
- Defect formation models

**Important Methods:**
- `calculateQuantumTunnelingProbability()`: Calculate quantum tunneling effects
- `applyQuantumFieldCorrections()`: Apply QFT corrections to classical models
- `calculateZeroPointEnergyContribution()`: Add zero-point energy effects

## Neural Network Protection

### `rad_ml_neural`

Specialized protection for neural networks and machine learning models.

**Key Components:**
- Weight protection
- Activation function protection
- Training resilience
- Selective hardening

**Important Classes:**
- `rad_ml::neural::ProtectedNeuralNetwork<T>`: Radiation-hardened neural network
- `rad_ml::neural::SelectiveHardening`: Selective protection based on criticality
- `rad_ml::neural::AdvancedReedSolomon<T>`: Reed-Solomon codes for weight protection
- `rad_ml::neural::AdaptiveProtection`: Environment-adaptive protection

**Usage Example:**
```cpp
#include "rad_ml/neural/protected_neural_network.hpp"

// Create a protected neural network with 3 layers
std::vector<size_t> layer_sizes = {784, 128, 10};
rad_ml::neural::ProtectedNeuralNetwork<float> network(
    layer_sizes, 
    rad_ml::neural::ProtectionLevel::ADAPTIVE_TMR
);

// Forward pass with protection
std::vector<float> input = /* input data */;
auto output = network.forward(input);
```

### `rad_ml::neural::AdaptiveProtection`

Adapts protection strategies based on criticality and environment.

**Key Features:**
- Criticality-based protection
- Environment adaptation
- Resource-constrained optimization

**Important Methods:**
- `protectValue()`: Protect a value based on current strategy
- `recoverValue()`: Recover a value with error correction
- `setRadiationEnvironment()`: Update the radiation environment
- `optimizeProtection()`: Optimize protection for resource constraints

## Simulation Environments

### `rad_ml_simulation`

Simulates various space radiation environments.

**Key Components:**
- Environment definitions (LEO, GEO, Mars, Jupiter)
- Radiation model parameters
- Mission profiles

**Important Classes:**
- `rad_ml::sim::RadiationEnvironment`: Radiation environment parameters
- `rad_ml::sim::PhysicsRadiationSimulator`: Physics-based simulator
- `rad_ml::sim::MissionParameters`: Space mission configuration

**Usage Example:**
```cpp
#include "rad_ml/sim/mission_environment.hpp"

// Create a radiation environment for Mars
auto mars_env = rad_ml::simulation::createRadiationEnvironment(
    rad_ml::sim::RadiationEnvironment::MARS, 0.8);

// Configure environment parameters
mars_env.solar_activity = 120; // Solar activity level
mars_env.shielding_thickness_mm = 2.5; // 2.5mm aluminum shielding

// Calculate radiation rates
auto rates = mars_env.calculateRadiationRates();
std::cout << "SEU rate: " << rates.seu_rate << " events/s" << std::endl;
```

### `rad_ml::testing::MissionSimulator`

Simulates a complete space mission with radiation effects.

**Key Features:**
- Complete mission timeline
- Environmental changes simulation
- Radiation event generation
- Protection effectiveness assessment

**Important Methods:**
- `configureMission()`: Set up mission parameters
- `runSimulation()`: Run the full mission simulation
- `getStatistics()`: Get mission performance statistics
- `exportResults()`: Export simulation results

## Memory Protection

### `rad_ml_memory`

Memory protection and management utilities.

**Key Components:**
- Protected memory allocators
- Memory scrubbing
- Error detection and correction
- Checkpoint/restore mechanisms

**Important Classes:**
- `rad_ml::memory::RadiationMappedAllocator<T>`: Radiation-aware allocator
- `rad_ml::memory::MemoryScrubber`: Periodic memory scrubbing
- `rad_ml::memory::CheckpointManager`: Memory state checkpointing

**Usage Example:**
```cpp
#include "rad_ml/memory/radiation_mapped_allocator.hpp"

// Create a vector using radiation-tolerant allocator
std::vector<float, rad_ml::memory::radiation_tolerant_allocator<float>> 
    protected_vector(1000);

// Fill the vector
for (size_t i = 0; i < protected_vector.size(); ++i) {
    protected_vector[i] = static_cast<float>(i);
}

// Memory is automatically protected
```

### `rad_ml::core::memory::UnifiedMemoryManager`

Central memory management for the framework.

**Key Features:**
- Memory registration and tracking
- Protection level management
- Memory usage statistics
- Leak detection

**Important Methods:**
- `registerMemory()`: Register memory for protection
- `unregisterMemory()`: Remove memory from protection
- `setProtectionLevel()`: Configure memory protection level
- `getMemoryStats()`: Get memory usage statistics

## Error Handling

### `rad_ml_error`

Error detection, reporting, and handling.

**Key Components:**
- Error types and severity levels
- Error handlers and callbacks
- Logging and reporting

**Important Classes:**
- `rad_ml::error::ErrorHandler`: Central error handling
- `rad_ml::error::ErrorCallback`: Custom error callbacks
- `rad_ml::error::ErrorInfo`: Detailed error information

**Usage Example:**
```cpp
#include "rad_ml/error/error_handling.hpp"

// Set up an error callback
rad_ml::error::ErrorHandler::setCallback([](const rad_ml::error::ErrorInfo& info) {
    std::cerr << "Error detected: " << info.message << std::endl;
    std::cerr << "Severity: " << static_cast<int>(info.severity) << std::endl;
    std::cerr << "Location: " << info.file << ":" << info.line << std::endl;
});

// Report an error
rad_ml::error::ErrorHandler::reportError(
    rad_ml::error::ErrorSeverity::WARNING,
    "Potential bit flip detected",
    __FILE__,
    __LINE__
);
```

### `rad_ml::error::ErrorCorrectionPolicy`

Base class for custom error correction policies.

**Key Features:**
- Policy-based error correction
- Customizable correction strategies
- Error assessment and prioritization

**Important Methods:**
- `shouldCorrect()`: Determine if an error should be corrected
- `correctError()`: Apply error correction
- `getPriority()`: Get correction priority
- `getScrubInterval()`: Get recommended scrubbing interval

## Hardware Abstraction Layer

### `rad_ml_hw`

Hardware abstraction layer for the framework.

**Key Components:**
- Hardware acceleration interfaces
- Platform-specific optimizations
- Device drivers

**Important Classes:**
- `rad_ml::hw::HardwareAccelerator`: Interface for hardware acceleration
- `rad_ml::hw::HardwareRegistry`: Registry for hardware components
- `rad_ml::hw::PlatformOptimization`: Platform-specific optimizations

**Usage Example:**
```cpp
#include "rad_ml/hw/hardware_accelerator.hpp"

// Check if hardware acceleration is available
if (rad_ml::hw::HardwareRegistry::hasAccelerator("ecc_generator")) {
    // Get the accelerator
    auto accelerator = rad_ml::hw::HardwareRegistry::getAccelerator("ecc_generator");
    
    // Use hardware acceleration
    std::vector<uint8_t> data = /* data to protect */;
    std::vector<uint8_t> ecc(16);
    
    accelerator->accelerateECC(
        data.data(), data.size(),
        ecc.data(), ecc.size()
    );
}
```

### `rad_ml::hw::HardwareRegistry`

Registry for hardware components and accelerators.

**Key Features:**
- Component discovery and registration
- Hardware capability querying
- Dynamic accelerator selection

**Important Methods:**
- `registerHardware()`: Register a hardware component
- `hasAccelerator()`: Check for hardware acceleration
- `getAccelerator()`: Get a hardware accelerator
- `getCapabilities()`: Query hardware capabilities

## Advanced Features

### `rad_ml_advanced`

Advanced features and experimental capabilities.

**Key Components:**
- Algorithmic diversity
- Error prediction
- Adaptive strategies
- Power-aware protection

**Important Classes:**
- `rad_ml::advanced::AlgorithmicDiversity`: Diverse algorithm implementation
- `rad_ml::advanced::ErrorPredictor`: Machine learning-based error prediction
- `rad_ml::advanced::AdaptiveStrategy`: Self-adapting protection strategies

**Usage Example:**
```cpp
#include "rad_ml/advanced/algorithmic_diversity.hpp"

// Create a diverse algorithm implementation
rad_ml::advanced::AlgorithmicDiversity<float> div_algorithm;

// Add multiple implementations of the same algorithm
div_algorithm.addImplementation("baseline", [](float x, float y) {
    return x * y;
});

div_algorithm.addImplementation("alternative", [](float x, float y) {
    return std::exp(std::log(x) + std::log(y));
});

// Execute with diversity
float result = div_algorithm.execute(3.14f, 2.71f);
```

### `rad_ml::power::PowerAwareProtection`

Power-optimized protection strategies.

**Key Features:**
- Power consumption modeling
- Dynamic power-reliability tradeoffs
- Adaptive protection based on power budget

**Important Methods:**
- `setPowerBudget()`: Set available power budget
- `optimizeProtection()`: Optimize protection for power constraints
- `getPowerConsumption()`: Get estimated power consumption
- `adaptToEnvironment()`: Adapt protection based on environment

## Power-Aware Protection

### `rad_ml::power::PowerAwareProtection`

Comprehensive power-optimized protection strategies designed specifically for resource-constrained space applications.

**Key Features:**
- Dynamic power consumption modeling and optimization
- Adaptive protection based on real-time power budget constraints
- Fine-grained power-reliability tradeoff management
- Mission phase-aware protection scaling
- Hardware-specific power profiling

**Important Methods:**
- `setPowerBudget(double budget_mW)`: Set available power budget in milliwatts
- `optimizeProtection(OptimizationGoal goal)`: Optimize protection based on goal (MAX_RELIABILITY, MIN_POWER, BALANCED)
- `getPowerConsumption()`: Get detailed power consumption breakdown by protection subsystem
- `adaptToEnvironment(RadiationEnvironment env)`: Dynamically adapt power allocation based on environment
- `setDeviceThermalModel(const ThermalModel& model)`: Configure device-specific thermal characteristics

**Usage Example:**
```cpp
#include "rad_ml/power/power_aware_protection.hpp"

// Create a power-aware protection manager
rad_ml::power::PowerAwareProtection pap;

// Set available power budget (10mW for protection)
pap.setPowerBudget(10.0);  // milliwatts

// Set device thermal characteristics
rad_ml::power::ThermalModel thermal_model;
thermal_model.max_operating_temp = 85.0;  // Celsius
thermal_model.thermal_resistance = 10.0;  // C/W
thermal_model.active_cooling = false;
pap.setDeviceThermalModel(thermal_model);

// Configure adaptive strategy
rad_ml::power::AdaptiveConfig config;
config.optimization_goal = rad_ml::power::OptimizationGoal::BALANCED;
config.min_reliability = 0.9999;  // 4-nines reliability minimum
config.critical_subsystems = {"memory", "neural_inference"};
pap.configureAdaptiveStrategy(config);

// At regular intervals or based on events
pap.adaptToEnvironment(rad_ml::sim::RadiationEnvironment::MARS);

// Get power consumption stats
auto power_stats = pap.getPowerConsumption();
std::cout << "Total power usage: " << power_stats.total_mW << " mW" << std::endl;
std::cout << "TMR subsystem: " << power_stats.by_subsystem["tmr"] << " mW" << std::endl;
std::cout << "ECC subsystem: " << power_stats.by_subsystem["ecc"] << " mW" << std::endl;
```

### `rad_ml::power::DynamicVoltageScaling`

Implements dynamic voltage and frequency scaling (DVFS) techniques for protection circuitry.

**Key Features:**
- Voltage-dependent bit error rate modeling
- Frequency-dependent computing capability modeling
- Energy-optimal protection circuit adaptation
- Fine-grained control of Vdd for different protection domains

**Important Methods:**
- `setVoltageRange(double min_v, double max_v)`: Set operating voltage range
- `setFrequencyRange(double min_hz, double max_hz)`: Set operating frequency range
- `getOptimalOperatingPoint(double error_tolerance)`: Calculate optimal V/F for given error tolerance
- `applyConfiguration(const DVFSConfig& config)`: Apply specific voltage/frequency settings

### `rad_ml::power::PowerConsumptionModel`

Sophisticated power consumption modeling for radiation-hardened components.

**Key Features:**
- Detailed power modeling for TMR registers, combinatorial logic, and memory
- Device-specific calibration capabilities
- Temperature-dependent leakage current models
- Activity-based dynamic power estimation

**Important Methods:**
- `calibrateWithRealDevice(const CalibrationData& data)`: Calibrate model with real device measurements
- `estimatePower(const ProtectionConfiguration& config)`: Estimate power for a protection configuration
- `analyzePowerBreakdown()`: Get detailed power breakdown by component type
- `estimateBatteryLifetime(double battery_capacity_mWh)`: Estimate battery lifetime

## Mission Simulation

### `rad_ml::mission::MissionProfile`

Comprehensive mission profile modeling for space applications.

**Key Features:**
- Detailed mission timeline with phase-specific radiation environments
- Orbital mechanics and trajectory modeling
- Multi-phase mission planning with varying protection requirements
- Failure probability calculations across mission phases

**Important Classes:**
- `rad_ml::mission::MissionPhase`: Individual mission phase configuration
- `rad_ml::mission::TrajectoryModel`: Orbital or interplanetary trajectory modeling
- `rad_ml::mission::ResourceBudget`: Power, memory, and compute budgets by phase
- `rad_ml::mission::RadioactiveSourceModel`: RTG or external radiation source modeling

**Usage Example:**
```cpp
#include "rad_ml/mission/mission_profile.hpp"

// Create a Mars mission profile
auto mars_mission = rad_ml::mission::MissionProfileFactory::createMarsMission();

// Add custom mission phases
mars_mission->addPhase(rad_ml::mission::MissionPhase{
    .name = "deep_space_cruise",
    .duration_days = 180,
    .environment = rad_ml::sim::RadiationEnvironment::DEEP_SPACE,
    .shielding_mm_al_eq = 10.0,
    .critical_systems = {"guidance_computer", "life_support"},
    .power_budget_mW = 5000
});

// Add landing phase with different requirements
mars_mission->addPhase(rad_ml::mission::MissionPhase{
    .name = "mars_entry_descent_landing",
    .duration_hours = 7,
    .environment = rad_ml::sim::RadiationEnvironment::MARS_ATMOSPHERE,
    .shielding_mm_al_eq = 8.0,
    .critical_systems = {"all"},
    .power_budget_mW = 25000,
    .required_reliability = 0.99999  // Five nines for EDL
});

// Run mission simulation
auto sim_results = mars_mission->simulate(
    1000,  // Monte Carlo iterations
    rad_ml::testing::FaultModelLibrary::getSolarParticleEvent(200)  // Add solar event
);

// Analyze results
std::cout << "Mission success probability: " << sim_results.success_probability << std::endl;
std::cout << "Critical failure points:" << std::endl;
for (const auto& failure : sim_results.failure_points) {
    std::cout << " - " << failure.phase_name << ": " << failure.probability << std::endl;
}
```

### `rad_ml::testing::MissionSimulator`

High-fidelity mission simulation with radiation environment effects.

**Key Features:**
- Full-system simulation of radiation effects on spacecraft electronics
- Monte Carlo analysis for mission success probability
- Component-level fault injection based on physics models
- Detailed telemetry recording and analysis
- Validation against real space mission data

**Important Methods:**
- `configureMission(const MissionProfile& profile)`: Configure mission parameters
- `setRadiationModels(const std::vector<RadiationModel>& models)`: Set radiation environment models
- `runSimulation(int iterations = 1000)`: Run Monte Carlo simulation
- `getDetailedResults()`: Get comprehensive simulation results
- `exportTelemetry(const std::string& filename)`: Export telemetry data

### `rad_ml::mission::ValidationFramework`

Framework for validating mission-critical algorithms against radiation effects.

**Key Features:**
- NASA/ESA standards compliance testing
- Automated validation of protection strategies
- Statistical confidence level reporting
- Mission-specific validation criteria

**Important Methods:**
- `validateAlgorithm(const Algorithm& algo, ValidationCriteria criteria)`: Validate an algorithm
- `compareProtectionStrategies(const std::vector<ProtectionStrategy>& strategies)`: Compare strategies
- `generateComplianceReport(ComplianceStandard standard)`: Generate standard compliance report

## Testing Infrastructure

### `rad_ml::testing::FaultInjector`

Sophisticated fault injection framework for comprehensive testing.

**Key Features:**
- Bit-flip, stuck-at-fault, and timing error injection
- Deterministic and stochastic fault models
- Hardware-accelerated fault injection for large-scale testing
- Realistic space radiation fault patterns
- Custom fault distribution modeling

**Important Classes:**
- `rad_ml::testing::SEUModel`: Single Event Upset model
- `rad_ml::testing::MBUModel`: Multiple Bit Upset model
- `rad_ml::testing::SELModel`: Single Event Latchup model
- `rad_ml::testing::TimingFaultModel`: Timing fault model

**Usage Example:**
```cpp
#include "rad_ml/testing/fault_injector.hpp"

// Create a fault injector with specific models
rad_ml::testing::FaultInjector injector;

// Configure SEU model based on LEO environment
auto seu_model = rad_ml::testing::SEUModelFactory::createForEnvironment(
    rad_ml::sim::RadiationEnvironment::LEO,
    0.8  // Solar activity factor (0.0-1.0)
);
injector.addFaultModel(seu_model);

// Add MBU model with custom parameters
auto mbu_model = std::make_shared<rad_ml::testing::MBUModel>();
mbu_model->setMultiplicity(4);  // 4-bit upsets
mbu_model->setGeometricDistribution(0.7);  // Adjacent bits more likely
mbu_model->setProbability(0.05);  // 5% of all upsets are MBUs
injector.addFaultModel(mbu_model);

// Create a target vector for testing
std::vector<float> test_data(1000, 3.14159f);

// Inject faults based on models
injector.injectFaults(test_data.data(), test_data.size() * sizeof(float));

// Analyze bit error statistics
auto error_stats = injector.getErrorStatistics();
std::cout << "Total bit flips: " << error_stats.total_bit_flips << std::endl;
std::cout << "MBUs: " << error_stats.multiple_bit_upsets << std::endl;
```

### `rad_ml::testing::BenchmarkFramework`

Comprehensive benchmarking suite for radiation-tolerant implementations.

**Key Features:**
- Standardized performance metrics for radiation-hardened algorithms
- Power consumption measurement infrastructure
- Resource utilization benchmarking (memory, CPU, accelerator)
- Cross-platform benchmark compatibility

**Important Methods:**
- `registerBenchmark(const std::string& name, BenchmarkFn fn)`: Register a benchmark function
- `runAll(BenchmarkConfig config)`: Run all registered benchmarks
- `compareImplementations(const std::vector<std::string>& impls)`: Compare different implementations
- `generateReport(ReportFormat format)`: Generate benchmark report

### `rad_ml::testing::ValidationSuite`

Industry-standard validation suite for radiation-tolerant systems.

**Key Features:**
- Automated validation against NASA/ESA radiation standards
- Comprehensive test scenarios based on real missions
- Statistical validation with confidence intervals
- Compliance reporting for certification

**Important Methods:**
- `configureValidation(ValidationConfig config)`: Configure validation parameters
- `validateComponent(const Component& component)`: Validate a specific component
- `runFullSuite()`: Run the complete validation suite
- `generateComplianceReport()`: Generate certification-ready report

**Usage Example:**
```cpp
#include "rad_ml/testing/validation_suite.hpp"

// Create validation suite with NASA standards
auto validation = rad_ml::testing::ValidationSuiteFactory::createNASACompliant();

// Configure for specific mission class
validation.configureValidation({
    .mission_class = rad_ml::testing::MissionClass::CLASS_B,
    .radiation_environment = rad_ml::sim::RadiationEnvironment::JUPITER,
    .mission_duration_years = 6.0,
    .confidence_level = 0.99,
    .enable_accelerated_testing = true
});

// Define component under test
rad_ml::testing::Component neural_inference_engine{
    .name = "Vision Neural Engine",
    .protection_strategy = "selective_tmr_with_ecc_memory",
    .critical_level = rad_ml::testing::CriticalityLevel::MISSION_CRITICAL
};

// Run validation
auto results = validation.validateComponent(neural_inference_engine);

// Check results
if (results.passed) {
    std::cout << "Validation passed with confidence level: " 
              << results.confidence_level << std::endl;
} else {
    std::cout << "Validation failed. Issues found:" << std::endl;
    for (const auto& issue : results.issues) {
        std::cout << " - " << issue.description 
                  << " (severity: " << issue.severity << ")" << std::endl;
    }
}
```

## Python Bindings

### `rad_ml` Python Module

Python bindings for the rad_ml framework.

**Key Components:**
- Python interfaces to C++ functionality
- Numpy integration
- Easy-to-use protection wrappers

**Important Classes:**
- `rad_ml.AdaptiveProtection`: Python wrapper for adaptive protection
- `rad_ml.StandardTMR`: Python TMR implementation
- `rad_ml.RadiationEnvironment`: Environment enumeration
- `rad_ml.SelectiveHardening`: Neural network hardening

**Usage Example:**
```python
import rad_ml
import numpy as np

# Initialize the framework
rad_ml.initialize()

# Create a protection object
protection = rad_ml.AdaptiveProtection()

# Set environment
protection.set_environment(rad_ml.RadiationEnvironment.LEO)

# Protect a numpy array
weights = np.array([0.1, 0.2, 0.3, 0.4])
protected = protection.protect_array(weights)

# Use in inference and recover later
recovered = protection.recover_array(protected)

# Shutdown the framework
rad_ml.shutdown()
```

### `rad_ml.tmr` Python Module

Python implementation of Triple Modular Redundancy.

**Key Classes:**
- `rad_ml.tmr.StandardTMR`: Basic TMR implementation
- `rad_ml.tmr.EnhancedTMR`: Enhanced TMR with extra features

**Important Methods:**
- `protect()`: Protect a value with TMR
- `recover()`: Recover a value with error correction
- `get_stats()`: Get error statistics
- `set_environment()`: Configure the radiation environment

---

This guide provides a comprehensive overview of the libraries in the Radiation-Tolerant Machine Learning Framework. For more detailed information on specific components, refer to the API documentation in the `docs/` directory. 