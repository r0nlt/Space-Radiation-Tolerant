# Radiation-Tolerant Machine Learning Framework

**Author:** Rishab Nuguru  
**Copyright:** © 2025 Rishab Nuguru  
**License:** MIT  
**Repository:** https://github.com/r0nlt/rad-tolerant-ml

A C++ framework for implementing machine learning models that can operate reliably in radiation environments, such as space.

## Features

- Zero dynamic memory allocation
- Triple Modular Redundancy (TMR) for fault tolerance
- Memory scrubbing and error correction
- Fixed-point arithmetic for deterministic execution
- Branchless operations for predictable code paths
- Optimized for low-latency inference
- Physics-based radiation simulation
- Adaptive protection levels based on radiation environment
- Mission profiles for different space environments
- Enhanced error detection with CRC checksums
- Full mission simulation with realistic radiation events
- Health-weighted voting for improved resilience

## Getting Started

### Prerequisites

- Modern C++ compiler with C++17 support
- CMake 3.15 or higher

### Building

```bash
mkdir build && cd build
cmake ..
make
```

### Running Tests

```bash
make test
```

### Running the Stress Test

To evaluate the framework's resilience under extreme radiation conditions:

```bash
# Run with default 5-minute duration
./stress_test

# Or specify a custom duration in seconds
./stress_test 120  # Run for 2 minutes
```

The stress test simulates extreme radiation environments beyond what would be encountered in real space missions. It provides detailed statistics on radiation events, error detection, correction rates, and overall system resilience.

### Running the Mission Simulator

The mission simulator demonstrates the adaptive radiation protection system and physics-based radiation simulation:

```bash
# Run the mission simulation test
./mission_simulation_test
```

Available mission profiles:
- ISS (International Space Station)
- ARTEMIS_I (Lunar mission with Van Allen belt transit)
- MARS_SCIENCE_LAB (Mars mission with deep space transit)
- VAN_ALLEN_PROBES (Radiation belt study with high radiation exposure)
- EUROPA_CLIPPER (Extreme radiation environment near Jupiter/Europa)

## Project Structure

- `include/rad_ml/`: Public headers
  - `tmr/`: Triple Modular Redundancy implementations
  - `memory/`: Memory protection implementations
  - `sim/`: Radiation simulation and testing tools
  - `power/`: Power-aware protection systems
  - `hw/`: Hardware acceleration interfaces
- `src/`: Implementation files
- `examples/`: Example applications
- `test/`: Unit and integration tests
- `docs/`: Documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA's radiation effects research
- ESA's fault-tolerant computing guidelines 


## Progress

### Foundation Layer (Completed):
- **Core Framework Components**:
  - Memory management (static_allocator.hpp)
  - Fault tolerance (tmr.hpp)
  - Error detection/correction (memory_scrubbing.hpp)
  - Deterministic math (fixed_point.hpp, branchless_ops.hpp)
- **Example Implementation**:
  - A simple neural network demonstrating TMR and fixed-point arithmetic
- **Testing Infrastructure**:
  - Basic unit tests for TMR implementation
- **Build System**:
  - CMake configuration for cross-platform building
- **Development Infrastructure**:
  - Code formatting (.clang-format)
  - Static analysis (.clang-tidy)
  - Editor configuration (.editorconfig)
  - Build presets (CMakePresets.json)
  - IDE settings (.cursor/settings.json)
  - Git hooks for code quality

### Neural Network Components (Completed):
- **Advanced Memory Protection**:
  - Robust memory scrubber implementation with background thread
  - Memory region registration and automated repair
- **Fault Injection Framework**:
  - Comprehensive fault simulation (bit flips, stuck bits, random values)
  - Component resilience testing capabilities
  - Statistical fault reporting
- **Neural Network Layers**:
  - Radiation-hardened convolution layer with TMR protection
  - Fixed-point computation for deterministic results
- **Model Interface**:
  - Abstract base class for radiation-tolerant models
  - Automatic memory scrubbing integration
  - Health monitoring capabilities

### Advanced Radiation Tolerance (Completed):
- **Enhanced TMR Implementation**:
  - CRC-32 checksum validation for stronger error detection
  - Health monitoring of redundant copies
  - Health-weighted voting for improved error correction
  - Adaptive regeneration of corrupted copies
  - Comprehensive error statistics and diagnostics
- **Physics-Based Radiation Simulation**:
  - Realistic modeling of different radiation environments (LEO, GEO, Deep Space)
  - Simulation of various radiation effects (SEUs, MBUs, SELs, SETs)
  - Configurable mission parameters (altitude, shielding, solar activity)
- **Mission Simulation System**:
  - Full mission profile simulation with physics-based radiation modeling
  - Pre-configured settings for common space missions
  - Dynamic environment changes including South Atlantic Anomaly and solar flares
  - Adaptive protection based on radiation environment
  - Detailed mission statistics with radiation event tracking
  - Resource usage and energy consumption monitoring

### Enhanced Features (Completed):
- **Health-Weighted TMR**:
  - Dynamic health score tracking for each redundant copy
  - Weighted voting based on historical reliability
  - Penalty and reward system for error frequency
  - Improved performance in radiation environments
- **Approximate TMR**:
  - Support for different approximation strategies
  - EXACT, REDUCED_PRECISION, and RANGE_LIMITED strategies
  - Custom approximation function capabilities
  - Improved size and power efficiency for non-critical data
- **Selective Hardening**:
  - Analysis of neural network components for criticality
  - Resource-constrained hardening strategy
  - Layerwise importance assessment
  - Gradient-based sensitivity analysis
  - Adaptive runtime protection

### Testing Framework (Completed):
- **Unit Tests**:
  - TMR functionality verification
  - Memory scrubber effectiveness tests
  - Fault injector validation
  - Convolutional layer correctness tests
  - Resilience to faults validation
- **Integration Tests**:
  - End-to-end mission simulation
  - Radiation environment simulation accuracy
  - Adaptive protection effectiveness
  - Statistical error correction verification
- **Benchmarking Framework**:
  - Performance and reliability metrics collection
  - Configurable test scenarios for different missions
  - Comparative analysis of protection strategies
  - Resource utilization tracking
  - Detailed reports with visualization capabilities
- **Extreme Radiation Stress Testing**:
  - Beyond-Jupiter radiation environment simulation
  - Large-scale testing with 100,000+ radiation events
  - Comparative analysis of TMR implementations
  - Per-error-type performance analysis
  - Long-duration mission simulation

## Recent Test Results

The framework has been subjected to rigorous testing using a comprehensive mission simulation that models real-world space radiation environments. The tests were conducted across five different mission profiles:

1. **ISS Mission (Low Earth Orbit)**
   - Moderate radiation with South Atlantic Anomaly passes
   - Average accuracy: 30%
   - Protection efficiency needs improvement

2. **Artemis I Mission (Lunar)**
   - Van Allen belt transit and lunar environment
   - Average accuracy: 30%
   - Showed vulnerability during high-radiation belt crossings

3. **Mars Science Laboratory Mission**
   - Interplanetary space and solar event simulation
   - Average accuracy: 20%
   - Lower power usage (10.87W) but still poor performance

4. **Van Allen Probes Mission**
   - Extended radiation belt exposure
   - Average accuracy: 30%
   - Framework showed limitations in high-flux environments

5. **Europa Clipper Mission**
   - Extreme Jupiter/Europa radiation environment
   - Average accuracy: 28.3%
   - Highest radiation scenario with billions of expected bit flips

These results indicate that while the framework has a solid theoretical foundation and architecture, there are significant opportunities for improvement in its implementation to handle real-world radiation environments effectively.

## Next Steps (Based on Test Results):

1. **Improve TMR Implementation**
   - Implement more sophisticated voting mechanisms that consider bit history patterns
   - Add checkpoint/rollback capabilities to restore from known good states when multiple errors are detected
   - Implement a "confidence score" for each TMR unit to give less weight to modules with detected stuck bits

2. **Enhance Memory Protection**
   - Implement more aggressive scrubbing techniques that periodically verify and correct memory contents
   - Add error-correcting codes (ECC) for critical memory regions
   - Implement block-level redundancy for highly critical data

3. **Optimize Power-Aware Protection**
   - Implement dynamic adjustment of protection levels based on detected error rates
   - Add more granular power states with corresponding protection levels
   - Implement predictive protection that increases protection before entering known high-radiation environments

4. **Improve Extreme Radiation Handling**
   - Add special handling for extreme radiation environments with stronger redundancy
   - Implement algorithm-based fault tolerance (ABFT) techniques specifically for neural networks
   - Consider selective hardening where critical neurons have higher protection levels

5. **Add Adaptive Recovery Mechanisms**
   - Implement learning-based error detection that adapts to specific bit error patterns
   - Add graceful degradation modes that sacrifice non-critical functions to maintain core functionality
   - Implement radiation-aware scheduling that postpones critical computations during high radiation periods

6. **Specialized Hardware Integration**
   - Better integrate with radiation-hardened hardware accelerators
   - Implement hybrid approaches that combine software TMR with hardware protection
   - Add support for partial reconfiguration to isolate damaged components

## Verification Results

The framework has been tested using mission simulations that model realistic radiation environments based on actual mission data. These simulations have revealed:

1. **Radiation Environment Modeling**: The physics-based radiation simulator accurately models different space environments from LEO to Europa, with radiation rates matching published scientific data.

2. **TMR Effectiveness**: Current TMR implementation shows limited effectiveness against stuck bits and multiple bit upsets, with accuracy ranging from 20-30% in high-radiation environments.

3. **Power-Aware Protection**: The power management system successfully adjusts power usage based on mission phase, but protection efficiency needs improvement.

4. **Memory Protection**: The radiation-mapped memory allocator correctly categorizes data by criticality, but more robust protection mechanisms are needed.

5. **Overall Framework Resilience**: Current implementation needs significant improvements to achieve the reliability required for actual space missions, particularly for extreme radiation environments like Europa.

To run the full mission simulation:

```bash
# Compile the mission simulation test
mkdir build && cd build
cmake ..
make mission_simulation_test

# Run the simulation
./mission_simulation_test
```