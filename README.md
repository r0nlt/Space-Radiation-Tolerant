# Radiation-Tolerant Machine Learning Framework

A C++ framework for implementing machine learning models that can operate reliably in radiation environments, such as space.

## Features

- Zero dynamic memory allocation
- Triple Modular Redundancy (TMR) for fault tolerance
- Memory scrubbing and error correction
- Fixed-point arithmetic for deterministic execution
- Branchless operations for predictable code paths
- Optimized for low-latency inference
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

## Project Structure

- `include/rad_ml/`: Public headers
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

### Testing Framework (Completed):
- **Unit Tests**:
  - TMR functionality verification
  - Memory scrubber effectiveness tests
  - Fault injector validation
  - Convolutional layer correctness tests
  - Resilience to faults validation

The core architecture is now in place with working implementations of key radiation-tolerance features. All components are integrated with proper test coverage, and all tests are passing. The project provides a solid foundation for building complete radiation-tolerant neural networks.

### Next Steps:
- Implement additional neural network layers (pooling, fully connected)
- Add model conversion tools for TensorFlow/PyTorch compatibility
- Create example applications for space-grade image processing
- Implement more advanced fault tolerance mechanisms
- Add performance optimization for resource-constrained environments