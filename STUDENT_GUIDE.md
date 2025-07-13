# Radiation Tolerant ML - Student Guide

This guide provides simple instructions for building and testing the Radiation Tolerant ML framework. The provided scripts and commands are designed to make the development process easier for students.

## Version Information

The current version is **v0.9.7**, which includes enhanced auto architecture search functionality for finding optimal neural network configurations for radiation environments.

## Installation

Before you can build and run the code, you need to install some dependencies:

```bash
./tools/install_dependencies.sh
```

This script will detect your operating system and install the required dependencies (CMake, Eigen3, Boost, and Google Test).

## Quick Start

1. **Build everything**:
   ```bash
   make -f Makefile.simple build
   ```

2. **Run all tests**:
   ```bash
   make -f Makefile.simple test
   ```

3. **Run all examples**:
   ```bash
   make -f Makefile.simple examples
   ```

4. **Clean build artifacts**:
   ```bash
   make -f Makefile.simple clean
   ```

## Available Commands

The simplified build system provides the following commands:

| Command | Description |
|---------|-------------|
| `make -f Makefile.simple` | Build all components (same as `make -f Makefile.simple build`) |
| `make -f Makefile.simple build` | Build all components |
| `make -f Makefile.simple test` | Run all tests |
| `make -f Makefile.simple examples` | Run all examples |
| `make -f Makefile.simple clean` | Clean build artifacts |
| `make -f Makefile.simple help` | Show help information |

## Running Individual Tests

To run a specific test, use:
```bash
make -f Makefile.simple test-NAME
```

For example:
```bash
make -f Makefile.simple test-monte_carlo_validation
```

## Running Individual Examples

To run a specific example, use:
```bash
make -f Makefile.simple example-NAME
```

For example:
```bash
make -f Makefile.simple example-quantum_field_example
```

## Available Tests

These are the main tests available in the framework:

- `monte_carlo_validation`
- `space_monte_carlo_validation`
- `realistic_space_validation`
- `framework_verification_test`
- `enhanced_tmr_test`
- `scientific_validation_test`
- `radiation_stress_test`
- `systematic_fault_test`
- `modern_features_test`
- `quantum_field_validation_test`
- `neural_network_validation`
- `monte_carlo_neuralnetwork`

## Available Examples

These are the main examples available in the framework:

- `quantum_field_example`
- `architecture_test`
- `auto_arch_search_example` (New in v0.9.7 - Test the auto architecture search enhancement)
- `residual_network_test`
- `simple_nn`
- `mission_simulator`

## New in v0.9.7: Auto Architecture Search

The v0.9.7 release adds a powerful auto architecture search capability that helps find optimal neural network architectures for specific radiation environments. To run this feature:

```bash
make -f Makefile.simple example-auto_arch_search_example
```

This will demonstrate:
- Finding optimal neural network configurations for radiation tolerance
- Monte Carlo testing of architectures with various parameters
- Performance comparison between different neural network designs
- Analysis of how architecture impacts radiation tolerance

The results will be saved to CSV files for further analysis.

## Manual Commands

If you prefer to use the scripts directly, you can use:

- Install dependencies: `./tools/install_dependencies.sh`
- Build everything: `./tools/build_all.sh`
- Run all tests: `./tools/run_tests.sh`
- Run specific test: `./tools/run_tests.sh TEST_NAME`
- Clean up: `./tools/clean.sh`

## Project Structure

Here's a brief overview of the project structure:

- `src/` - Source code for the core libraries
- `include/` - Header files
- `examples/` - Example applications
- `test/` - Test files
- `tools/` - Helper scripts and tools
- `build/` - Build directory (created during build)

## Troubleshooting

- If you encounter build errors, try running `make -f Makefile.simple clean` followed by `make -f Makefile.simple build`.
- Make sure you have all required dependencies installed (run `./tools/install_dependencies.sh`).
- If a test is failing, check the output for error messages and report them to your instructor.
- On macOS, you might need to install Xcode command-line tools: `xcode-select --install`

For any other issues, refer to the complete documentation or contact your instructor. 