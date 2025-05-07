# Radiation-Tolerant ML Framework Verification Tests

This directory contains verification tests for the radiation-tolerant machine learning framework, designed to evaluate the framework's robustness and error correction capabilities under various conditions.

## Available Tests

### 1. Framework Verification Test
Located in `framework_verification_test.cpp`, this test verifies the basic functionality of the framework components, including protection mechanisms and error handling.

### 2. Scientific Validation Test
Located in `scientific_validation_test.cpp`, this test validates the scientific accuracy of the framework under standard space radiation environments (LEO, GEO, ISS, etc.).

### 3. Radiation Stress Test
Located in `radiation_stress_test.cpp`, this test simulates extreme radiation conditions to evaluate the framework's performance under severe radiation environments.

## Radiation Stress Test

The radiation stress test is specifically designed to evaluate the system under extreme conditions by deliberately introducing severe bit flips that simulate high-intensity radiation environments.

### Key Features

- **Bit-flip Simulation**: The test implements precise bit-level flips in floating-point values, simulating radiation-induced Single Event Upsets (SEUs) and Multiple Bit Upsets (MBUs).
- **Triple Modular Redundancy (TMR)**: Implements a custom neural network with TMR protection that can be toggled on/off to compare performance.
- **Majority Voting**: Includes a bit-level majority voter that can correct errors across the redundant copies.
- **Configurable Radiation Levels**: Tests are performed at multiple radiation intensities (10, 50, and 100 units) to observe degradation patterns.

### Test Methodology

1. **Network Initialization**: Creates a neural network with triple redundant copies of all weights and biases.
2. **Reference Pass**: Performs a forward pass with no radiation to establish baseline outputs.
3. **Radiation Exposure**: Applies bit flips to the network parameters based on radiation intensity.
4. **Protection Evaluation**: Runs tests with protection enabled and disabled to measure effectiveness.
5. **Error Analysis**: Records and analyzes raw errors, corrected errors, and output accuracy (MSE).

### Visualization

The test generates CSV files with detailed results, which can be visualized using the provided Python script:

```bash
cd build
./radiation_stress_test
python fixed_radiation_visualization.py
```

The visualizations are saved in the `radiation_plots` directory and include:
- MSE over samples
- Errors over samples
- MSE distribution
- Error correction effectiveness (for protection-enabled tests)

### Key Findings

1. **Protection Effectiveness**: The TMR protection demonstrates significant error correction capability, with correction rates approaching 100% even under extreme radiation.

2. **Error Accumulation**: Without protection, errors accumulate rapidly as radiation intensity increases, leading to complete system failure at high intensities.

3. **Performance vs. Radiation Intensity**:
   - At low radiation levels (10 units), the protected system maintains low MSE.
   - At medium levels (50 units), some output degradation occurs but remains manageable.
   - At high levels (100 units), even the protected system shows significant stress, but still outperforms the unprotected version.

4. **Error Correction Mechanism**: The bit-level majority voting proves effective at correcting both single and multiple bit errors.

## Running the Tests

To build and run all verification tests:

```bash
cd build
cmake ..
make
./framework_verification_test
./scientific_validation_test
./radiation_stress_test
```

For visualization of results:

```bash
cd build
python fixed_radiation_visualization.py
```

## Overview

The verification tests in this directory are designed to demonstrate and validate the framework's capabilities in protecting machine learning applications from radiation-induced errors. These tests serve both as examples of how to use the framework's protection mechanisms and as scientific validation tools.

## For Scientific Researchers

### Understanding Validation Metrics

1. **Success Rate**: Percentage of trials where the framework maintained accuracy within acceptable bounds
2. **Error Reduction**: Percentage reduction in error magnitude with TMR compared to without protection
3. **Error Correction Efficiency**: Percentage of detected errors that were successfully corrected
4. **Confidence Interval**: 95% confidence interval for the error measurements
5. **MTBF (Mean Time Between Failures)**: Average time between uncorrectable errors

### Theoretical Model

The validation tests include a simplified physics-based theoretical model for predicting error rates based on:
- Radiation environment parameters (particle flux, energy spectrum)
- Shielding effectiveness
- Mission parameters (altitude, orbit, etc.)

The measured error rates from the tests can be compared against this theoretical model to validate both the framework's performance and the underlying physics model.

### Customizing Tests for Specific Missions

Researchers can modify the validation tests to model specific mission profiles by:
1. Adjusting the radiation environment parameters in the `RadiationSimulator::getMissionEnvironment()` function
2. Modifying the network architecture in the `ValidationNetwork` constructor
3. Changing the error thresholds and success criteria in the validation functions

### Adding New Radiation Environments

To test with a custom radiation environment:
1. Create a new environment parameter set following the structure in `RadiationSimulator::getMissionEnvironment()`
2. Add your environment to the test parameters in the `run_validation_suite()` function

## Performance Considerations

The scientific validation test can be computationally intensive, especially when running the full validation suite across multiple environments. For quick tests during development, you can reduce the number of trials or use a less intensive environment like LEO or ISS instead of JUPITER.

## References

For researchers needing to understand the underlying physics models and radiation effects being simulated, refer to:
- NASA Single Event Effects Program documentation
- ESA Radiation Environment Models documentation
- The framework's implementation of physics models in `rad_ml/physics/` and `rad_ml/radiation/` 