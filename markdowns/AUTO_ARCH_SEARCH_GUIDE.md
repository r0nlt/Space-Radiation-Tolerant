# Auto Architecture Search Guide (v0.9.7)

This guide explains how to use the Auto Architecture Search feature introduced in v0.9.7 of the Radiation-Tolerant Machine Learning Framework. This feature helps you find optimal neural network architectures for specific radiation environments.

## Overview

The Auto Architecture Search capability allows you to:

1. Automatically test multiple neural network architectures with different configurations
2. Evaluate performance under various radiation environments
3. Compare protection strategies for each architecture
4. Find the optimal configuration for your mission requirements

## Getting Started

### Running the Example

The easiest way to see the Auto Architecture Search in action is to run the included example:

```bash
make -f Makefile.simple example-auto_arch_search_example
```

Or build and run it directly:

```bash
make
./examples/auto_arch_search_example
```

This will:
- Create a synthetic dataset for testing
- Configure the search parameters for various architectures
- Run an evolutionary search algorithm to find optimal configurations
- Compare results with a random search approach
- Save detailed results to CSV files

### Understanding the Results

After running the example, check the following files:
- `leo_arch_search_results.csv` - Contains detailed results for each tested architecture
- `leo_combined_results.csv` - Contains combined results from all search methods

These files include metrics such as:
- Baseline accuracy (without radiation)
- Radiation accuracy (under radiation conditions)
- Accuracy preservation percentage
- Execution time
- Error detection and correction statistics

## Using in Your Own Projects

### Basic Usage

Here's how to integrate the Auto Architecture Search into your own projects:

```cpp
#include <rad_ml/research/auto_arch_search.hpp>

// Create search instance with your dataset
rad_ml::research::AutoArchSearch searcher(
    train_data, train_labels, test_data, test_labels,
    rad_ml::sim::Environment::MARS,  // Target environment
    {32, 64, 128, 256},              // Width options to test
    {0.3, 0.4, 0.5, 0.6},            // Dropout options to test
    "results.csv"                    // Output file
);

// Configure the search parameters
searcher.setFixedParameters(
    input_size,   // Input size
    output_size,  // Output size
    2             // Number of hidden layers
);

// Set protection levels to test
searcher.setProtectionLevels({
    rad_ml::neural::ProtectionLevel::NONE,
    rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY,
    rad_ml::neural::ProtectionLevel::SELECTIVE_TMR,
    rad_ml::neural::ProtectionLevel::FULL_TMR
});

// Enable residual connections testing
searcher.setTestResidualConnections(true);

// Run the search (evolutionary search is recommended)
auto result = searcher.evolutionarySearch(
    10,    // Population size
    5,     // Number of generations
    0.2,   // Mutation rate
    5,     // Epochs for training
    true,  // Use Monte Carlo testing
    3      // Number of Monte Carlo trials
);

// Print the best architecture found
std::cout << "Best architecture for MARS environment:" << std::endl;
// Access result.config to see the details
```

### Advanced Configuration

#### Monte Carlo Testing

The Monte Carlo testing feature allows for more reliable results by running multiple trials with different random seeds:

```cpp
// Enable Monte Carlo testing with 10 trials per architecture
auto result = searcher.randomSearch(
    20,    // Number of iterations
    5,     // Epochs for training
    true,  // Enable Monte Carlo testing
    10     // Number of Monte Carlo trials
);
```

#### Search Methods

The framework offers three search methods:

1. **Grid Search** - Exhaustively tests all configurations:
```cpp
auto result = searcher.findOptimalArchitecture(
    5,     // Epochs for training
    true,  // Use Monte Carlo
    3      // Monte Carlo trials
);
```

2. **Random Search** - Randomly samples the configuration space:
```cpp
auto result = searcher.randomSearch(
    20,    // Number of iterations
    5,     // Epochs for training
    true,  // Use Monte Carlo
    3      // Monte Carlo trials
);
```

3. **Evolutionary Search** - Uses genetic algorithms to efficiently search:
```cpp
auto result = searcher.evolutionarySearch(
    10,    // Population size
    5,     // Number of generations
    0.2,   // Mutation rate
    5,     // Epochs for training
    true,  // Use Monte Carlo
    3      // Monte Carlo trials
);
```

#### Custom Width Options

You can specify your own layer width options to test:

```cpp
// Test specific width options
std::vector<size_t> width_options = {16, 32, 64, 128, 256, 512};
rad_ml::research::AutoArchSearch searcher(
    train_data, train_labels, test_data, test_labels,
    rad_ml::sim::Environment::EARTH_ORBIT,
    width_options,                    // Custom width options
    {0.1, 0.2, 0.3, 0.4, 0.5},        // Dropout options
    "custom_search_results.csv"
);
```

#### Custom Environment

Test your architecture in different radiation environments:

```cpp
// Available environments:
// rad_ml::sim::Environment::EARTH
// rad_ml::sim::Environment::EARTH_ORBIT
// rad_ml::sim::Environment::MOON
// rad_ml::sim::Environment::MARS
// rad_ml::sim::Environment::JUPITER
// rad_ml::sim::Environment::DEEP_SPACE
// rad_ml::sim::Environment::EXTREME

// Test for Jupiter environment
rad_ml::research::AutoArchSearch jupiter_searcher(
    train_data, train_labels, test_data, test_labels,
    rad_ml::sim::Environment::JUPITER,
    width_options,
    dropout_options,
    "jupiter_results.csv"
);
```

## Best Practices

### Effective Search Strategy

For best results:

1. **Start with random search** to quickly explore the search space
2. **Follow with evolutionary search** to fine-tune the most promising configurations
3. **Run multiple Monte Carlo trials** (at least 10 for final results) to ensure statistical significance
4. **Test multiple environments** if your mission will encounter varying radiation conditions

### Resource Optimization

To balance search quality with computing resources:

1. **For initial exploration**: 
   - Use fewer Monte Carlo trials (3-5)
   - Test fewer width options
   - Use smaller epochs (5-10)

2. **For final optimization**:
   - Increase Monte Carlo trials (10+)
   - Focus on promising width ranges
   - Increase epochs (20+)
   - Use evolutionary search with larger populations and generations

## Understanding the Results

### Key Metrics

When analyzing the results, focus on these key metrics:

- **Accuracy Preservation**: The percentage of accuracy retained under radiation conditions
- **Radiation Accuracy**: The absolute accuracy under radiation conditions
- **Standard Deviation**: Low values indicate more reliable performance

### Interpreting Architecture Impact

From our comprehensive testing, we've found:

1. **Layer Width Impact**: Wider architectures (32-16 nodes) often show greater radiation tolerance
2. **Dropout Effect**: Higher dropout rates (0.5) significantly enhance radiation resilience
3. **Residual Connections**: These help with deeper networks under radiation
4. **Protection Level**: Sometimes, architectures with proper width/dropout perform better with NONE or minimal protection

## Troubleshooting

### Common Issues

1. **Identical Performance Metrics**: If all tested configurations produce identical metrics, there may be an issue with the testing environment or simulation parameters.

2. **Long Execution Times**: Using many Monte Carlo trials with large architectures can be time-consuming. Start with fewer trials and simpler architectures for initial testing.

3. **Unexpected Results**: If results seem counter-intuitive, try:
   - Verifying dataset integrity
   - Checking that environment parameters are correct
   - Ensuring proper random seed generation

### Debug Output

To enable detailed debug output:

```cpp
// Set environment variable
export RAD_ML_LOG_LEVEL=DEBUG

// Then run your search
./examples/auto_arch_search_example
```

## Conclusion

The Auto Architecture Search feature provides a powerful way to optimize neural networks for radiation environments. By leveraging this capability, you can design more efficient and robust systems for space applications with significantly less manual effort.

For more detailed information about this enhancement, see [autoarchsearchwriteup.md](autoarchsearchwriteup.md).

For a complete version history, refer to [VERSION_HISTORY.md](VERSION_HISTORY.md). 