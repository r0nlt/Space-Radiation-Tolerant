# VAE Auto Tuning Guide with Monte Carlo Testing

This guide explains how to use your existing VAE auto tuning infrastructure to systematically optimize VAE configurations for your AI Native Database.

## Overview

Your project includes a sophisticated `VAEAutoTuner` class that provides:

- **Grid Search**: Exhaustive parameter space exploration
- **Random Search**: Efficient sampling of configuration space
- **Evolutionary Search**: Genetic algorithm-based optimization
- **Monte Carlo Testing**: Statistical reliability with confidence intervals
- **Database Integration**: Real-world performance testing

## Quick Start

### 1. Basic Usage

```bash
# Build the examples
cd build
make vae_quick_tuning_demo

# Run quick demonstration
./examples/vae_quick_tuning_demo
```

This will:
- Generate test data
- Find optimal VAE configurations for compression and anomaly detection
- Test the configurations with your AI Native Database
- Complete in under 1 minute

### 2. Comprehensive Tuning

```bash
# Build comprehensive tuning example
make vae_monte_carlo_tuning_example

# Run full optimization (takes longer but more thorough)
./examples/vae_monte_carlo_tuning_example
```

This will:
- Use larger datasets and more Monte Carlo trials
- Test multiple search strategies
- Generate detailed reports and CSV files
- Take 10-30 minutes depending on your hardware

## Understanding the Results

### Key Metrics

**Compression Score**: Higher is better for database storage
- Balances compression ratio vs. reconstruction quality
- Optimal for `storage::AINativeDatabase` use cases

**Anomaly Score**: Higher is better for anomaly detection
- Balances true positive rate vs. false positive rate
- Optimal for monitoring and fault detection

**Balanced Score**: Weighted combination of both
- Good general-purpose configuration
- 60% compression + 40% anomaly detection

### Configuration Parameters

**Latent Dimension**: Controls compression ratio
- Lower = more compression, potential quality loss
- Higher = better quality, less compression
- Sweet spot usually 4-8 for 12D telemetry data

**Beta Parameter**: Controls VAE behavior
- β < 1.0: Prioritizes reconstruction quality (compression)
- β > 1.0: Prioritizes structured latent space (anomaly detection)
- β = 1.0: Standard VAE behavior

**Architecture**: Hidden layer sizes
- Deeper/wider = more capacity, slower training
- Shallower/narrower = faster, may underfit
- Funnel shapes (e.g., 128-64-32) work well

## Monte Carlo Testing

### Why Use Monte Carlo?

VAE training has inherent randomness from:
- Weight initialization
- Mini-batch sampling
- Stochastic gradient descent

Monte Carlo testing runs multiple trials to get:
- **Mean performance**: Average expected result
- **Standard deviation**: Reliability measure
- **Convergence rate**: Training stability

### Interpreting Statistics

```
Reconstruction Error: 0.0045 ± 0.0012
```

This means:
- Average error: 0.0045
- 68% of trials fall within 0.0033-0.0057
- 95% of trials fall within 0.0021-0.0069

Lower standard deviation = more reliable configuration.

## Practical Workflow

### Step 1: Quick Exploration

Start with the quick demo to get baseline results:

```cpp
// Generate your data
auto training_data = generateYourData(1000);
auto validation_data = generateYourData(200);

// Create tuner
VAEAutoTuner tuner(training_data, validation_data);

// Quick random search
auto result = tuner.randomSearch(20, 5, "compression");
```

### Step 2: Focused Optimization

Use grid search around promising regions:

```cpp
// Focus on best latent dimensions found
std::vector<size_t> focused_latents = {4, 6, 8};
std::vector<float> focused_betas = {0.5f, 1.0f, 1.5f};

auto result = tuner.gridSearch(focused_latents, focused_betas,
                              architectures, learning_rates,
                              10, "compression");
```

### Step 3: Production Validation

Test with your actual database:

```cpp
// Apply optimal configuration
AINativeDatabase::Config db_config;
db_config.default_latent_dim = result.config.latent_dim;
db_config.vae_hidden_dims = result.config.hidden_dims;

AINativeDatabase db(db_config);
// Test with real data...
```

## Advanced Usage

### Custom Search Ranges

```cpp
tuner.setSearchRanges(
    {2, 4, 6, 8, 12, 16},           // latent_dims
    {0.1f, 0.5f, 1.0f, 2.0f, 3.0f}, // beta_values
    {0.0001f, 0.001f, 0.01f},       // learning_rates
    {20, 50, 100}                   // epochs
);
```

### Evolutionary Search

For complex optimization landscapes:

```cpp
auto result = tuner.evolutionarySearch(
    30,    // population_size
    20,    // generations
    0.15,  // mutation_rate
    8,     // monte_carlo_trials
    "balanced"
);
```

### Database Integration Testing

Test configurations directly with database:

```cpp
auto db_result = tuner.testDatabaseIntegration(config, 10);
std::cout << "Storage efficiency: " << db_result.storage_efficiency_mean
          << " ± " << db_result.storage_efficiency_stddev << std::endl;
```

## Use Case Specific Recommendations

### For Maximum Compression

```cpp
// Optimize for storage efficiency
std::vector<size_t> latent_dims = {2, 3, 4};     // High compression
std::vector<float> beta_values = {0.1f, 0.5f};   // Low beta for quality
auto result = tuner.gridSearch(latent_dims, beta_values,
                              architectures, learning_rates,
                              15, "compression");
```

### For Anomaly Detection

```cpp
// Optimize for anomaly detection
std::vector<size_t> latent_dims = {6, 8, 12};      // Moderate compression
std::vector<float> beta_values = {1.5f, 2.0f, 3.0f}; // High beta for structure
auto result = tuner.gridSearch(latent_dims, beta_values,
                              architectures, learning_rates,
                              12, "anomaly_detection");
```

### For Balanced Performance

```cpp
// Optimize for both use cases
auto result = tuner.randomSearch(50, 10, "balanced");
```

## Output Files

The tuning process generates several useful files:

### CSV Results (`*_results.csv`)
- All tested configurations
- Performance metrics with statistics
- Can be imported into Excel/Python for analysis

### Reports (`*_report.md`)
- Human-readable summary
- Best configurations for each use case
- Recommendations and next steps

### Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('comprehensive_vae_tuning_results.csv')

# Plot compression ratio vs. reconstruction error
plt.scatter(df['compression_ratio_mean'], df['reconstruction_error_mean'])
plt.xlabel('Compression Ratio')
plt.ylabel('Reconstruction Error')
plt.title('VAE Performance Trade-offs')
plt.show()
```

## Performance Tips

### Speed Up Tuning

1. **Start Small**: Use fewer Monte Carlo trials (3-5) for initial exploration
2. **Use Random Search**: More efficient than grid search for large spaces
3. **Reduce Data Size**: Use 500-1000 samples for tuning, full dataset for final validation
4. **Parallel Processing**: Run multiple tuning jobs with different seeds

### Improve Results

1. **More Trials**: Use 10-15 Monte Carlo trials for final configurations
2. **Better Data**: Ensure training data represents real usage patterns
3. **Longer Training**: Increase epochs for complex architectures
4. **Cross-Validation**: Test on completely separate datasets

## Troubleshooting

### Common Issues

**Low Convergence Rate**:
- Reduce learning rate
- Increase training epochs
- Simplify architecture

**High Reconstruction Error**:
- Increase latent dimension
- Reduce beta parameter
- Use larger/deeper networks

**Inconsistent Results**:
- Increase Monte Carlo trials
- Check data quality
- Ensure sufficient training data

## Integration with Your Database

### Applying Optimal Configurations

```cpp
// Get best configuration from tuning
auto best_config = tuning_result.config;

// Apply to database
AINativeDatabase::Config db_config;
db_config.default_latent_dim = best_config.latent_dim;
db_config.vae_hidden_dims = best_config.hidden_dims;
db_config.max_reconstruction_error = 0.01f;  // Based on tuning results

// Create optimized database
AINativeDatabase optimized_db(db_config);
```

### Monitoring Performance

```cpp
// Monitor database performance
auto stats = db.get_statistics();
std::cout << "Average compression: " << stats.average_compression_ratio << std::endl;
std::cout << "Average error: " << stats.average_reconstruction_error << std::endl;

// Re-tune if performance degrades
if (stats.average_reconstruction_error > threshold) {
    // Run tuning again with new data
}
```

## Next Steps

1. **Run the Quick Demo**: Get familiar with the system
2. **Analyze Your Data**: Understand your specific data characteristics
3. **Custom Tuning**: Adapt the examples to your specific use case
4. **Production Deployment**: Apply optimal configurations to your database
5. **Continuous Monitoring**: Set up automated re-tuning as needed

## Files to Explore

- `examples/vae_quick_tuning_demo.cpp` - Quick start example
- `examples/vae_monte_carlo_tuning_example.cpp` - Comprehensive tuning
- `include/rad_ml/research/vae_auto_tuner.hpp` - Full API documentation
- `examples/vae_space_mission_test.cpp` - Real-world usage example
