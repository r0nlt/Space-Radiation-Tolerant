#!/bin/bash

# Create results directory
mkdir -p results

# Run the enhanced Monte Carlo test with a reduced number of trials
python enhanced_monte_carlo.py \
  --trials 100 \
  --min-rate 0.0005 \
  --max-rate 0.05 \
  --num-rates 10 \
  --output-dir results \
  --methods all

# Display the results
echo "Results are available in the 'results' directory"
echo "- Comparative plot: results/protection_methods_comparison.png"
echo "- Data CSV: results/protection_methods_comparison.csv"
echo "- Recommendations: results/protection_recommendations.md"
