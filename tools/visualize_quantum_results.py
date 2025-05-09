#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main():
    # Check if results file exists
    results_file = "quantum_enhancement_results.csv"
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        return 1
    
    # Read results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} test results from {results_file}")
    
    # Create output directory if it doesn't exist
    output_dir = "quantum_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot defect differences by material and scenario
    plt.figure(figsize=(12, 8))
    
    # Group by material and scenario
    grouped = df.groupby(['Material', 'Scenario'])
    
    materials = df['Material'].unique()
    scenarios = df['Scenario'].unique()
    
    bar_width = 0.25
    index = np.arange(len(materials))
    
    for i, scenario in enumerate(scenarios):
        percent_diffs = [grouped.get_group((mat, scenario))['Percent Difference'].values[0] 
                         if (mat, scenario) in grouped.groups else 0 
                         for mat in materials]
        
        plt.bar(index + i*bar_width, percent_diffs, bar_width, 
                label=f'Scenario: {scenario}')
    
    plt.xlabel('Material')
    plt.ylabel('Defect Count Difference (%)')
    plt.title('Quantum vs. Classical Defect Count Differences')
    plt.xticks(index + bar_width, materials, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/defect_differences.png", dpi=300)
    
    # 2. Plot quantum contributions
    plt.figure(figsize=(12, 8))
    
    for i, material in enumerate(materials):
        material_data = df[df['Material'] == material]
        
        tunneling = material_data['Tunneling Contribution (%)']
        zero_point = material_data['Zero-Point Contribution (%)']
        
        plt.scatter(tunneling, zero_point, label=material, s=100, alpha=0.7)
    
    plt.xlabel('Tunneling Contribution (%)')
    plt.ylabel('Zero-Point Energy Contribution (%)')
    plt.title('Quantum Contributions by Material')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quantum_contributions.png", dpi=300)
    
    # 3. Plot performance impact
    plt.figure(figsize=(12, 8))
    
    # Calculate average by scenario
    scenario_perf = df.groupby('Scenario')['Execution Time (ms)'].mean().reset_index()
    
    plt.bar(scenario_perf['Scenario'], scenario_perf['Execution Time (ms)'], color='skyblue')
    plt.xlabel('Scenario')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Impact by Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_impact.png", dpi=300)
    
    # 4. Generate summary statistics
    summary = pd.DataFrame({
        'Metric': [
            'Average Defect Difference (%)', 
            'Max Defect Difference (%)',
            'Average Tunneling Contribution (%)',
            'Average Zero-Point Contribution (%)',
            'Average Execution Time (ms)'
        ],
        'Value': [
            df['Percent Difference'].mean(),
            df['Percent Difference'].max(),
            df['Tunneling Contribution (%)'].mean(),
            df['Zero-Point Contribution (%)'].mean(),
            df['Execution Time (ms)'].mean()
        ]
    })
    
    summary.to_csv(f"{output_dir}/summary_stats.csv", index=False)
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    print("\nSummary Statistics:")
    print(summary.to_string(index=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 