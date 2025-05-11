# Copyright (C) 2025 Rishab Nuguru
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import LinearSegmentedColormap

def load_data(file_path):
    """Load the CSV data from the visualizer"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    return pd.read_csv(file_path)

def create_protection_level_plot(data, save_path=None):
    """Create plot showing protection levels by environment"""
    # Get unique environments in order of increasing radiation
    env_order = ["NONE", "LEO", "GEO", "SAA", "MARS", "SOLAR_STORM", "JUPITER"]
    
    # Filter to get one row per environment (last row for each)
    env_data = []
    for env in env_order:
        if env in data['Environment'].values:
            env_data.append(data[data['Environment'] == env].iloc[-1])
    
    env_df = pd.DataFrame(env_data)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create color map
    cmap = plt.cm.viridis
    
    # Plot protection levels
    bars = plt.bar(
        env_df['Environment'], 
        env_df['ProtectionLevel'],
        color=cmap(env_df['ProtectionLevel']/5)  # Normalize by max protection level
    )
    
    # Add protection level names as text
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            env_df['ProtectionName'].iloc[i],
            ha='center', va='bottom',
            rotation=45, fontsize=9
        )
    
    # Add checkpoint interval as second axis
    ax2 = plt.twinx()
    ax2.plot(
        env_df['Environment'],
        env_df['CheckpointInterval'],
        'r-o', 
        linewidth=2, 
        label='Checkpoint Interval (s)'
    )
    
    # Labels and title
    plt.title('Protection Levels and Checkpoint Intervals by Environment', fontsize=14)
    plt.xlabel('Radiation Environment', fontsize=12)
    plt.ylabel('Protection Level', fontsize=12)
    ax2.set_ylabel('Checkpoint Interval (s)', fontsize=12, color='r')
    ax2.tick_params(axis='y', colors='r')
    
    # Legend
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def create_success_rate_plot(data, save_path=None):
    """Create plot showing success rates by environment"""
    # Get unique environments
    environments = data['Environment'].unique()
    
    # Filter to get the final row for each environment
    success_data = []
    for env in environments:
        env_data = data[data['Environment'] == env]
        if not env_data.empty:
            success_data.append(env_data.iloc[-1])
    
    success_df = pd.DataFrame(success_data)
    
    # Sort by radiation level (approximated by ProtectionLevel)
    success_df = success_df.sort_values('ProtectionLevel')
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list(
        'success_cmap', 
        [(0, 'lightcoral'), (0.7, 'gold'), (1, 'forestgreen')]
    )
    
    # Plot success rates
    bars = plt.bar(
        success_df['Environment'], 
        success_df['SuccessRate'],
        color=cmap(success_df['SuccessRate']/100)  # Normalize by 100%
    )
    
    # Add error rate as text
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"Error Rate: {success_df['ErrorRate'].iloc[i]:.4f}",
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Labels and title
    plt.title('Success Rates by Environment', fontsize=14)
    plt.xlabel('Radiation Environment', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 105)  # Allow space for text
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def create_time_series_plot(data, save_path=None):
    """Create time series plot showing protection and success rates over time"""
    plt.figure(figsize=(14, 8))
    
    # Setup subplots
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Define environment transition points
    env_changes = []
    current_env = None
    
    for i, row in data.iterrows():
        if row['Environment'] != current_env:
            current_env = row['Environment']
            env_changes.append((row['Time'], current_env))
    
    # Plot Protection Level over time
    ax1.plot(data['Time'], data['ProtectionLevel'], 'b-', linewidth=2)
    ax1.set_ylabel('Protection Level')
    ax1.set_title('Adaptive Protection and Success Rate Over Time')
    
    # Plot Success Rate over time
    ax2.plot(data['Time'], data['SuccessRate'], 'g-', linewidth=2)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xlabel('Simulation Time')
    
    # Add environment transition markers
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    
    for time, env in env_changes:
        ax1.axvline(x=time, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=time, color='r', linestyle='--', alpha=0.5)
        ax1.text(time, ymax1*0.9, env, rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def create_factor_comparison_plot(data, save_path=None):
    """Create plot comparing temperature and radiation factors"""
    # Get unique environments
    environments = data['Environment'].unique()
    
    # Filter to get one row per environment
    factor_data = []
    for env in environments:
        env_data = data[data['Environment'] == env]
        if not env_data.empty:
            factor_data.append(env_data.iloc[-1])
    
    factor_df = pd.DataFrame(factor_data)
    
    # Sort by radiation factor
    factor_df = factor_df.sort_values('RadiationFactor')
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Width of bars
    width = 0.35
    
    # Positions
    pos = np.arange(len(factor_df))
    
    # Plot bars
    plt.bar(pos - width/2, factor_df['TemperatureFactor'], width, color='orange', label='Temperature Factor')
    plt.bar(pos + width/2, factor_df['RadiationFactor'], width, color='purple', label='Radiation Factor')
    
    # Add synergy factor as a line
    plt.plot(pos, factor_df['SynergyFactor'], 'r-o', linewidth=2, label='Synergy Factor')
    
    # Labels and title
    plt.title('Environmental Factor Comparison', fontsize=14)
    plt.xlabel('Environment', fontsize=12)
    plt.ylabel('Factor Value', fontsize=12)
    plt.xticks(pos, factor_df['Environment'])
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function to create all visualizations"""
    # Path to the CSV file
    csv_file = "protection_visualization.csv"
    
    # Load data
    data = load_data(csv_file)
    
    # Create output directory if it doesn't exist
    output_dir = "visualization_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate plots
    print("Generating protection level plot...")
    create_protection_level_plot(data, os.path.join(output_dir, "protection_levels.png"))
    
    print("Generating success rate plot...")
    create_success_rate_plot(data, os.path.join(output_dir, "success_rates.png"))
    
    print("Generating time series plot...")
    create_time_series_plot(data, os.path.join(output_dir, "time_series.png"))
    
    print("Generating environmental factor comparison plot...")
    create_factor_comparison_plot(data, os.path.join(output_dir, "factor_comparison.png"))
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main() 
