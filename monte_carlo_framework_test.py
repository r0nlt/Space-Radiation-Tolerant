#!/usr/bin/env python3
"""
Comprehensive Monte Carlo Radiation Test
=======================================

This script demonstrates a full Monte Carlo testing suite for the
Space Labs AI Radiation-Tolerant ML Framework, testing different
neural network architectures across various radiation environments
and protection strategies.

Author: Rishab Nuguru
Copyright: © 2025 Space Labs AI
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import the radiation-tolerant ML framework
try:
    import rad_ml_minimal as rad_ml
    from rad_ml_minimal.rad_ml.tmr import EnhancedTMR, StandardTMR
    from rad_ml_minimal.rad_ml.sim import Environment, RadiationEnvironment
    from rad_ml_minimal.rad_ml.neural import ProtectionLevel, AdvancedReedSolomon
except ImportError:
    print("Error: rad_ml_minimal package not found.")
    print("Please install it using: pip install rad_ml_minimal")
    sys.exit(1)

# Optional: Check for GPU acceleration
try:
    import torch

    HAS_TORCH = True
    print(f"Using PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Using CPU-only mode.")

# Constants
NUM_MONTE_CARLO_RUNS = 100  # Number of Monte Carlo simulation runs
TEST_DURATION_MINUTES = 5  # Duration of each test in simulated minutes
PLOT_RESULTS = True  # Whether to generate plots

# Define test environments
ENVIRONMENTS = {
    "LEO": Environment.LEO,  # Low Earth Orbit
    "GEO": Environment.GEO,  # Geostationary Orbit
    "LUNAR": Environment.LUNAR,  # Lunar orbit
    "MARS": Environment.MARS,  # Mars orbit
    "JUPITER": Environment.JUPITER,  # Jupiter orbit (extreme radiation)
    "SAA": Environment.SAA,  # South Atlantic Anomaly (high radiation area)
}

# Define protection levels
PROTECTION_LEVELS = {
    "NONE": ProtectionLevel.NONE,
    "MINIMAL": ProtectionLevel.MINIMAL,
    "MODERATE": ProtectionLevel.MODERATE,
    "HIGH": ProtectionLevel.HIGH,
    "ADAPTIVE": ProtectionLevel.ADAPTIVE,
    "SPACE_OPTIMIZED": ProtectionLevel.SPACE_OPTIMIZED,
}

# Define neural network architectures to test
ARCHITECTURES = {
    "standard": [8, 8, 4],  # Standard architecture
    "wide": [32, 16, 4],  # Wide architecture (better radiation tolerance)
    "deep": [8, 8, 8, 8, 4],  # Deep architecture
    "ultrawide": [64, 32, 4],  # Ultra-wide architecture (best for extreme radiation)
    "minimal": [4, 4],  # Minimal architecture
}


# Define test dataset generation function
def generate_test_data(input_size, num_samples=1000):
    """Generate synthetic test data for neural network evaluation."""
    inputs = np.random.normal(0, 1, (num_samples, input_size))
    # Create a simple classification problem (4 classes)
    labels = np.zeros((num_samples, 4))
    for i in range(num_samples):
        # Assign a class based on the sum of inputs
        class_idx = min(3, max(0, int(np.sum(inputs[i]) / 2) + 2))
        labels[i, class_idx] = 1
    return inputs, labels


class SimpleNeuralNetwork:
    """Simple neural network implementation for testing radiation effects."""

    def __init__(
        self,
        architecture,
        protection_level=ProtectionLevel.NONE,
        dropout_rate=0.0,
        use_torch=False,
    ):
        """Initialize a simple neural network with the given architecture and protection level."""
        self.architecture = architecture
        self.protection_level = protection_level
        self.dropout_rate = dropout_rate
        self.use_torch = use_torch and HAS_TORCH

        # Initialize weights with protection
        self.layers = []
        for i in range(len(architecture) - 1):
            if self.use_torch:
                # PyTorch implementation
                weights = torch.randn(architecture[i], architecture[i + 1]) * 0.1
                bias = torch.zeros(architecture[i + 1])
                if protection_level != ProtectionLevel.NONE:
                    # Apply protection using our framework's PyTorch integration
                    weights = rad_ml.torch_protect(weights, protection_level)
                    bias = rad_ml.torch_protect(bias, protection_level)
                self.layers.append({"weights": weights, "bias": bias})
            else:
                # NumPy implementation with explicit protection
                weights = np.random.randn(architecture[i], architecture[i + 1]) * 0.1
                bias = np.zeros(architecture[i + 1])

                # Apply protection based on level
                if protection_level != ProtectionLevel.NONE:
                    protected_weights = []
                    for row in weights:
                        protected_row = []
                        for val in row:
                            if protection_level == ProtectionLevel.MINIMAL:
                                protected_row.append(StandardTMR(val))
                            else:
                                protected_row.append(EnhancedTMR(val))
                        protected_weights.append(protected_row)

                    protected_bias = []
                    for val in bias:
                        if protection_level == ProtectionLevel.MINIMAL:
                            protected_bias.append(StandardTMR(val))
                        else:
                            protected_bias.append(EnhancedTMR(val))

                    self.layers.append(
                        {"weights": protected_weights, "bias": protected_bias}
                    )
                else:
                    self.layers.append({"weights": weights, "bias": bias})

    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            # Matrix multiplication and bias addition
            if self.use_torch:
                x = x @ layer["weights"] + layer["bias"]
                # Apply ReLU activation except for the last layer
                if i < len(self.layers) - 1:
                    x = torch.nn.functional.relu(x)
                    # Apply dropout if specified
                    if self.dropout_rate > 0:
                        x = torch.nn.functional.dropout(
                            x, p=self.dropout_rate, training=True
                        )
            else:
                # For NumPy implementation with protection
                if self.protection_level != ProtectionLevel.NONE:
                    # Protected forward pass
                    result = np.zeros(len(layer["bias"]))
                    for j in range(x.shape[0]):
                        for k in range(len(layer["bias"])):
                            sum_val = layer["bias"][k].value
                            for l in range(len(x)):
                                # Access protected weights and multiply
                                weight_val = layer["weights"][l][k].value
                                sum_val += x[j, l] * weight_val
                            result[j, k] = sum_val
                    x = result
                else:
                    # Standard forward pass
                    x = x @ layer["weights"] + layer["bias"]

                # Apply ReLU activation except for the last layer
                if i < len(self.layers) - 1:
                    x = np.maximum(0, x)
                    # Apply dropout if specified
                    if self.dropout_rate > 0:
                        dropout_mask = np.random.binomial(
                            1, 1 - self.dropout_rate, x.shape
                        )
                        x = x * dropout_mask / (1 - self.dropout_rate)

        # Apply softmax to final layer output
        if self.use_torch:
            return torch.nn.functional.softmax(x, dim=1)
        else:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def evaluate(self, inputs, labels):
        """Evaluate the network on the given inputs and labels."""
        outputs = self.forward(inputs)
        if self.use_torch:
            predictions = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(torch.tensor(labels), dim=1)
            accuracy = (predictions == true_labels).float().mean().item()
        else:
            predictions = np.argmax(outputs, axis=1)
            true_labels = np.argmax(labels, axis=1)
            accuracy = np.mean(predictions == true_labels)
        return accuracy

    def simulate_radiation_effects(self, bit_error_rate, environment):
        """Simulate radiation effects on network weights."""
        stats = {"bit_flips": 0, "errors_detected": 0, "errors_corrected": 0}

        # Configure radiation environment
        rad_env = rad_ml.sim.createEnvironment(environment)

        for layer in self.layers:
            if self.protection_level != ProtectionLevel.NONE:
                # For protected weights
                for i in range(len(layer["weights"])):
                    for j in range(len(layer["weights"][i])):
                        # Simulate radiation effect
                        if np.random.random() < bit_error_rate:
                            stats["bit_flips"] += 1
                            # Corrupt one of the redundant copies
                            if isinstance(
                                layer["weights"][i][j], StandardTMR
                            ) or isinstance(layer["weights"][i][j], EnhancedTMR):
                                # Corrupt a random bit in the first copy
                                layer["weights"][i][j]._v1 = flip_random_bit(
                                    layer["weights"][i][j]._v1
                                )

                                # Check and attempt correction
                                if not layer["weights"][i][j].check_integrity():
                                    stats["errors_detected"] += 1
                                    if layer["weights"][i][j].correct():
                                        stats["errors_corrected"] += 1

                # For protected biases
                for i in range(len(layer["bias"])):
                    # Simulate radiation effect
                    if np.random.random() < bit_error_rate:
                        stats["bit_flips"] += 1
                        # Corrupt one of the redundant copies
                        if isinstance(layer["bias"][i], StandardTMR) or isinstance(
                            layer["bias"][i], EnhancedTMR
                        ):
                            # Corrupt a random bit in the first copy
                            layer["bias"][i]._v1 = flip_random_bit(layer["bias"][i]._v1)

                            # Check and attempt correction
                            if not layer["bias"][i].check_integrity():
                                stats["errors_detected"] += 1
                                if layer["bias"][i].correct():
                                    stats["errors_corrected"] += 1
            else:
                # For unprotected weights, directly corrupt the values
                if self.use_torch:
                    # Apply bit errors to PyTorch tensors
                    mask = torch.rand_like(layer["weights"]) < bit_error_rate
                    bit_pos = torch.randint(0, 32, mask.sum().item())
                    layer["weights"].flatten()[mask.flatten()] = flip_bits_tensor(
                        layer["weights"].flatten()[mask.flatten()], bit_pos
                    )

                    mask = torch.rand_like(layer["bias"]) < bit_error_rate
                    bit_pos = torch.randint(0, 32, mask.sum().item())
                    layer["bias"].flatten()[mask.flatten()] = flip_bits_tensor(
                        layer["bias"].flatten()[mask.flatten()], bit_pos
                    )

                    stats["bit_flips"] += mask.sum().item()
                else:
                    # Apply bit errors to NumPy arrays
                    mask = np.random.random(layer["weights"].shape) < bit_error_rate
                    stats["bit_flips"] += np.sum(mask)
                    for i, j in zip(*np.where(mask)):
                        layer["weights"][i, j] = flip_random_bit(layer["weights"][i, j])

                    mask = np.random.random(layer["bias"].shape) < bit_error_rate
                    stats["bit_flips"] += np.sum(mask)
                    for i in np.where(mask)[0]:
                        layer["bias"][i] = flip_random_bit(layer["bias"][i])

        return stats


def flip_random_bit(value):
    """Flip a random bit in a float value."""
    # Convert float to its binary representation (as 32-bit int)
    import struct

    binary = struct.unpack("!I", struct.pack("!f", value))[0]

    # Flip a random bit (avoid the sign bit for simplicity)
    bit_to_flip = np.random.randint(0, 31)
    binary ^= 1 << bit_to_flip

    # Convert back to float
    result = struct.unpack("!f", struct.pack("!I", binary))[0]
    return result


def flip_bits_tensor(tensor, bit_positions):
    """Flip specified bits in a PyTorch tensor."""
    if not HAS_TORCH:
        return tensor

    # Convert to binary, flip bits, convert back
    float_vals = tensor.cpu().numpy().astype(np.float32)
    result = np.zeros_like(float_vals)

    for i, (val, bit_pos) in enumerate(zip(float_vals, bit_positions)):
        # Convert float to its binary representation
        binary = struct.unpack("!I", struct.pack("!f", val))[0]

        # Flip the specified bit
        binary ^= 1 << bit_pos.item()

        # Convert back to float
        result[i] = struct.unpack("!f", struct.pack("!I", binary))[0]

    return torch.tensor(result, device=tensor.device, dtype=tensor.dtype)


def run_monte_carlo_test(
    architecture_name,
    architecture,
    protection_level_name,
    protection_level,
    environment_name,
    environment,
    dropout_rate=0.0,
    num_runs=50,
):
    """Run a full Monte Carlo test and return results."""
    print(f"\nRunning Monte Carlo test with:")
    print(f"  Architecture: {architecture_name} {architecture}")
    print(f"  Protection: {protection_level_name}")
    print(f"  Environment: {environment_name}")
    print(f"  Dropout Rate: {dropout_rate}")

    # Initialize the framework
    rad_ml.initialize()

    # Create the radiation environment
    rad_env = rad_ml.sim.createEnvironment(environment)

    # Generate test data
    input_size = architecture[0]
    inputs, labels = generate_test_data(input_size)

    # Create a network with the specified architecture and protection
    network = SimpleNeuralNetwork(
        architecture,
        protection_level=protection_level,
        dropout_rate=dropout_rate,
        use_torch=HAS_TORCH,
    )

    # Calculate baseline accuracy without radiation effects
    baseline_accuracy = network.evaluate(inputs, labels)

    # Results storage
    results = {
        "run": [],
        "bit_error_rate": [],
        "accuracy": [],
        "accuracy_percentage": [],
        "bit_flips": [],
        "errors_detected": [],
        "errors_corrected": [],
        "correction_rate": [],
    }

    # Run multiple Monte Carlo simulations with increasing bit error rates
    bit_error_rates = np.logspace(-6, -2, num_runs)  # From 0.000001 to 0.01

    for run, bit_error_rate in enumerate(
        tqdm(bit_error_rates, desc="Running simulations")
    ):
        # Create a fresh network for each test to avoid accumulating errors
        test_network = SimpleNeuralNetwork(
            architecture,
            protection_level=protection_level,
            dropout_rate=dropout_rate,
            use_torch=HAS_TORCH,
        )

        # Simulate radiation effects
        stats = test_network.simulate_radiation_effects(bit_error_rate, environment)

        # Evaluate accuracy after radiation effects
        accuracy = test_network.evaluate(inputs, labels)

        # Calculate accuracy percentage relative to baseline
        accuracy_percentage = (
            (accuracy / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
        )

        # Calculate error correction rate
        correction_rate = (
            (stats["errors_corrected"] / stats["errors_detected"]) * 100
            if stats["errors_detected"] > 0
            else 100
        )

        # Store results
        results["run"].append(run)
        results["bit_error_rate"].append(bit_error_rate)
        results["accuracy"].append(accuracy)
        results["accuracy_percentage"].append(accuracy_percentage)
        results["bit_flips"].append(stats["bit_flips"])
        results["errors_detected"].append(stats["errors_detected"])
        results["errors_corrected"].append(stats["errors_corrected"])
        results["correction_rate"].append(correction_rate)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\nResults Summary:")
    print(f"  Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"  Mean Accuracy: {results_df['accuracy'].mean():.4f}")
    print(
        f"  Mean Accuracy Preservation: {results_df['accuracy_percentage'].mean():.2f}%"
    )
    print(f"  Mean Error Correction Rate: {results_df['correction_rate'].mean():.2f}%")

    return results_df, baseline_accuracy


def plot_monte_carlo_results(
    results_df,
    baseline_accuracy,
    architecture_name,
    protection_level_name,
    environment_name,
):
    """Plot the Monte Carlo test results."""
    if not PLOT_RESULTS:
        return

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot accuracy vs bit error rate
    ax1.semilogx(
        results_df["bit_error_rate"],
        results_df["accuracy_percentage"],
        marker="o",
        linestyle="-",
        markersize=4,
    )
    ax1.axhline(y=100, color="g", linestyle="--", alpha=0.5, label="Baseline")
    ax1.axhline(y=90, color="y", linestyle="--", alpha=0.5, label="90% threshold")
    ax1.set_xlabel("Bit Error Rate (log scale)")
    ax1.set_ylabel("Accuracy Preservation (%)")
    ax1.set_title(
        f"Accuracy Preservation vs Bit Error Rate\n{architecture_name} / {protection_level_name} / {environment_name}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0, top=105)
    ax1.legend()

    # Plot error detection and correction
    ax2.semilogx(
        results_df["bit_error_rate"],
        results_df["correction_rate"],
        marker="s",
        linestyle="-",
        color="r",
        markersize=4,
        label="Correction Rate",
    )
    ax2.set_xlabel("Bit Error Rate (log scale)")
    ax2.set_ylabel("Error Correction Rate (%)")
    ax2.set_title("Error Correction Performance")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0, top=105)
    ax2.legend()

    plt.tight_layout()

    # Save the figure
    filename = f"results/mc_test_{architecture_name}_{protection_level_name}_{environment_name}.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")

    # Also save the raw data
    csv_filename = f"results/mc_test_{architecture_name}_{protection_level_name}_{environment_name}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


def run_batch_monte_carlo_tests(config_list, num_runs=50):
    """Run a batch of Monte Carlo tests with different configurations."""
    all_results = []

    for config in config_list:
        arch_name = config["architecture"]
        arch = ARCHITECTURES[arch_name]
        prot_name = config["protection"]
        prot = PROTECTION_LEVELS[prot_name]
        env_name = config["environment"]
        env = ENVIRONMENTS[env_name]
        dropout = config.get("dropout", 0.0)

        results_df, baseline_accuracy = run_monte_carlo_test(
            arch_name, arch, prot_name, prot, env_name, env, dropout, num_runs
        )

        # Plot individual results
        plot_monte_carlo_results(
            results_df, baseline_accuracy, arch_name, prot_name, env_name
        )

        # Store summary for comparison
        summary = {
            "architecture": arch_name,
            "protection": prot_name,
            "environment": env_name,
            "dropout": dropout,
            "baseline_accuracy": baseline_accuracy,
            "mean_accuracy": results_df["accuracy"].mean(),
            "mean_accuracy_preservation": results_df["accuracy_percentage"].mean(),
            "mean_correction_rate": results_df["correction_rate"].mean(),
            "max_sustainable_error_rate": (
                results_df.loc[
                    results_df["accuracy_percentage"] >= 90, "bit_error_rate"
                ].max()
                if any(results_df["accuracy_percentage"] >= 90)
                else 0
            ),
        }
        all_results.append(summary)

    # Convert to DataFrame and save comparison
    comparison_df = pd.DataFrame(all_results)
    csv_filename = "results/monte_carlo_comparison.csv"
    comparison_df.to_csv(csv_filename, index=False)
    print(f"\nComparison saved to {csv_filename}")

    # Create comparison plots
    if PLOT_RESULTS:
        # Create bar chart of accuracy preservation
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            [
                f"{row['architecture']}/{row['protection']}/{row['environment']}"
                for _, row in comparison_df.iterrows()
            ],
            comparison_df["mean_accuracy_preservation"],
        )
        plt.axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% threshold")
        plt.xlabel("Configuration")
        plt.ylabel("Mean Accuracy Preservation (%)")
        plt.title("Comparison of Configurations: Accuracy Preservation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.savefig("results/accuracy_comparison.png", dpi=150)

        # Create bar chart of sustainable error rates
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            [
                f"{row['architecture']}/{row['protection']}/{row['environment']}"
                for _, row in comparison_df.iterrows()
            ],
            comparison_df["max_sustainable_error_rate"] * 100,  # Convert to percentage
        )
        plt.xlabel("Configuration")
        plt.ylabel("Max Sustainable Error Rate (%)")
        plt.title("Comparison of Configurations: Radiation Tolerance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig("results/error_rate_comparison.png", dpi=150)

        # Print max sustainable error rate for each configuration
        print("\nMax Sustainable Error Rates:")
        for _, row in comparison_df.iterrows():
            print(
                f"  {row['architecture']}/{row['protection']}/{row['environment']}: {row['max_sustainable_error_rate']*100:.4f}%"
            )

    return comparison_df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Monte Carlo Radiation Test for Neural Networks"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_MONTE_CARLO_RUNS,
        help="Number of Monte Carlo runs",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with fewer configurations",
    )
    args = parser.parse_args()

    global PLOT_RESULTS
    PLOT_RESULTS = not args.no_plot
    num_runs = args.runs

    # Print test information
    print(f"Starting Monte Carlo Test with {num_runs} runs per configuration")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Define test configurations
    if args.quick_test:
        print("Running quick test with limited configurations")
        configurations = [
            {"architecture": "standard", "protection": "NONE", "environment": "LEO"},
            {
                "architecture": "standard",
                "protection": "MODERATE",
                "environment": "LEO",
            },
            {
                "architecture": "wide",
                "protection": "SPACE_OPTIMIZED",
                "environment": "MARS",
                "dropout": 0.5,
            },
        ]
    else:
        configurations = [
            # LEO configurations
            {"architecture": "standard", "protection": "NONE", "environment": "LEO"},
            {"architecture": "standard", "protection": "MINIMAL", "environment": "LEO"},
            {
                "architecture": "standard",
                "protection": "MODERATE",
                "environment": "LEO",
            },
            {
                "architecture": "wide",
                "protection": "MODERATE",
                "environment": "LEO",
                "dropout": 0.3,
            },
            # GEO configurations
            {"architecture": "standard", "protection": "HIGH", "environment": "GEO"},
            {
                "architecture": "wide",
                "protection": "HIGH",
                "environment": "GEO",
                "dropout": 0.4,
            },
            # MARS configurations
            {
                "architecture": "wide",
                "protection": "SPACE_OPTIMIZED",
                "environment": "MARS",
                "dropout": 0.5,
            },
            {
                "architecture": "deep",
                "protection": "SPACE_OPTIMIZED",
                "environment": "MARS",
                "dropout": 0.5,
            },
            # JUPITER configurations (extreme radiation)
            {
                "architecture": "ultrawide",
                "protection": "SPACE_OPTIMIZED",
                "environment": "JUPITER",
                "dropout": 0.6,
            },
        ]

    # Run the batch Monte Carlo tests
    start_time = time.time()
    comparison_results = run_batch_monte_carlo_tests(configurations, num_runs)
    end_time = time.time()

    # Print execution time
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

    # Print the best configuration
    best_config = comparison_results.loc[
        comparison_results["mean_accuracy_preservation"].idxmax()
    ]
    print(f"\nBest Configuration:")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Protection Level: {best_config['protection']}")
    print(f"  Environment: {best_config['environment']}")
    print(f"  Dropout Rate: {best_config['dropout']}")
    print(f"  Accuracy Preservation: {best_config['mean_accuracy_preservation']:.2f}%")
    print(f"  Error Correction Rate: {best_config['mean_correction_rate']:.2f}%")
    print(
        f"  Max Sustainable Error Rate: {best_config['max_sustainable_error_rate']*100:.4f}%"
    )

    print(
        "\nMonte Carlo testing complete. See 'results' directory for detailed outputs."
    )


if __name__ == "__main__":
    main()
