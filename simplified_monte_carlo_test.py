#!/usr/bin/env python3
"""
Simplified Monte Carlo Radiation Test
=====================================

A simplified version of the Monte Carlo framework test that
uses patterns from the advanced_radiation_comparison.py example.

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
import random
import struct
from collections import defaultdict

# Add rad_ml_minimal to the path directly
sys.path.append(os.path.dirname(__file__))

# Constants
NUM_MONTE_CARLO_RUNS = 50  # Number of runs per bit error rate
TEST_DURATION_MINUTES = 5  # Duration of each test in simulated minutes
PLOT_RESULTS = True  # Whether to generate plots
NUM_SAMPLES = 5000  # Increased from 1000 for more stable results
NUM_SEEDS = 3  # Number of random seeds to average over


# Create mock Environment and ProtectionLevel enums
class Environment:
    EARTH = "EARTH"
    EARTH_ORBIT = "EARTH_ORBIT"
    LEO = "LEO"
    GEO = "GEO"
    LUNAR = "LUNAR"
    MARS = "MARS"
    JUPITER = "JUPITER"
    SAA = "SAA"
    SOLAR_STORM = "SOLAR_STORM"


class ProtectionLevel:
    NONE = "NONE"
    MINIMAL = "MINIMAL"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    ADAPTIVE = "ADAPTIVE"
    SPACE_OPTIMIZED = "SPACE_OPTIMIZED"


class DefenseStrategy:
    NONE = "NONE"
    BASIC = "BASIC"
    MULTI_LAYERED = "MULTI_LAYERED"
    REED_SOLOMON = "REED_SOLOMON"
    PHYSICS_DRIVEN = "PHYSICS_DRIVEN"
    ADAPTIVE_PROTECTION = "ADAPTIVE_PROTECTION"


# Mock TMR classes
class StandardTMR:
    def __init__(self, value):
        self._v1 = value
        self._v2 = value
        self._v3 = value
        self.value = value

    def check_integrity(self):
        return self._v1 == self._v2 or self._v1 == self._v3 or self._v2 == self._v3

    def correct(self):
        if self._v1 == self._v2:
            self._v3 = self._v1
            self.value = self._v1
            return True
        elif self._v1 == self._v3:
            self._v2 = self._v1
            self.value = self._v1
            return True
        elif self._v2 == self._v3:
            self._v1 = self._v2
            self.value = self._v2
            return True
        return False


class EnhancedTMR(StandardTMR):
    def __init__(self, value):
        super().__init__(value)
        self._checksum = value * 3  # Simple checksum

    def check_integrity(self):
        basic_check = super().check_integrity()
        checksum_valid = abs((self._v1 + self._v2 + self._v3) - self._checksum) < 1e-5
        return basic_check and checksum_valid


# Monte Carlo Radiation Simulator - based on the example
class MonteCarloRadiationSimulator:
    def __init__(self, environment=Environment.LEO):
        self.environment = environment
        # Configure radiation profile based on environment
        self.configure_radiation_profile()
        # Initialize quantum field effects
        self.enable_quantum_effects = True
        self.quantum_field_intensity = 0.0

    def configure_radiation_profile(self):
        """Set radiation characteristics based on environment"""
        if self.environment == Environment.EARTH:
            # Low radiation
            self.bit_flip_base_prob = 0.005
            self.multi_bit_upset_prob = 0.01
            self.memory_corruption_prob = 0.005
            self.quantum_field_intensity = 0.001
        elif self.environment == Environment.EARTH_ORBIT:
            # Low-moderate radiation
            self.bit_flip_base_prob = 0.008
            self.multi_bit_upset_prob = 0.02
            self.memory_corruption_prob = 0.01
            self.quantum_field_intensity = 0.002
        elif self.environment == Environment.LEO:
            # Moderate radiation
            self.bit_flip_base_prob = 0.01
            self.multi_bit_upset_prob = 0.05
            self.memory_corruption_prob = 0.02
            self.quantum_field_intensity = 0.005
        elif self.environment == Environment.GEO:
            # Moderate-high radiation
            self.bit_flip_base_prob = 0.02
            self.multi_bit_upset_prob = 0.08
            self.memory_corruption_prob = 0.03
            self.quantum_field_intensity = 0.01
        elif self.environment == Environment.LUNAR:
            # Variable radiation (no magnetic field)
            self.bit_flip_base_prob = 0.03
            self.multi_bit_upset_prob = 0.06
            self.memory_corruption_prob = 0.025
            self.quantum_field_intensity = 0.007
        elif self.environment == Environment.MARS:
            # Moderate-high radiation (thin atmosphere)
            self.bit_flip_base_prob = 0.035
            self.multi_bit_upset_prob = 0.07
            self.memory_corruption_prob = 0.03
            self.quantum_field_intensity = 0.008
        elif self.environment == Environment.SAA:
            # South Atlantic Anomaly
            self.bit_flip_base_prob = 0.05
            self.multi_bit_upset_prob = 0.1
            self.memory_corruption_prob = 0.05
            self.quantum_field_intensity = 0.015
        elif self.environment == Environment.JUPITER:
            # Harsh radiation
            self.bit_flip_base_prob = 0.1
            self.multi_bit_upset_prob = 0.2
            self.memory_corruption_prob = 0.1
            self.quantum_field_intensity = 0.05
        elif self.environment == Environment.SOLAR_STORM:
            # Extreme radiation from solar event
            self.bit_flip_base_prob = 0.15
            self.multi_bit_upset_prob = 0.25
            self.memory_corruption_prob = 0.15
            self.quantum_field_intensity = 0.1
        else:
            # Default medium radiation
            self.bit_flip_base_prob = 0.02
            self.multi_bit_upset_prob = 0.05
            self.memory_corruption_prob = 0.02
            self.quantum_field_intensity = 0.01

    def apply_radiation_effects(self, value, radiation_strength=1.0):
        """Apply radiation effects to a value (can be a number or tensor)"""
        # Scale probabilities by radiation strength
        bit_flip_prob = self.bit_flip_base_prob * radiation_strength

        # For a simple implementation, just apply bit flips
        if isinstance(value, (list, np.ndarray)):
            result = np.array(value).copy()
            # Apply bit flips to array
            mask = np.random.random(result.shape) < bit_flip_prob
            for idx in np.where(mask.flatten())[0]:
                flat_idx = np.unravel_index(idx, result.shape)
                result[flat_idx] = self._flip_random_bit(result[flat_idx])
            return result
        else:
            # Apply bit flip to single value
            if random.random() < bit_flip_prob:
                return self._flip_random_bit(value)
            return value

    def _flip_random_bit(self, value):
        """Flip a random bit in a value"""
        # Convert float to its binary representation (as 32-bit int)
        binary = struct.unpack("!I", struct.pack("!f", float(value)))[0]

        # Flip a random bit (avoid the sign bit for simplicity)
        bit_to_flip = random.randint(0, 31)
        binary ^= 1 << bit_to_flip

        # Convert back to float
        result = struct.unpack("!f", struct.pack("!I", binary))[0]
        return result


# Helper function for radiation framework
def initialize():
    """Mock function to initialize the radiation framework"""
    print("Initializing radiation framework...")
    return True


def createEnvironment(env_type):
    """Mock function to create a radiation environment"""
    return MonteCarloRadiationSimulator(env_type)


# Mock the rad_ml.sim module
class MockRadML:
    def __init__(self):
        self.sim = type("obj", (object,), {"createEnvironment": createEnvironment})

    def initialize(self):
        return initialize()

    def torch_protect(self, tensor, protection_level):
        """Mock function to protect PyTorch tensors"""
        return tensor  # Just return the tensor unchanged in this mock


# Create a global instance of the mock rad_ml
rad_ml = MockRadML()


# Define test dataset generation function
def generate_test_data(input_size, num_samples=NUM_SAMPLES):
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
        self.use_torch = False  # Force to False for simplified version

        # Initialize weights with protection
        self.layers = []
        for i in range(len(architecture) - 1):
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
            # For NumPy implementation with protection
            if self.protection_level != ProtectionLevel.NONE:
                # Protected forward pass
                result = np.zeros((x.shape[0], len(layer["bias"])))
                for j in range(x.shape[0]):
                    for k in range(len(layer["bias"])):
                        sum_val = layer["bias"][k].value
                        for l in range(len(x[j])):
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
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
                    x = x * dropout_mask / (1 - self.dropout_rate)

        # Apply softmax to final layer output
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def evaluate(self, inputs, labels):
        """Evaluate the network on the given inputs and labels."""
        outputs = self.forward(inputs)
        predictions = np.argmax(outputs, axis=1)
        true_labels = np.argmax(labels, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def simulate_radiation_effects(self, bit_error_rate, environment):
        """Simulate radiation effects on network weights."""
        stats = {"bit_flips": 0, "errors_detected": 0, "errors_corrected": 0}

        # Configure radiation environment
        rad_env = MonteCarloRadiationSimulator(environment)

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
                                layer["weights"][i][j]._v1 = rad_env._flip_random_bit(
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
                            layer["bias"][i]._v1 = rad_env._flip_random_bit(
                                layer["bias"][i]._v1
                            )

                            # Check and attempt correction
                            if not layer["bias"][i].check_integrity():
                                stats["errors_detected"] += 1
                                if layer["bias"][i].correct():
                                    stats["errors_corrected"] += 1
            else:
                # For unprotected weights, directly corrupt the values
                # Apply bit errors to NumPy arrays
                mask = np.random.random(layer["weights"].shape) < bit_error_rate
                stats["bit_flips"] += np.sum(mask)
                for i, j in zip(*np.where(mask)):
                    layer["weights"][i, j] = rad_env._flip_random_bit(
                        layer["weights"][i, j]
                    )

                mask = np.random.random(layer["bias"].shape) < bit_error_rate
                stats["bit_flips"] += np.sum(mask)
                for i in np.where(mask)[0]:
                    layer["bias"][i] = rad_env._flip_random_bit(layer["bias"][i])

        return stats


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
    initialize()

    # Create the radiation environment
    rad_env = MonteCarloRadiationSimulator(environment)

    # Generate test data
    input_size = architecture[0]
    inputs, labels = generate_test_data(input_size)

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

    # First create a baseline network and measure its accuracy
    # This will be used for all tests to ensure a stable baseline
    baseline_network = SimpleNeuralNetwork(
        architecture,
        protection_level=protection_level,
        dropout_rate=dropout_rate,
        use_torch=False,
    )
    baseline_accuracy = baseline_network.evaluate(inputs, labels)

    for run, bit_error_rate in enumerate(
        tqdm(bit_error_rates, desc="Running simulations")
    ):
        # Run multiple trials with different seeds and average the results
        seed_accuracies = []
        seed_stats = {"bit_flips": 0, "errors_detected": 0, "errors_corrected": 0}

        for seed in range(NUM_SEEDS):
            # Set the random seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Create a copy of the baseline network for consistent starting point
            test_network = SimpleNeuralNetwork(
                architecture,
                protection_level=protection_level,
                dropout_rate=dropout_rate,
                use_torch=False,
            )

            # Make a deep copy of the baseline network weights
            # Instead of attempting to copy the entire network structure
            if protection_level == ProtectionLevel.NONE:
                # For unprotected networks, simply copy the weights and biases
                for i, layer in enumerate(baseline_network.layers):
                    test_network.layers[i]["weights"] = np.copy(layer["weights"])
                    test_network.layers[i]["bias"] = np.copy(layer["bias"])
            else:
                # For protected networks, we need to be more careful with the TMR objects
                # We'll initialize with the same values but keep the new TMR objects
                pass  # The protection objects are already initialized with random values

            # Simulate radiation effects
            stats = test_network.simulate_radiation_effects(bit_error_rate, environment)

            # Evaluate accuracy after radiation effects
            accuracy = test_network.evaluate(inputs, labels)
            seed_accuracies.append(accuracy)

            # Accumulate stats
            seed_stats["bit_flips"] += stats["bit_flips"]
            seed_stats["errors_detected"] += stats["errors_detected"]
            seed_stats["errors_corrected"] += stats["errors_corrected"]

        # Average results across seeds
        avg_accuracy = np.mean(seed_accuracies)
        avg_stats = {k: v / NUM_SEEDS for k, v in seed_stats.items()}

        # Calculate accuracy percentage relative to baseline
        accuracy_percentage = (
            (avg_accuracy / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
        )

        # Calculate error correction rate
        correction_rate = (
            (avg_stats["errors_corrected"] / avg_stats["errors_detected"]) * 100
            if avg_stats["errors_detected"] > 0
            else 100
        )

        # Store results
        results["run"].append(run)
        results["bit_error_rate"].append(bit_error_rate)
        results["accuracy"].append(avg_accuracy)
        results["accuracy_percentage"].append(accuracy_percentage)
        results["bit_flips"].append(avg_stats["bit_flips"])
        results["errors_detected"].append(avg_stats["errors_detected"])
        results["errors_corrected"].append(avg_stats["errors_corrected"])
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
    global PLOT_RESULTS
    if not PLOT_RESULTS:
        return

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate y-axis limits based on data
    max_accuracy_pct = results_df["accuracy_percentage"].max()
    y_max = max(105, np.ceil(max_accuracy_pct / 10) * 10)  # Round up to nearest 10%

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
    ax1.set_ylim(bottom=0, top=y_max)  # Dynamic y-axis limit
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


# Define test environments
ENVIRONMENTS = {
    "LEO": Environment.LEO,  # Low Earth Orbit
    "GEO": Environment.GEO,  # Geostationary Orbit
    "LUNAR": Environment.LUNAR,  # Lunar orbit
    "MARS": Environment.MARS,  # Mars orbit
    "JUPITER": Environment.JUPITER,  # Jupiter orbit (extreme radiation)
    "SAA": Environment.SAA,  # South Atlantic Anomaly (high radiation area)
    "EARTH": Environment.EARTH,  # Earth (low radiation)
    "SOLAR_STORM": Environment.SOLAR_STORM,  # Solar storm (extreme radiation)
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
