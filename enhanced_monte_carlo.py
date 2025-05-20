#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import argparse
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from tabulate import tabulate
import os

# Import our protection methods
from protection_methods import create_protection_method
from rs_monte_carlo import apply_bit_errors


def run_monte_carlo_trials_for_method(
    protection_method_name: str,
    num_trials: int,
    error_rates: List[float],
    min_weight: float = -2.0,
    max_weight: float = 2.0,
) -> dict:
    """
    Run Monte Carlo trials to test a specific protection method.

    Args:
        protection_method_name: Name of the protection method to test
        num_trials: Number of trials per error rate
        error_rates: List of bit error rates to test
        min_weight: Minimum neural network weight to simulate
        max_weight: Maximum neural network weight to simulate

    Returns:
        Dictionary of results
    """
    # Create the protection method
    protection_method = create_protection_method(protection_method_name)

    results = {
        "method": protection_method.name,
        "overhead": protection_method.overhead,
        "error_rates": error_rates,
        "success_rates": [],
        "confidence_intervals": [],
    }

    for error_rate in tqdm(error_rates, desc=f"Testing {protection_method.name}"):
        successes = 0
        trial_results = []

        for _ in range(num_trials):
            # Generate random neural network weight
            weight = np.random.uniform(min_weight, max_weight)

            # Protect the value using the current method
            protected_data = protection_method.protect(weight)

            # Apply random bit errors
            corrupted_data = apply_bit_errors(protected_data, error_rate)

            # Attempt to recover
            recovered_value, success = protection_method.recover(corrupted_data)

            # Check if recovery was successful (within small epsilon)
            if success and abs(recovered_value - weight) < 1e-6:
                successes += 1
                trial_results.append(1)
            else:
                trial_results.append(0)

        # Calculate success rate
        success_rate = successes / num_trials
        results["success_rates"].append(success_rate)

        # Calculate 95% confidence interval
        if num_trials > 30:
            # For large number of trials, use normal approximation
            std_error = np.sqrt((success_rate * (1 - success_rate)) / num_trials)
            confidence_interval = (
                max(0, success_rate - 1.96 * std_error),
                min(1, success_rate + 1.96 * std_error),
            )
        else:
            # For small sample sizes, use Wilson score interval
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / num_trials
            center = (success_rate + z**2 / (2 * num_trials)) / denominator
            margin = (
                z
                * np.sqrt(
                    (success_rate * (1 - success_rate) + z**2 / (4 * num_trials))
                    / num_trials
                )
                / denominator
            )
            confidence_interval = (max(0, center - margin), min(1, center + margin))

        results["confidence_intervals"].append(confidence_interval)

    # Find the error correction threshold (where success rate drops below 50%)
    threshold_idx = next(
        (i for i, rate in enumerate(results["success_rates"]) if rate < 0.5), None
    )
    if threshold_idx is not None and threshold_idx > 0:
        # Linear interpolation to find more precise threshold
        y1, y2 = (
            results["success_rates"][threshold_idx - 1],
            results["success_rates"][threshold_idx],
        )
        x1, x2 = (
            results["error_rates"][threshold_idx - 1],
            results["error_rates"][threshold_idx],
        )

        if y1 != y2:
            threshold = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
        else:
            threshold = x1

        results["threshold"] = threshold
    else:
        results["threshold"] = None

    return results


def plot_comparative_results(
    all_results: List[Dict[str, Any]], output_file: str = None
):
    """
    Plot comparative results for all protection methods.

    Args:
        all_results: List of result dictionaries from run_monte_carlo_trials_for_method()
        output_file: Path to save the plot (if None, display instead)
    """
    plt.figure(figsize=(12, 8))

    # Use different colors and styles for each method
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]

    # Convert error rates to percentages for display
    for i, results in enumerate(all_results):
        error_rates_pct = [rate * 100 for rate in results["error_rates"]]

        # Plot success rates for this method
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        linestyle_idx = i % len(linestyles)

        plt.plot(
            error_rates_pct,
            results["success_rates"],
            linestyles[linestyle_idx] + markers[marker_idx],
            label=f"{results['method']} (OH: {results['overhead']*100:.0f}%)",
            color=colors[color_idx],
            markersize=8,
            alpha=0.8,
        )

        # Add confidence intervals as shaded area
        lower_bounds = [ci[0] for ci in results["confidence_intervals"]]
        upper_bounds = [ci[1] for ci in results["confidence_intervals"]]
        plt.fill_between(
            error_rates_pct,
            lower_bounds,
            upper_bounds,
            color=colors[color_idx],
            alpha=0.2,
        )

    # Add threshold line at 50%
    plt.axhline(
        y=0.5, color="black", linestyle="--", alpha=0.5, label="50% Success Threshold"
    )

    plt.xlabel("Bit Error Rate (%)")
    plt.ylabel("Correction Success Rate")
    plt.title("Comparative Protection Method Performance")
    plt.grid(True)
    plt.legend(loc="best")
    plt.xscale("log")  # Use log scale for x-axis
    plt.ylim(0, 1.05)

    # Add annotations for thresholds
    for results in all_results:
        if results["threshold"] is not None:
            plt.annotate(
                f"{results['threshold']*100:.3f}%",
                xy=(results["threshold"] * 100, 0.5),
                xytext=(results["threshold"] * 100 * 1.2, 0.55),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def print_comparative_table(all_results: List[Dict[str, Any]]):
    """
    Print a comparative table of protection method results.

    Args:
        all_results: List of result dictionaries from run_monte_carlo_trials_for_method()
    """
    # Prepare table data
    table_data = []
    headers = [
        "Protection Method",
        "Memory Overhead",
        "50% Threshold",
        "0.1% BER Success",
        "1% BER Success",
        "5% BER Success",
    ]

    for results in all_results:
        # Find success rates at specific error rates
        success_at_01 = None
        success_at_1 = None
        success_at_5 = None

        for i, rate in enumerate(results["error_rates"]):
            if abs(rate - 0.001) < 0.0001:  # 0.1%
                success_at_01 = results["success_rates"][i] * 100
            elif abs(rate - 0.01) < 0.0001:  # 1%
                success_at_1 = results["success_rates"][i] * 100
            elif abs(rate - 0.05) < 0.0001:  # 5%
                success_at_5 = results["success_rates"][i] * 100

        # Format threshold value
        if results["threshold"] is not None:
            threshold = f"{results['threshold']*100:.3f}%"
        else:
            threshold = "N/A"

        # Add row to table
        table_data.append(
            [
                results["method"],
                f"{results['overhead']*100:.1f}%",
                threshold,
                f"{success_at_01:.2f}%" if success_at_01 is not None else "N/A",
                f"{success_at_1:.2f}%" if success_at_1 is not None else "N/A",
                f"{success_at_5:.2f}%" if success_at_5 is not None else "N/A",
            ]
        )

    # Sort by threshold (best performance first)
    table_data.sort(
        key=lambda x: float(x[2].replace("%", "")) if x[2] != "N/A" else float("inf")
    )

    # Print table
    print("\nComparative Protection Method Performance")
    print("==========================================")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_results_to_csv(all_results: List[Dict[str, Any]], output_file: str):
    """
    Save all results to a CSV file for further analysis.

    Args:
        all_results: List of result dictionaries from run_monte_carlo_trials_for_method()
        output_file: Path to save the CSV file
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(
            [
                "Method",
                "Overhead",
                "Error Rate",
                "Success Rate",
                "CI Lower",
                "CI Upper",
                "Threshold",
            ]
        )

        # Write data rows
        for results in all_results:
            for i, rate in enumerate(results["error_rates"]):
                writer.writerow(
                    [
                        results["method"],
                        results["overhead"],
                        rate,
                        results["success_rates"][i],
                        results["confidence_intervals"][i][0],
                        results["confidence_intervals"][i][1],
                        (
                            results["threshold"] if i == 0 else ""
                        ),  # Only write threshold once per method
                    ]
                )

    print(f"Results saved to {output_file}")


def write_recommendations(all_results: List[Dict[str, Any]], output_file: str):
    """
    Write practical recommendations based on performance in space environments.

    Args:
        all_results: List of result dictionaries
        output_file: Path to save the recommendations markdown file
    """
    # Create lookup tables for space environments and their typical error rates
    environments = {
        "Low Earth Orbit (LEO)": 0.0001,  # 0.01% BER
        "Medium Earth Orbit (MEO)": 0.0005,  # 0.05% BER
        "Geosynchronous Orbit (GEO)": 0.001,  # 0.1% BER
        "Lunar": 0.002,  # 0.2% BER
        "Mars": 0.005,  # 0.5% BER
        "Solar Probe": 0.01,  # 1% BER
        "Solar Flare Event": 0.05,  # 5% BER
    }

    # Find best method for each environment
    recommendations = {}
    for env_name, error_rate in environments.items():
        best_method = None
        best_success = -1
        best_overhead = float("inf")

        for results in all_results:
            # Find closest error rate
            idx = min(
                range(len(results["error_rates"])),
                key=lambda i: abs(results["error_rates"][i] - error_rate),
            )
            success_rate = results["success_rates"][idx]

            # Consider both success rate and overhead
            # Prefer method with higher success rate; if very close, prefer lower overhead
            if success_rate > best_success + 0.05 or (
                abs(success_rate - best_success) < 0.05
                and results["overhead"] < best_overhead
            ):
                best_method = results["method"]
                best_success = success_rate
                best_overhead = results["overhead"]

        recommendations[env_name] = {
            "method": best_method,
            "success_rate": best_success,
            "overhead": best_overhead,
        }

    # Write recommendations to file
    with open(output_file, "w") as f:
        f.write("# Protection Method Recommendations for Space Environments\n\n")
        f.write("## 1. Overview\n\n")
        f.write(
            "This document provides recommendations for error correction methods in various space radiation environments, "
        )
        f.write(
            "based on Monte Carlo simulation results. The recommendations balance error correction effectiveness "
        )
        f.write("against memory overhead requirements.\n\n")

        f.write("## 2. Summary of Methods\n\n")
        f.write(
            "| Method | Memory Overhead | Effective Error Range | Best Use Case |\n"
        )
        f.write("|--------|----------------|----------------------|---------------|\n")

        method_summaries = {
            "No Protection": {
                "overhead": "0%",
                "range": "< 0.001%",
                "use_case": "Non-critical data in very low radiation",
            },
            "Hamming Code": {
                "overhead": "~31%",
                "range": "< 0.1%",
                "use_case": "Low radiation, memory-constrained systems",
            },
            "SEC-DED": {
                "overhead": "~38%",
                "range": "< 0.2%",
                "use_case": "Low radiation with double-error detection needs",
            },
            "Reed-Solomon (RS4)": {
                "overhead": "100%",
                "range": "< 0.5%",
                "use_case": "Medium radiation, balanced protection",
            },
            "Reed-Solomon (RS8)": {
                "overhead": "200%",
                "range": "< 1%",
                "use_case": "High radiation, burst error scenarios",
            },
            "Triple Modular Redundancy (TMR)": {
                "overhead": "200%",
                "range": "< 5%",
                "use_case": "Mission-critical data in high radiation",
            },
            "Hybrid (RS4 + TMR)": {
                "overhead": "150%",
                "range": "< 2%",
                "use_case": "Critical systems with mixed sensitivity data",
            },
            "Hybrid (Hamming + RS4)": {
                "overhead": "~48%",
                "range": "< 0.3%",
                "use_case": "Medium radiation, overhead-constrained",
            },
        }

        for method, info in method_summaries.items():
            f.write(
                f"| {method} | {info['overhead']} | {info['range']} | {info['use_case']} |\n"
            )

        f.write("\n## 3. Environment-Specific Recommendations\n\n")
        f.write(
            "| Environment | Recommended Method | Expected Success Rate | Memory Overhead |\n"
        )
        f.write(
            "|-------------|-------------------|----------------------|----------------|\n"
        )

        for env_name, rec in recommendations.items():
            f.write(
                f"| {env_name} | {rec['method']} | {rec['success_rate']*100:.1f}% | {rec['overhead']*100:.1f}% |\n"
            )

        f.write("\n## 4. Detailed Recommendations\n\n")

        for env_name, rec in recommendations.items():
            f.write(f"### {env_name}\n\n")
            f.write(f"**Recommended Method:** {rec['method']}\n\n")
            f.write(f"**Success Rate:** {rec['success_rate']*100:.1f}%\n\n")
            f.write(f"**Memory Overhead:** {rec['overhead']*100:.1f}%\n\n")

            # Write environment-specific justification
            f.write("**Justification:** ")
            error_rate = environments[env_name]
            if error_rate <= 0.001:
                f.write(
                    f"In the relatively benign radiation environment of {env_name} (BER ~{error_rate*100:.3f}%), "
                )
                if "Hamming" in rec["method"] or "SEC-DED" in rec["method"]:
                    f.write(
                        "lighter-weight methods provide sufficient protection while minimizing resource usage. "
                    )
                else:
                    f.write(
                        "this method provides an optimal balance of protection and resource efficiency. "
                    )
            elif error_rate <= 0.01:
                f.write(
                    f"The moderate radiation levels in {env_name} (BER ~{error_rate*100:.3f}%) require robust error correction. "
                )
                if "Reed-Solomon" in rec["method"]:
                    f.write(
                        "Reed-Solomon provides excellent multi-bit error correction capabilities for these conditions. "
                    )
                elif "Hybrid" in rec["method"]:
                    f.write(
                        "A hybrid approach offers balanced protection for different types of data sensitivity. "
                    )
                else:
                    f.write(
                        "This method provides the best balance of overhead and protection for this environment. "
                    )
            else:
                f.write(
                    f"The harsh radiation environment of {env_name} (BER ~{error_rate*100:.3f}%) demands the strongest protection. "
                )
                if "TMR" in rec["method"]:
                    f.write(
                        "TMR's robust majority-voting approach can withstand high error rates necessary for critical data. "
                    )
                else:
                    f.write(
                        "This method showed the highest success rate in these extreme conditions. "
                    )

            # Add weight protection strategy for hybrid methods
            if "Hybrid" in rec["method"]:
                if "RS4 + TMR" in rec["method"]:
                    f.write(
                        "\n\n**Weight Protection Strategy:** TMR is applied to the most significant bits (50% of weights) to ensure critical parameters maintain accuracy, while Reed-Solomon protects the remaining 50% of weights with lower overhead."
                    )
                elif "Hamming + RS4" in rec["method"]:
                    f.write(
                        "\n\n**Weight Protection Strategy:** Hamming code protects 75% of weights (less critical parameters) with low overhead, while Reed-Solomon provides stronger protection for the 25% most critical weights."
                    )

            f.write("\n\n")

        f.write("## 5. Implementation Considerations\n\n")
        f.write(
            "1. **Critical Weight Identification:** For hybrid methods, identify the most critical neural network weights through sensitivity analysis.\n"
        )
        f.write(
            "2. **Adaptive Protection:** Consider implementing adaptive protection that adjusts based on the current radiation environment.\n"
        )
        f.write(
            "3. **Performance Impact:** Higher protection levels increase computation time; balance protection needs with mission performance requirements.\n"
        )
        f.write(
            "4. **Power Consumption:** More complex protection schemes increase power usage; critical for power-constrained missions.\n"
        )
        f.write(
            "5. **Verification Testing:** Validate protection effectiveness through hardware-in-the-loop testing with radiation sources.\n\n"
        )

        f.write("## 6. Future Research Directions\n\n")
        f.write(
            "1. **Optimized Hybrid Schemes:** Further research into optimal partitioning strategies for hybrid protection.\n"
        )
        f.write(
            "2. **Hardware-Accelerated Protection:** Explore FPGA implementations of these protection schemes for performance gains.\n"
        )
        f.write(
            "3. **Dynamic Protection Adjustment:** Develop algorithms to dynamically adjust protection based on real-time radiation measurements.\n"
        )
        f.write(
            "4. **Application-Specific Tuning:** Fine-tune protection strategies for specific neural network architectures and applications.\n"
        )

    print(f"Recommendations saved to {output_file}")


def main():
    """Main function to run enhanced Monte Carlo simulation."""
    parser = argparse.ArgumentParser(
        description="Run enhanced Monte Carlo simulation for various protection methods"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of trials per error rate (default: 1000)",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.0001,
        help="Minimum bit error rate as decimal (default: 0.0001)",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=0.1,
        help="Maximum bit error rate as decimal (default: 0.1)",
    )
    parser.add_argument(
        "--num-rates",
        type=int,
        default=20,
        help="Number of error rates to test (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help='Comma-separated list of methods to test or "all" (default: all)',
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate logarithmically spaced error rates
    error_rates = np.logspace(
        np.log10(args.min_rate), np.log10(args.max_rate), args.num_rates
    ).tolist()

    # Determine which methods to test
    if args.methods.lower() == "all":
        method_names = [
            "none",  # No protection (baseline)
            "hamming",  # Hamming code
            "secded",  # SEC-DED
            "tmr",  # Triple Modular Redundancy
            "rs4",  # Reed-Solomon with 4 ECC symbols
            "rs8",  # Reed-Solomon with 8 ECC symbols
            "hybrid_rs4_tmr",  # Hybrid: RS4 + TMR
            "hybrid_hamming_rs4",  # Hybrid: Hamming + RS4
        ]
    else:
        method_names = [name.strip() for name in args.methods.split(",")]

    print(f"Running Monte Carlo simulation with {args.trials} trials per error rate")
    print(
        f"Testing {len(error_rates)} error rates from {args.min_rate*100:.4f}% to {args.max_rate*100:.1f}%"
    )
    print(f"Testing protection methods: {', '.join(method_names)}")

    # Run Monte Carlo trials for each method
    all_results = []
    start_time = time.time()

    for method_name in method_names:
        results = run_monte_carlo_trials_for_method(
            method_name, num_trials=args.trials, error_rates=error_rates
        )
        all_results.append(results)

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.1f} seconds")

    # Create comparative plot
    plot_file = os.path.join(args.output_dir, "protection_methods_comparison.png")
    plot_comparative_results(all_results, plot_file)

    # Print comparative table
    print_comparative_table(all_results)

    # Save results to CSV
    csv_file = os.path.join(args.output_dir, "protection_methods_comparison.csv")
    save_results_to_csv(all_results, csv_file)

    # Generate recommendations
    recommendations_file = os.path.join(
        args.output_dir, "protection_recommendations.md"
    )
    write_recommendations(all_results, recommendations_file)


if __name__ == "__main__":
    main()
