#!/usr/bin/env python3
import numpy as np
import struct
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import argparse
from tqdm import tqdm


# Galois Field (GF(2^8)) implementation
class GF256:
    """Galois Field GF(2^8) implementation."""

    # Pre-computed tables for GF(2^8) arithmetic
    exp_table = [0] * 256
    log_table = [0] * 256

    def __init__(self):
        """Initialize the exp and log tables for GF(2^8)."""
        x = 1
        for i in range(255):
            self.exp_table[i] = x
            # Multiply by primitive element (0x03 for this field)
            x = self.mult_no_table(x, 0x03)

        # Set element 255 in exp table for calculation simplicity
        self.exp_table[255] = self.exp_table[0]

        # Generate log table from exp table
        for i in range(1, 256):
            for j in range(255):
                if self.exp_table[j] == i:
                    self.log_table[i] = j
                    break

    def add(self, a: int, b: int) -> int:
        """Addition in GF(2^8) (XOR operation)."""
        return a ^ b

    def subtract(self, a: int, b: int) -> int:
        """Subtraction in GF(2^8) (same as addition)."""
        return a ^ b

    def mult_no_table(self, a: int, b: int) -> int:
        """Multiplication in GF(2^8) without lookup tables."""
        result = 0
        for i in range(8):
            if (b & 1) == 1:
                result ^= a
            high_bit_set = a & 0x80
            a <<= 1
            if high_bit_set:
                a ^= 0x1B  # The irreducible polynomial x^8 + x^4 + x^3 + x + 1
            b >>= 1
        return result & 0xFF

    def mult(self, a: int, b: int) -> int:
        """Multiplication in GF(2^8) using log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[(self.log_table[a] + self.log_table[b]) % 255]

    def div(self, a: int, b: int) -> int:
        """Division in GF(2^8)."""
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(256)")
        return self.exp_table[(self.log_table[a] - self.log_table[b] + 255) % 255]

    def pow(self, a: int, power: int) -> int:
        """Exponentiation in GF(2^8)."""
        if a == 0:
            return 0
        if power == 0:
            return 1
        return self.exp_table[(self.log_table[a] * power) % 255]


# Reed-Solomon encoder/decoder implementation
class RS8Bit8Sym:
    """Reed-Solomon implementation with 8-bit symbols and 8 ECC symbols."""

    def __init__(self):
        """Initialize Reed-Solomon encoder/decoder."""
        self.gf = GF256()
        self.n_ecc_symbols = 8  # Number of ECC symbols
        # Generate generator polynomial
        self.gen_poly = self._generate_generator_polynomial()

    def _generate_generator_polynomial(self) -> List[int]:
        """Generate the generator polynomial for Reed-Solomon encoding."""
        g = [1]
        for i in range(self.n_ecc_symbols):
            # Multiply (x + α^i) term
            g = self._polynomial_multiply(g, [1, self.gf.exp_table[i]])
        return g

    def _polynomial_multiply(self, p1: List[int], p2: List[int]) -> List[int]:
        """Multiply two polynomials in GF(2^8)."""
        result = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                result[i + j] = self.gf.add(result[i + j], self.gf.mult(p1[i], p2[j]))
        return result

    def encode(self, data: List[int]) -> List[int]:
        """
        Encode data using Reed-Solomon algorithm.

        Args:
            data: List of data bytes to encode

        Returns:
            List of encoded bytes (original data + ECC symbols)
        """
        # Convert data to polynomial form (with zeros for ECC)
        poly = list(data) + [0] * self.n_ecc_symbols

        # Perform polynomial division to get remainder (ECC symbols)
        divisor = self.gen_poly
        for i in range(len(data)):
            if poly[i] != 0:
                coef = poly[i]
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:
                        poly[i + j] = self.gf.add(
                            poly[i + j], self.gf.mult(divisor[j], coef)
                        )

        # Extract remainder (last n_ecc_symbols bytes)
        remainder = poly[-self.n_ecc_symbols :]

        # Return data + ECC
        return list(data) + remainder

    def _calculate_syndromes(self, data: List[int]) -> List[int]:
        """
        Calculate syndromes to detect errors.

        Args:
            data: Encoded data (including ECC symbols)

        Returns:
            List of syndrome values (all zeros means no errors)
        """
        syndromes = []
        for i in range(self.n_ecc_symbols):
            result = 0
            for j in range(len(data)):
                # Evaluate polynomial at α^i
                term = self.gf.mult(
                    data[j], self.gf.pow(self.gf.exp_table[i], len(data) - 1 - j)
                )
                result = self.gf.add(result, term)
            syndromes.append(result)
        return syndromes

    def _find_error_locator_polynomial(self, syndromes: List[int]) -> List[int]:
        """
        Find error locator polynomial using Berlekamp-Massey algorithm.

        Args:
            syndromes: List of syndrome values

        Returns:
            Coefficients of the error locator polynomial
        """
        # Initialize with simple polynomial Λ(x) = 1
        error_loc = [1]
        old_loc = [1]

        # Berlekamp-Massey algorithm
        for i in range(self.n_ecc_symbols):
            delta = syndromes[i]
            for j in range(1, len(error_loc)):
                delta = self.gf.add(delta, self.gf.mult(error_loc[j], syndromes[i - j]))

            # Update error locator polynomial if needed
            if delta != 0:
                if len(old_loc) > len(error_loc):
                    new_loc = [0] * (len(old_loc) + 1)
                    term = [delta]
                    for j in range(len(old_loc)):
                        term.append(
                            self.gf.mult(
                                old_loc[j],
                                self.gf.div(self.gf.exp_table[255 - i], delta),
                            )
                        )

                    for j in range(len(new_loc)):
                        if j < len(error_loc):
                            new_loc[j] = error_loc[j]
                        if j < len(term):
                            new_loc[j] = self.gf.add(new_loc[j], term[j])

                    old_loc = list(error_loc)
                    error_loc = list(new_loc)
                else:
                    temp = [0] * len(error_loc)
                    for j in range(len(old_loc)):
                        temp[j] = self.gf.mult(
                            old_loc[j], self.gf.div(self.gf.exp_table[255 - i], delta)
                        )

                    for j in range(1, len(error_loc)):
                        temp[j] = self.gf.add(error_loc[j], temp[j - 1])

                    old_loc = list(error_loc)
                    error_loc = list(temp)

        return error_loc

    def _find_error_positions(self, error_loc: List[int], data_len: int) -> List[int]:
        """
        Find error positions using Chien search.

        Args:
            error_loc: Error locator polynomial coefficients
            data_len: Length of the encoded data

        Returns:
            List of positions where errors occurred (in reverse order)
        """
        # Reverse error locator polynomial for easier Chien search
        error_loc = list(reversed(error_loc))

        # Find roots using Chien search
        error_positions = []
        for i in range(data_len):
            x_inv = self.gf.exp_table[255 - i]
            result = error_loc[0]
            for j in range(1, len(error_loc)):
                result = self.gf.add(
                    result, self.gf.mult(error_loc[j], self.gf.pow(x_inv, j))
                )
            if result == 0:
                error_positions.append(data_len - 1 - i)

        return error_positions

    def _calculate_error_values(
        self, error_positions: List[int], syndromes: List[int]
    ) -> List[int]:
        """
        Calculate error values using Forney algorithm.

        Args:
            error_positions: Positions of errors
            syndromes: Syndrome values

        Returns:
            List of error values at each position
        """
        error_values = []
        for pos in error_positions:
            x = self.gf.exp_table[pos]

            # Calculate error evaluator polynomial
            omega = 0
            for i in range(len(syndromes)):
                omega = self.gf.add(
                    omega, self.gf.mult(syndromes[i], self.gf.pow(x, i + 1))
                )

            # Calculate formal derivative of error locator
            derivative = 0
            for i in range(len(error_positions)):
                if error_positions[i] != pos:
                    term = 1
                    for j in range(len(error_positions)):
                        if j != i and error_positions[j] != pos:
                            term = self.gf.mult(
                                term,
                                self.gf.div(
                                    self.gf.add(
                                        x, self.gf.exp_table[error_positions[j]]
                                    ),
                                    self.gf.add(
                                        self.gf.exp_table[pos],
                                        self.gf.exp_table[error_positions[j]],
                                    ),
                                ),
                            )
                    derivative = self.gf.add(derivative, term)

            # Calculate error value
            error_value = self.gf.div(omega, derivative)
            error_values.append(error_value)

        return error_values

    def decode(self, data: List[int]) -> Tuple[List[int], bool]:
        """
        Decode Reed-Solomon encoded data, correcting errors if possible.

        Args:
            data: Encoded data with possible errors

        Returns:
            Tuple of (corrected data, success flag)
        """
        # Make a copy of data to avoid modifying the original
        data_copy = list(data)
        data_len = len(data_copy)

        # Calculate syndromes
        syndromes = self._calculate_syndromes(data_copy)

        # Check if all syndromes are zero (no errors)
        if all(s == 0 for s in syndromes):
            return data_copy[: -self.n_ecc_symbols], True

        # Find error locator polynomial
        error_loc = self._find_error_locator_polynomial(syndromes)

        # Find error positions
        error_positions = self._find_error_positions(error_loc, data_len)

        # If error positions couldn't be found, decoding failed
        if len(error_positions) == 0:
            return data_copy[: -self.n_ecc_symbols], False

        # Calculate error values
        error_values = self._calculate_error_values(error_positions, syndromes)

        # Correct errors
        for i in range(len(error_positions)):
            position = error_positions[i]
            value = error_values[i]
            data_copy[position] = self.gf.add(data_copy[position], value)

        # Verify correction by recalculating syndromes
        verification_syndromes = self._calculate_syndromes(data_copy)
        success = all(s == 0 for s in verification_syndromes)

        # Return corrected data (without ECC symbols)
        return data_copy[: -self.n_ecc_symbols], success


# Helper functions for neural network weight simulation
def float_to_bytes(value: float) -> List[int]:
    """Convert float to list of bytes."""
    # IEEE 754 binary32 format
    return list(struct.pack("!f", value))


def bytes_to_float(byte_list: List[int]) -> float:
    """Convert list of bytes to float."""
    # IEEE 754 binary32 format
    return struct.unpack("!f", bytes(byte_list))[0]


def apply_bit_errors(data: List[int], error_rate: float) -> List[int]:
    """
    Apply random bit errors to data.

    Args:
        data: List of bytes to corrupt
        error_rate: Probability of each bit flipping (0.0 to 1.0)

    Returns:
        List of corrupted bytes
    """
    result = list(data)
    for i in range(len(result)):
        byte = result[i]
        new_byte = 0
        for bit in range(8):
            # Extract bit
            bit_value = (byte >> bit) & 1
            # Randomly flip with probability error_rate
            if np.random.random() < error_rate:
                bit_value = 1 - bit_value
            # Set bit in new byte
            new_byte |= bit_value << bit
        result[i] = new_byte
    return result


def count_bit_errors(original: List[int], corrupted: List[int]) -> int:
    """
    Count the number of bit errors between two byte lists.

    Args:
        original: Original bytes
        corrupted: Corrupted bytes

    Returns:
        Count of bit errors
    """
    error_count = 0
    for i in range(len(original)):
        # XOR shows differences
        xor_result = original[i] ^ corrupted[i]
        # Count set bits in XOR result
        for bit in range(8):
            if (xor_result >> bit) & 1:
                error_count += 1
    return error_count


def run_monte_carlo_trials(
    num_trials: int,
    error_rates: List[float],
    rs_codec: RS8Bit8Sym,
    min_weight: float = -2.0,
    max_weight: float = 2.0,
) -> dict:
    """
    Run Monte Carlo trials to test Reed-Solomon error correction.

    Args:
        num_trials: Number of trials per error rate
        error_rates: List of bit error rates to test
        rs_codec: Reed-Solomon codec instance
        min_weight: Minimum neural network weight to simulate
        max_weight: Maximum neural network weight to simulate

    Returns:
        Dictionary of results
    """
    results = {
        "error_rates": error_rates,
        "success_rates": [],
        "confidence_intervals": [],
        "avg_bit_errors": [],
        "corrected_errors": [],
    }

    for error_rate in tqdm(error_rates, desc="Testing error rates"):
        successes = 0
        total_bit_errors = 0
        total_corrected_errors = 0
        trial_results = []

        for _ in range(num_trials):
            # Generate random neural network weight
            weight = np.random.uniform(min_weight, max_weight)

            # Convert to bytes and encode
            original_bytes = float_to_bytes(weight)
            encoded_bytes = rs_codec.encode(original_bytes)

            # Apply random bit errors
            corrupted_bytes = apply_bit_errors(encoded_bytes, error_rate)

            # Count actual bit errors
            bit_errors = count_bit_errors(encoded_bytes, corrupted_bytes)
            total_bit_errors += bit_errors

            # Attempt to decode and correct
            decoded_bytes, success = rs_codec.decode(corrupted_bytes)

            if success:
                # Count corrected errors if successful
                corrected_errors = count_bit_errors(
                    encoded_bytes[: len(decoded_bytes)],
                    corrupted_bytes[: len(decoded_bytes)],
                )
                total_corrected_errors += corrected_errors

                # Verify if the decoded value matches the original
                try:
                    decoded_weight = bytes_to_float(decoded_bytes)
                    if abs(decoded_weight - weight) < 1e-6:
                        successes += 1
                        trial_results.append(1)
                    else:
                        trial_results.append(0)
                except:
                    # If there's an error converting bytes to float
                    trial_results.append(0)
            else:
                trial_results.append(0)

        # Calculate success rate
        success_rate = successes / num_trials
        results["success_rates"].append(success_rate)

        # Calculate 95% confidence interval using normal approximation
        # (valid for large number of trials)
        if num_trials > 30:
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

        # Average bit errors per trial
        avg_bit_errors = total_bit_errors / num_trials
        results["avg_bit_errors"].append(avg_bit_errors)

        # Average corrected errors per trial
        avg_corrected = total_corrected_errors / num_trials
        results["corrected_errors"].append(avg_corrected)

    return results


def plot_monte_carlo_results(results: dict, output_file: str = None):
    """
    Plot Monte Carlo simulation results.

    Args:
        results: Dictionary of results from run_monte_carlo_trials
        output_file: Path to save the plot (if None, display instead)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Convert error rates to percentages for display
    error_rates_pct = [rate * 100 for rate in results["error_rates"]]

    # Success rates with confidence intervals
    ax1.plot(
        error_rates_pct,
        results["success_rates"],
        "o-",
        label="Success Rate",
        color="blue",
    )

    # Plot confidence intervals
    for i, (lower, upper) in enumerate(results["confidence_intervals"]):
        ax1.vlines(error_rates_pct[i], lower, upper, color="blue", alpha=0.3)

    ax1.set_xlabel("Bit Error Rate (%)")
    ax1.set_ylabel("Correction Success Rate")
    ax1.set_title("Reed-Solomon Error Correction Performance")
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    # Plot bit errors and corrected errors
    ax2.plot(
        error_rates_pct,
        results["avg_bit_errors"],
        "o-",
        label="Average Bit Errors",
        color="red",
    )
    ax2.plot(
        error_rates_pct,
        results["corrected_errors"],
        "o-",
        label="Average Corrected Errors",
        color="green",
    )

    ax2.set_xlabel("Bit Error Rate (%)")
    ax2.set_ylabel("Number of Bit Errors")
    ax2.set_title("Error Count Analysis")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to run Monte Carlo simulation."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation for Reed-Solomon error correction"
    )
    parser.add_argument(
        "--trials", type=int, default=1000, help="Number of trials per error rate"
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.001,
        help="Minimum bit error rate (as a decimal)",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=0.3,
        help="Maximum bit error rate (as a decimal)",
    )
    parser.add_argument(
        "--num-rates", type=int, default=20, help="Number of error rates to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rs_monte_carlo_results.png",
        help="Output file for plot (PNG)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="rs_monte_carlo_results.csv",
        help="Output file for CSV data",
    )
    args = parser.parse_args()

    # Initialize Reed-Solomon codec
    rs_codec = RS8Bit8Sym()

    # Generate logarithmically spaced error rates
    error_rates = np.logspace(
        np.log10(args.min_rate), np.log10(args.max_rate), args.num_rates
    ).tolist()

    print(f"Running Monte Carlo simulation with {args.trials} trials per error rate")
    print(
        f"Testing {len(error_rates)} error rates from {args.min_rate*100:.3f}% to {args.max_rate*100:.1f}%"
    )

    start_time = time.time()
    results = run_monte_carlo_trials(
        num_trials=args.trials, error_rates=error_rates, rs_codec=rs_codec
    )
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.1f} seconds")

    # Save results to CSV
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(
                "error_rate,success_rate,ci_lower,ci_upper,avg_bit_errors,avg_corrected_errors\n"
            )
            for i, rate in enumerate(results["error_rates"]):
                f.write(
                    f"{rate},{results['success_rates'][i]},{results['confidence_intervals'][i][0]},{results['confidence_intervals'][i][1]},{results['avg_bit_errors'][i]},{results['corrected_errors'][i]}\n"
                )
        print(f"Results saved to {args.csv}")

    # Plot results
    plot_monte_carlo_results(results, args.output)

    # Print summary of results at key error rates
    print("\nSummary of Results:")
    print("-" * 80)
    print(
        "Error Rate | Success Rate | 95% Confidence Interval | Avg Bit Errors | Avg Corrected"
    )
    print("-" * 80)

    for i, rate in enumerate(results["error_rates"]):
        if (
            i % 3 == 0 or i == len(results["error_rates"]) - 1
        ):  # Print every 3rd rate plus the last one
            ci_lower, ci_upper = results["confidence_intervals"][i]
            print(
                f"{rate*100:9.2f}% | {results['success_rates'][i]*100:11.2f}% | ({ci_lower*100:.2f}%, {ci_upper*100:.2f}%) | {results['avg_bit_errors'][i]:13.2f} | {results['corrected_errors'][i]:12.2f}"
            )

    # Find the error rate threshold (where success rate drops below 50%)
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

        print(f"\nError correction threshold (50% success): {threshold*100:.3f}%")
    else:
        print("\nCould not determine error correction threshold.")


if __name__ == "__main__":
    main()
