#!/usr/bin/env python3
"""
TMR Example for Radiation-Tolerant ML Framework

This example demonstrates how to use Triple Modular Redundancy (TMR)
in Python with the rad_ml framework.

Author: Rishab Nuguru
Copyright: © 2025 Rishab Nuguru
License: AGPL v3 license
"""

import random


def simulate_radiation_effect(value, bit_pos):
    """Simulate a radiation-induced bit flip"""
    # Convert to binary
    binary = bin(value)[2:].zfill(32)
    binary_list = list(binary)

    # Flip the bit
    binary_list[bit_pos] = "1" if binary_list[bit_pos] == "0" else "0"

    # Convert back to integer
    return int("".join(binary_list), 2)


def main():
    try:
        # Try importing rad_ml
        import rad_ml
    except ImportError as e:
        print(f"Failed to import rad_ml: {e}")
        print("Make sure you've built the Python bindings correctly.")
        return

    try:
        # Initialize the rad_ml framework
        rad_ml.initialize(enable_logging=True)

        print(f"Using rad_ml version: {rad_ml.__version__}")

        # Create a simple TMR-protected integer
        counter = rad_ml.StandardTMR(42)
        print(f"Initial counter value: {counter.value}")

        # Create an enhanced TMR-protected float
        sensor_reading = rad_ml.EnhancedTMR(98.6)
        print(f"Initial sensor reading: {sensor_reading.value}")

        # Simulate a radiation event affecting our counter's internal representation
        print("\nSimulating radiation event affecting counter...")

        # Get the raw C++ TMR object and modify the internal state to simulate corruption
        try:
            # This might not work if using the minimal bindings
            counter._tmr.set_value(
                simulate_radiation_effect(counter.value, random.randint(0, 31))
            )
        except Exception:
            # Fallback approach if we're using the fallback implementation
            print("Using fallback radiation simulation approach")
            # Instead, just modify the value directly (in a real app, this would be done by the C++ code)
            counter.value = simulate_radiation_effect(
                counter.value, random.randint(0, 31)
            )

        # Check integrity and correct if needed
        if not counter.check_integrity():
            print("Integrity check failed, attempting correction...")
            corrected = counter.correct()
            if corrected:
                print(f"Successfully corrected value to: {counter.value}")
            else:
                print("Failed to correct value")
        else:
            print("No corruption detected")

        try:
            # Try to use the radiation simulator
            # Create a radiation simulator
            sim = rad_ml.PhysicsRadiationSimulator(
                environment=rad_ml.RadiationEnvironment.MARS, intensity=0.75
            )

            print(
                f"\nCreated radiation simulator for environment: {sim.get_environment()}"
            )
            print(f"Intensity: {sim.get_intensity()}")

            # Run simulation
            sim.simulate()
        except Exception as e:
            print(f"\nFailed to create radiation simulator: {e}")
            print("This feature may not be available in the minimal bindings.")

        # Shutdown the framework
        rad_ml.shutdown()
        print("\nFramework shutdown successfully")

    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()
