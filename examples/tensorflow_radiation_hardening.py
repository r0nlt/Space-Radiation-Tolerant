#!/usr/bin/env python3
"""
TensorFlow Radiation Hardening Example

This example demonstrates how to use the Unified Defense API with TensorFlow
to create radiation-tolerant neural networks.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Add the rad_ml package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

# Import the rad_ml unified defense API
from rad_ml.unified_defense import (
    initialize,
    shutdown,
    DefenseConfig,
    Environment,
    DefenseStrategy,
    ProtectionLevel,
    UnifiedDefenseSystem,
    ProtectedArray,
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Define a function to create a simple TensorFlow model
def create_simple_model(input_shape=(10,), num_classes=2):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(20, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Create a wrapper for TensorFlow models
class RadiationHardenedTFModel:
    def __init__(self, model, config=None):
        if config is None:
            config = DefenseConfig()

        self.model = model
        self.config = config
        self.defense_system = UnifiedDefenseSystem(config)

        # Save original weights for recovery
        self.original_weights = self.model.get_weights()

    def predict(self, inputs):
        """Make predictions with radiation protection"""
        # Protect the input data
        protected_inputs = self.defense_system.protect_array(inputs)

        # Execute prediction with protection
        result = self.defense_system.execute_protected(
            lambda: self.model.predict(protected_inputs.get())
        )

        return result["value"]

    def repair(self):
        """Restore original weights"""
        self.model.set_weights(self.original_weights)
        return True

    def update_environment(self, environment):
        """Update the radiation environment"""
        self.defense_system.update_environment(environment)


# Function to protect a TensorFlow model
def protect_tf_model(model, config=None):
    """Wrap a TensorFlow model with radiation protection"""
    return RadiationHardenedTFModel(model, config)


# Function to simulate radiation-induced bit flips
def simulate_bit_flips(array, bit_flip_prob=0.001):
    """Simulate random bit flips in a numpy array"""
    # Get the byte representation
    bytes_view = array.view(np.uint8)

    # Generate random bit flips
    num_bytes = bytes_view.size
    num_flips = int(num_bytes * 8 * bit_flip_prob)

    if num_flips > 0:
        # Select random byte positions
        byte_positions = np.random.randint(0, num_bytes, num_flips)

        # Select random bit positions within each byte
        bit_positions = np.random.randint(0, 8, num_flips)

        # Flip the bits
        for byte_pos, bit_pos in zip(byte_positions, bit_positions):
            bytes_view.flat[byte_pos] ^= 1 << bit_pos

    return array


# Function to generate synthetic data
def generate_data(num_samples=100, input_size=10, num_classes=2):
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y


# Main function
def main():
    # Initialize the rad_ml framework
    initialize()

    print("Creating models and configurations...")

    # Create defense configurations for different environments
    earth_config = DefenseConfig.for_environment(Environment.EARTH)
    leo_config = DefenseConfig.for_environment(Environment.LEO)
    jupiter_config = DefenseConfig.for_environment(Environment.JUPITER)

    # Create a custom configuration
    custom_config = DefenseConfig()
    custom_config.strategy = DefenseStrategy.MULTI_LAYERED
    custom_config.protection_level = ProtectionLevel.FULL_TMR

    # Generate synthetic data
    X_train, y_train = generate_data(500, 10, 2)
    X_test, y_test = generate_data(100, 10, 2)

    # Create and train a standard model
    print("Training the standard model...")
    standard_model = create_simple_model()
    standard_model.fit(
        X_train, y_train, epochs=5, verbose=1, batch_size=32, validation_split=0.2
    )

    # Evaluate the standard model
    baseline_loss, baseline_accuracy = standard_model.evaluate(
        X_test, y_test, verbose=0
    )
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    # Create protected models with different configurations
    protected_models = {
        "Standard": standard_model,
        "Earth Protection": protect_tf_model(create_simple_model(), earth_config),
        "LEO Protection": protect_tf_model(create_simple_model(), leo_config),
        "Jupiter Protection": protect_tf_model(create_simple_model(), jupiter_config),
        "Custom Protection": protect_tf_model(create_simple_model(), custom_config),
    }

    # Copy trained weights to protected models
    for name, model in protected_models.items():
        if name != "Standard":
            model.model.set_weights(standard_model.get_weights())
            # Update original weights reference
            model.original_weights = model.model.get_weights()

    # Evaluate models under different radiation conditions
    print("\nEvaluating models under radiation conditions...")

    # Store results
    accuracy_results = {name: [] for name in protected_models}

    # Test with different radiation strengths
    radiation_strengths = np.linspace(0, 0.02, 10)  # From 0 to 2% bit flip probability

    for rad_strength in radiation_strengths:
        print(f"Testing with {rad_strength:.4f} bit flip probability...")

        # Corrupt test data with radiation
        X_test_corrupted = simulate_bit_flips(X_test.copy(), rad_strength)

        # Evaluate each model
        model_accuracies = {}

        # Standard model (no protection)
        standard_preds = standard_model.predict(X_test_corrupted, verbose=0)
        standard_accuracy = np.mean(np.argmax(standard_preds, axis=1) == y_test)
        accuracy_results["Standard"].append(standard_accuracy)
        model_accuracies["Standard"] = standard_accuracy

        # Protected models
        for name, model in protected_models.items():
            if name != "Standard":
                # Make predictions with protection
                protected_preds = model.predict(X_test_corrupted)
                protected_accuracy = np.mean(
                    np.argmax(protected_preds, axis=1) == y_test
                )
                accuracy_results[name].append(protected_accuracy)
                model_accuracies[name] = protected_accuracy

        # Log accuracies
        accuracy_str = ", ".join(
            [f"{name}: {acc:.4f}" for name, acc in model_accuracies.items()]
        )
        print(f"  {accuracy_str}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in accuracy_results.items():
        plt.plot(radiation_strengths * 100, accuracies, marker="o", label=model_name)

    plt.xlabel("Radiation Strength (% bit flip probability)")
    plt.ylabel("Accuracy")
    plt.title("TensorFlow Model Accuracy Under Radiation")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("tensorflow_radiation_comparison.png")
    print("\nResults plot saved as 'tensorflow_radiation_comparison.png'")

    # Demonstrate environment updates
    print("\nDemonstrating dynamic environment updates...")

    # Get a protected model
    leo_model = protected_models["LEO Protection"]

    # Make a prediction in LEO environment
    leo_result = leo_model.predict(X_test[:1])
    print(f"Prediction in LEO environment: {np.argmax(leo_result[0])}")

    # Update to Jupiter environment (stronger radiation)
    print("Updating to Jupiter environment (higher radiation)...")
    leo_model.update_environment(Environment.JUPITER)

    # Make a prediction in Jupiter environment
    jupiter_result = leo_model.predict(X_test[:1])
    print(f"Prediction in Jupiter environment: {np.argmax(jupiter_result[0])}")

    # Demonstrate value protection
    print("\nDemonstrating value protection...")
    defense = UnifiedDefenseSystem(jupiter_config)

    # Protect model weights (first layer only for demonstration)
    first_layer_weights = standard_model.layers[0].get_weights()[0]
    protected_weights = defense.protect_array(first_layer_weights)

    # Corrupt weights
    corrupted_weights = simulate_bit_flips(first_layer_weights.copy(), 0.1)

    # Calculate error
    error = np.sum(np.abs(corrupted_weights - first_layer_weights))
    print(f"Error introduced by bit flips: {error:.4f}")

    # Shutdown the framework
    shutdown()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
