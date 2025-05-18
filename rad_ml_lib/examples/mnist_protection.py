#!/usr/bin/env python3
"""
MNIST with Radiation Protection Example

This example demonstrates how to use the rad_ml_lib library to protect a neural network
against radiation effects while training and evaluating on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

from rad_ml_lib import (
    initialize,
    shutdown,
    Environment,
    DefenseStrategy,
    ProtectionLevel,
    DefenseConfig,
    protect_network,
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleNN(nn.Module):
    """A simple neural network for MNIST classification."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_data():
    """Load the MNIST dataset."""
    # Define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    return train_loader, test_loader


def train(model, train_loader, epochs=5):
    """Train the model on the MNIST dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.3f}"
                )
                running_loss = 0.0

    print("Finished Training")
    return model


def evaluate(model, test_loader, radiation_strength=0.0):
    """Evaluate the model on the test dataset with optional radiation effects."""
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    errors_detected = 0
    errors_corrected = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply radiation effects if strength > 0
            if radiation_strength > 0 and hasattr(model, "detect_and_correct_errors"):
                # This works for protected models
                outputs, stats = model.detect_and_correct_errors(inputs)
                errors_detected += stats["errors_detected"]
                errors_corrected += stats["errors_corrected"]
            else:
                # Standard forward pass
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy:.2f}% [Radiation Strength: {radiation_strength:.1f}x]")
    if radiation_strength > 0 and hasattr(model, "detect_and_correct_errors"):
        print(
            f"Errors Detected: {errors_detected}, Errors Corrected: {errors_corrected}"
        )

    return accuracy, errors_detected, errors_corrected


def compare_protection_strategies():
    """Compare different protection strategies under radiation."""
    # Load data
    train_loader, test_loader = load_data()

    # Create baseline model
    base_model = SimpleNN()
    trained_model = train(base_model, train_loader, epochs=3)

    # Define radiation strengths to test
    radiation_strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Define protection strategies to compare
    strategies = {
        "None": None,
        "FULL_TMR": DefenseConfig(
            strategy=DefenseStrategy.MULTI_LAYERED,
            protection_level=ProtectionLevel.FULL_TMR,
        ),
        "Reed-Solomon": DefenseConfig(
            strategy=DefenseStrategy.REED_SOLOMON,
            protection_level=ProtectionLevel.FULL_TMR,
        ),
        "Jupiter-Optimized": DefenseConfig.for_environment(Environment.JUPITER),
    }

    # Store results
    results = {}

    # Test each strategy
    for name, config in strategies.items():
        print(f"\nTesting {name} protection strategy")

        # Create a new model copy with the same weights
        model_copy = SimpleNN()
        model_copy.load_state_dict(trained_model.state_dict())

        # Apply protection if config is not None
        if config is not None:
            protected_model = protect_network(model_copy, config)
        else:
            protected_model = model_copy

        # Evaluate under different radiation strengths
        accuracies = []
        detections = []
        corrections = []

        for strength in radiation_strengths:
            print(f"\nEvaluating with radiation strength {strength:.1f}x")
            accuracy, detected, corrected = evaluate(
                protected_model, test_loader, strength
            )
            accuracies.append(accuracy)
            detections.append(detected)
            corrections.append(corrected)

        results[name] = {
            "accuracies": accuracies,
            "detections": detections,
            "corrections": corrections,
        }

    # Plot results
    plt.figure(figsize=(10, 6))

    for name, data in results.items():
        plt.plot(radiation_strengths, data["accuracies"], marker="o", label=name)

    plt.xlabel("Radiation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title("Protection Strategy Comparison under Radiation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.savefig("protection_comparison.png")
    plt.close()

    print("\nResults saved to protection_comparison.png")

    # Plot error detection and correction
    plt.figure(figsize=(12, 8))

    # Skip unprotected model
    for name, data in results.items():
        if name != "None":
            plt.plot(
                radiation_strengths,
                data["detections"],
                marker="o",
                linestyle="-",
                label=f"{name} Detected",
            )
            plt.plot(
                radiation_strengths,
                data["corrections"],
                marker="x",
                linestyle="--",
                label=f"{name} Corrected",
            )

    plt.xlabel("Radiation Strength")
    plt.ylabel("Number of Errors")
    plt.title("Error Detection and Correction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.savefig("error_correction.png")
    plt.close()

    print("Error correction results saved to error_correction.png")


def main():
    parser = argparse.ArgumentParser(
        description="MNIST with Radiation Protection Example"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare protection strategies"
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="JUPITER",
        choices=["EARTH", "LEO", "GEO", "LUNAR", "MARS", "JUPITER", "SOLAR_STORM"],
        help="Radiation environment",
    )
    args = parser.parse_args()

    # Initialize the rad_ml_lib library
    initialize()

    if args.compare:
        compare_protection_strategies()
    else:
        # Load data
        train_loader, test_loader = load_data()

        # Create and train base model
        base_model = SimpleNN()
        trained_model = train(base_model, train_loader, epochs=3)

        # Get environment
        env = getattr(Environment, args.environment)

        # Create protected model for the specified environment
        config = DefenseConfig.for_environment(env)
        protected_model = protect_network(trained_model, config)

        print(f"\nEvaluating base model (no radiation)")
        evaluate(trained_model, test_loader)

        print(
            f"\nEvaluating protected model with {args.environment} config (no radiation)"
        )
        evaluate(protected_model, test_loader)

        print(f"\nEvaluating under radiation effects:")
        for strength in [1.0, 3.0, 5.0]:
            print(f"\nWith radiation strength {strength:.1f}x:")
            print("Base model:")
            evaluate(trained_model, test_loader, strength)
            print(f"\nProtected model ({args.environment} config):")
            evaluate(protected_model, test_loader, strength)

    # Shutdown the rad_ml_lib library
    shutdown()


if __name__ == "__main__":
    main()
