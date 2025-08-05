#!/usr/bin/env python3


import rad_ml
import numpy as np


class SimpleNetwork:
    """Simple neural network implementation for demonstration"""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Softmax activation
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs


def main():
    # Initialize the rad_ml framework
    rad_ml.initialize(enable_logging=True)

    print(f"Using rad_ml version: {rad_ml.__version__}")

    # Create a simple neural network
    net = SimpleNetwork(input_size=10, hidden_size=20, output_size=2)

    # Create a hardened version with selective protection
    hardened_net = rad_ml.create_protected_network(
        model=net,
        strategy=rad_ml.HardeningStrategy.CRITICAL_LAYERS,
        protection_level=rad_ml.ProtectionLevel.MODERATE,
    )

    # Generate a sample input
    sample_input = np.random.rand(1, 10)

    # Run inference on both networks
    standard_output = net.forward(sample_input)
    hardened_output = hardened_net.forward(sample_input)

    print("\nStandard Network Output:")
    print(standard_output)

    print("\nHardened Network Output:")
    print(hardened_output)

    # Create a fault injector to simulate radiation effects
    fault_injector = rad_ml.create_fault_injector(fault_rate=0.05)

    # Inject faults into both networks
    print("\nInjecting faults...")
    fault_injector.inject_fault(net.W1)
    fault_injector.inject_fault(hardened_net.W1)

    # Run inference again after fault injection
    standard_output_after = net.forward(sample_input)
    hardened_output_after = hardened_net.forward(sample_input)

    print("\nStandard Network Output After Fault Injection:")
    print(standard_output_after)

    print("\nHardened Network Output After Fault Injection:")
    print(hardened_output_after)

    # Calculate the difference to see the impact
    standard_diff = np.abs(standard_output - standard_output_after).mean()
    hardened_diff = np.abs(hardened_output - hardened_output_after).mean()

    print(f"\nStandard Network Average Output Change: {standard_diff:.6f}")
    print(f"Hardened Network Average Output Change: {hardened_diff:.6f}")
    print(
        f"Improvement Factor: {standard_diff/hardened_diff if hardened_diff > 0 else 'infinite':.2f}x"
    )

    # Shutdown the framework
    rad_ml.shutdown()
    print("\nFramework shutdown successfully")


if __name__ == "__main__":
    main()
