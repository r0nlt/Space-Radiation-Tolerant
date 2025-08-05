#!/usr/bin/env python3
"""
Test suite for the unified defense API

This module contains tests for all components of the rad_ml unified defense system.
Run with pytest: `pytest test_unified_defense.py -v`
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn

from rad_ml.unified_defense import (
    initialize,
    shutdown,
    DefenseConfig,
    Environment,
    DefenseStrategy,
    ProtectionLevel,
    UnifiedDefenseSystem,
    TMRProtectedValue,
    ProtectedArray,
    protect_network,
    create_protected_network,
    CustomProtectionStrategy,
)


# Test fixture to initialize and shutdown rad_ml for each test
@pytest.fixture(scope="function")
def rad_ml_env():
    initialize()
    yield
    shutdown()


# Simple PyTorch model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Function to simulate radiation effects by flipping bits
def simulate_bit_flips(tensor, bit_flip_prob=0.01):
    # Convert to numpy for bit manipulation
    array = tensor.clone().detach().numpy()

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

    # Convert back to tensor
    return torch.from_numpy(array)


# Custom protection strategy for testing
class TestProtectionStrategy(CustomProtectionStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.protection_used = False

    def protect(self, value):
        self.protection_used = True
        return value

    def was_used(self):
        return self.protection_used


# Test cases
class TestUnifiedDefense:
    def test_defense_config_creation(self, rad_ml_env):
        # Test default config
        config = DefenseConfig()
        assert config.strategy == DefenseStrategy.ENHANCED_TMR
        assert config.environment == Environment.LEO

        # Test environment-specific config
        jupiter_config = DefenseConfig.for_environment(Environment.JUPITER)
        assert jupiter_config.strategy == DefenseStrategy.MULTI_LAYERED
        assert jupiter_config.protection_level == ProtectionLevel.FULL_TMR

        # Test custom config
        custom_config = DefenseConfig(
            strategy=DefenseStrategy.REED_SOLOMON,
            environment=Environment.MARS,
            protection_level=ProtectionLevel.CHECKSUM_ONLY,
        )
        assert custom_config.strategy == DefenseStrategy.REED_SOLOMON
        assert custom_config.environment == Environment.MARS
        assert custom_config.protection_level == ProtectionLevel.CHECKSUM_ONLY

    def test_tmr_protected_value(self, rad_ml_env):
        # Test protecting an integer
        protected_int = TMRProtectedValue(42, DefenseConfig())
        assert protected_int.get() == 42

        # Test setting a new value
        protected_int.set(100)
        assert protected_int.get() == 100

        # Test protecting a float
        protected_float = TMRProtectedValue(3.14159, DefenseConfig())
        assert protected_float.get() == 3.14159

        # Test manual corruption and repair
        # Directly manipulate the copies to simulate bit flips
        protected_float.copies[0] = 0.0  # Corrupt first copy
        assert (
            protected_float.get() == 3.14159
        )  # Should still return correct value (majority)

        # Corrupt two copies
        protected_float.copies[1] = 1.0
        # Now the majority is corrupted, so get() will return a wrong value
        assert protected_float.get() != 3.14159

        # Repair should fix it
        protected_float.copies[2] = 2.0  # Corrupt all three differently to test repair
        protected_float.set(3.14159)  # Reset to original
        assert protected_float.get() == 3.14159

    def test_protected_array(self, rad_ml_env):
        # Create a test array
        original = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Protect the array
        protected = ProtectedArray(original, DefenseConfig())

        # Check that we can get the array
        result = protected.get()
        assert np.array_equal(result, original)

        # Test corruption and repair
        # Modify one copy
        protected.copies[0][0, 0] = 99.0

        # Should still return correct values
        result = protected.get()
        assert np.array_equal(result, original)

        # Corrupt two copies
        protected.copies[1][0, 0] = 88.0
        protected.copies[2][0, 0] = 88.0

        # Now majority is corrupted
        result = protected.get()
        assert result[0, 0] == 88.0

        # Repair should fix it
        protected.repair()

        # After repair, all copies should be the same
        assert np.array_equal(protected.copies[0], protected.copies[1])
        assert np.array_equal(protected.copies[1], protected.copies[2])

    def test_unified_defense_system(self, rad_ml_env):
        # Create a defense system
        system = UnifiedDefenseSystem()

        # Test protecting a value
        protected_value = system.protect_value(3.14159)
        assert protected_value.get() == 3.14159

        # Test protecting an array
        original = np.array([1.0, 2.0, 3.0])
        protected_array = system.protect_array(original)
        assert np.array_equal(protected_array.get(), original)

        # Test executing a protected function
        result = system.execute_protected(lambda: np.sqrt(2.0))
        assert np.isclose(result["value"], 1.4142, atol=1e-4)
        assert not result["error_detected"]

        # Test updating environment
        system.update_environment(Environment.JUPITER)
        # No easy way to verify the change, but it should not raise exceptions

    def test_network_protection(self, rad_ml_env):
        # Create a model
        model = SimpleModel()

        # Create a defense configuration
        config = DefenseConfig.for_environment(Environment.JUPITER)

        # Protect the model
        protected_model = protect_network(model, config)

        # Test forward pass
        input_tensor = torch.randn(1, 10)
        output = protected_model(input_tensor)

        # Output should be the expected shape
        assert output.shape == (1, 2)

        # Test corruption and recovery (simplified)
        # For a real test, we would need more sophisticated validation
        # Save original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Create corrupted input
        corrupted_input = simulate_bit_flips(input_tensor, 0.1)

        # Should still produce output without errors
        output_corrupted = protected_model(corrupted_input)
        assert output_corrupted.shape == (1, 2)

        # Repair should restore original state
        protected_model.repair()

        # Verify state is restored (comparing one parameter is sufficient for test)
        for key in original_state:
            assert torch.allclose(model.state_dict()[key], original_state[key])

    def test_create_protected_network(self, rad_ml_env):
        # Create a network using the helper function
        config = DefenseConfig()
        protected_model = create_protected_network(SimpleModel, config, 5, 3, 1)

        # Test the network
        input_tensor = torch.randn(1, 5)
        output = protected_model(input_tensor)

        # Check output shape
        assert output.shape == (1, 1)

    def test_environment_updates(self, rad_ml_env):
        # Create protected model
        model = SimpleModel()
        protected_model = protect_network(model)

        # Test environment updates
        protected_model.update_environment(Environment.LEO)
        protected_model.update_environment(Environment.SAA)
        protected_model.update_environment(Environment.EARTH)

        # No exceptions should be raised
        assert True

    def test_multiple_models(self, rad_ml_env):
        # Create different configs
        earth_config = DefenseConfig.for_environment(Environment.EARTH)
        jupiter_config = DefenseConfig.for_environment(Environment.JUPITER)

        # Create multiple protected models
        model1 = protect_network(SimpleModel(), earth_config)
        model2 = protect_network(SimpleModel(), jupiter_config)

        # Both should work independently
        input_tensor = torch.randn(1, 10)
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)

        assert output1.shape == (1, 2)
        assert output2.shape == (1, 2)

    def test_custom_strategy(self, rad_ml_env):
        # Create a custom strategy
        strategy = TestProtectionStrategy(DefenseConfig())

        # Use it to protect a value
        value = 42
        protected = strategy.protect(value)

        # Check that the strategy was used
        assert strategy.was_used()
        assert protected == value


if __name__ == "__main__":
    # Manual test execution
    initialize()

    # Create a configuration for Jupiter environment
    jupiter_config = DefenseConfig.for_environment(Environment.JUPITER)
    print(f"Created Jupiter config with strategy: {jupiter_config.strategy}")

    # Create a protected neural network
    model = SimpleModel()
    protected_model = protect_network(model, jupiter_config)

    # Test the protected network
    input_tensor = torch.randn(1, 10)
    output = protected_model(input_tensor)
    print(f"Protected network output shape: {output.shape}")

    # Test protection of a single value
    defense = UnifiedDefenseSystem(jupiter_config)
    protected_pi = defense.protect_value(3.14159)
    print(f"Protected value: {protected_pi.get()}")

    # Cleanup
    shutdown()
    print("Tests completed manually.")
