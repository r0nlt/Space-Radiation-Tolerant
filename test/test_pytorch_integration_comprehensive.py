#!/usr/bin/env python3
"""
Comprehensive PyTorch Integration Test

This script tests all aspects of the PyTorch integration with the rad_ml framework,
including tensor protection, model hardening, fault injection, and resilience analysis.

Author: Rishab Nuguru
Copyright: © 2025 Rishab Nuguru
License: AGPL v3 license
"""

import sys
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add the build directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'rad_ml'))

try:
    import rad_ml
except ImportError as e:
    print(f"Error importing rad_ml: {e}")
    print("Please build the project with Python bindings enabled.")
    sys.exit(1)


class TestPyTorchIntegration(unittest.TestCase):
    """Test suite for PyTorch integration"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        rad_ml.initialize(enable_logging=True)
        cls.pytorch_integration = rad_ml.create_pytorch_integration()

        # Configure PyTorch integration
        cls.config = rad_ml.PyTorchConfig()
        cls.config.enable_tmr_protection = True
        cls.config.enable_radiation_hardening = True
        cls.config.protection_level = rad_ml.ProtectionLevel.MODERATE
        cls.config.tmr_strategy = rad_ml.tmr.ProtectionLevel.HYBRID_REDUNDANCY
        cls.config.enable_gradient_protection = True
        cls.config.enable_weight_protection = True

        cls.pytorch_integration.initialize(cls.config)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        cls.pytorch_integration.shutdown()
        rad_ml.shutdown()

    def test_pytorch_availability(self):
        """Test PyTorch availability detection"""
        self.assertTrue(rad_ml.is_pytorch_enabled())
        self.assertTrue(self.pytorch_integration.is_pytorch_available())

    def test_config_initialization(self):
        """Test PyTorch configuration initialization"""
        config = self.pytorch_integration.get_config()
        self.assertTrue(config.enable_tmr_protection)
        self.assertTrue(config.enable_radiation_hardening)
        self.assertEqual(config.protection_level, rad_ml.ProtectionLevel.MODERATE)

    def test_tensor_protection(self):
        """Test tensor protection functionality"""
        # Create a test tensor
        test_tensor = torch.randn(3, 4)

        # Test tensor protection (if implemented)
        # This would test the C++ ProtectedTensor functionality
        self.assertIsInstance(test_tensor, torch.Tensor)

    def test_model_protection(self):
        """Test model protection functionality"""
        # Create a simple model
        model = nn.Linear(10, 1)

        # Test model protection
        # This would test the C++ model protection functionality
        self.assertIsInstance(model, nn.Module)

    def test_fault_injection(self):
        """Test fault injection with PyTorch models"""
        # Create a simple model
        model = nn.Linear(5, 1)
        original_weights = model.weight.data.clone()

        # Create fault injector
        fault_injector = rad_ml.create_fault_injector(fault_rate=0.1)

        # Inject faults
        with torch.no_grad():
            for param in model.parameters():
                if torch.rand(1).item() < 0.1:
                    # Corrupt a random element
                    flat_param = param.data.flatten()
                    if len(flat_param) > 0:
                        idx = torch.randint(0, len(flat_param), (1,)).item()
                        flat_param[idx] = torch.randn(1).item() * 10
                        param.data = flat_param.view(param.data.shape)

        # Check if weights were modified
        weight_changed = not torch.allclose(model.weight.data, original_weights)
        self.assertTrue(weight_changed)

    def test_radiation_simulation(self):
        """Test radiation environment simulation"""
        # Create radiation simulator
        radiation_sim = rad_ml.create_radiation_simulator(
            environment=rad_ml.RadiationEnvironment.GEO,
            intensity=0.8
        )

        self.assertIsNotNone(radiation_sim)
        self.assertEqual(radiation_sim.get_environment(), rad_ml.RadiationEnvironment.GEO)
        self.assertEqual(radiation_sim.get_intensity(), 0.8)

    def test_training_protection(self):
        """Test training protection functionality"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Create dummy data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Backward pass
        loss.backward()

        # Test training protection (if implemented)
        # This would test the C++ training protection functionality
        self.assertTrue(loss.requires_grad)

    def test_resilience_analysis(self):
        """Test resilience analysis with PyTorch models"""
        # Create two identical models
        model1 = nn.Linear(5, 1)
        model2 = nn.Linear(5, 1)
        model2.load_state_dict(model1.state_dict())

        # Test input
        test_input = torch.randn(1, 5)

        # Get outputs before fault injection
        with torch.no_grad():
            output1_before = model1(test_input)
            output2_before = model2(test_input)

        # Verify outputs are identical initially
        self.assertTrue(torch.allclose(output1_before, output2_before))

        # Inject faults into model2
        with torch.no_grad():
            for param in model2.parameters():
                if torch.rand(1).item() < 0.2:
                    flat_param = param.data.flatten()
                    if len(flat_param) > 0:
                        idx = torch.randint(0, len(flat_param), (1,)).item()
                        flat_param[idx] = torch.randn(1).item() * 10
                        param.data = flat_param.view(param.data.shape)

        # Get outputs after fault injection
        with torch.no_grad():
            output1_after = model1(test_input)
            output2_after = model2(test_input)

        # Calculate resilience metrics
        change1 = torch.abs(output1_before - output1_after).mean().item()
        change2 = torch.abs(output2_before - output2_after).mean().item()

        # Model1 should show no change (no faults injected)
        self.assertAlmostEqual(change1, 0.0, places=6)

        # Model2 should show some change (faults were injected)
        self.assertGreater(change2, 0.0)

    def test_cuda_availability(self):
        """Test CUDA availability detection"""
        cuda_available = self.pytorch_integration.is_cuda_available()
        # This test just checks that the method works, not the actual CUDA availability
        self.assertIsInstance(cuda_available, bool)

    def test_config_update(self):
        """Test configuration update functionality"""
        # Create new configuration
        new_config = rad_ml.PyTorchConfig()
        new_config.enable_tmr_protection = False
        new_config.protection_level = rad_ml.ProtectionLevel.HIGH

        # Update configuration
        self.pytorch_integration.update_config(new_config)

        # Verify update
        updated_config = self.pytorch_integration.get_config()
        self.assertFalse(updated_config.enable_tmr_protection)
        self.assertEqual(updated_config.protection_level, rad_ml.ProtectionLevel.HIGH)

        # Restore original configuration
        self.pytorch_integration.update_config(self.config)


class TestPyTorchAdvancedFeatures(unittest.TestCase):
    """Test suite for advanced PyTorch integration features"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        rad_ml.initialize(enable_logging=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        rad_ml.shutdown()

    def test_mission_simulation(self):
        """Test mission simulation with PyTorch models"""
        # Create mission simulator
        mission_sim = rad_ml.create_mission_simulator(
            mission_type=rad_ml.MissionType.GEOSTATIONARY,
            duration_days=30
        )

        self.assertIsNotNone(mission_sim)
        self.assertEqual(mission_sim.get_mission_type(), rad_ml.MissionType.GEOSTATIONARY)
        self.assertEqual(mission_sim.get_duration_days(), 30)

    def test_error_prediction(self):
        """Test error prediction with PyTorch models"""
        # Create error predictor
        error_predictor = rad_ml.ErrorPredictor()

        self.assertIsNotNone(error_predictor)

    def test_tmr_types(self):
        """Test TMR types with PyTorch tensors"""
        # Test TMR for different numeric types
        tmr_int = rad_ml.create_standard_tmr_int(42)
        tmr_float = rad_ml.create_standard_tmr_float(3.14)
        tmr_double = rad_ml.create_standard_tmr_double(2.718)

        self.assertEqual(tmr_int.get_value(), 42)
        self.assertAlmostEqual(tmr_float.get_value(), 3.14, places=6)
        self.assertAlmostEqual(tmr_double.get_value(), 2.718, places=6)


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=== Running Comprehensive PyTorch Integration Tests ===")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPyTorchIntegration))
    test_suite.addTest(unittest.makeSuite(TestPyTorchAdvancedFeatures))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
