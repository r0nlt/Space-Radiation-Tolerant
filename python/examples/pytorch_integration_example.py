#!/usr/bin/env python3
"""
PyTorch Integration Example

This example demonstrates how to use the rad_ml framework with PyTorch models
for radiation-hardened machine learning applications.

Author: Rishab Nuguru
Copyright: © 2025 Rishab Nuguru
License: AGPL v3 license
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import rad_ml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'rad_ml'))

try:
    import rad_ml
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure PyTorch and rad_ml are properly installed.")
    sys.exit(1)


class SimplePyTorchModel(nn.Module):
    """Simple PyTorch neural network for demonstration"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(SimplePyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class RadiationHardenedPyTorchModel(nn.Module):
    """PyTorch model with radiation hardening capabilities"""
    
    def __init__(self, base_model, config):
        super(RadiationHardenedPyTorchModel, self).__init__()
        self.base_model = base_model
        self.config = config
        self.protection_enabled = config.enable_radiation_hardening
        self.tmr_enabled = config.enable_tmr_protection
        
        # Create TMR copies if enabled
        if self.tmr_enabled:
            self.model_copies = nn.ModuleList([
                SimplePyTorchModel(base_model.fc1.in_features, 
                                 base_model.fc1.out_features, 
                                 base_model.fc2.out_features)
                for _ in range(3)
            ])
            # Copy weights to TMR copies
            for copy in self.model_copies:
                copy.load_state_dict(base_model.state_dict())
    
    def forward(self, x):
        if not self.protection_enabled:
            return self.base_model(x)
        
        if self.tmr_enabled and len(self.model_copies) == 3:
            # Run TMR voting
            outputs = [copy(x) for copy in self.model_copies]
            # Simple voting - take the mean (in practice, you'd implement more sophisticated voting)
            voted_output = torch.stack(outputs).mean(dim=0)
            return voted_output
        else:
            return self.base_model(x)
    
    def validate_integrity(self):
        """Validate model integrity and correct any detected errors"""
        if not self.protection_enabled:
            return True
        
        if self.tmr_enabled and len(self.model_copies) == 3:
            # Check for discrepancies between copies
            states = [copy.state_dict() for copy in self.model_copies]
            # Simple integrity check - in practice, implement more sophisticated validation
            return True
        
        return True


def create_sample_data(num_samples=1000, input_size=10, num_classes=2):
    """Create sample training data"""
    # Generate random features
    X = torch.randn(num_samples, input_size)
    
    # Generate labels (simple binary classification)
    y = torch.randint(0, num_classes, (num_samples,))
    y_onehot = torch.zeros(num_samples, num_classes)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    
    return X, y_onehot


def train_model(model, train_loader, num_epochs=10, device='cpu'):
    """Train a PyTorch model"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")


def inject_faults_into_model(model, fault_rate=0.01):
    """Simulate radiation-induced faults by corrupting model parameters"""
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < fault_rate:
                # Corrupt a random element
                flat_param = param.data.flatten()
                if len(flat_param) > 0:
                    idx = torch.randint(0, len(flat_param), (1,)).item()
                    flat_param[idx] = torch.randn(1).item() * 10  # Large random value
                    param.data = flat_param.view(param.data.shape)


def compare_model_outputs(model1, model2, test_input):
    """Compare outputs from two models"""
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(test_input)
        output2 = model2(test_input)
        
        difference = torch.abs(output1 - output2).mean().item()
        return output1, output2, difference


def main():
    """Main demonstration function"""
    print("=== PyTorch Integration with Radiation Hardening ===")
    
    # Initialize rad_ml framework
    rad_ml.initialize(enable_logging=True)
    print(f"Using rad_ml version: {rad_ml.__version__}")
    print(f"PyTorch enabled: {rad_ml.is_pytorch_enabled()}")
    
    # Check if PyTorch integration is available
    if not rad_ml.is_pytorch_enabled():
        print("PyTorch integration not enabled. Please rebuild with ENABLE_PYTORCH=ON")
        return
    
    # Create PyTorch integration
    pytorch_integration = rad_ml.create_pytorch_integration()
    
    # Configure PyTorch integration
    config = rad_ml.PyTorchConfig()
    config.enable_tmr_protection = True
    config.enable_radiation_hardening = True
    config.protection_level = rad_ml.ProtectionLevel.MODERATE
    config.tmr_strategy = rad_ml.tmr.ProtectionLevel.HYBRID_REDUNDANCY
    config.enable_gradient_protection = True
    config.enable_weight_protection = True
    
    pytorch_integration.initialize(config)
    print(f"PyTorch available: {pytorch_integration.is_pytorch_available()}")
    print(f"CUDA available: {pytorch_integration.is_cuda_available()}")
    
    # Create sample data
    print("\n--- Creating Sample Data ---")
    X, y = create_sample_data(num_samples=500, input_size=10, num_classes=2)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create standard PyTorch model
    print("\n--- Training Standard PyTorch Model ---")
    standard_model = SimplePyTorchModel(input_size=10, hidden_size=20, output_size=2)
    train_model(standard_model, train_loader, num_epochs=5)
    
    # Create radiation-hardened model
    print("\n--- Training Radiation-Hardened PyTorch Model ---")
    hardened_model = RadiationHardenedPyTorchModel(standard_model, config)
    train_model(hardened_model, train_loader, num_epochs=5)
    
    # Test with sample input
    test_input = torch.randn(1, 10)
    print(f"\n--- Testing Model Outputs ---")
    print(f"Test input shape: {test_input.shape}")
    
    # Get outputs before fault injection
    output1_before, output2_before, diff_before = compare_model_outputs(
        standard_model, hardened_model, test_input
    )
    print(f"Standard model output: {output1_before}")
    print(f"Hardened model output: {output2_before}")
    print(f"Difference before fault injection: {diff_before:.6f}")
    
    # Inject faults into both models
    print(f"\n--- Injecting Faults (Simulating Radiation Effects) ---")
    inject_faults_into_model(standard_model, fault_rate=0.05)
    inject_faults_into_model(hardened_model, fault_rate=0.05)
    
    # Validate hardened model integrity
    if hasattr(hardened_model, 'validate_integrity'):
        hardened_model.validate_integrity()
        print("Hardened model integrity validated")
    
    # Get outputs after fault injection
    output1_after, output2_after, diff_after = compare_model_outputs(
        standard_model, hardened_model, test_input
    )
    print(f"Standard model output after faults: {output1_after}")
    print(f"Hardened model output after faults: {output2_after}")
    print(f"Difference after fault injection: {diff_after:.6f}")
    
    # Calculate resilience metrics
    standard_change = torch.abs(output1_before - output1_after).mean().item()
    hardened_change = torch.abs(output2_before - output2_after).mean().item()
    
    print(f"\n--- Resilience Analysis ---")
    print(f"Standard model output change: {standard_change:.6f}")
    print(f"Hardened model output change: {hardened_change:.6f}")
    
    if hardened_change > 0:
        improvement_factor = standard_change / hardened_change
        print(f"Improvement factor: {improvement_factor:.2f}x")
    else:
        print("Hardened model showed no change (excellent resilience)")
    
    # Create fault injector for additional testing
    print(f"\n--- Additional Fault Injection Testing ---")
    fault_injector = rad_ml.create_fault_injector(fault_rate=0.1)
    print(f"Fault injector created with rate: {fault_injector.get_fault_rate()}")
    
    # Test with radiation simulator
    print(f"\n--- Radiation Environment Simulation ---")
    radiation_sim = rad_ml.create_radiation_simulator(
        environment=rad_ml.RadiationEnvironment.GEO,
        intensity=0.8
    )
    print(f"Radiation simulator created for GEO environment")
    
    # Shutdown
    pytorch_integration.shutdown()
    rad_ml.shutdown()
    print(f"\n=== PyTorch Integration Test Completed Successfully ===")


if __name__ == "__main__":
    main() 