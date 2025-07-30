#!/usr/bin/env python3
"""
LibTorch Python Standalone Test

This script tests basic LibTorch functionality from Python to ensure
the installation is working correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def test_basic_tensor_operations():
    """Test basic tensor operations"""
    print("1. Testing basic tensor operations...")

    # Create tensors
    tensor = torch.randn(3, 4)
    print(f"Created tensor:\n{tensor}")

    tensor2 = torch.ones(3, 4)
    print(f"Created ones tensor:\n{tensor2}")

    # Basic operations
    result = tensor + tensor2
    print(f"Addition result:\n{result}")

    # Mathematical operations
    sin_result = torch.sin(tensor)
    print(f"Sin of tensor:\n{sin_result}")

    mean_val = torch.mean(tensor)
    print(f"Mean of tensor: {mean_val.item():.6f}")

def test_neural_network():
    """Test neural network functionality"""
    print("\n2. Testing neural network...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(4, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )
    print(f"Created model: {model}")

    # Create input
    input_tensor = torch.randn(2, 4)
    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

def test_optimizer():
    """Test optimizer functionality"""
    print("\n3. Testing optimizer...")

    model = nn.Linear(4, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"Created SGD optimizer with lr=0.01")

    # Create dummy data
    x = torch.randn(10, 4)
    y = torch.randn(10, 2)

    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer.step()

    print(f"Training step completed, loss: {loss.item():.6f}")

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n4. Testing CUDA availability...")

    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

        # Test moving tensor to GPU
        tensor = torch.randn(3, 4)
        gpu_tensor = tensor.cuda()
        print(f"Moved tensor to GPU successfully")
        print(f"GPU tensor device: {gpu_tensor.device}")
    else:
        print("CUDA is not available, using CPU only")

def test_complex_operations():
    """Test complex mathematical operations"""
    print("\n5. Testing complex operations...")

    # Matrix operations
    matrix = torch.randn(5, 5)
    eigenvals = torch.linalg.eigvals(matrix)
    print(f"Eigenvalues of 5x5 matrix:\n{eigenvals}")

    # SVD
    U, S, V = torch.linalg.svd(matrix)
    print(f"SVD completed, singular values:\n{S}")

def test_serialization():
    """Test model serialization"""
    print("\n6. Testing serialization...")

    model = nn.Linear(4, 2)

    # Save model
    torch.save(model.state_dict(), "test_model_python.pt")
    print("Saved model to test_model_python.pt")

    # Load model
    new_model = nn.Linear(4, 2)
    new_model.load_state_dict(torch.load("test_model_python.pt"))
    print("Loaded model from test_model_python.pt")

    # Test that models are the same
    test_input = torch.randn(1, 4)
    output1 = model(test_input)
    output2 = new_model(test_input)

    if torch.allclose(output1, output2):
        print("Serialization test passed - models produce same output")
    else:
        print("Serialization test failed - models produce different output")

def test_version_info():
    """Test version information"""
    print("\n7. PyTorch version info...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")

def main():
    """Main test function"""
    print("=== LibTorch Python Standalone Test ===")

    try:
        test_basic_tensor_operations()
        test_neural_network()
        test_optimizer()
        test_cuda_availability()
        test_complex_operations()
        test_serialization()
        test_version_info()

        print("\n✅ All LibTorch Python tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    # Clean up
    import os
    if os.path.exists("test_model_python.pt"):
        os.remove("test_model_python.pt")
        print("Cleaned up test model file")

    return 0

if __name__ == "__main__":
    exit(main())
