#!/usr/bin/env python3
"""
LibTorch macOS Example
Simple demonstration of LibTorch functionality on macOS

@author Rishab Nuguru
@copyright © 2025 Rishab Nuguru
@license AGPL v3 license
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

def demonstrate_basic_operations():
    """Demonstrate basic tensor operations"""
    print("=== Basic Tensor Operations ===")

    # Create tensors
    tensor1 = torch.randn(5, 5)
    tensor2 = torch.ones(5, 5)

    print(f"Tensor 1:\n{tensor1}")
    print(f"Tensor 2:\n{tensor2}")

    # Basic operations
    sum_tensor = tensor1 + tensor2
    prod_tensor = tensor1 * tensor2
    matmul_result = torch.matmul(tensor1, tensor1.t())

    print(f"Sum:\n{sum_tensor}")
    print(f"Product:\n{prod_tensor}")
    print(f"Matrix multiplication:\n{matmul_result}")

    # Mathematical functions
    sin_tensor = torch.sin(tensor1)
    exp_tensor = torch.exp(tensor1)

    print(f"Sin:\n{sin_tensor}")
    print(f"Exponential:\n{exp_tensor}")

def demonstrate_neural_network():
    """Demonstrate neural network operations"""
    print("\n=== Neural Network Operations ===")

    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    print(f"Model architecture:\n{model}")

    # Create input data
    input_data = torch.randn(32, 10)
    target_data = torch.randn(32, 1)

    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target_data.shape}")

    # Forward pass
    output = model(input_data)
    print(f"Output shape: {output.shape}")

    # Calculate loss
    loss = F.mse_loss(output, target_data)
    print(f"Initial loss: {loss.item():.6f}")

    # Backward pass and optimization
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining for 10 epochs...")
    for epoch in range(10):
        # Forward pass
        output = model(input_data)
        loss = F.mse_loss(output, target_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}")

def demonstrate_performance():
    """Demonstrate performance characteristics"""
    print("\n=== Performance Demonstration ===")

    # Test different tensor sizes
    sizes = [50, 100, 200]

    for size in sizes:
        start_time = time.time()

        # Create tensors and perform operations
        tensor1 = torch.randn(size, size)
        tensor2 = torch.randn(size, size)

        # Matrix multiplication
        result = torch.matmul(tensor1, tensor2)

        # Additional operations
        mean_val = torch.mean(result)
        std_val = torch.std(result)

        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds

        print(f"Size {size}x{size}: {duration:.2f}ms")

    # Test threading performance
    print("\nTesting threading performance...")

    import threading

    def worker_function(worker_id, results):
        tensor = torch.randn(30, 30)
        result = torch.matmul(tensor, tensor.t())
        mean_val = torch.mean(result)
        results[worker_id] = mean_val.item()

    # Run with multiple threads
    threads = []
    results = [0] * 4

    start_time = time.time()

    for i in range(4):
        thread = threading.Thread(target=worker_function, args=(i, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    duration = (end_time - start_time) * 1000

    print(f"Threading test: {duration:.2f}ms")
    print(f"Results: {[f'{r:.4f}' for r in results]}")

def demonstrate_memory_management():
    """Demonstrate memory management"""
    print("\n=== Memory Management ===")

    import gc

    # Test memory allocation and deallocation
    print("Creating tensors...")
    tensors = []

    for i in range(10):
        tensor = torch.randn(100, 100)
        tensors.append(tensor)
        print(f"Created tensor {i+1}")

    # Perform operations
    print("Performing operations...")
    for i in range(5):
        result = torch.matmul(tensors[i], tensors[i+1])
        mean_val = torch.mean(result)
        print(f"Operation {i+1}: Mean = {mean_val.item():.4f}")

    # Clean up
    print("Cleaning up...")
    del tensors
    gc.collect()
    print("Memory cleanup completed")

def demonstrate_serialization():
    """Demonstrate model serialization"""
    print("\n=== Model Serialization ===")

    # Create a model
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # Create some data
    input_data = torch.randn(10, 5)
    output = model(input_data)

    print(f"Original model output:\n{output}")

    # Save the model
    torch.save(model.state_dict(), "macos_example_model.pt")
    print("Model saved successfully")

    # Load the model
    loaded_model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    loaded_model.load_state_dict(torch.load("macos_example_model.pt"))

    # Test the loaded model
    loaded_output = loaded_model(input_data)
    print(f"Loaded model output:\n{loaded_output}")

    # Verify they're the same
    diff = torch.abs(output - loaded_output)
    if torch.sum(diff).item() < 1e-6:
        print("✅ Serialization test passed!")
    else:
        print("❌ Serialization test failed!")

    # Clean up
    import os
    os.remove("macos_example_model.pt")

def main():
    """Main demonstration function"""
    print("🎉 LibTorch macOS Demonstration")
    print("=" * 50)

    # System information
    print(f"Platform: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()

    try:
        # Run demonstrations
        demonstrate_basic_operations()
        demonstrate_neural_network()
        demonstrate_performance()
        demonstrate_memory_management()
        demonstrate_serialization()

        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("Your macOS LibTorch setup is working perfectly!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
