#!/usr/bin/env python3
"""
LibTorch Python Resilience Test Suite

This script provides comprehensive testing of PyTorch functionality with focus on
resilience, error handling, and integration with the rad_ml framework for
radiation-hardened machine learning applications.

@author Rishab Nuguru
@copyright © 2025 Rishab Nuguru
@license AGPL v3 license
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
import time
import os
import sys
import traceback
from typing import List, Tuple, Optional
import gc
import psutil
import multiprocessing as mp

class LibTorchResilienceTest:
    """Comprehensive LibTorch resilience testing suite"""

    def __init__(self):
        self.test_counter = 0
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        print(f"Testing on device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        if self.cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results with consistent formatting"""
        self.test_counter += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{self.test_counter:2d}] {test_name}: {status}")
        if details:
            print(f"    {details}")
        return passed

    def test_basic_tensor_operations(self) -> bool:
        """Test basic tensor operations and mathematical functions"""
        try:
            # Test tensor creation
            tensor1 = torch.randn(10, 10, device=self.device)
            tensor2 = torch.ones(10, 10, device=self.device)
            tensor3 = torch.zeros(10, 10, device=self.device)
            tensor4 = torch.eye(10, device=self.device)

            # Test arithmetic operations
            sum_tensor = tensor1 + tensor2
            diff_tensor = tensor1 - tensor2
            prod_tensor = tensor1 * tensor2
            div_tensor = tensor1 / (tensor2 + 1e-8)

            # Test mathematical functions
            sin_tensor = torch.sin(tensor1)
            cos_tensor = torch.cos(tensor1)
            exp_tensor = torch.exp(tensor1)
            log_tensor = torch.log(torch.abs(tensor1) + 1e-8)

            # Test reduction operations
            mean_val = torch.mean(tensor1)
            sum_val = torch.sum(tensor1)
            max_val = torch.max(tensor1)
            min_val = torch.min(tensor1)

            # Test shape operations
            reshaped = tensor1.reshape(5, 20)
            transposed = tensor1.t()
            flattened = tensor1.flatten()

            return True
        except Exception as e:
            print(f"Basic tensor operations failed: {e}")
            return False

    def test_advanced_tensor_operations(self) -> bool:
        """Test advanced tensor operations and linear algebra"""
        try:
            # Test matrix operations
            A = torch.randn(100, 100, device=self.device)
            B = torch.randn(100, 100, device=self.device)

            # Matrix multiplication
            C = torch.matmul(A, B)
            C_alt = A @ B

            # Test linear algebra operations
            eigenvals = torch.linalg.eigvals(A)
            U, S, V = torch.linalg.svd(A)
            det_A = torch.linalg.det(A)
            inv_A = torch.linalg.inv(A)

            # Test broadcasting
            tensor_3d = torch.randn(5, 10, 15, device=self.device)
            tensor_2d = torch.randn(10, 15, device=self.device)
            broadcasted = tensor_3d + tensor_2d

            # Test advanced indexing
            indices = torch.randint(0, 10, (5,), device=self.device)
            indexed = A[indices, :]

            return True
        except Exception as e:
            print(f"Advanced tensor operations failed: {e}")
            return False

    def test_neural_network_operations(self) -> bool:
        """Test neural network layers and operations"""
        try:
            # Test various layer types
            linear = nn.Linear(100, 50).to(self.device)
            conv2d = nn.Conv2d(3, 16, 3, padding=1).to(self.device)
            conv1d = nn.Conv1d(10, 20, 3, padding=1).to(self.device)
            lstm = nn.LSTM(10, 20, 2, batch_first=True).to(self.device)
            gru = nn.GRU(10, 20, 2, batch_first=True).to(self.device)

            # Test forward passes
            linear_input = torch.randn(32, 100, device=self.device)
            linear_output = linear(linear_input)

            conv_input = torch.randn(4, 3, 32, 32, device=self.device)
            conv_output = conv2d(conv_input)

            conv1d_input = torch.randn(4, 10, 50, device=self.device)
            conv1d_output = conv1d(conv1d_input)

            lstm_input = torch.randn(4, 10, 10, device=self.device)
            lstm_output, (h_n, c_n) = lstm(lstm_input)

            gru_input = torch.randn(4, 10, 10, device=self.device)
            gru_output, h_n = gru(gru_input)

            # Test activation functions
            relu_output = F.relu(linear_output)
            sigmoid_output = torch.sigmoid(linear_output)
            tanh_output = torch.tanh(linear_output)
            softmax_output = F.softmax(linear_output, dim=1)

            # Test dropout
            dropout = nn.Dropout(0.5)
            dropout_output = dropout(linear_output)

            # Test batch normalization
            bn = nn.BatchNorm1d(50).to(self.device)
            bn_output = bn(linear_output)

            return True
        except Exception as e:
            print(f"Neural network operations failed: {e}")
            return False

    def test_model_training(self) -> bool:
        """Test complete model training pipeline"""
        try:
            # Create a simple model
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            ).to(self.device)

            # Create optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(5):
                # Generate dummy data
                inputs = torch.randn(32, 10, device=self.device)
                targets = torch.randn(32, 1, device=self.device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if epoch % 2 == 0:
                    print(f"    Epoch {epoch}, Loss: {loss.item():.6f}")

            return True
        except Exception as e:
            print(f"Model training failed: {e}")
            return False

    def test_memory_management(self) -> bool:
        """Test memory allocation, deallocation, and garbage collection"""
        try:
            # Test memory allocation
            tensors = []
            for i in range(50):
                tensor = torch.randn(100, 100, device=self.device)
                tensors.append(tensor)

            # Test memory deallocation
            del tensors
            gc.collect()

            if self.cuda_available:
                torch.cuda.empty_cache()

            # Test large tensor allocation
            large_tensor = torch.randn(500, 500, device=self.device)
            del large_tensor
            gc.collect()

            # Test memory fragmentation resistance
            for i in range(20):
                temp_tensor = torch.randn(50, 50, device=self.device)
                result = torch.matmul(temp_tensor, temp_tensor.t())
                del temp_tensor, result
                gc.collect()

            return True
        except Exception as e:
            print(f"Memory management failed: {e}")
            return False

    def test_cuda_operations(self) -> bool:
        """Test CUDA-specific operations"""
        if not self.cuda_available:
            return self.log_test("CUDA Operations", True, "Skipped (CUDA not available)")

        try:
            # Test CUDA tensor creation
            cpu_tensor = torch.randn(100, 100)
            cuda_tensor = cpu_tensor.cuda()

            # Test CUDA operations
            cuda_result = torch.matmul(cuda_tensor, cuda_tensor.t())

            # Test CPU-CUDA transfers
            back_to_cpu = cuda_result.cpu()

            # Test multiple CUDA streams
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()

            with torch.cuda.stream(stream1):
                tensor1 = torch.randn(50, 50, device='cuda')
                result1 = torch.matmul(tensor1, tensor1.t())

            with torch.cuda.stream(stream2):
                tensor2 = torch.randn(50, 50, device='cuda')
                result2 = torch.matmul(tensor2, tensor2.t())

            # Synchronize streams
            stream1.synchronize()
            stream2.synchronize()

            # Test CUDA memory management
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"CUDA operations failed: {e}")
            return False

    def test_serialization(self) -> bool:
        """Test model and tensor serialization"""
        try:
            # Create a model to serialize
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)

            # Test model saving
            torch.save(model.state_dict(), "test_model_python.pt")

            # Test model loading
            loaded_model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)
            loaded_model.load_state_dict(torch.load("test_model_python.pt"))

            # Test tensor serialization
            tensor = torch.randn(10, 10, device=self.device)
            torch.save(tensor, "test_tensor_python.pt")

            loaded_tensor = torch.load("test_tensor_python.pt", map_location=self.device)

            # Verify loaded tensor matches original
            diff = torch.abs(tensor - loaded_tensor)
            tensors_match = torch.sum(diff).item() < 1e-6

            # Clean up
            os.remove("test_model_python.pt")
            os.remove("test_tensor_python.pt")

            return tensors_match
        except Exception as e:
            print(f"Serialization failed: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling and edge cases"""
        try:
            # Test invalid tensor operations
            try:
                tensor1 = torch.randn(3, 4, device=self.device)
                tensor2 = torch.randn(5, 6, device=self.device)
                result = tensor1 + tensor2  # Should raise error
                return False  # Should not reach here
            except RuntimeError:
                pass  # Expected error

            # Test invalid index access
            try:
                tensor = torch.randn(5, 5, device=self.device)
                invalid_access = tensor[10, 10]  # Should raise error
                return False  # Should not reach here
            except IndexError:
                pass  # Expected error

            # Test division by zero handling
            try:
                tensor = torch.zeros(2, 2, device=self.device)
                result = 1.0 / tensor  # Should handle gracefully
            except RuntimeError:
                pass  # May raise error, which is acceptable

            # Test NaN handling
            tensor_with_nan = torch.tensor([1.0, float('nan'), 3.0], device=self.device)
            has_nan = torch.isnan(tensor_with_nan).any().item()

            return True
        except Exception as e:
            print(f"Error handling failed: {e}")
            return False

    def test_concurrent_operations(self) -> bool:
        """Test concurrent tensor operations"""
        try:
            def worker_function(worker_id: int, results: List[bool]):
                try:
                    tensor = torch.randn(50, 50, device=self.device)
                    result = torch.matmul(tensor, tensor.t())
                    mean_val = torch.mean(result)
                    results[worker_id] = True
                except Exception:
                    results[worker_id] = False

            # Test with multiple threads
            num_threads = 4
            results = [False] * num_threads
            threads = []

            for i in range(num_threads):
                thread = threading.Thread(target=worker_function, args=(i, results))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            return all(results)
        except Exception as e:
            print(f"Concurrent operations failed: {e}")
            return False

    def test_stress_operations(self) -> bool:
        """Test operations under stress conditions"""
        try:
            # Test with large tensors
            large_tensor = torch.randn(300, 300, device=self.device)
            large_result = torch.matmul(large_tensor, large_tensor.t())

            # Test repeated operations
            for i in range(50):
                temp_tensor = torch.randn(50, 50, device=self.device)
                temp_result = torch.matmul(temp_tensor, temp_tensor.t())
                temp_mean = torch.mean(temp_result)
                del temp_tensor, temp_result, temp_mean
                gc.collect()

            # Test memory pressure
            tensor_pool = []
            for i in range(10):
                tensor_pool.append(torch.randn(100, 100, device=self.device))

            # Perform operations under memory pressure
            for i in range(20):
                result = torch.matmul(tensor_pool[i % 10], tensor_pool[(i + 1) % 10])

            return True
        except Exception as e:
            print(f"Stress operations failed: {e}")
            return False

    def test_data_loading_operations(self) -> bool:
        """Test data loading and preprocessing operations"""
        try:
            # Create dummy dataset
            dataset_size = 1000
            input_size = 10
            output_size = 1

            # Generate data
            inputs = torch.randn(dataset_size, input_size, device=self.device)
            targets = torch.randn(dataset_size, output_size, device=self.device)

            # Create DataLoader-like functionality
            batch_size = 32
            num_batches = dataset_size // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_inputs = inputs[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]

                # Process batch
                batch_mean = torch.mean(batch_inputs, dim=0)
                batch_std = torch.std(batch_inputs, dim=0)

                # Normalize batch
                normalized_batch = (batch_inputs - batch_mean) / (batch_std + 1e-8)

            return True
        except Exception as e:
            print(f"Data loading operations failed: {e}")
            return False

    def test_advanced_optimization(self) -> bool:
        """Test advanced optimization techniques"""
        try:
            # Create model
            model = nn.Sequential(
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            ).to(self.device)

            # Test different optimizers
            optimizers = [
                optim.SGD(model.parameters(), lr=0.01),
                optim.Adam(model.parameters(), lr=0.001),
                optim.AdamW(model.parameters(), lr=0.001),
                optim.RMSprop(model.parameters(), lr=0.01)
            ]

            for optimizer in optimizers:
                # Reset model parameters
                for param in model.parameters():
                    param.data.normal_()

                # Training step
                inputs = torch.randn(16, 20, device=self.device)
                targets = torch.randn(16, 1, device=self.device)

                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            return True
        except Exception as e:
            print(f"Advanced optimization failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all resilience tests"""
        print("\n=== LibTorch Python Resilience Test Suite ===")

        tests = [
            ("Basic Tensor Operations", self.test_basic_tensor_operations),
            ("Advanced Tensor Operations", self.test_advanced_tensor_operations),
            ("Neural Network Operations", self.test_neural_network_operations),
            ("Model Training", self.test_model_training),
            ("Memory Management", self.test_memory_management),
            ("CUDA Operations", self.test_cuda_operations),
            ("Serialization", self.test_serialization),
            ("Error Handling", self.test_error_handling),
            ("Concurrent Operations", self.test_concurrent_operations),
            ("Stress Operations", self.test_stress_operations),
            ("Data Loading Operations", self.test_data_loading_operations),
            ("Advanced Optimization", self.test_advanced_optimization)
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            try:
                result = test_func()
                if self.log_test(test_name, result):
                    passed_tests += 1
            except Exception as e:
                print(f"Test {test_name} crashed: {e}")
                traceback.print_exc()

        print(f"\n=== Test Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        return passed_tests == total_tests

def main():
    """Main test function"""
    try:
        test_suite = LibTorchResilienceTest()
        success = test_suite.run_all_tests()

        if success:
            print("\n🎉 All LibTorch Python resilience tests passed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
