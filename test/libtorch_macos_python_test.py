#!/usr/bin/env python3
"""
LibTorch macOS Python Compatibility Test

This script provides macOS-specific testing of PyTorch functionality,
optimized for CPU-only operations and macOS system characteristics.

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
import multiprocessing as mp
import platform

class LibTorchMacOSTest:
    """macOS-specific LibTorch testing suite"""
    
    def __init__(self):
        self.test_counter = 0
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cpu')  # Force CPU usage on macOS
        self.num_threads = mp.cpu_count()
        
        print(f"=== LibTorch macOS Python Test ===")
        print(f"Platform: {platform.platform()}")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {self.cuda_available}")
        print(f"Number of CPU threads: {self.num_threads}")
        print(f"Using device: {self.device}")
        
        # Set PyTorch to use CPU threads efficiently
        torch.set_num_threads(self.num_threads)
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results with consistent formatting"""
        self.test_counter += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{self.test_counter:2d}] {test_name}: {status}")
        if details:
            print(f"    {details}")
        return passed
    
    def test_cpu_tensor_operations(self) -> bool:
        """Test basic tensor operations optimized for CPU"""
        try:
            # Test tensor creation
            tensor1 = torch.randn(20, 20, device=self.device)
            tensor2 = torch.ones(20, 20, device=self.device)
            tensor3 = torch.zeros(20, 20, device=self.device)
            
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
            
            # Test matrix operations
            matmul_result = torch.matmul(tensor1, tensor1.t())
            transpose_result = tensor1.t()
            
            return True
        except Exception as e:
            print(f"CPU tensor operations failed: {e}")
            return False
    
    def test_memory_efficient_operations(self) -> bool:
        """Test memory-efficient operations suitable for macOS"""
        try:
            # Create tensors with reasonable sizes for macOS
            tensors = []
            for i in range(20):
                tensor = torch.randn(50, 50, device=self.device)
                tensors.append(tensor)
            
            # Test operations on these tensors
            for i in range(10):
                result = torch.matmul(tensors[i], tensors[i + 1])
                mean_val = torch.mean(result)
            
            # Clear tensors to test memory management
            del tensors
            gc.collect()
            
            # Test with larger tensors but fewer of them
            large_tensor = torch.randn(200, 200, device=self.device)
            large_result = torch.matmul(large_tensor, large_tensor.t())
            
            return True
        except Exception as e:
            print(f"Memory efficient operations failed: {e}")
            return False
    
    def test_neural_network_cpu_operations(self) -> bool:
        """Test neural network operations optimized for CPU"""
        try:
            # Create a model optimized for CPU
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 10)
            ).to(self.device)
            
            # Test forward pass
            input_tensor = torch.randn(32, 100, device=self.device)
            output = model(input_tensor)
            
            # Test loss computation
            target = torch.randn_like(output)
            criterion = nn.MSELoss()
            loss = criterion(output, target)
            
            # Test backward pass
            loss.backward()
            
            # Test optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()
            
            return True
        except Exception as e:
            print(f"Neural network CPU operations failed: {e}")
            return False
    
    def test_serialization_macos(self) -> bool:
        """Test serialization with macOS-compatible file paths"""
        try:
            # Create a tensor to serialize
            tensor = torch.randn(10, 10, device=self.device)
            
            # Use a simple filename for macOS
            filename = "test_tensor_macos_python.pt"
            torch.save(tensor, filename)
            
            # Load the tensor back
            loaded_tensor = torch.load(filename, map_location=self.device)
            
            # Verify the loaded tensor matches
            diff = torch.abs(tensor - loaded_tensor)
            tensors_match = torch.sum(diff).item() < 1e-6
            
            # Clean up
            os.remove(filename)
            
            return tensors_match
        except Exception as e:
            print(f"Serialization failed: {e}")
            return False
    
    def test_threading_performance(self) -> bool:
        """Test threading performance on macOS"""
        try:
            def worker_function(worker_id: int, results: List[bool]):
                try:
                    # Each thread creates its own tensors
                    tensor = torch.randn(30, 30, device=self.device)
                    result = torch.matmul(tensor, tensor.t())
                    mean_val = torch.mean(result)
                    results[worker_id] = True
                except Exception:
                    results[worker_id] = False
            
            # Test with multiple threads
            num_threads = min(4, self.num_threads)
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
            print(f"Threading performance failed: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Benchmark performance on macOS"""
        try:
            start_time = time.time()
            
            # Perform a series of operations
            for i in range(50):
                tensor = torch.randn(50, 50, device=self.device)
                result = torch.matmul(tensor, tensor.t())
                mean_val = torch.mean(result)
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            print(f"    Performance benchmark: {duration:.2f}ms for 50 operations")
            
            return duration < 10000  # Should complete within 10 seconds
        except Exception as e:
            print(f"Performance benchmark failed: {e}")
            return False
    
    def test_memory_management_macos(self) -> bool:
        """Test memory management optimized for macOS"""
        try:
            # Allocate tensors
            tensors = []
            for i in range(15):
                tensor = torch.randn(40, 40, device=self.device)
                tensors.append(tensor)
            
            # Perform operations
            for i in range(10):
                result = torch.matmul(tensors[i], tensors[i + 1])
            
            # Clear tensors
            del tensors
            gc.collect()
            
            # Test large tensor allocation
            large_tensor = torch.randn(150, 150, device=self.device)
            large_result = torch.matmul(large_tensor, large_tensor.t())
            
            return True
        except Exception as e:
            print(f"Memory management failed: {e}")
            return False
    
    def test_error_handling_macos(self) -> bool:
        """Test error handling specific to macOS"""
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
            
            return True
        except Exception as e:
            print(f"Error handling failed: {e}")
            return False
    
    def test_advanced_optimization_macos(self) -> bool:
        """Test advanced optimization techniques for macOS"""
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
    
    def test_data_loading_macos(self) -> bool:
        """Test data loading operations optimized for macOS"""
        try:
            # Create dummy dataset
            dataset_size = 500  # Smaller dataset for macOS
            input_size = 10
            output_size = 1
            
            # Generate data
            inputs = torch.randn(dataset_size, input_size, device=self.device)
            targets = torch.randn(dataset_size, output_size, device=self.device)
            
            # Create DataLoader-like functionality
            batch_size = 16  # Smaller batch size for macOS
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
    
    def run_all_tests(self) -> bool:
        """Run all macOS compatibility tests"""
        print("\n=== Running macOS Python Tests ===")
        
        tests = [
            ("CPU Tensor Operations", self.test_cpu_tensor_operations),
            ("Memory Efficient Operations", self.test_memory_efficient_operations),
            ("Neural Network CPU Operations", self.test_neural_network_cpu_operations),
            ("macOS Serialization", self.test_serialization_macos),
            ("Threading Performance", self.test_threading_performance),
            ("Performance Benchmark", self.test_performance_benchmark),
            ("Memory Management (macOS)", self.test_memory_management_macos),
            ("Error Handling (macOS)", self.test_error_handling_macos),
            ("Advanced Optimization (macOS)", self.test_advanced_optimization_macos),
            ("Data Loading (macOS)", self.test_data_loading_macos)
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
        test_suite = LibTorchMacOSTest()
        success = test_suite.run_all_tests()
        
        if success:
            print("\n🎉 All macOS LibTorch Python tests passed successfully!")
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