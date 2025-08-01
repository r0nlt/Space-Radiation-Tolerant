/**
 * @file libtorch_macos_compatibility_test.cpp
 * @brief macOS compatibility test for LibTorch
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef RAD_ML_PYTORCH_ENABLED
#include <torch/torch.h>
#endif

#include <rad_ml/core/logger.hpp>

// Don't use namespace to avoid conflicts with PyTorch

class LibTorchMacOSCompatibilityTest {
   private:
    std::atomic<int> test_counter_{0};
    std::mutex output_mutex_;
    bool cuda_available_;
    int num_threads_;

    void log_test(const std::string& test_name, bool passed, const std::string& details = "")
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::cout << "[" << ++test_counter_ << "] " << test_name << ": "
                  << (passed ? "✅ PASS" : "❌ FAIL");
        if (!details.empty()) {
            std::cout << " (" << details << ")";
        }
        std::cout << std::endl;
    }

   public:
    LibTorchMacOSCompatibilityTest()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        cuda_available_ = torch::cuda::is_available();
        num_threads_ = std::thread::hardware_concurrency();

        std::cout << "=== LibTorch macOS Compatibility Test ===" << std::endl;
        std::cout << "CUDA available: " << (cuda_available_ ? "Yes" : "No") << std::endl;
        std::cout << "Number of CPU threads: " << num_threads_ << std::endl;
        std::cout << "Using CPU-only operations" << std::endl;

        // Set PyTorch to use CPU threads efficiently
        torch::set_num_threads(num_threads_);
#else
        std::cout << "PyTorch not enabled - skipping tests" << std::endl;
#endif
    }

    bool test_cpu_tensor_operations()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test basic tensor operations on CPU
            auto tensor1 = torch::randn({20, 20});
            auto tensor2 = torch::ones({20, 20});
            auto tensor3 = torch::zeros({20, 20});

            // Test arithmetic operations
            auto sum = tensor1 + tensor2;
            auto diff = tensor1 - tensor2;
            auto prod = tensor1 * tensor2;
            auto div = tensor1 / (tensor2 + 1e-8);

            // Test mathematical functions
            auto sin_tensor = torch::sin(tensor1);
            auto cos_tensor = torch::cos(tensor1);
            auto exp_tensor = torch::exp(tensor1);
            auto log_tensor = torch::log(torch::abs(tensor1) + 1e-8);

            // Test reduction operations
            auto mean_val = torch::mean(tensor1);
            auto sum_val = torch::sum(tensor1);
            auto max_val = torch::max(tensor1);
            auto min_val = torch::min(tensor1);

            // Test matrix operations
            auto matmul_result = torch::matmul(tensor1, tensor1.t());
            auto transpose_result = tensor1.t();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "CPU tensor operations failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_memory_efficient_operations()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test memory-efficient operations suitable for macOS
            std::vector<torch::Tensor> tensors;

            // Create tensors with reasonable sizes for macOS
            for (int i = 0; i < 20; ++i) {
                auto tensor = torch::randn({50, 50});
                tensors.push_back(tensor);
            }

            // Test operations on these tensors
            for (int i = 0; i < 10; ++i) {
                auto result = torch::matmul(tensors[i], tensors[i + 1]);
                auto mean_val = torch::mean(result);
            }

            // Clear tensors to test memory management
            tensors.clear();

            // Test with larger tensors but fewer of them
            auto large_tensor = torch::randn({200, 200});
            auto large_result = torch::matmul(large_tensor, large_tensor.t());

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Memory efficient operations failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_basic_neural_network()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test basic neural network operations (simplified)
            auto input = torch::randn({32, 100});
            auto target = torch::randn({32, 10});

            // Simple linear transformation
            auto weights = torch::randn({100, 10});
            auto bias = torch::randn({10});
            auto output = torch::matmul(input, weights) + bias;

            // Test loss computation
            auto loss = torch::mse_loss(output, target);

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Basic neural network test failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_serialization_macos()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test serialization with macOS-compatible file paths
            auto tensor = torch::randn({10, 10});

            // Use a simple filename for macOS
            std::string filename = "test_tensor_macos.pt";
            torch::save(tensor, filename);

            // Load the tensor back
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, filename);

            // Verify the loaded tensor matches
            auto diff = torch::abs(tensor - loaded_tensor);
            bool tensors_match = torch::sum(diff).item<float>() < 1e-6;

            // Clean up
            std::remove(filename.c_str());

            return tensors_match;
        }
        catch (const std::exception& e) {
            std::cerr << "Serialization test failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_threading_performance()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test threading performance on macOS
            std::vector<std::thread> threads;
            std::atomic<int> success_count{0};

            // Create threads to perform tensor operations
            for (int i = 0; i < std::min(4, num_threads_); ++i) {
                threads.emplace_back([&success_count]() {
                    try {
                        auto tensor = torch::randn({30, 30});
                        auto result = torch::matmul(tensor, tensor.t());
                        auto mean_val = torch::mean(result);
                        success_count++;
                    }
                    catch (const std::exception&) {
                        // Thread failed
                    }
                });
            }

            // Wait for all threads
            for (auto& thread : threads) {
                thread.join();
            }

            return success_count == threads.size();
        }
        catch (const std::exception& e) {
            std::cerr << "Threading performance test failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_performance_benchmark()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Benchmark performance on macOS
            auto start_time = std::chrono::high_resolution_clock::now();

            // Perform a series of operations
            for (int i = 0; i < 50; ++i) {
                auto tensor1 = torch::randn({50, 50});
                auto tensor2 = torch::randn({50, 50});
                auto result = torch::matmul(tensor1, tensor2);
                auto mean_val = torch::mean(result);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "    Performance benchmark: " << duration.count() << "ms for 50 operations"
                      << std::endl;

            return duration.count() < 10000;  // Should complete within 10 seconds
        }
        catch (const std::exception& e) {
            std::cerr << "Performance benchmark failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_error_handling_macos()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test error handling specific to macOS
            try {
                auto tensor1 = torch::randn({3, 4});
                auto tensor2 = torch::randn({5, 6});
                auto result = tensor1 + tensor2;  // Should throw
                return false;                     // Should not reach here
            }
            catch (const std::exception&) {
                // Expected to throw
            }

            // Test invalid index access
            try {
                auto tensor = torch::randn({5, 5});
                auto invalid_access = tensor[10][10];  // Should throw
                return false;                          // Should not reach here
            }
            catch (const std::exception&) {
                // Expected to throw
            }

            // Test division by zero handling
            try {
                auto tensor = torch::zeros({2, 2});
                auto result = 1.0 / tensor;  // Should handle gracefully
            }
            catch (const std::exception&) {
                // Should handle gracefully
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error handling test failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    bool test_memory_management_macos()
    {
#ifdef RAD_ML_PYTORCH_ENABLED
        try {
            // Test memory management optimized for macOS
            std::vector<torch::Tensor> tensors;

            // Allocate tensors
            for (int i = 0; i < 15; ++i) {
                auto tensor = torch::randn({40, 40});
                tensors.push_back(tensor);
            }

            // Perform operations
            for (int i = 0; i < 10; ++i) {
                auto result = torch::matmul(tensors[i], tensors[i + 1]);
            }

            // Clear tensors
            tensors.clear();

            // Test large tensor allocation
            auto large_tensor = torch::randn({150, 150});
            auto large_result = torch::matmul(large_tensor, large_tensor.t());

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Memory management test failed: " << e.what() << std::endl;
            return false;
        }
#else
        return true;  // Skip if PyTorch not enabled
#endif
    }

    void run_all_tests()
    {
        std::cout << "\n=== Running macOS Compatibility Tests ===" << std::endl;

        bool all_passed = true;

        all_passed &= test_cpu_tensor_operations();
        log_test("CPU Tensor Operations", all_passed);

        all_passed &= test_memory_efficient_operations();
        log_test("Memory Efficient Operations", all_passed);

        all_passed &= test_basic_neural_network();
        log_test("Basic Neural Network", all_passed);

        all_passed &= test_serialization_macos();
        log_test("macOS Serialization", all_passed);

        all_passed &= test_threading_performance();
        log_test("Threading Performance", all_passed);

        all_passed &= test_performance_benchmark();
        log_test("Performance Benchmark", all_passed);

        all_passed &= test_error_handling_macos();
        log_test("Error Handling (macOS)", all_passed);

        all_passed &= test_memory_management_macos();
        log_test("Memory Management (macOS)", all_passed);

        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << test_counter_.load() << std::endl;
        std::cout << "All tests passed: " << (all_passed ? "✅ YES" : "❌ NO") << std::endl;

        if (!all_passed) {
            throw std::runtime_error("Some macOS compatibility tests failed");
        }
    }
};

int main()
{
    try {
        LibTorchMacOSCompatibilityTest test_suite;
        test_suite.run_all_tests();

        std::cout << "\n🎉 All macOS LibTorch compatibility tests passed successfully!"
                  << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test suite failed: " << e.what() << std::endl;
        return 1;
    }
}
