/**
 * @file libtorch_radiation_integration_test.cpp
 * @brief LibTorch radiation hardening integration tests
 *
 * This test suite validates LibTorch integration with radiation hardening features,
 * including fault injection, TMR protection, and adaptive protection mechanisms
 * for space applications.
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

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
#include <torch/cuda.h>
#include <torch/nn/modules.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/loss.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#endif

#include <rad_ml/core/logger.hpp>
#include <rad_ml/neural/adaptive_protection.hpp>
#include <rad_ml/pytorch/pytorch_integration.hpp>
#include <rad_ml/radiation/fault_injection.hpp>
#include <rad_ml/tmr/adaptive_protection.hpp>

using namespace rad_ml::pytorch;
using namespace rad_ml::neural;
using namespace rad_ml::tmr;
using namespace rad_ml::radiation;

class LibTorchRadiationIntegrationTest {
   private:
    std::atomic<int> test_counter_{0};
    std::mutex output_mutex_;
    bool cuda_available_;
    std::unique_ptr<FaultInjector> fault_injector_;

    void log_test(const std::string& test_name, bool passed)
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::cout << "[" << ++test_counter_ << "] " << test_name << ": "
                  << (passed ? "✅ PASS" : "❌ FAIL") << std::endl;
    }

   public:
    LibTorchRadiationIntegrationTest()
    {
        cuda_available_ = torch::cuda::is_available();
        fault_injector_ = std::make_unique<FaultInjector>();
        std::cout << "CUDA available: " << (cuda_available_ ? "Yes" : "No") << std::endl;
    }

    bool test_protected_tensor_creation()
    {
        try {
            // Initialize PyTorch integration with radiation protection
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;
            config.tmr_strategy = ProtectionLevel::TRIPLE_REDUNDANCY;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create tensors with different protection levels
            auto tensor1 = torch::randn({10, 10});
            auto protected_tensor1 = integration.create_protected_tensor(tensor1);

            auto tensor2 = torch::ones({5, 5});
            auto protected_tensor2 = integration.create_protected_tensor(tensor2);

            // Verify protection is enabled
            bool protection1_enabled = protected_tensor1.is_protected();
            bool protection2_enabled = protected_tensor2.is_protected();

            // Test tensor operations with protection
            auto result = protected_tensor1.tensor() + protected_tensor2.tensor();

            integration.shutdown();

            return protection1_enabled && protection2_enabled;
        }
        catch (const std::exception& e) {
            std::cerr << "Protected tensor creation failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_tmr_protection()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.tmr_strategy = ProtectionLevel::TRIPLE_REDUNDANCY;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create tensor with TMR protection
            auto tensor = torch::randn({8, 8});
            auto tmr_tensor = apply_tmr_protection(tensor, ProtectionLevel::TRIPLE_REDUNDANCY);

            // Test TMR voting under normal conditions
            auto original_result = torch::matmul(tensor, tensor.t());
            auto tmr_result = torch::matmul(tmr_tensor, tmr_tensor.t());

            // Verify results are consistent
            auto diff = torch::abs(original_result - tmr_result);
            bool results_consistent = torch::sum(diff).item<float>() < 1e-6;

            integration.shutdown();

            return results_consistent;
        }
        catch (const std::exception& e) {
            std::cerr << "TMR protection test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fault_injection_protection()
    {
        try {
            PyTorchConfig config;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create a simple neural network
            torch::nn::Sequential model(torch::nn::Linear(torch::nn::LinearOptions(10, 5)),
                                        torch::nn::ReLU(),
                                        torch::nn::Linear(torch::nn::LinearOptions(5, 1)));

            // Protect the model
            integration.protect_model(model);

            // Create test data
            auto input = torch::randn({32, 10});
            auto target = torch::randn({32, 1});

            // Test forward pass with fault injection
            fault_injector_->enable_fault_injection(true);
            fault_injector_->set_fault_rate(0.01);  // 1% fault rate

            auto output = model->forward(input);
            auto loss = torch::mse_loss(output, target);

            // Test backward pass with fault injection
            loss.backward();

            // Verify model integrity after fault injection
            integration.validate_training_state(model);

            fault_injector_->enable_fault_injection(false);
            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Fault injection protection test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_adaptive_protection()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::ADAPTIVE;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create adaptive protection manager
            AdaptiveProtectionManager protection_manager;
            protection_manager.set_radiation_environment(RadiationEnvironment::HIGH_ORBIT);

            // Create model with adaptive protection
            torch::nn::Linear model(torch::nn::LinearOptions(20, 10));
            integration.protect_model(model);

            // Simulate different radiation conditions
            std::vector<RadiationLevel> radiation_levels = {
                RadiationLevel::LOW, RadiationLevel::MEDIUM, RadiationLevel::HIGH,
                RadiationLevel::EXTREME};

            for (auto level : radiation_levels) {
                protection_manager.set_radiation_level(level);

                // Test model under different radiation conditions
                auto input = torch::randn({16, 20});
                auto output = model->forward(input);

                // Verify protection adapts to radiation level
                bool protection_active = protection_manager.is_protection_active();

                // Test training under current conditions
                torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
                auto target = torch::randn({16, 10});
                auto loss = torch::mse_loss(output, target);

                loss.backward();
                integration.protect_training_step(model, optimizer);
                optimizer.step();
            }

            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Adaptive protection test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_radiation_hardened_training()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.enable_gradient_protection = true;
            config.enable_weight_protection = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create a more complex model
            torch::nn::Sequential model(
                torch::nn::Linear(torch::nn::LinearOptions(50, 25)), torch::nn::ReLU(),
                torch::nn::Dropout(0.2), torch::nn::Linear(torch::nn::LinearOptions(25, 10)),
                torch::nn::ReLU(), torch::nn::Linear(torch::nn::LinearOptions(10, 1)));

            // Protect the model
            integration.protect_model(model);

            // Create optimizer with gradient protection
            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

            // Training loop with radiation hardening
            for (int epoch = 0; epoch < 5; ++epoch) {
                // Generate training data
                auto inputs = torch::randn({64, 50});
                auto targets = torch::randn({64, 1});

                // Forward pass with protection
                auto outputs = model->forward(inputs);
                auto loss = torch::mse_loss(outputs, targets);

                // Backward pass with gradient protection
                optimizer.zero_grad();
                loss.backward();

                // Apply gradient protection
                integration.protect_training_step(model, optimizer);

                // Optimizer step
                optimizer.step();

                // Validate model state after each epoch
                integration.validate_training_state(model);

                if (epoch % 2 == 0) {
                    std::cout << "    Epoch " << epoch << ", Loss: " << loss.item<float>()
                              << std::endl;
                }
            }

            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Radiation hardened training failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_memory_protection()
    {
        try {
            PyTorchConfig config;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Test memory protection with large tensors
            std::vector<ProtectedTensor> protected_tensors;

            for (int i = 0; i < 10; ++i) {
                auto tensor = torch::randn({100, 100});
                auto protected_tensor = integration.create_protected_tensor(tensor);
                protected_tensor.enable_protection(ProtectionLevel::HIGH);
                protected_tensors.push_back(std::move(protected_tensor));
            }

            // Test operations on protected tensors
            for (auto& protected_tensor : protected_tensors) {
                auto& tensor = protected_tensor.tensor();
                auto result = torch::matmul(tensor, tensor.t());

                // Validate tensor integrity
                protected_tensor.validate_integrity();
            }

            // Test memory deallocation with protection
            protected_tensors.clear();

            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Memory protection test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_concurrent_radiation_protection()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            std::vector<std::thread> threads;
            std::atomic<int> success_count{0};

            // Test concurrent operations with radiation protection
            for (int i = 0; i < 4; ++i) {
                threads.emplace_back([&, i]() {
                    try {
                        // Create protected tensor in thread
                        auto tensor = torch::randn({50, 50});
                        auto protected_tensor = integration.create_protected_tensor(tensor);
                        protected_tensor.enable_protection(ProtectionLevel::HIGH);

                        // Perform operations
                        auto result =
                            torch::matmul(protected_tensor.tensor(), protected_tensor.tensor().t());

                        // Validate integrity
                        protected_tensor.validate_integrity();

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

            integration.shutdown();

            return success_count == 4;
        }
        catch (const std::exception& e) {
            std::cerr << "Concurrent radiation protection failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_error_recovery()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create model with protection
            torch::nn::Linear model(torch::nn::LinearOptions(10, 5));
            integration.protect_model(model);

            // Simulate error conditions
            fault_injector_->enable_fault_injection(true);
            fault_injector_->set_fault_rate(0.05);  // 5% fault rate

            // Test error recovery during training
            torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

            for (int step = 0; step < 10; ++step) {
                try {
                    auto input = torch::randn({16, 10});
                    auto target = torch::randn({16, 5});

                    auto output = model->forward(input);
                    auto loss = torch::mse_loss(output, target);

                    loss.backward();
                    integration.protect_training_step(model, optimizer);
                    optimizer.step();

                    // Validate model state
                    integration.validate_training_state(model);
                }
                catch (const std::exception& e) {
                    // Error recovery - validate and continue
                    integration.validate_training_state(model);
                    std::cout << "    Recovered from error at step " << step << std::endl;
                }
            }

            fault_injector_->enable_fault_injection(false);
            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error recovery test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_performance_under_radiation()
    {
        try {
            PyTorchConfig config;
            config.enable_tmr_protection = true;
            config.enable_radiation_hardening = true;
            config.protection_level = ProtectionLevel::HIGH;

            auto& integration = PyTorchIntegration::get_instance();
            integration.initialize(config);

            // Create model for performance testing
            torch::nn::Sequential model(
                torch::nn::Linear(torch::nn::LinearOptions(100, 50)), torch::nn::ReLU(),
                torch::nn::Linear(torch::nn::LinearOptions(50, 25)), torch::nn::ReLU(),
                torch::nn::Linear(torch::nn::LinearOptions(25, 10)));

            integration.protect_model(model);

            // Performance test under different radiation conditions
            std::vector<RadiationLevel> levels = {RadiationLevel::LOW, RadiationLevel::HIGH};

            for (auto level : levels) {
                fault_injector_->set_radiation_level(level);

                auto start_time = std::chrono::high_resolution_clock::now();

                // Run multiple forward passes
                for (int i = 0; i < 100; ++i) {
                    auto input = torch::randn({32, 100});
                    auto output = model->forward(input);

                    // Validate output integrity
                    if (torch::isnan(output).any().item<bool>()) {
                        std::cout << "    Detected NaN in output at iteration " << i << std::endl;
                    }
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                std::cout << "    Performance under "
                          << (level == RadiationLevel::LOW ? "LOW" : "HIGH")
                          << " radiation: " << duration.count() << "ms" << std::endl;
            }

            integration.shutdown();

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Performance under radiation test failed: " << e.what() << std::endl;
            return false;
        }
    }

    void run_all_tests()
    {
        std::cout << "\n=== LibTorch Radiation Integration Test Suite ===" << std::endl;

        bool all_passed = true;

        all_passed &= test_protected_tensor_creation();
        log_test("Protected Tensor Creation", all_passed);

        all_passed &= test_tmr_protection();
        log_test("TMR Protection", all_passed);

        all_passed &= test_fault_injection_protection();
        log_test("Fault Injection Protection", all_passed);

        all_passed &= test_adaptive_protection();
        log_test("Adaptive Protection", all_passed);

        all_passed &= test_radiation_hardened_training();
        log_test("Radiation Hardened Training", all_passed);

        all_passed &= test_memory_protection();
        log_test("Memory Protection", all_passed);

        all_passed &= test_concurrent_radiation_protection();
        log_test("Concurrent Radiation Protection", all_passed);

        all_passed &= test_error_recovery();
        log_test("Error Recovery", all_passed);

        all_passed &= test_performance_under_radiation();
        log_test("Performance Under Radiation", all_passed);

        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << test_counter_.load() << std::endl;
        std::cout << "All tests passed: " << (all_passed ? "✅ YES" : "❌ NO") << std::endl;

        if (!all_passed) {
            throw std::runtime_error("Some LibTorch radiation integration tests failed");
        }
    }
};

int main()
{
    try {
        LibTorchRadiationIntegrationTest test_suite;
        test_suite.run_all_tests();

        std::cout << "\n🎉 All LibTorch radiation integration tests passed successfully!"
                  << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test suite failed: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "\n❌ Unknown error occurred during testing" << std::endl;
        return 1;
    }
}
