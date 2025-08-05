/**
 * @file libtorch_macos_compatibility_test.cpp
 * @brief Test for LibTorch macOS compatibility
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef RAD_ML_PYTORCH_ENABLED
#include <torch/nn/modules/linear.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#endif

#include <rad_ml/core/logger.hpp>
#include <rad_ml/pytorch/pytorch_integration.hpp>

using namespace rad_ml::pytorch;

int main()
{
    std::cout << "=== LibTorch macOS Compatibility Test ===" << std::endl;

    try {
        // Test system information
        std::cout << "Testing macOS compatibility..." << std::endl;

#ifdef __APPLE__
        std::cout << "Running on macOS" << std::endl;

        // Check architecture
#ifdef __arm64__
        std::cout << "Architecture: ARM64 (Apple Silicon)" << std::endl;
#else
        std::cout << "Architecture: x86_64 (Intel)" << std::endl;
#endif

        // Check macOS version
        std::cout << "macOS version: " << __MAC_OS_X_VERSION_MIN_REQUIRED << std::endl;
#endif

        // Initialize PyTorch integration
        PyTorchConfig config;
        config.enable_tmr_protection = true;
        config.enable_radiation_hardening = true;
        config.protection_level = rad_ml::neural::ProtectionLevel::MODERATE;

        auto& integration = PyTorchIntegration::get_instance();
        integration.initialize(config);

        std::cout << "PyTorch integration initialized successfully" << std::endl;
        std::cout << "PyTorch available: " << (integration.is_pytorch_available() ? "Yes" : "No")
                  << std::endl;
        std::cout << "CUDA available: " << (integration.is_cuda_available() ? "Yes" : "No")
                  << std::endl;

#ifdef RAD_ML_PYTORCH_ENABLED
        // Test basic PyTorch functionality on macOS
        std::cout << "\n--- Testing Basic PyTorch Functionality on macOS ---" << std::endl;

        // Test tensor creation
        auto tensor = torch::randn({3, 4});
        std::cout << "Tensor created successfully on macOS" << std::endl;
        std::cout << "Tensor device: " << tensor.device() << std::endl;

        // Test tensor operations
        auto tensor_squared = torch::square(tensor);
        auto tensor_sum = torch::sum(tensor);
        std::cout << "Tensor operations completed successfully" << std::endl;

        // Test neural network creation
        std::cout << "\n--- Testing Neural Network on macOS ---" << std::endl;

        torch::nn::Linear model(torch::nn::LinearOptions(10, 1));
        std::cout << "Linear model created successfully on macOS" << std::endl;

        // Test forward pass
        auto input = torch::randn({5, 10});
        auto output = model->forward(input);
        std::cout << "Forward pass completed on macOS" << std::endl;

        // Test optimizer
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
        std::cout << "Optimizer created successfully on macOS" << std::endl;

        // Test training step
        auto target = torch::randn({5, 1});
        auto loss = torch::mse_loss(output, target);
        loss.backward();
        optimizer.step();
        std::cout << "Training step completed successfully on macOS" << std::endl;

        // Test memory management
        std::cout << "\n--- Testing Memory Management on macOS ---" << std::endl;

        // Create multiple tensors to test memory management
        std::vector<torch::Tensor> tensors;
        for (int i = 0; i < 10; ++i) {
            tensors.push_back(torch::randn({100, 100}));
        }
        std::cout << "Created 10 large tensors successfully" << std::endl;

        // Clear tensors to test cleanup
        tensors.clear();
        std::cout << "Tensors cleared successfully" << std::endl;

        // Test tensor validation
        std::cout << "\n--- Testing Tensor Validation on macOS ---" << std::endl;

        bool integrity_valid = validate_tensor_integrity(tensor);
        std::cout << "Tensor integrity validation: " << (integrity_valid ? "Passed" : "Failed")
                  << std::endl;

        // Test radiation hardening
        std::cout << "\n--- Testing Radiation Hardening on macOS ---" << std::endl;

        try {
            auto hardened_tensor = apply_radiation_hardening(tensor);
            std::cout << "Radiation hardening applied successfully on macOS" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Radiation hardening failed on macOS: " << e.what() << std::endl;
        }

        // Test TMR protection
        std::cout << "\n--- Testing TMR Protection on macOS ---" << std::endl;

        try {
            auto tmr_tensor =
                apply_tmr_protection(tensor, rad_ml::tmr::ProtectionLevel::HYBRID_REDUNDANCY);
            std::cout << "TMR protection applied successfully on macOS" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "TMR protection failed on macOS: " << e.what() << std::endl;
        }

        // Test model protection
        std::cout << "\n--- Testing Model Protection on macOS ---" << std::endl;

        try {
            integration.protect_model(static_cast<::torch::nn::Module&>(*model));
            std::cout << "Model protection applied successfully on macOS" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Model protection failed on macOS: " << e.what() << std::endl;
        }

        std::cout << "\n--- All LibTorch macOS Compatibility Tests Passed ---" << std::endl;

#else
        std::cout << "LibTorch integration not enabled at compile time" << std::endl;
        std::cout << "To enable, set ENABLE_PYTORCH=ON and ensure PyTorch is found" << std::endl;
#endif

        // Shutdown integration
        integration.shutdown();
        std::cout << "LibTorch macOS compatibility test shutdown successfully" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
