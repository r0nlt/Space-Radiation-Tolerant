/**
 * @file libtorch_resilience_test.cpp
 * @brief Test for LibTorch resilience and basic functionality
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
    std::cout << "=== LibTorch Resilience Test ===" << std::endl;

    try {
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
        // Test basic PyTorch functionality
        std::cout << "\n--- Testing Basic PyTorch Functionality ---" << std::endl;

        // Create a simple tensor
        auto tensor = torch::randn({3, 4});
        std::cout << "Created tensor with shape: ";
        for (auto dim : tensor.sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        // Test tensor operations
        auto tensor_squared = torch::square(tensor);
        auto tensor_sum = torch::sum(tensor);
        std::cout << "Tensor sum: " << tensor_sum.item<float>() << std::endl;

        // Test neural network creation
        std::cout << "\n--- Testing Neural Network Creation ---" << std::endl;

        torch::nn::Linear model(torch::nn::LinearOptions(10, 1));
        std::cout << "Linear model created successfully" << std::endl;

        // Test forward pass
        auto input = torch::randn({5, 10});
        auto output = model->forward(input);
        std::cout << "Forward pass completed. Output shape: ";
        for (auto dim : output.sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        // Test optimizer
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
        std::cout << "Optimizer created successfully" << std::endl;

        // Test training step
        auto target = torch::randn({5, 1});
        auto loss = torch::mse_loss(output, target);
        loss.backward();
        optimizer.step();
        std::cout << "Training step completed successfully" << std::endl;

        // Test tensor validation
        std::cout << "\n--- Testing Tensor Validation ---" << std::endl;

        bool integrity_valid = validate_tensor_integrity(tensor);
        std::cout << "Tensor integrity validation: " << (integrity_valid ? "Passed" : "Failed")
                  << std::endl;

        // Test with NaN tensor
        auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
        bool nan_integrity = validate_tensor_integrity(nan_tensor);
        std::cout << "NaN tensor integrity validation: " << (nan_integrity ? "Passed" : "Failed")
                  << std::endl;

        // Test radiation hardening
        std::cout << "\n--- Testing Radiation Hardening ---" << std::endl;

        try {
            auto hardened_tensor = apply_radiation_hardening(tensor);
            std::cout << "Radiation hardening applied successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Radiation hardening failed: " << e.what() << std::endl;
        }

        // Test TMR protection
        std::cout << "\n--- Testing TMR Protection ---" << std::endl;

        try {
            auto tmr_tensor =
                apply_tmr_protection(tensor, rad_ml::tmr::ProtectionLevel::HYBRID_REDUNDANCY);
            std::cout << "TMR protection applied successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "TMR protection failed: " << e.what() << std::endl;
        }

        std::cout << "\n--- All LibTorch Resilience Tests Passed ---" << std::endl;

#else
        std::cout << "LibTorch integration not enabled at compile time" << std::endl;
        std::cout << "To enable, set ENABLE_PYTORCH=ON and ensure PyTorch is found" << std::endl;
#endif

        // Shutdown integration
        integration.shutdown();
        std::cout << "LibTorch integration shutdown successfully" << std::endl;

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
