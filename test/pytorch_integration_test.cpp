/**
 * @file pytorch_integration_test.cpp
 * @brief Test for PyTorch integration
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef RAD_ML_PYTORCH_ENABLED
#include <torch/nn/modules.h>
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
    std::cout << "=== PyTorch Integration Test ===" << std::endl;

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
        // Test tensor protection
        std::cout << "\n--- Testing Tensor Protection ---" << std::endl;

        // Create a simple tensor
        auto tensor = torch::randn({3, 4});
        std::cout << "Original tensor:\n" << tensor << std::endl;

        // Create protected tensor
        auto protected_tensor = integration.create_protected_tensor(tensor);
        std::cout << "Protected tensor created successfully" << std::endl;
        std::cout << "Protection enabled: " << (protected_tensor.is_protected() ? "Yes" : "No")
                  << std::endl;

        // Test tensor validation
        integration.protect_tensor(tensor);
        std::cout << "Tensor validation passed" << std::endl;

        // Test radiation hardening
        auto hardened_tensor = apply_radiation_hardening(tensor);
        std::cout << "Radiation hardening applied successfully" << std::endl;

        // Test TMR protection
        auto tmr_tensor = apply_tmr_protection(tensor, rad_ml::tmr::ProtectionLevel::HYBRID_REDUNDANCY);
        std::cout << "TMR protection applied successfully" << std::endl;

        // Test tensor integrity validation
        bool integrity_valid = validate_tensor_integrity(tensor);
        std::cout << "Tensor integrity validation: " << (integrity_valid ? "Passed" : "Failed")
                  << std::endl;

        // Test with a simple neural network
        std::cout << "\n--- Testing Neural Network Protection ---" << std::endl;

        // Create a simple linear model
        torch::nn::Linear model(torch::nn::LinearOptions(10, 1));
        std::cout << "Linear model created" << std::endl;

        // Protect the model
        integration.protect_model(model);
        std::cout << "Model protection applied" << std::endl;

        // Test training protection
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
        std::cout << "Optimizer created" << std::endl;

        // Create dummy data
        auto input = torch::randn({5, 10});
        auto target = torch::randn({5, 1});

        // Forward pass
        auto output = model->forward(input);
        std::cout << "Forward pass completed" << std::endl;

        // Backward pass
        auto loss = torch::mse_loss(output, target);
        loss.backward();
        std::cout << "Backward pass completed" << std::endl;

        // Protect training step
        integration.protect_training_step(model, optimizer);
        std::cout << "Training step protection applied" << std::endl;

        // Optimizer step
        optimizer.step();
        std::cout << "Optimizer step completed" << std::endl;

        // Validate training state
        integration.validate_training_state(model);
        std::cout << "Training state validation passed" << std::endl;

        std::cout << "\n--- All PyTorch Integration Tests Passed ---" << std::endl;

#else
        std::cout << "PyTorch integration not enabled at compile time" << std::endl;
        std::cout << "To enable, set ENABLE_PYTORCH=ON and ensure PyTorch is found" << std::endl;
#endif

        // Shutdown integration
        integration.shutdown();
        std::cout << "PyTorch integration shutdown successfully" << std::endl;

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
