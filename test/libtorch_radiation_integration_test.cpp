/**
 * @file libtorch_radiation_integration_test.cpp
 * @brief Test for LibTorch radiation integration and hardening
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

#include <iostream>
#include <memory>
#include <random>
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

// Function to simulate bit flips in a tensor (for testing)
#ifdef RAD_ML_PYTORCH_ENABLED
torch::Tensor simulate_bit_flips(const torch::Tensor& tensor, double flip_probability = 0.001)
{
    auto corrupted = tensor.clone();
    auto flat = corrupted.flatten();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int64_t i = 0; i < flat.numel(); ++i) {
        if (dis(gen) < flip_probability) {
            // Simulate bit flip by adding small random noise
            flat[i] += torch::randn(1).item<float>() * 0.1;
        }
    }

    return corrupted;
}
#endif

int main()
{
    std::cout << "=== LibTorch Radiation Integration Test ===" << std::endl;

    try {
        // Initialize PyTorch integration
        PyTorchConfig config;
        config.enable_tmr_protection = true;
        config.enable_radiation_hardening = true;
        config.protection_level = rad_ml::neural::ProtectionLevel::HIGH;

        auto& integration = PyTorchIntegration::get_instance();
        integration.initialize(config);

        std::cout << "PyTorch integration initialized successfully" << std::endl;

#ifdef RAD_ML_PYTORCH_ENABLED
        // Test radiation hardening with corrupted tensors
        std::cout << "\n--- Testing Radiation Hardening with Corrupted Data ---" << std::endl;

        // Create a clean tensor
        auto clean_tensor = torch::randn({4, 4});
        std::cout << "Clean tensor created" << std::endl;

        // Simulate radiation corruption
        auto corrupted_tensor = simulate_bit_flips(clean_tensor, 0.01);
        std::cout << "Simulated radiation corruption applied" << std::endl;

        // Test radiation hardening on corrupted tensor
        try {
            auto hardened_tensor = apply_radiation_hardening(corrupted_tensor);
            std::cout << "Radiation hardening applied to corrupted tensor" << std::endl;

            // Check if hardening detected corruption
            bool hardened_integrity = validate_tensor_integrity(hardened_tensor);
            std::cout << "Hardened tensor integrity: " << (hardened_integrity ? "Valid" : "Invalid")
                      << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Radiation hardening caught corruption: " << e.what() << std::endl;
        }

        // Test TMR protection with corrupted data
        std::cout << "\n--- Testing TMR Protection with Corrupted Data ---" << std::endl;

        try {
            auto tmr_protected = apply_tmr_protection(
                corrupted_tensor, rad_ml::tmr::ProtectionLevel::HYBRID_REDUNDANCY);
            std::cout << "TMR protection applied to corrupted tensor" << std::endl;

            bool tmr_integrity = validate_tensor_integrity(tmr_protected);
            std::cout << "TMR protected tensor integrity: " << (tmr_integrity ? "Valid" : "Invalid")
                      << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "TMR protection caught corruption: " << e.what() << std::endl;
        }

        // Test neural network with radiation hardening
        std::cout << "\n--- Testing Neural Network Radiation Hardening ---" << std::endl;

        // Create a simple neural network
        torch::nn::Linear model(torch::nn::LinearOptions(5, 2));
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

        std::cout << "Neural network created" << std::endl;

        // Test with clean data
        auto clean_input = torch::randn({3, 5});
        auto clean_target = torch::randn({3, 2});

        auto clean_output = model->forward(clean_input);
        auto clean_loss = torch::mse_loss(clean_output, clean_target);
        clean_loss.backward();

        // Apply radiation hardening to training step
        try {
            integration.protect_training_step(static_cast<::torch::nn::Module&>(*model), optimizer);
            optimizer.step();
            std::cout << "Training step with radiation protection completed" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Training protection failed: " << e.what() << std::endl;
        }

        // Test with corrupted weights
        std::cout << "\n--- Testing Corrupted Weight Detection ---" << std::endl;

        for (auto& param : model->parameters()) {
            if (param.grad().defined()) {
                // Corrupt some gradients
                auto corrupted_grad = simulate_bit_flips(param.grad(), 0.005);
                param.grad().copy_(corrupted_grad);
                break;
            }
        }

        // Try to protect the corrupted model
        try {
            integration.protect_model(static_cast<::torch::nn::Module&>(*model));
            std::cout << "Model protection applied successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Model protection detected corruption: " << e.what() << std::endl;
        }

        // Test tensor validation with various data types
        std::cout << "\n--- Testing Tensor Validation with Various Data ---" << std::endl;

        // Test with normal tensor
        auto normal_tensor = torch::randn({2, 3});
        bool normal_valid = validate_tensor_integrity(normal_tensor);
        std::cout << "Normal tensor validation: " << (normal_valid ? "Passed" : "Failed")
                  << std::endl;

        // Test with NaN tensor
        auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
        bool nan_valid = validate_tensor_integrity(nan_tensor);
        std::cout << "NaN tensor validation: " << (nan_valid ? "Passed" : "Failed") << std::endl;

        // Test with infinite tensor
        auto inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity());
        bool inf_valid = validate_tensor_integrity(inf_tensor);
        std::cout << "Infinite tensor validation: " << (inf_valid ? "Passed" : "Failed")
                  << std::endl;

        std::cout << "\n--- All LibTorch Radiation Integration Tests Completed ---" << std::endl;

#else
        std::cout << "LibTorch integration not enabled at compile time" << std::endl;
#endif

        // Shutdown integration
        integration.shutdown();
        std::cout << "LibTorch radiation integration shutdown successfully" << std::endl;

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
