/**
 * @file minimal_adaptive_test.cpp
 * @brief Minimal test to isolate adaptive protection issues
 */

#include <iostream>
#include <rad_ml/neural/adaptive_protection.hpp>

#ifdef RAD_ML_PYTORCH_ENABLED
#include <torch/torch.h>
#endif

int main()
{
    std::cout << "Starting minimal adaptive protection test..." << std::endl;

    try {
        std::cout << "Including adaptive protection header..." << std::endl;

        // Just create the object to see if it crashes
        rad_ml::neural::AdaptiveProtection<float> protection;
        std::cout << "AdaptiveProtection object created successfully!" << std::endl;

        // Test the specific methods used in the radiation integration test
        std::cout << "Testing set_environment..." << std::endl;
        protection.set_environment(
            rad_ml::neural::RadiationEnvironment(rad_ml::neural::SpaceMission::LEO_EQUATORIAL));
        std::cout << "set_environment completed!" << std::endl;

        std::cout << "Testing set_protection_level..." << std::endl;
        protection.set_protection_level(rad_ml::neural::ProtectionLevel::HIGH);
        std::cout << "set_protection_level completed!" << std::endl;

        std::cout << "Testing protect_value..." << std::endl;
        float test_value = 3.14159f;
        float protected_value = protection.protect_value(test_value, 1.0f);
        std::cout << "protect_value completed!" << std::endl;

        std::cout << "Testing recover_value..." << std::endl;
        auto [recovered_value, was_corrected] = protection.recover_value(protected_value, 1.0f);
        std::cout << "recover_value completed!" << std::endl;

#ifdef RAD_ML_PYTORCH_ENABLED
        std::cout << "Testing PyTorch integration..." << std::endl;
        bool cuda_available = torch::cuda::is_available();
        std::cout << "CUDA available: " << (cuda_available ? "Yes" : "No") << std::endl;
        std::cout << "PyTorch integration completed!" << std::endl;
#endif

        std::cout << "Test completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
