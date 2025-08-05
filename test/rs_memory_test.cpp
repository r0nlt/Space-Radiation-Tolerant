/**
 * @file rs_memory_test.cpp
 * @brief Test to isolate RS8Bit8Sym memory management issues
 */

#include <iostream>
#include <rad_ml/neural/adaptive_protection.hpp>
#include <rad_ml/neural/advanced_reed_solomon.hpp>
#include <vector>

using namespace rad_ml::neural;

int main()
{
    std::cout << "Testing AdvancedReedSolomon memory management..." << std::endl;

    try {
        // Test 1: Create and destroy single AdvancedReedSolomon object
        std::cout << "Test 1: Single AdvancedReedSolomon object..." << std::endl;
        {
            RS8Bit8Sym<float> rs1;
            std::cout << "  AdvancedReedSolomon object created successfully!" << std::endl;

            float test_data = 3.14159f;
            auto encoded = rs1.encode(test_data);
            std::cout << "  Encoding completed successfully!" << std::endl;

            auto decoded = rs1.decode(encoded);
            std::cout << "  Decoding completed successfully!" << std::endl;
        }
        std::cout << "  Single object test completed!" << std::endl;

        // Test 2: Create multiple AdvancedReedSolomon objects
        std::cout << "Test 2: Multiple AdvancedReedSolomon objects..." << std::endl;
        {
            std::vector<RS8Bit8Sym<float>> rs_objects;
            for (int i = 0; i < 5; ++i) {
                rs_objects.emplace_back();
                std::cout << "  Created AdvancedReedSolomon object " << i << std::endl;
            }
            std::cout << "  All objects created successfully!" << std::endl;
        }
        std::cout << "  Multiple objects test completed!" << std::endl;

        // Test 3: Create multiple AdaptiveProtection objects
        std::cout << "Test 3: Multiple AdaptiveProtection objects..." << std::endl;
        {
            std::vector<AdaptiveProtection<float>> protection_objects;
            for (int i = 0; i < 3; ++i) {
                protection_objects.emplace_back();
                std::cout << "  Created AdaptiveProtection object " << i << std::endl;

                // Test basic functionality
                float test_value = 42.0f + i;
                float protected_value = protection_objects[i].protect_value(test_value, 1.0f);
                std::cout << "  Protection test " << i << " completed!" << std::endl;
            }
            std::cout << "  All protection objects created and tested successfully!" << std::endl;
        }
        std::cout << "  Multiple protection objects test completed!" << std::endl;

        std::cout << "All memory management tests passed!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Memory management test failed: " << e.what() << std::endl;
        return 1;
    }
}
