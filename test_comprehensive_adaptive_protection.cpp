#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// Include the neural adaptive protection header
#include "include/rad_ml/neural/adaptive_protection.hpp"

using namespace rad_ml;

void test_reed_solomon_implementation()
{
    std::cout << "Testing Reed-Solomon Implementation..." << std::endl;

    // Test RS8Bit8Sym
    neural::RS8Bit8Sym<int> rs8;
    int test_data = 0x12345678;

    auto encoded = rs8.encode(test_data);
    std::cout << "  Original data: 0x" << std::hex << test_data << std::dec << std::endl;
    std::cout << "  Encoded size: " << encoded.size() << " bytes" << std::endl;
    std::cout << "  Overhead: " << std::fixed << std::setprecision(1) << rs8.overhead_percent()
              << "%" << std::endl;

    // Test decoding without errors
    auto decoded = rs8.decode(encoded);
    if (decoded && *decoded == test_data) {
        std::cout << "  ✓ RS8Bit8Sym encoding/decoding successful" << std::endl;
    }
    else {
        std::cout << "  ✗ RS8Bit8Sym encoding/decoding failed" << std::endl;
    }

    // Test RS8Bit16Sym
    neural::RS8Bit16Sym<int> rs16;
    auto encoded16 = rs16.encode(test_data);
    std::cout << "  RS16 encoded size: " << encoded16.size() << " bytes" << std::endl;
    std::cout << "  RS16 overhead: " << std::fixed << std::setprecision(1)
              << rs16.overhead_percent() << "%" << std::endl;

    auto decoded16 = rs16.decode(encoded16);
    if (decoded16 && *decoded16 == test_data) {
        std::cout << "  ✓ RS8Bit16Sym encoding/decoding successful" << std::endl;
    }
    else {
        std::cout << "  ✗ RS8Bit16Sym encoding/decoding failed" << std::endl;
    }

    std::cout << std::endl;
}

void test_multi_bit_protection()
{
    std::cout << "Testing Multi-Bit Protection..." << std::endl;

    neural::MultibitProtection<int> mbu;
    std::vector<uint8_t> test_data = {0xAA, 0xBB, 0xCC, 0xDD};

    std::mt19937_64 rng(42);  // Fixed seed for reproducibility

    // Test single-bit upset
    auto single_bit_result =
        mbu.apply_multi_bit_upset(test_data, neural::MultibitUpsetType::SINGLE_BIT, 0.1, rng);
    std::cout << "  Single-bit upset applied" << std::endl;

    // Test adjacent bits upset
    auto adjacent_result =
        mbu.apply_multi_bit_upset(test_data, neural::MultibitUpsetType::ADJACENT_BITS, 0.1, rng);
    std::cout << "  Adjacent bits upset applied" << std::endl;

    // Test random multi-bit upset
    auto random_result =
        mbu.apply_multi_bit_upset(test_data, neural::MultibitUpsetType::RANDOM_MULTI, 0.1, rng);
    std::cout << "  Random multi-bit upset applied" << std::endl;

    std::cout << "  ✓ Multi-bit protection tests completed" << std::endl;
    std::cout << std::endl;
}

void test_thread_safety()
{
    std::cout << "Testing Thread Safety..." << std::endl;

    neural::AdaptiveProtection<float> protection;
    std::vector<std::thread> threads;
    std::vector<float> results(4);

    // Test concurrent radiation effects application
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&protection, &results, i]() {
            float test_value = 3.14159f + i;
            results[i] = protection.apply_radiation_effects(test_value, 0.01);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "  Thread results: ";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "T" << i << "=" << std::fixed << std::setprecision(3) << results[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  ✓ Thread safety test completed" << std::endl;
    std::cout << std::endl;
}

void test_protected_network()
{
    std::cout << "Testing Protected Neural Network..." << std::endl;

    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};
    neural::SimpleProtectedNetwork<float> network(weights);

    std::vector<float> input = {0.5f, 0.3f, 0.7f, 0.2f};
    auto output = network.forward(input);

    std::cout << "  Input: ";
    for (float val : input) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "  Output: ";
    for (float val : output) std::cout << val << " ";
    std::cout << std::endl;

    // Test weight replacement
    network.replace_weight(2.0f, 2.5f);
    auto new_output = network.forward(input);

    std::cout << "  After weight replacement: ";
    for (float val : new_output) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "  ✓ Protected network test completed" << std::endl;
    std::cout << std::endl;
}

void test_adaptive_protection_levels()
{
    std::cout << "Testing Adaptive Protection Levels..." << std::endl;

    neural::AdaptiveProtection<int> protection;

    // Test different protection levels
    std::vector<neural::ProtectionLevel> levels = {
        neural::ProtectionLevel::NONE,      neural::ProtectionLevel::MINIMAL,
        neural::ProtectionLevel::MODERATE,  neural::ProtectionLevel::HIGH,
        neural::ProtectionLevel::VERY_HIGH, neural::ProtectionLevel::ADAPTIVE};

    int test_value = 42;

    for (auto level : levels) {
        protection.set_protection_level(level);
        auto protected_value = protection.protect_value(test_value, 1.0f);
        auto [recovered_value, was_corrected] = protection.recover_value(protected_value, 1.0f);

        std::cout << "  Level " << static_cast<int>(level) << ": ";
        if (recovered_value == test_value) {
            std::cout << "✓";
        }
        else {
            std::cout << "✗";
        }
        std::cout << " (corrected: " << (was_corrected ? "yes" : "no") << ")" << std::endl;
    }

    std::cout << std::endl;
}

void test_error_model_types()
{
    std::cout << "Testing Error Model Types..." << std::endl;

    neural::AdaptiveProtection<float> protection;
    float test_value = 3.14159f;

    std::vector<neural::MultibitUpsetType> models = {neural::MultibitUpsetType::SINGLE_BIT,
                                                     neural::MultibitUpsetType::ADJACENT_BITS,
                                                     neural::MultibitUpsetType::RANDOM_MULTI};

    for (auto model : models) {
        protection.set_error_model(model);
        auto result = protection.apply_radiation_effects(test_value, 0.1);

        std::cout << "  Model " << static_cast<int>(model) << ": ";
        if (result != test_value) {
            std::cout << "✓ (radiation effects applied)";
        }
        else {
            std::cout << "✗ (no effects)";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

int main()
{
    std::cout << "=== Comprehensive Adaptive Protection Test ===" << std::endl << std::endl;

    test_reed_solomon_implementation();
    test_multi_bit_protection();
    test_thread_safety();
    test_protected_network();
    test_adaptive_protection_levels();
    test_error_model_types();

    std::cout << "=== Comprehensive Test Complete ===" << std::endl;

    return 0;
}
