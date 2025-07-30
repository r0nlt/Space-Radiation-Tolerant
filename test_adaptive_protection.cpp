#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

// Include only the neural adaptive protection header (no Eigen dependency)
#include "include/rad_ml/neural/adaptive_protection.hpp"

using namespace rad_ml;

void test_hamming_code()
{
    std::cout << "Testing Hamming Code Implementation..." << std::endl;

    neural::AdaptiveProtection<float> protection;

    // Test data
    float test_value = 3.14159f;

    // Apply Hamming protection
    auto protected_value = protection.apply_hamming_protection(test_value);

    // Verify protection was applied (values should be different)
    if (protected_value == test_value) {
        std::cout << "  WARNING: Hamming protection returned unchanged value!" << std::endl;
    }
    else {
        std::cout << "  ✓ Hamming protection applied successfully" << std::endl;
    }

    // Test error correction
    auto [recovered_value, was_corrected] = protection.recover_with_hamming(protected_value);

    if (recovered_value == test_value) {
        std::cout << "  ✓ Hamming recovery successful" << std::endl;
    }
    else {
        std::cout << "  ✗ Hamming recovery failed" << std::endl;
    }

    std::cout << std::endl;
}

void test_parity_protection()
{
    std::cout << "Testing Parity Protection..." << std::endl;

    neural::AdaptiveProtection<int> protection;

    int test_value = 42;

    // Compute parity
    bool parity = protection.compute_parity(test_value);
    std::cout << "  Original value: " << test_value << ", Parity: " << (parity ? "1" : "0")
              << std::endl;

    // Add parity bit
    auto value_with_parity = protection.add_parity_bit(test_value, parity);

    // Extract parity bit
    bool extracted_parity = protection.extract_parity_bit(value_with_parity);

    if (parity == extracted_parity) {
        std::cout << "  ✓ Parity bit correctly added and extracted" << std::endl;
    }
    else {
        std::cout << "  ✗ Parity bit extraction failed" << std::endl;
    }

    // Remove parity bit
    auto value_without_parity = protection.remove_parity_bit(value_with_parity);

    if (value_without_parity == test_value) {
        std::cout << "  ✓ Parity bit correctly removed" << std::endl;
    }
    else {
        std::cout << "  ✗ Parity bit removal failed" << std::endl;
    }

    std::cout << std::endl;
}

void test_hamming_byte_encoding()
{
    std::cout << "Testing Hamming Byte Encoding..." << std::endl;

    // Test the static Hamming encoding/decoding functions directly
    uint8_t test_data = 0x0A;  // 4-bit data: 1010

    // Encode
    uint8_t encoded = neural::AdaptiveProtection<float>::hamming_encode_byte(test_data);
    std::cout << "  Original 4-bit data: 0x" << std::hex << (int)test_data << std::dec << std::endl;
    std::cout << "  Encoded 7-bit codeword: 0x" << std::hex << (int)encoded << std::dec
              << std::endl;

    // Decode without errors
    auto [decoded, was_corrected] = neural::AdaptiveProtection<float>::hamming_decode_byte(encoded);
    std::cout << "  Decoded data: 0x" << std::hex << (int)decoded << std::dec << std::endl;
    std::cout << "  Error correction applied: " << (was_corrected ? "yes" : "no") << std::endl;

    if (decoded == test_data) {
        std::cout << "  ✓ Hamming encoding/decoding successful" << std::endl;
    }
    else {
        std::cout << "  ✗ Hamming encoding/decoding failed" << std::endl;
    }

    // Test with a single-bit error
    uint8_t corrupted = encoded ^ 0x01;  // Flip first bit
    auto [error_corrected, error_fixed] =
        neural::AdaptiveProtection<float>::hamming_decode_byte(corrupted);

    std::cout << "  Corrupted codeword: 0x" << std::hex << (int)corrupted << std::dec << std::endl;
    std::cout << "  Error-corrected data: 0x" << std::hex << (int)error_corrected << std::dec
              << std::endl;
    std::cout << "  Error was corrected: " << (error_fixed ? "yes" : "no") << std::endl;

    if (error_corrected == test_data && error_fixed) {
        std::cout << "  ✓ Single-bit error correction successful" << std::endl;
    }
    else {
        std::cout << "  ✗ Single-bit error correction failed" << std::endl;
    }

    std::cout << std::endl;
}

int main()
{
    std::cout << "=== Adaptive Protection Implementation Test ===" << std::endl << std::endl;

    test_hamming_code();
    test_parity_protection();
    test_hamming_byte_encoding();

    std::cout << "=== Test Complete ===" << std::endl;

    return 0;
}
