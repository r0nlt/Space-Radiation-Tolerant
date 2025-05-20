#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "include/rad_ml/neural/advanced_reed_solomon.hpp"

// Print the binary representation of a float
void print_binary(float value)
{
    union {
        float f;
        uint32_t i;
    } data;
    data.f = value;

    std::cout << "Binary: ";
    for (int bit = 31; bit >= 0; bit--) {
        std::cout << ((data.i >> bit) & 1);
        if (bit % 8 == 0) std::cout << ' ';
    }
    std::cout << std::endl;
}

// Test Reed-Solomon at a specific error rate
bool test_rs_at_error_rate(double error_rate, unsigned int seed = 42)
{
    using RS = rad_ml::neural::RS8Bit8Sym<float>;
    RS rs;

    float original_weight = 0.7853f;

    // Encode
    auto encoded = rs.encode(original_weight);

    // Corrupt
    auto corrupted = rs.apply_bit_errors(encoded, error_rate, seed);

    // Count the number of corrupted bytes
    int corrupted_bytes = 0;
    for (size_t i = 0; i < encoded.size(); i++) {
        if (encoded[i] != corrupted[i]) {
            corrupted_bytes++;
        }
    }

    // Decode
    auto decoded_opt = rs.decode(corrupted);

    // Return true if successfully recovered
    if (decoded_opt) {
        float corrected_weight = *decoded_opt;
        return std::abs(corrected_weight - original_weight) < 1e-6;
    }

    return false;
}

int main()
{
    using RS = rad_ml::neural::RS8Bit8Sym<float>;
    RS rs;

    // First, run the basic test at 10% error rate
    float original_weight = 0.7853f;
    std::cout << "BASIC TEST WITH 10% ERROR RATE\n";
    std::cout << "==============================\n";
    std::cout << "Original weight: " << original_weight << std::endl;
    print_binary(original_weight);

    // 1. Encode with Reed-Solomon
    auto encoded = rs.encode(original_weight);

    // 2. Simulate random bit errors
    double error_rate = 0.10;  // 10% bit error rate
    auto corrupted = rs.apply_bit_errors(encoded, error_rate, 42);

    std::cout << "\nEncoded data bytes (hex): ";
    for (uint8_t b : encoded)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b << " ";
    std::cout << std::dec << "\nCorrupted data bytes (hex): ";
    for (uint8_t b : corrupted)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b << " ";
    std::cout << std::dec << std::endl;

    // Count corrupted bytes
    int corrupted_bytes = 0;
    for (size_t i = 0; i < encoded.size(); i++) {
        if (encoded[i] != corrupted[i]) {
            corrupted_bytes++;
        }
    }
    std::cout << "Number of corrupted bytes: " << corrupted_bytes << " out of " << encoded.size()
              << std::endl;

    // 3. Try to decode and correct
    auto decoded_opt = rs.decode(corrupted);

    if (decoded_opt) {
        float corrected_weight = *decoded_opt;
        std::cout << "\nCorrected weight: " << corrected_weight << std::endl;
        print_binary(corrected_weight);
        bool recovered = std::abs(corrected_weight - original_weight) < 1e-6;
        std::cout << "Original value recovered: " << (recovered ? "Yes" : "No") << std::endl;
    }
    else {
        std::cout
            << "\nError correction failed: Unrecoverable data (too many errors for ECC capability)"
            << std::endl;
    }

    std::cout << "\nCorrection capability: up to " << rs.correction_capability() << " symbol errors"
              << std::endl;
    std::cout << "Storage overhead: " << rs.overhead_percent() << "%" << std::endl;

    // Now test multiple error rates to find failure threshold
    std::cout << "\n\nERROR RATE THRESHOLD TEST\n";
    std::cout << "========================\n";

    const int NUM_TRIALS = 100;
    std::vector<double> error_rates = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};

    for (double rate : error_rates) {
        int success_count = 0;

        // Run multiple trials at this error rate with different seeds
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            if (test_rs_at_error_rate(rate, trial)) {
                success_count++;
            }
        }

        double success_rate = static_cast<double>(success_count) / NUM_TRIALS * 100.0;
        std::cout << "Error rate " << std::fixed << std::setprecision(2) << rate * 100.0
                  << "%: " << success_count << "/" << NUM_TRIALS << " successful corrections ("
                  << success_rate << "% success rate)" << std::endl;
    }

    return 0;
}
