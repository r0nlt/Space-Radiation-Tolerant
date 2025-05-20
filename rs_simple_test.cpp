#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "include/rad_ml/neural/advanced_reed_solomon.hpp"

void print_hex(const std::vector<uint8_t>& data, const std::string& label)
{
    std::cout << label << ": ";
    for (uint8_t b : data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
    }
    std::cout << std::dec << std::endl;
}

int main()
{
    using RS = rad_ml::neural::RS8Bit8Sym<float>;
    RS rs;

    // Original value
    float original_weight = 3.14159f;
    std::cout << "Original weight: " << original_weight << std::endl;

    try {
        // Encode
        auto encoded = rs.encode(original_weight);
        print_hex(encoded, "Encoded data");

        // Apply errors (10% bit error rate)
        double error_rate = 0.10;
        auto corrupted = rs.apply_bit_errors(encoded, error_rate, 42);
        print_hex(corrupted, "Corrupted data");

        // Count errors
        int bit_errors = 0;
        for (size_t i = 0; i < encoded.size(); i++) {
            uint8_t xor_result = encoded[i] ^ corrupted[i];
            bit_errors += std::bitset<8>(xor_result).count();
        }
        std::cout << "Number of bit errors: " << bit_errors << " out of " << (encoded.size() * 8)
                  << std::endl;

        // Decode
        auto decoded_opt = rs.decode(corrupted);

        if (decoded_opt) {
            float corrected_weight = *decoded_opt;
            std::cout << "Corrected weight: " << corrected_weight << std::endl;
            bool recovered = std::abs(corrected_weight - original_weight) < 1e-6;
            std::cout << "Original value recovered: " << (recovered ? "Yes" : "No") << std::endl;
        }
        else {
            std::cout << "Error correction failed: Unrecoverable data" << std::endl;
        }

        // Print ECC information
        std::cout << "Correction capability: up to " << rs.correction_capability()
                  << " symbol errors" << std::endl;
        std::cout << "Storage overhead: " << rs.overhead_percent() << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
