#include <cmath>
#include <iomanip>
#include <iostream>
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

int main()
{
    using RS = rad_ml::neural::RS8Bit8Sym<float>;  // 8-bit symbols, 8 ECC symbols (can correct 4
                                                   // symbol errors)
    RS rs;

    float original_weight = 0.7853f;
    std::cout << "Original weight: " << original_weight << std::endl;
    print_binary(original_weight);

    // 1. Encode with Reed-Solomon
    auto encoded = rs.encode(original_weight);

    // 2. Simulate random bit errors directly in the encoded data (realistic ECC test)
    double error_rate = 0.10;  // 10% bit error rate
    auto corrupted = rs.apply_bit_errors(encoded, error_rate, 42);

    std::cout << "\nEncoded data bytes (hex): ";
    for (uint8_t b : encoded)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b << " ";
    std::cout << std::dec << "\nCorrupted data bytes (hex): ";
    for (uint8_t b : corrupted)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b << " ";
    std::cout << std::dec << std::endl;

    // 3. Try to decode and correct the errors
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

    // 4. Print ECC correction capability and overhead
    std::cout << "\nCorrection capability: up to " << rs.correction_capability() << " symbol errors"
              << std::endl;
    std::cout << "Storage overhead: " << rs.overhead_percent() << "%" << std::endl;

    return 0;
}
