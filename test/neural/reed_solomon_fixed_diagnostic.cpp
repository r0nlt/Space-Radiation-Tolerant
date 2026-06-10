/**
 * @file reed_solomon_fixed_diagnostic.cpp
 * @brief Manual diagnostic: RS systematic encoding, syndromes, and single-error correction
 *
 * Verbose step-by-step output for debugging GF(256) encode/decode. Run directly;
 * not an automated pass/fail regression test (see galois_field_bm_noheap_parity_test).
 */

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../include/rad_ml/neural/galois_field.hpp"

using namespace rad_ml::neural;

void print_vector(const std::vector<uint8_t>& vec, const std::string& name)
{
    std::cout << "    " << name << ": ";
    for (auto v : vec) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)v << " ";
    }
    std::cout << std::dec << "\n";
}

// Helper function to perform polynomial division for systematic encoding
std::vector<uint8_t> polynomial_division(const std::vector<uint8_t>& dividend,
                                         const std::vector<uint8_t>& divisor, const GF256& gf)
{
    // Guard against invalid input sizes
    if (dividend.size() < divisor.size()) {
        std::cerr << "Error: dividend size (" << dividend.size()
                  << ") is smaller than divisor size (" << divisor.size()
                  << "). Cannot perform polynomial division." << std::endl;
        return std::vector<uint8_t>(divisor.size() - 1, 0);  // Return zero remainder
    }

    std::vector<uint8_t> remainder = dividend;

    // Perform polynomial long division
    for (size_t i = 0; i <= dividend.size() - divisor.size(); ++i) {
        if (remainder[i] != 0) {
            uint8_t coeff = remainder[i];

            // Subtract (divisor * coeff) from remainder
            for (size_t j = 0; j < divisor.size(); ++j) {
                remainder[i + j] = gf.add(remainder[i + j], gf.multiply(divisor[j], coeff));
            }
        }
    }

    // Return the remainder (last nsym elements)
    return std::vector<uint8_t>(remainder.end() - divisor.size() + 1, remainder.end());
}

// Alternative implementation using the shift-register approach
std::vector<uint8_t> systematic_encoding_shift_register(const std::vector<uint8_t>& data,
                                                        const std::vector<uint8_t>& gen_poly,
                                                        const GF256& gf)
{
    const size_t nsym = gen_poly.size() - 1;
    std::vector<uint8_t> remainder(nsym, 0);

    // Process each data symbol
    for (size_t i = 0; i < data.size(); ++i) {
        uint8_t feedback = gf.add(data[i], remainder[0]);

        // Shift remainder left and apply generator polynomial
        for (size_t j = 0; j < nsym - 1; ++j) {
            remainder[j] = gf.add(remainder[j + 1], gf.multiply(gen_poly[j + 1], feedback));
        }
        remainder[nsym - 1] = gf.multiply(gen_poly[nsym], feedback);
    }

    return remainder;
}

int main()
{
    std::cout << "=== CORRECTED Reed-Solomon Encoding Test ===\n\n";

    try {
        GF256 gf;

        // Test data
        std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04};
        const uint8_t nsym = 4;

        std::cout << "STEP 1: Input Data\n";
        print_vector(data, "Original data");

        std::cout << "\nSTEP 2: Generator Polynomial\n";
        auto gen_poly = gf.rs_generator_poly(nsym);
        std::cout << "    Generator polynomial: ";
        for (auto coeff : gen_poly) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)coeff << " ";
        }
        std::cout << std::dec << " (size: " << gen_poly.size() << ")\n";

        // Verify generator polynomial by checking it has roots at α^0, α^1, ..., α^(nsym-1)
        std::cout << "    Verifying generator polynomial roots:\n";
        for (uint8_t i = 0; i < nsym; ++i) {
            uint8_t alpha_i = 1;
            for (uint8_t j = 0; j < i; ++j) {
                alpha_i = gf.multiply(alpha_i, 2);
            }

            uint8_t eval_result = 0;
            for (const auto& coeff : gen_poly) {
                eval_result = gf.add(gf.multiply(eval_result, alpha_i), coeff);
            }

            std::cout << "      g(α^" << (int)i << ") = g(" << std::hex << (int)alpha_i
                      << ") = " << (int)eval_result << std::dec;
            if (eval_result == 0) {
                std::cout << " ✓\n";
            }
            else {
                std::cout << " ✗\n";
            }
        }

        std::cout << "\nSTEP 3: CORRECTED Systematic Encoding\n";

        // Method 1: Polynomial division approach
        std::cout << "    Method 1: Polynomial division approach\n";
        // Prepare dividend: data * x^nsym (shifted left by nsym positions)
        std::vector<uint8_t> dividend = data;
        dividend.insert(dividend.end(), nsym, 0);  // Add nsym zeros

        std::cout << "      Dividend (data * x^" << (int)nsym << "): ";
        for (auto d : dividend) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)d << " ";
        }
        std::cout << std::dec << "\n";

        // Perform polynomial division to get remainder
        std::vector<uint8_t> remainder1 = polynomial_division(dividend, gen_poly, gf);

        std::cout << "      Remainder from division: ";
        for (auto r : remainder1) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)r << " ";
        }
        std::cout << std::dec << "\n";

        // Method 2: Shift-register approach
        std::cout << "    Method 2: Shift-register approach\n";
        std::vector<uint8_t> remainder2 = systematic_encoding_shift_register(data, gen_poly, gf);

        std::cout << "      Remainder from shift-register: ";
        for (auto r : remainder2) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)r << " ";
        }
        std::cout << std::dec << "\n";

        // Use the shift-register result (more reliable for systematic encoding)
        std::vector<uint8_t> remainder = remainder2;

        // Create systematic codeword: [data | remainder]
        std::vector<uint8_t> codeword = data;
        codeword.insert(codeword.end(), remainder.begin(), remainder.end());

        print_vector(codeword, "CORRECTED codeword");

        std::cout << "\nSTEP 4: Syndrome Test\n";
        auto syndromes = gf.rs_calc_syndromes(codeword, nsym);
        std::cout << "    Syndromes: ";
        for (size_t i = 0; i < syndromes.size(); ++i) {
            std::cout << "S" << i << "=" << std::hex << (int)syndromes[i] << " ";
        }
        std::cout << std::dec << "\n";

        // Debug: Manual syndrome calculation
        std::cout << "    Manual syndrome calculation:\n";
        for (uint8_t i = 0; i <= nsym; ++i) {
            // Calculate α^i manually
            uint8_t alpha_i = 1;
            for (uint8_t j = 0; j < i; ++j) {
                alpha_i = gf.multiply(alpha_i, 2);  // α = 2 in GF(256)
            }

            uint8_t manual_syndrome = 0;

            // Evaluate codeword at alpha^i using Horner's method
            for (const auto& coeff : codeword) {
                manual_syndrome = gf.add(gf.multiply(manual_syndrome, alpha_i), coeff);
            }

            std::cout << "      S" << i << " at α^" << i << " (α^" << i << "=" << std::hex
                      << (int)alpha_i << "): " << (int)manual_syndrome << std::dec << "\n";
        }

        // Check if the relevant syndromes are zero
        // Generator roots at α⁰..α^(nsym-1) → syndromes[0]..syndromes[nsym-1] must be zero
        bool all_relevant_zero = true;
        std::cout << "    Checking syndromes S₀ through S" << (nsym - 1) << ":\n";
        for (size_t i = 0; i < static_cast<size_t>(nsym); ++i) {
            std::cout << "      S" << i << " = " << std::hex << (int)syndromes[i] << std::dec;
            if (syndromes[i] != 0) {
                all_relevant_zero = false;
                std::cout << " ✗\n";
            }
            else {
                std::cout << " ✓\n";
            }
        }

        // Also note what S₀ and S_nsym are (for reference)
        std::cout << "    Reference syndromes:\n";
        std::cout << "      S₀ = " << std::hex << (int)syndromes[0] << std::dec
                  << " (should be 0)\n";
        std::cout << "      S" << nsym << " = " << std::hex << (int)syndromes[nsym] << std::dec
                  << " (can be non-zero)\n";

        if (all_relevant_zero) {
            std::cout << "    ✓ SUCCESS: All relevant syndromes are zero - VALID CODEWORD!\n";
        }
        else {
            std::cout << "    ✗ FAILURE: Non-zero syndromes detected in S₀ through S" << (int)nsym - 1
                      << "\n";
        }

        std::cout << "\nSTEP 5: Complete Pipeline Test (with corrected library)\n";
        auto result = gf.rs_correct_errors(codeword, nsym);
        if (result.has_value()) {
            std::cout << "    ✓ rs_correct_errors succeeded!\n";
            print_vector(result.value(), "Decoded result");

            // Check data integrity
            bool data_match = true;
            for (size_t i = 0; i < data.size(); ++i) {
                if (i >= result->size() || (*result)[i] != data[i]) {
                    data_match = false;
                    break;
                }
            }
            std::cout << "    Data integrity: " << (data_match ? "✓ PERFECT" : "✗ CORRUPTED")
                      << "\n";
        }
        else {
            std::cout << "    ✗ rs_correct_errors still failed - need to investigate further\n";
        }

        std::cout << "\nSTEP 6: Error Correction Test\n";
        std::vector<uint8_t> corrupted = codeword;
        corrupted[1] ^= 0x55;  // Inject error
        std::cout << "    Injected error at position 1: " << std::hex << (int)codeword[1] << " -> "
                  << (int)corrupted[1] << std::dec << "\n";
        print_vector(corrupted, "Corrupted codeword");

        // Check syndromes of corrupted message
        auto corrupted_syndromes = gf.rs_calc_syndromes(corrupted, nsym);
        std::cout << "    Corrupted syndromes: ";
        for (size_t i = 0; i < corrupted_syndromes.size(); ++i) {
            std::cout << "S" << i << "=" << std::hex << (int)corrupted_syndromes[i] << " ";
        }
        std::cout << std::dec << "\n";

        auto corrected = gf.rs_correct_errors(corrupted, nsym);
        if (corrected.has_value()) {
            std::cout << "    ✓ Error correction succeeded!\n";
            print_vector(corrected.value(), "Corrected result");

            // Check if correction is perfect
            bool perfect = true;
            for (size_t i = 0; i < data.size(); ++i) {
                if (i >= corrected->size() || (*corrected)[i] != data[i]) {
                    perfect = false;
                    break;
                }
            }
            std::cout << "    Error correction: " << (perfect ? "✓ PERFECT" : "✗ IMPERFECT")
                      << "\n";
        }
        else {
            std::cout << "    ✗ Error correction failed - investigating further issues in error "
                         "correction pipeline\n";
        }
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
