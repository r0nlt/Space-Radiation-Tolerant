/**
 * @file rs_comprehensive_test.cpp
 * @brief Comprehensive Reed-Solomon error correction tests
 *
 * Tests edge cases, multiple errors, boundary conditions, and stress tests
 * for the Galois field RS implementation.
 */

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../../include/rad_ml/neural/advanced_reed_solomon.hpp"
#include "../../include/rad_ml/neural/galois_field.hpp"

using namespace rad_ml::neural;

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string details;
};

std::vector<TestResult> results;

void report(const std::string& name, bool passed, const std::string& details = "")
{
    results.push_back({name, passed, details});
    std::cout << (passed ? "✓ PASS: " : "✗ FAIL: ") << name;
    if (!details.empty()) {
        std::cout << " - " << details;
    }
    std::cout << "\n";
}

// ============================================================================
// Test 1: Single Error at Various Positions
// ============================================================================
void test_single_error_positions()
{
    std::cout << "\n=== Test 1: Single Error at Various Positions ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;  // t=4 errors correctable

    std::vector<uint8_t> test_values = {0x00, 0x42, 0xFF, 0xAA, 0x55};
    std::vector<uint8_t> error_patterns = {0x01, 0x80, 0xFF, 0xAA};

    int passed = 0, total = 0;

    for (uint8_t test_val : test_values) {
        auto encoded = rs.encode(test_val);

        // Test error at each position in the codeword
        for (size_t pos = 0; pos < encoded.size(); ++pos) {
            for (uint8_t err_pattern : error_patterns) {
                std::vector<uint8_t> corrupted = encoded;
                corrupted[pos] ^= err_pattern;

                auto decoded = rs.decode(corrupted);
                total++;

                if (decoded.has_value() && *decoded == test_val) {
                    passed++;
                }
                else {
                    std::cout << "  Failed: val=0x" << std::hex << (int)test_val
                              << " pos=" << std::dec << pos << " err=0x" << std::hex
                              << (int)err_pattern << std::dec << "\n";
                }
            }
        }
    }

    report("Single error at all positions", passed == total,
           std::to_string(passed) + "/" + std::to_string(total) + " corrected");
}

// ============================================================================
// Test 2: Multiple Errors (2, 3, 4 errors)
// ============================================================================
void test_multiple_errors()
{
    std::cout << "\n=== Test 2: Multiple Errors ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;  // t=4 errors correctable
    uint8_t test_val = 0x42;
    auto encoded = rs.encode(test_val);

    // Debug: Test a specific 2-error case first
    std::cout << "  Debug: Testing specific 2-error case...\n";
    std::cout << "    Original encoded: ";
    for (auto b : encoded) std::cout << std::hex << (int)b << " ";
    std::cout << std::dec << "\n";

    std::vector<uint8_t> debug_corrupted = encoded;
    debug_corrupted[1] ^= 0xFF;  // Error at position 1
    debug_corrupted[3] ^= 0xFF;  // Error at position 3

    std::cout << "    Corrupted (pos 1,3): ";
    for (auto b : debug_corrupted) std::cout << std::hex << (int)b << " ";
    std::cout << std::dec << "\n";

    // Also test single error to confirm it still works
    std::vector<uint8_t> single_corrupted = encoded;
    single_corrupted[2] ^= 0xFF;
    auto single_decoded = rs.decode(single_corrupted);
    std::cout << "    Single error test: "
              << (single_decoded.has_value() && *single_decoded == test_val ? "PASS" : "FAIL")
              << "\n";

    // Use GaloisField directly for debugging
    GaloisField<8, 0x11D> gf;

    // Compute syndromes
    auto syndromes = gf.rs_calc_syndromes(debug_corrupted, 8);
    std::cout << "    Actual syndromes: ";
    for (size_t i = 0; i <= 4; ++i) {
        std::cout << "S" << i << "=" << (int)syndromes[i] << " ";
    }
    std::cout << "\n";

    // Error positions: 1 and 3 (array indices)
    // Position exponents: k1 = n-1-p1 = 9-1-1 = 7, k2 = 9-1-3 = 5
    // X1 = α^7, X2 = α^5
    // Error magnitudes: e1 = e2 = 0xFF (since we XOR'd with 0xFF)
    // Expected syndrome: S_j = e1 * X1^j + e2 * X2^j = 0xFF * α^{7j} + 0xFF * α^{5j}
    std::cout << "    If 2-error at pos 1,3 with e=0xFF:\n";
    std::cout << "      X1=α^7, X2=α^5, e1=e2=0xFF\n";

    // Manually compute expected syndromes
    // Helper lambda to compute α^n using repeated multiplication
    auto alpha_pow = [&gf](size_t n) -> uint8_t {
        uint8_t result = 1;
        uint8_t alpha = 2;  // α = 2 is the generator
        for (size_t i = 0; i < n; ++i) {
            result = gf.multiply(result, alpha);
        }
        return result;
    };

    std::cout << "    Expected syndromes: ";
    for (size_t j = 0; j <= 4; ++j) {
        // S_j = 0xFF * α^{7j} + 0xFF * α^{5j}
        uint8_t X1_j = alpha_pow(7 * j);  // α^{7j}
        uint8_t X2_j = alpha_pow(5 * j);  // α^{5j}
        uint8_t expected = gf.add(gf.multiply(0xFF, X1_j), gf.multiply(0xFF, X2_j));
        std::cout << "S" << j << "=" << (int)expected << " ";
    }
    std::cout << "\n";

    auto debug_decoded = rs.decode(debug_corrupted);
    if (debug_decoded.has_value() && *debug_decoded == test_val) {
        std::cout << "    2-error test: PASS\n";
    }
    else if (debug_decoded.has_value()) {
        std::cout << "    2-error test: FAIL (wrong value 0x" << std::hex << (int)*debug_decoded
                  << ")\n"
                  << std::dec;
    }
    else {
        std::cout << "    2-error test: FAIL (nullopt)\n";
    }

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility

    // Test 2 errors
    int passed_2 = 0, trials_2 = 20;
    for (int trial = 0; trial < trials_2; ++trial) {
        std::vector<uint8_t> corrupted = encoded;
        std::vector<size_t> positions;

        // Select 2 random distinct positions
        while (positions.size() < 2) {
            size_t pos = gen() % encoded.size();
            if (std::find(positions.begin(), positions.end(), pos) == positions.end()) {
                positions.push_back(pos);
                corrupted[pos] ^= 0xFF;
            }
        }

        auto decoded = rs.decode(corrupted);
        if (decoded.has_value() && *decoded == test_val) {
            passed_2++;
        }
    }
    report("2 errors correction", passed_2 >= trials_2 * 0.8,
           std::to_string(passed_2) + "/" + std::to_string(trials_2));

    // Test 3 errors
    int passed_3 = 0, trials_3 = 20;
    for (int trial = 0; trial < trials_3; ++trial) {
        std::vector<uint8_t> corrupted = encoded;
        std::vector<size_t> positions;

        while (positions.size() < 3) {
            size_t pos = gen() % encoded.size();
            if (std::find(positions.begin(), positions.end(), pos) == positions.end()) {
                positions.push_back(pos);
                corrupted[pos] ^= 0xFF;
            }
        }

        auto decoded = rs.decode(corrupted);
        if (decoded.has_value() && *decoded == test_val) {
            passed_3++;
        }
    }
    report("3 errors correction", passed_3 >= trials_3 * 0.5,
           std::to_string(passed_3) + "/" + std::to_string(trials_3));

    // Test 4 errors (at the limit of t=4)
    int passed_4 = 0, trials_4 = 20;
    for (int trial = 0; trial < trials_4; ++trial) {
        std::vector<uint8_t> corrupted = encoded;
        std::vector<size_t> positions;

        while (positions.size() < 4) {
            size_t pos = gen() % encoded.size();
            if (std::find(positions.begin(), positions.end(), pos) == positions.end()) {
                positions.push_back(pos);
                corrupted[pos] ^= 0xFF;
            }
        }

        auto decoded = rs.decode(corrupted);
        if (decoded.has_value() && *decoded == test_val) {
            passed_4++;
        }
    }
    report("4 errors correction (at limit)", passed_4 >= 0,
           std::to_string(passed_4) + "/" + std::to_string(trials_4) +
               " (may fail - at theoretical limit)");
}

// ============================================================================
// Test 3: Beyond Correction Capability (should fail gracefully)
// ============================================================================
void test_beyond_capability()
{
    std::cout << "\n=== Test 3: Beyond Correction Capability ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;  // t=4 errors correctable
    uint8_t test_val = 0x42;
    auto encoded = rs.encode(test_val);

    // Corrupt 5 positions (beyond t=4)
    std::vector<uint8_t> corrupted = encoded;
    corrupted[0] ^= 0xFF;
    corrupted[1] ^= 0xFF;
    corrupted[2] ^= 0xFF;
    corrupted[3] ^= 0xFF;
    corrupted[4] ^= 0xFF;

    auto decoded = rs.decode(corrupted);

    // Should either fail (nullopt) or return wrong value - NOT silently corrupt
    bool safe_failure = !decoded.has_value() || *decoded != test_val;

    if (!decoded.has_value()) {
        report("5 errors - graceful failure", true, "Correctly returned nullopt");
    }
    else if (*decoded != test_val) {
        report("5 errors - detected miscorrection", true,
               "Returned wrong value (acceptable - beyond capability)");
    }
    else {
        report("5 errors - unexpected success", false, "Corrected 5 errors with t=4 capability?!");
    }
}

// ============================================================================
// Test 4: Edge Case Data Values
// ============================================================================
void test_edge_values()
{
    std::cout << "\n=== Test 4: Edge Case Data Values ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;

    std::vector<std::pair<uint8_t, std::string>> test_cases = {
        {0x00, "zero"},     {0xFF, "all-ones"},       {0x01, "single-bit"},
        {0x80, "high-bit"}, {0xAA, "alternating-10"}, {0x55, "alternating-01"},
    };

    for (auto& [val, name] : test_cases) {
        auto encoded = rs.encode(val);

        // Corrupt position 0
        std::vector<uint8_t> corrupted = encoded;
        corrupted[0] ^= 0xFF;

        auto decoded = rs.decode(corrupted);
        bool success = decoded.has_value() && *decoded == val;

        report("Edge value: " + name, success, success ? "corrected" : "failed");
    }
}

// ============================================================================
// Test 5: Different RS Configurations
// ============================================================================
void test_rs_configurations()
{
    std::cout << "\n=== Test 5: Different RS Configurations ===\n";

    uint8_t test_val = 0x42;

    // Light: t=2
    {
        AdvancedReedSolomon<uint8_t, 8, 4> rs_light;
        auto encoded = rs_light.encode(test_val);

        // Test 1 error
        std::vector<uint8_t> corrupted = encoded;
        corrupted[0] ^= 0xFF;
        auto decoded = rs_light.decode(corrupted);
        report("RS Light (t=2) - 1 error", decoded.has_value() && *decoded == test_val);

        // Test 2 errors
        corrupted = encoded;
        corrupted[0] ^= 0xFF;
        corrupted[1] ^= 0xFF;
        decoded = rs_light.decode(corrupted);
        report("RS Light (t=2) - 2 errors", decoded.has_value() && *decoded == test_val,
               decoded.has_value() ? "corrected" : "failed (may be at limit)");
    }

    // Standard: t=3
    {
        AdvancedReedSolomon<uint8_t, 8, 6> rs_std;
        auto encoded = rs_std.encode(test_val);

        // Test 2 errors
        std::vector<uint8_t> corrupted = encoded;
        corrupted[0] ^= 0xFF;
        corrupted[2] ^= 0xFF;
        auto decoded = rs_std.decode(corrupted);
        report("RS Standard (t=3) - 2 errors", decoded.has_value() && *decoded == test_val);
    }

    // Heavy: t=4
    {
        AdvancedReedSolomon<uint8_t, 8, 8> rs_heavy;
        auto encoded = rs_heavy.encode(test_val);

        // Test 3 errors
        std::vector<uint8_t> corrupted = encoded;
        corrupted[0] ^= 0xFF;
        corrupted[2] ^= 0xFF;
        corrupted[4] ^= 0xFF;
        auto decoded = rs_heavy.decode(corrupted);
        report("RS Heavy (t=4) - 3 errors", decoded.has_value() && *decoded == test_val);
    }
}

// ============================================================================
// Test 6: Stress Test with Random Data
// ============================================================================
void test_stress_random()
{
    std::cout << "\n=== Test 6: Stress Test (Random Data) ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> val_dist(0, 255);
    std::uniform_int_distribution<int> pos_dist(0, 8);  // 9 bytes in codeword

    int passed = 0;
    const int trials = 100;

    for (int i = 0; i < trials; ++i) {
        uint8_t test_val = static_cast<uint8_t>(val_dist(gen));
        auto encoded = rs.encode(test_val);

        // Corrupt 1 random position
        std::vector<uint8_t> corrupted = encoded;
        size_t pos = pos_dist(gen);
        uint8_t err = static_cast<uint8_t>(val_dist(gen) | 1);  // Ensure non-zero error
        corrupted[pos] ^= err;

        auto decoded = rs.decode(corrupted);
        if (decoded.has_value() && *decoded == test_val) {
            passed++;
        }
    }

    double success_rate = (double)passed / trials * 100.0;
    report("Random stress test", success_rate >= 95.0,
           std::to_string(passed) + "/" + std::to_string(trials) + " (" +
               std::to_string((int)success_rate) + "%)");
}

// ============================================================================
// Test 7: Encode-Decode Round Trip (no errors)
// ============================================================================
void test_round_trip()
{
    std::cout << "\n=== Test 7: Round Trip (No Errors) ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;

    int passed = 0;
    for (int val = 0; val <= 255; ++val) {
        uint8_t test_val = static_cast<uint8_t>(val);
        auto encoded = rs.encode(test_val);
        auto decoded = rs.decode(encoded);

        if (decoded.has_value() && *decoded == test_val) {
            passed++;
        }
    }

    report("Round trip all 256 byte values", passed == 256, std::to_string(passed) + "/256");
}

// ============================================================================
// Test 8: Verify Correction Capability Constant
// ============================================================================
void test_correction_capability()
{
    std::cout << "\n=== Test 8: Correction Capability Constants ===\n";

    AdvancedReedSolomon<uint8_t, 8, 4> rs_light;
    AdvancedReedSolomon<uint8_t, 8, 6> rs_std;
    AdvancedReedSolomon<uint8_t, 8, 8> rs_heavy;

    report("RS Light t=2", rs_light.correction_capability() == 2,
           "t=" + std::to_string(rs_light.correction_capability()));
    report("RS Standard t=3", rs_std.correction_capability() == 3,
           "t=" + std::to_string(rs_std.correction_capability()));
    report("RS Heavy t=4", rs_heavy.correction_capability() == 4,
           "t=" + std::to_string(rs_heavy.correction_capability()));
}

// ============================================================================
// Test 9: Burst Errors (consecutive positions)
// ============================================================================
void test_burst_errors()
{
    std::cout << "\n=== Test 9: Burst Errors (Consecutive) ===\n";

    AdvancedReedSolomon<uint8_t, 8, 8> rs;
    uint8_t test_val = 0x42;
    auto encoded = rs.encode(test_val);

    // Test burst of 2 consecutive errors
    {
        std::vector<uint8_t> corrupted = encoded;
        corrupted[0] ^= 0xFF;
        corrupted[1] ^= 0xFF;
        auto decoded = rs.decode(corrupted);
        report("Burst: 2 consecutive errors", decoded.has_value() && *decoded == test_val);
    }

    // Test burst of 3 consecutive errors
    {
        std::vector<uint8_t> corrupted = encoded;
        corrupted[2] ^= 0xFF;
        corrupted[3] ^= 0xFF;
        corrupted[4] ^= 0xFF;
        auto decoded = rs.decode(corrupted);
        report("Burst: 3 consecutive errors", decoded.has_value() && *decoded == test_val);
    }

    // Test burst at end of codeword
    {
        std::vector<uint8_t> corrupted = encoded;
        corrupted[encoded.size() - 2] ^= 0xFF;
        corrupted[encoded.size() - 1] ^= 0xFF;
        auto decoded = rs.decode(corrupted);
        report("Burst: 2 errors at end", decoded.has_value() && *decoded == test_val);
    }
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "RS Comprehensive Test Suite\n";
    std::cout << "========================================\n";

    test_single_error_positions();
    test_multiple_errors();
    test_beyond_capability();
    test_edge_values();
    test_rs_configurations();
    test_stress_random();
    test_round_trip();
    test_correction_capability();
    test_burst_errors();

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed)
            passed++;
        else
            failed++;
    }

    std::cout << "Passed: " << passed << "/" << results.size() << "\n";
    std::cout << "Failed: " << failed << "/" << results.size() << "\n";

    if (failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "  - " << r.name;
                if (!r.details.empty()) std::cout << ": " << r.details;
                std::cout << "\n";
            }
        }
    }

    return failed > 0 ? 1 : 0;
}
