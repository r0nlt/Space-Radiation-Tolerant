/**
 * @file adaptive_protection_validation.cpp
 * @brief Monte Carlo validation of AdaptiveProtection class (Hamming, Reed-Solomon, ECC)
 *
 * This test validates the fixed AdaptiveProtection implementation:
 * - Full-byte Hamming encoding (two 7,4 codes per byte)
 * - Position-based RS storage (no hash collisions)
 * - Master seed for reproducible RNG
 * - Conservative stuck-bit detection
 */

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../../include/rad_ml/neural/adaptive_protection.hpp"

using namespace rad_ml::neural;

// Test configuration
constexpr int NUM_TRIALS = 10000;  // Fewer trials for faster execution
constexpr int NUM_ENVIRONMENTS = 4;

struct EnvironmentConfig {
    std::string name;
    double error_severity;
    AdaptiveProtectionLevel recommended_level;
};

const EnvironmentConfig ENVIRONMENTS[NUM_ENVIRONMENTS] = {
    {"LOW_RADIATION", 0.1, AdaptiveProtectionLevel::MODERATE},
    {"MEDIUM_RADIATION", 0.4, AdaptiveProtectionLevel::HIGH},
    {"HIGH_RADIATION", 0.7, AdaptiveProtectionLevel::VERY_HIGH},
    {"EXTREME_RADIATION", 0.95, AdaptiveProtectionLevel::VERY_HIGH}};

struct TestResults {
    int total_trials = 0;
    int hamming_success = 0;
    int hamming_correction = 0;
    int rs_high_success = 0;
    int rs_high_correction = 0;
    int rs_very_high_success = 0;
    int rs_very_high_correction = 0;
    int overall_success = 0;
};

template <typename T>
void runAdaptiveProtectionTests(std::mt19937& gen, std::map<std::string, TestResults>& results)
{
    std::string type_name = typeid(T).name();

    std::cout << "\nTesting type: " << type_name << "\n";
    std::cout << "=========================================\n";

    for (const auto& env : ENVIRONMENTS) {
        TestResults test_results;
        test_results.total_trials = NUM_TRIALS;

        std::cout << "  Environment: " << env.name << " (severity: " << env.error_severity << ")\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        // Set master seed for reproducibility
        AdaptiveProtection<T>::set_master_seed(42);

        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            // Generate random test value
            std::uniform_real_distribution<double> value_dist(-100.0, 100.0);
            T original_value;
            if constexpr (std::is_floating_point_v<T>) {
                original_value = static_cast<T>(value_dist(gen));
            }
            else {
                original_value =
                    static_cast<T>(std::uniform_int_distribution<int64_t>(-1000000, 1000000)(gen));
            }

            // Create AdaptiveProtection instance
            AdaptiveProtection<T> protection(RadiationEnvironment(SpaceMission::LEO_EQUATORIAL),
                                             env.recommended_level);

            // Test based on protection level
            // NOTE: We use IN-PLACE corruption to simulate real radiation effects.
            // The protection stores encoded data keyed by memory address, so we must
            // corrupt the SAME variable that was protected, not a copy.

            if (env.recommended_level == AdaptiveProtectionLevel::MODERATE) {
                // Test Hamming protection - in-place corruption
                T test_value = original_value;  // Value to protect and corrupt
                protection.protect_value(
                    test_value);  // Store encoded data for test_value's address

                // Simulate single-bit corruption IN PLACE (like radiation would)
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
                int bit_to_flip = std::uniform_int_distribution<int>(0, 7)(gen);
                bytes[0] ^= (1 << bit_to_flip);

                // Now recover using the SAME variable (same memory address)
                auto [recovered, was_corrected] = protection.recover_value(test_value);

                if (recovered == original_value) {
                    test_results.hamming_success++;
                    test_results.overall_success++;
                }
                if (was_corrected) {
                    test_results.hamming_correction++;
                }
            }
            else if (env.recommended_level == AdaptiveProtectionLevel::HIGH) {
                // Test RS-8 protection - in-place corruption
                T test_value = original_value;
                protection.protect_value(test_value);

                // Simulate multi-byte corruption IN PLACE
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
                int bytes_to_corrupt = std::min(static_cast<int>(sizeof(T)), 2);
                for (int i = 0; i < bytes_to_corrupt; ++i) {
                    bytes[i] ^= static_cast<uint8_t>(0xFF - i);
                }

                auto [recovered, was_corrected] = protection.recover_value(test_value);

                if (recovered == original_value) {
                    test_results.rs_high_success++;
                    test_results.overall_success++;
                }
                if (was_corrected) {
                    test_results.rs_high_correction++;
                }
            }
            else if (env.recommended_level == AdaptiveProtectionLevel::VERY_HIGH) {
                // Test RS-16 protection - in-place corruption
                T test_value = original_value;
                protection.protect_value(test_value);

                // Simulate severe corruption IN PLACE
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
                int bytes_to_corrupt = std::min(static_cast<int>(sizeof(T)), 4);
                for (int i = 0; i < bytes_to_corrupt; ++i) {
                    bytes[i] ^= static_cast<uint8_t>(0xAA ^ i);
                }

                auto [recovered, was_corrected] = protection.recover_value(test_value);

                if (recovered == original_value) {
                    test_results.rs_very_high_success++;
                    test_results.overall_success++;
                }
                if (was_corrected) {
                    test_results.rs_very_high_correction++;
                }
            }

            // Progress indicator
            if ((trial + 1) % 2000 == 0) {
                std::cout << "    Progress: " << (trial + 1) << "/" << NUM_TRIALS << "\r"
                          << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Store results
        std::string key = type_name + "_" + env.name;
        results[key] = test_results;

        // Print results
        std::cout << "    Completed in " << elapsed << "ms\n";
        std::cout << "    Overall Success: " << std::fixed << std::setprecision(2)
                  << (test_results.overall_success * 100.0 / NUM_TRIALS) << "%\n";

        if (test_results.hamming_success > 0) {
            std::cout << "    Hamming Success: "
                      << (test_results.hamming_success * 100.0 / NUM_TRIALS) << "%\n";
            std::cout << "    Hamming Corrections: " << test_results.hamming_correction << "\n";
        }
        if (test_results.rs_high_success > 0) {
            std::cout << "    RS-8 Success: " << (test_results.rs_high_success * 100.0 / NUM_TRIALS)
                      << "%\n";
            std::cout << "    RS-8 Corrections: " << test_results.rs_high_correction << "\n";
        }
        if (test_results.rs_very_high_success > 0) {
            std::cout << "    RS-16 Success: "
                      << (test_results.rs_very_high_success * 100.0 / NUM_TRIALS) << "%\n";
            std::cout << "    RS-16 Corrections: " << test_results.rs_very_high_correction << "\n";
        }
    }
}

// Verification test to PROVE protection is working
template <typename T>
void runVerificationTest()
{
    std::cout << "\n=== VERIFICATION TEST (Proving Protection Works) ===\n";
    std::cout << "Type: " << typeid(T).name() << "\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    AdaptiveProtection<T>::set_master_seed(12345);

    int corruption_verified = 0;
    int recovery_success = 0;

    for (int trial = 0; trial < 100; ++trial) {
        // Generate a random value
        T original;
        if constexpr (std::is_floating_point_v<T>) {
            original = static_cast<T>(std::uniform_real_distribution<double>(-100.0, 100.0)(gen));
        }
        else {
            original = static_cast<T>(std::uniform_int_distribution<int64_t>(-10000, 10000)(gen));
        }

        // Create protection and protect the value
        AdaptiveProtection<T> protection(RadiationEnvironment(SpaceMission::LEO_EQUATORIAL),
                                         AdaptiveProtectionLevel::MODERATE  // Hamming
        );

        T test_value = original;
        protection.protect_value(test_value);

        // Remember the value before corruption
        T before_corruption = test_value;

        // Corrupt the value (flip a random bit)
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
        int byte_idx = std::uniform_int_distribution<int>(0, static_cast<int>(sizeof(T)) - 1)(gen);
        int bit_idx = std::uniform_int_distribution<int>(0, 7)(gen);
        bytes[byte_idx] ^= (1 << bit_idx);

        T after_corruption = test_value;

        // Verify corruption actually happened
        if (before_corruption != after_corruption) {
            corruption_verified++;
        }

        // Try to recover
        auto [recovered, was_error] = protection.recover_value(test_value);

        // Check if we recovered the original
        if (recovered == original) {
            recovery_success++;
        }

        if (trial < 5) {
            // Show details for first 5 trials
            std::cout << "  Trial " << trial << ": ";
            std::cout << "original=" << original;
            std::cout << ", after_corrupt=" << after_corruption;
            std::cout << ", recovered=" << recovered;
            std::cout << ", match=" << (recovered == original ? "YES" : "NO");
            std::cout << "\n";
        }
    }

    std::cout << "\nResults for 100 trials:\n";
    std::cout << "  Corruptions verified: " << corruption_verified << "/100\n";
    std::cout << "  Recovered original:   " << recovery_success << "/100\n";

    if (recovery_success == corruption_verified && corruption_verified > 0) {
        std::cout << "  ✅ VERIFIED: Corrupted " << corruption_verified
                  << " values, recovered ALL!\n";
    }
    else if (corruption_verified == 0) {
        std::cout << "  ❌ ERROR: No corruption detected\n";
    }
    else {
        std::cout << "  ⚠️  Recovered " << recovery_success << "/" << corruption_verified
                  << " corrupted values\n";
    }
}

void printSummary(const std::map<std::string, TestResults>& results)
{
    std::cout << "\n";
    std::cout << "=================================================================\n";
    std::cout << "                    SUMMARY RESULTS                              \n";
    std::cout << "=================================================================\n\n";

    double total_success = 0.0;
    double total_hamming = 0.0;
    double total_rs_high = 0.0;
    double total_rs_very_high = 0.0;
    int hamming_count = 0;
    int rs_high_count = 0;
    int rs_very_high_count = 0;

    for (const auto& [key, res] : results) {
        double success_rate = res.overall_success * 100.0 / res.total_trials;
        total_success += success_rate;

        if (res.hamming_success > 0) {
            total_hamming += res.hamming_success * 100.0 / res.total_trials;
            hamming_count++;
        }
        if (res.rs_high_success > 0) {
            total_rs_high += res.rs_high_success * 100.0 / res.total_trials;
            rs_high_count++;
        }
        if (res.rs_very_high_success > 0) {
            total_rs_very_high += res.rs_very_high_success * 100.0 / res.total_trials;
            rs_very_high_count++;
        }
    }

    int total_tests = results.size();

    std::cout << "ADAPTIVE ECC PROTECTION RESULTS:\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "  Overall Average Success: " << std::fixed << std::setprecision(4)
              << (total_success / total_tests) << "%\n";

    if (hamming_count > 0) {
        std::cout << "  Hamming (Moderate) Avg:  " << (total_hamming / hamming_count) << "%\n";
    }
    if (rs_high_count > 0) {
        std::cout << "  RS-8 (High) Average:     " << (total_rs_high / rs_high_count) << "%\n";
    }
    if (rs_very_high_count > 0) {
        std::cout << "  RS-16 (Very High) Avg:   " << (total_rs_very_high / rs_very_high_count)
                  << "%\n";
    }

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Total test configurations: " << total_tests << "\n";
    std::cout << "Trials per configuration:  " << NUM_TRIALS << "\n";
    std::cout << "Total trials executed:     " << (total_tests * NUM_TRIALS) << "\n";
}

int main()
{
    std::cout << "=================================================================\n";
    std::cout << "   ADAPTIVE PROTECTION VALIDATION (Hamming, Reed-Solomon, ECC)   \n";
    std::cout << "=================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  • Trials per test: " << NUM_TRIALS << "\n";
    std::cout << "  • Environments: " << NUM_ENVIRONMENTS << "\n";
    std::cout << "  • Data types: float, double, int32_t, int64_t\n";
    std::cout << "  • Protection levels: MODERATE (Hamming), HIGH (RS-8), VERY_HIGH (RS-16)\n";
    std::cout << "=================================================================\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    std::map<std::string, TestResults> all_results;

    auto total_start = std::chrono::high_resolution_clock::now();

    runAdaptiveProtectionTests<float>(gen, all_results);
    runAdaptiveProtectionTests<double>(gen, all_results);
    runAdaptiveProtectionTests<int32_t>(gen, all_results);
    runAdaptiveProtectionTests<int64_t>(gen, all_results);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();

    printSummary(all_results);

    std::cout << "\nTotal validation time: " << total_elapsed << " seconds\n";

    // Run detailed verification to PROVE protection works
    std::cout << "\n=================================================================\n";
    std::cout << "                    PROOF OF PROTECTION                          \n";
    std::cout << "=================================================================\n";
    runVerificationTest<float>();
    runVerificationTest<int32_t>();
    std::cout << "=================================================================\n";

    return 0;
}
