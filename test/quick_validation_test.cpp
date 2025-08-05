#include <bitset>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

// Include the voting mechanisms
#include "../include/rad_ml/core/redundancy/enhanced_voting.hpp"

using namespace rad_ml::core::redundancy;

// Copy the error injection functions from monte_carlo_validation.cpp
template <typename T>
T injectSingleBitError(T value, std::mt19937& gen)
{
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));

    // Select random bit to flip
    std::uniform_int_distribution<int> dist(0, sizeof(T) * 8 - 1);
    int bit_pos = dist(gen);

    // Flip the bit
    bits ^= (UintType(1) << bit_pos);

    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

template <typename T>
T injectMultiBitError(T value, std::mt19937& gen)
{
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));

    // Select random starting bit
    std::uniform_int_distribution<int> start_dist(0, sizeof(T) * 8 - 4);
    std::uniform_int_distribution<int> num_bits_dist(2, 3);

    int start_bit = start_dist(gen);
    int num_bits = num_bits_dist(gen);

    // Flip consecutive bits
    for (int i = 0; i < num_bits; i++) {
        int bit_pos = (start_bit + i) % (sizeof(T) * 8);
        bits ^= (UintType(1) << bit_pos);
    }

    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Helper function to print binary representation
template <typename T>
void printBinary(const T& value, const std::string& label)
{
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));

    std::cout << label << " (decimal): " << value << std::endl;
    std::cout << label << " (binary):  " << std::bitset<sizeof(T) * 8>(bits) << std::endl;
}

int main()
{
    std::cout << "=== QUICK VALIDATION TEST ===" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Test 1: Check Error Injection
    std::cout << "\n1. CHECKING ERROR INJECTION:" << std::endl;
    std::cout << "=============================" << std::endl;

    float original_value = 42.5f;
    printBinary(original_value, "Original");

    float corrupted_single = injectSingleBitError(original_value, gen);
    printBinary(corrupted_single, "Single Bit Corrupted");

    float corrupted_multi = injectMultiBitError(original_value, gen);
    printBinary(corrupted_multi, "Multi Bit Corrupted");

    // Check if corruption actually happened
    bool single_changed = (original_value != corrupted_single);
    bool multi_changed = (original_value != corrupted_multi);

    std::cout << "\nError Injection Results:" << std::endl;
    std::cout << "Single bit corruption: " << (single_changed ? "✅ WORKING" : "❌ FAILED")
              << std::endl;
    std::cout << "Multi bit corruption:  " << (multi_changed ? "✅ WORKING" : "❌ FAILED")
              << std::endl;

    // Test 2: Verify Voting Logic
    std::cout << "\n2. VERIFYING VOTING LOGIC:" << std::endl;
    std::cout << "===========================" << std::endl;

    // Create test scenario: 2 good copies, 1 corrupted
    float copy1 = original_value;    // Good
    float copy2 = original_value;    // Good
    float copy3 = corrupted_single;  // Corrupted

    std::cout << "Test scenario: 2 good copies, 1 corrupted" << std::endl;
    printBinary(copy1, "Copy1 (Good)");
    printBinary(copy2, "Copy2 (Good)");
    printBinary(copy3, "Copy3 (Corrupted)");

    // Test different voting methods
    float standard_result = EnhancedVoting::standardVote(copy1, copy2, copy3);
    float bit_level_result = EnhancedVoting::bitLevelVote(copy1, copy2, copy3);
    float adaptive_result =
        EnhancedVoting::adaptiveVote(copy1, copy2, copy3, FaultPattern::SINGLE_BIT);

    std::cout << "\nVoting Results:" << std::endl;
    printBinary(standard_result, "Standard Vote");
    printBinary(bit_level_result, "Bit-Level Vote");
    printBinary(adaptive_result, "Adaptive Vote");

    // Check if voting methods are different
    bool standard_correct = (standard_result == original_value);
    bool bit_level_correct = (bit_level_result == original_value);
    bool adaptive_correct = (adaptive_result == original_value);

    std::cout << "\nVoting Logic Results:" << std::endl;
    std::cout << "Standard vote correct: " << (standard_correct ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Bit-level vote correct: " << (bit_level_correct ? "✅ YES" : "❌ NO")
              << std::endl;
    std::cout << "Adaptive vote correct: " << (adaptive_correct ? "✅ YES" : "❌ NO") << std::endl;

    // Test 3: Edge Cases
    std::cout << "\n3. TESTING EDGE CASES:" << std::endl;
    std::cout << "=====================" << std::endl;

    // Edge case 1: All copies corrupted
    float all_corrupted1 = injectSingleBitError(original_value, gen);
    float all_corrupted2 = injectSingleBitError(original_value, gen);
    float all_corrupted3 = injectSingleBitError(original_value, gen);

    std::cout << "Edge case: All copies corrupted" << std::endl;
    printBinary(all_corrupted1, "All Corrupted 1");
    printBinary(all_corrupted2, "All Corrupted 2");
    printBinary(all_corrupted3, "All Corrupted 3");

    float edge_result =
        EnhancedVoting::standardVote(all_corrupted1, all_corrupted2, all_corrupted3);
    printBinary(edge_result, "Edge Case Result");

    // Define expected behavior: When all copies are corrupted,
    // the voting algorithm should return one of the corrupted values
    // (not necessarily the original value)
    bool edge_returns_corrupted_value =
        (edge_result == all_corrupted1 || edge_result == all_corrupted2 ||
         edge_result == all_corrupted3);

    std::cout << "Edge case behavior: "
              << (edge_returns_corrupted_value ? "✅ CORRECT" : "❌ UNEXPECTED") << std::endl;
    std::cout << "  - Expected: Return one of the corrupted values" << std::endl;
    std::cout << "  - Actual: Returned " << edge_result << std::endl;
    std::cout << "  - Note: This is correct behavior - when all copies are corrupted," << std::endl;
    std::cout << "    the algorithm cannot recover the original value" << std::endl;

    // Edge case 2: All copies identical (no corruption)
    float edge_result2 =
        EnhancedVoting::standardVote(original_value, original_value, original_value);
    bool edge2_correct = (edge_result2 == original_value);
    std::cout << "No corruption case handled correctly: " << (edge2_correct ? "✅ YES" : "❌ NO")
              << std::endl;

    // Test 4: Check if voting methods are actually different
    std::cout << "\n4. CHECKING IF VOTING METHODS ARE DIFFERENT:" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Create a scenario where methods might differ
    float diff1 = original_value;
    float diff2 = injectSingleBitError(original_value, gen);
    float diff3 = injectMultiBitError(original_value, gen);

    float std_diff = EnhancedVoting::standardVote(diff1, diff2, diff3);
    float bit_diff = EnhancedVoting::bitLevelVote(diff1, diff2, diff3);
    float adapt_diff =
        EnhancedVoting::adaptiveVote(diff1, diff2, diff3, FaultPattern::ADJACENT_BITS);

    bool methods_different =
        (std_diff != bit_diff) || (std_diff != adapt_diff) || (bit_diff != adapt_diff);

    std::cout << "Different voting methods produce different results: "
              << (methods_different ? "✅ YES" : "❌ NO (all identical)") << std::endl;

    if (methods_different) {
        std::cout << "Standard: " << std_diff << std::endl;
        std::cout << "Bit-level: " << bit_diff << std::endl;
        std::cout << "Adaptive: " << adapt_diff << std::endl;
    }

    // Summary
    std::cout << "\n=== VALIDATION SUMMARY ===" << std::endl;
    std::cout << "Error injection working: "
              << (single_changed && multi_changed ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Voting logic working: "
              << (standard_correct && bit_level_correct && adaptive_correct ? "✅ YES" : "❌ NO")
              << std::endl;
    std::cout << "Edge cases handled: " << (edge2_correct ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Voting methods different: " << (methods_different ? "✅ YES" : "❌ NO")
              << std::endl;

    if (!methods_different) {
        std::cout << "\n🚨 RED FLAG: All voting methods produce identical results!" << std::endl;
        std::cout << "This suggests they might all be using the same algorithm." << std::endl;
    }

    if (!single_changed || !multi_changed) {
        std::cout << "\n🚨 RED FLAG: Error injection is not working!" << std::endl;
        std::cout << "This would explain the 100% success rates in Monte Carlo test." << std::endl;
    }

    return 0;
}
