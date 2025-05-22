/**
 * @file ieee754_tmr_test.cpp
 * @brief Test the IEEE-754 aware bit-level voting implementation
 */

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <rad_ml/core/redundancy/space_enhanced_tmr.hpp>
#include <vector>

using namespace rad_ml::core::redundancy;

// Define special values for testing
constexpr float POS_INF = std::numeric_limits<float>::infinity();
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
constexpr float NAN_VAL = std::numeric_limits<float>::quiet_NaN();
constexpr float DENORM_MIN = std::numeric_limits<float>::denorm_min();
constexpr double DBL_POS_INF = std::numeric_limits<double>::infinity();

// Utility to inject bit errors
template <typename T>
T injectBitError(T value, int bit_position)
{
    using BitsType =
        typename std::conditional<sizeof(T) == sizeof(float), uint32_t, uint64_t>::type;

    BitsType bits;
    std::memcpy(&bits, &value, sizeof(T));

    // Flip the bit at the specified position
    bits ^= (BitsType(1) << bit_position);

    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

// Utility function to print bit representation
template <typename T>
void printBits(const T& value)
{
    using BitsType =
        typename std::conditional<sizeof(T) == sizeof(float), uint32_t, uint64_t>::type;

    BitsType bits;
    std::memcpy(&bits, &value, sizeof(T));

    std::cout << "Bits: ";
    BitsType mask = BitsType(1) << ((sizeof(T) * 8) - 1);

    // Print sign bit
    std::cout << ((bits & mask) ? '1' : '0') << " | ";
    mask >>= 1;

    // Print exponent bits
    int exp_bits = (sizeof(T) == sizeof(float)) ? 8 : 11;
    for (int i = 0; i < exp_bits; i++) {
        std::cout << ((bits & mask) ? '1' : '0');
        mask >>= 1;
    }

    std::cout << " | ";

    // Print mantissa bits
    int mantissa_bits = (sizeof(T) == sizeof(float)) ? 23 : 52;
    for (int i = 0; i < mantissa_bits; i++) {
        std::cout << ((bits & mask) ? '1' : '0');
        mask >>= 1;
    }

    std::cout << std::endl;
}

// Test helper to validate float TMR voting
bool testFloatTMR(float a, float b, float c, const std::string& test_name)
{
    SpaceEnhancedTMR<float> tmr;

    // Manually set the three values
#ifdef ENABLE_TESTING
    tmr.setForTesting(0, a);
    tmr.setForTesting(1, b);
    tmr.setForTesting(2, c);
    tmr.recalculateChecksumsForTesting();
#else
    // If testing is not enabled, we need a workaround
    // For example, create a new TMR with one value and then inject bit errors
    tmr = SpaceEnhancedTMR<float>(a);
    // This approach won't work well for testing without the testing methods
    std::cout << "WARNING: ENABLE_TESTING not defined, test may not be accurate\n";
#endif

    float result;
    auto status = tmr.get(result);

    std::cout << "Test: " << test_name << "\n";
    std::cout << "Input values: " << a << ", " << b << ", " << c << "\n";
    std::cout << "Result: " << result << "\n";

    // Check if result is valid (not NaN unless all inputs were NaN)
    bool all_nan = std::isnan(a) && std::isnan(b) && std::isnan(c);
    bool result_valid = !std::isnan(result) || all_nan;

    // Check if result matches one of the inputs when they agree
    bool inputs_agree = (a == b) || (a == c) || (b == c);
    bool result_matches = (result == a) || (result == b) || (result == c);

    bool test_passed = result_valid && (!inputs_agree || result_matches);

    std::cout << "Status: " << (test_passed ? "PASS" : "FAIL") << "\n\n";
    return test_passed;
}

// Test helper to validate double TMR voting with bit flips
bool testDoubleTMRWithBitFlips(double base_value, int bit_position1, int bit_position2,
                               const std::string& test_name)
{
    double a = base_value;
    double b = injectBitError(base_value, bit_position1);
    double c = injectBitError(base_value, bit_position2);

    SpaceEnhancedTMR<double> tmr;

#ifdef ENABLE_TESTING
    tmr.setForTesting(0, a);
    tmr.setForTesting(1, b);
    tmr.setForTesting(2, c);
    tmr.recalculateChecksumsForTesting();
#else
    tmr = SpaceEnhancedTMR<double>(a);
    std::cout << "WARNING: ENABLE_TESTING not defined, test may not be accurate\n";
#endif

    double result;
    auto status = tmr.get(result);

    std::cout << "Test: " << test_name << "\n";
    std::cout << "Base value: " << a << "\n";
    std::cout << "Bit positions flipped: " << bit_position1 << ", " << bit_position2 << "\n";
    std::cout << "Input values: " << a << ", " << b << ", " << c << "\n";
    std::cout << "Result: " << result << "\n";

    // For sign bit tests, we must handle differently since majority voting should take precedence
    if (bit_position1 == 63 || bit_position2 == 63) {
        // Count how many negative values we have
        int negative_count = 0;
        if (a < 0) negative_count++;
        if (b < 0) negative_count++;
        if (c < 0) negative_count++;

        // If majority are negative, result should be negative
        bool should_be_negative = negative_count >= 2;
        bool result_is_negative = result < 0;

        // Test is successful if sign matches the majority
        bool test_passed = (should_be_negative == result_is_negative);
        std::cout << "Status: " << (test_passed ? "PASS" : "FAIL") << "\n\n";
        return test_passed;
    }

    // For other tests, use the original criteria
    bool test_passed = std::abs(result - base_value) < std::abs(base_value) * 0.001;

    std::cout << "Status: " << (test_passed ? "PASS" : "FAIL") << "\n\n";
    return test_passed;
}

int main()
{
    std::cout << "===== IEEE-754 TMR Voting Test =====\n\n";

    int passed_tests = 0;
    int total_tests = 0;

    // Test 1: Basic majority voting with 3 valid values
    if (testFloatTMR(1.0f, 1.0f, 2.0f, "Basic Majority Voting")) passed_tests++;
    total_tests++;

    // Test 2: All values different but finite
    if (testFloatTMR(1.0f, 2.0f, 3.0f, "All Values Different")) passed_tests++;
    total_tests++;

    // Test 3: Special values - one NaN
    if (testFloatTMR(1.0f, NAN_VAL, 3.0f, "One NaN Value")) passed_tests++;
    total_tests++;

    // Test 4: Special values - one Infinity
    if (testFloatTMR(1.0f, POS_INF, 3.0f, "One Infinity Value")) passed_tests++;
    total_tests++;

    // Test 5: Special values - all NaN
    if (testFloatTMR(NAN_VAL, NAN_VAL, NAN_VAL, "All NaN Values")) passed_tests++;
    total_tests++;

    // Test 6: Mix of special values
    if (testFloatTMR(POS_INF, NAN_VAL, NEG_INF, "Mix of Special Values")) passed_tests++;
    total_tests++;

    // Test 7: Denormal values
    if (testFloatTMR(DENORM_MIN, DENORM_MIN, 0.0f, "Denormal Values")) passed_tests++;
    total_tests++;

    // Test 8: Double with bit flips in mantissa
    if (testDoubleTMRWithBitFlips(3.14159265359, 0, 1, "Double with Mantissa Bit Flips"))
        passed_tests++;
    total_tests++;

    // Test 9: Double with bit flips in exponent
    if (testDoubleTMRWithBitFlips(3.14159265359, 52, 53, "Double with Exponent Bit Flips"))
        passed_tests++;
    total_tests++;

    // Test 10: Double with bit flips in sign
    if (testDoubleTMRWithBitFlips(3.14159265359, 63, 63, "Double with Sign Bit Flips"))
        passed_tests++;
    total_tests++;

    // Test 11: Float with multiple bit errors
    float multi_error_base = 2.71828f;
    float a = multi_error_base;
    float b = injectBitError(injectBitError(multi_error_base, 5), 10);
    float c = injectBitError(injectBitError(multi_error_base, 15), 20);
    if (testFloatTMR(a, b, c, "Float with Multiple Bit Errors")) passed_tests++;
    total_tests++;

    // Test 12: Near-zero values
    if (testFloatTMR(1e-30f, 1e-30f, 1e-31f, "Near-zero Values")) passed_tests++;
    total_tests++;

    // Test 13: Large values near max float
    float large_val = std::numeric_limits<float>::max() / 2;
    if (testFloatTMR(large_val, large_val, large_val * 0.9f, "Large Values")) passed_tests++;
    total_tests++;

    // Summary
    std::cout << "===== Test Summary =====\n";
    std::cout << "Passed: " << passed_tests << " / " << total_tests << " tests\n";
    std::cout << "Success Rate: " << (passed_tests * 100 / total_tests) << "%\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
