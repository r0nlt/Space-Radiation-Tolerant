/**
 * @file multibit_rs_test.cpp
 * @brief Comprehensive tests for RS-backed MultibitProtection
 *
 * Tests the integration of AdvancedReedSolomon with MultibitProtection,
 * verifying Tour of C++ principles:
 * - RAII initialization
 * - Type safety
 * - Zero-cost abstractions
 * - Value semantics
 *
 * @author Space Labs AI
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/neural/multi_bit_protection.hpp"

using namespace rad_ml::neural;

// Test utilities
namespace {

constexpr const char* GREEN = "\033[32m";
constexpr const char* RED = "\033[31m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* RESET = "\033[0m";
constexpr const char* BOLD = "\033[1m";

int tests_passed = 0;
int tests_failed = 0;

void print_test_header(const char* name)
{
    std::cout << "\n" << BOLD << "━━━ " << name << " ━━━" << RESET << "\n";
}

void check(bool condition, const char* test_name)
{
    if (condition) {
        std::cout << GREEN << "  ✓ " << RESET << test_name << "\n";
        tests_passed++;
    }
    else {
        std::cout << RED << "  ✗ " << RESET << test_name << "\n";
        tests_failed++;
    }
}

template <typename T>
bool approx_equal(T a, T b, T epsilon = static_cast<T>(1e-6))
{
    return std::abs(a - b) < epsilon;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: RSProtectionBackend - Basic RAII and Value Semantics
// ═══════════════════════════════════════════════════════════════════════════
void test_rs_backend_basic()
{
    print_test_header("RSProtectionBackend Basic Operations");

    // Test default construction (RAII - should be valid immediately)
    RSProtectionBackend<float> backend;
    check(true, "Default construction succeeds");

    // Test value construction
    RSProtectionBackend<float, RSCorrectionTier::STANDARD> backend_with_value(3.14159f);
    check(!backend_with_value.codeword().empty(), "Codeword generated on construction");

    // Test encode and decode round-trip
    backend.encode(2.71828f);
    auto decoded = backend.decode();
    check(decoded.has_value(), "Decode returns value for clean codeword");
    check(decoded && approx_equal(*decoded, 2.71828f), "Round-trip preserves value");

    // Test get() method
    float value = backend.get();
    check(approx_equal(value, 2.71828f), "get() returns correct value");

    // Test cached_value() fast path
    float cached = backend.cached_value();
    check(approx_equal(cached, 2.71828f), "cached_value() returns correct value");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: RSProtectionBackend - Error Injection and Correction
// ═══════════════════════════════════════════════════════════════════════════
void test_rs_backend_error_correction()
{
    print_test_header("RSProtectionBackend Error Correction");

    // Create backend with known value
    RSProtectionBackend<float, RSCorrectionTier::STANDARD> backend(42.0f);

    // Verify clean state
    check(!backend.has_error(), "No error in clean state");

    // Inject single error
    backend.inject_error(0, 0);  // Flip bit 0 of byte 0
    check(backend.has_error(), "Error detected after injection");

    // Attempt correction
    auto corrected = backend.decode();
    check(corrected.has_value(), "Single error corrected");
    check(corrected && approx_equal(*corrected, 42.0f), "Corrected value is accurate");

    // Test correction capability (t=3 for STANDARD tier)
    RSProtectionBackend<float, RSCorrectionTier::STANDARD> backend2(123.456f);

    // Inject 3 errors (should be correctable)
    backend2.inject_error(0, 0);
    backend2.inject_error(1, 3);
    backend2.inject_error(2, 7);

    auto corrected2 = backend2.decode();
    check(corrected2.has_value(), "3 errors corrected (at limit for t=3)");

    // Test beyond correction capability
    RSProtectionBackend<float, RSCorrectionTier::LIGHT> backend3(99.9f);  // t=2
    backend3.inject_error(0, 0);
    backend3.inject_error(1, 1);
    backend3.inject_error(2, 2);  // 3 errors, but t=2 can only correct 2

    auto uncorrectable = backend3.decode();
    // This may or may not succeed depending on error pattern
    std::cout << YELLOW << "  ℹ " << RESET << "3 errors on t=2 tier: "
              << (uncorrectable ? "corrected (lucky pattern)" : "uncorrectable as expected")
              << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: RSProtectedValue - Compile-Time Tier Selection
// ═══════════════════════════════════════════════════════════════════════════
void test_rs_protected_value()
{
    print_test_header("RSProtectedValue Compile-Time Tiers");

    // Test different tiers (compile-time selection)
    RSLight<float> light_protected(1.0f);
    RSStandard<float> standard_protected(2.0f);
    RSHeavy<float> heavy_protected(3.0f);

    // Verify correction capabilities are correct
    check(RSLight<float>::correction_capability == 2, "LIGHT tier: t=2");
    check(RSStandard<float>::correction_capability == 3, "STANDARD tier: t=3");
    check(RSHeavy<float>::correction_capability == 4, "HEAVY tier: t=4");

    // Test value semantics (implicit conversion)
    float v1 = light_protected;
    float v2 = standard_protected;
    float v3 = heavy_protected;

    check(approx_equal(v1, 1.0f), "LIGHT: implicit conversion works");
    check(approx_equal(v2, 2.0f), "STANDARD: implicit conversion works");
    check(approx_equal(v3, 3.0f), "HEAVY: implicit conversion works");

    // Test assignment operator
    light_protected = 4.0f;
    check(approx_equal(light_protected.get(), 4.0f), "Assignment operator works");

    // Test set() method
    standard_protected.set(5.0f);
    check(approx_equal(standard_protected.get(), 5.0f), "set() method works");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: MultibitProtection with REED_SOLOMON scheme
// ═══════════════════════════════════════════════════════════════════════════
void test_multibit_protection_rs()
{
    print_test_header("MultibitProtection with RS Scheme");

    // Create protection with RS scheme
    MultibitProtection<float> protected_value(3.14159f, ECCCodingScheme::REED_SOLOMON);

    // Test getValue()
    float v = protected_value.getValue();
    check(approx_equal(v, 3.14159f), "getValue() returns correct value");

    // Test hasError() on clean value
    check(!protected_value.hasError(), "No error on clean value");

    // Test isValid()
    check(protected_value.isValid(), "isValid() returns true for clean value");

    // Test setValue()
    protected_value.setValue(2.71828f);
    check(approx_equal(protected_value.getValue(), 2.71828f), "setValue() works");

    // Test operator=
    protected_value = 1.41421f;
    check(approx_equal(protected_value.getValue(), 1.41421f), "operator= works");

    // Test implicit conversion
    float converted = protected_value;
    check(approx_equal(converted, 1.41421f), "Implicit conversion works");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: HybridRSTMRProtection
// ═══════════════════════════════════════════════════════════════════════════
void test_hybrid_protection()
{
    print_test_header("HybridRSTMRProtection");

    // Create hybrid protection
    HybridProtection<float> hybrid(100.0f);

    // Test get()
    float v = hybrid.get();
    check(approx_equal(v, 100.0f), "get() returns correct value");

    // Test set()
    hybrid.set(200.0f);
    check(approx_equal(hybrid.get(), 200.0f), "set() works");

    // Test has_error() on clean value
    check(!hybrid.has_error(), "No error on clean value");

    // Test implicit conversion
    float converted = hybrid;
    check(approx_equal(converted, 200.0f), "Implicit conversion works");

    // Test rs_correction_capability()
    check(HybridProtection<float>::rs_correction_capability() == 3,
          "RS correction capability is t=3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Type Trait Validation
// ═══════════════════════════════════════════════════════════════════════════
void test_type_traits()
{
    print_test_header("Type Traits (Compile-Time Safety)");

    // Test is_rs_protectable
    check(is_rs_protectable_v<float>, "float is RS-protectable");
    check(is_rs_protectable_v<double>, "double is RS-protectable");
    check(is_rs_protectable_v<int>, "int is RS-protectable");
    check(is_rs_protectable_v<uint64_t>, "uint64_t is RS-protectable");
    check(is_rs_protectable_v<char>, "char is RS-protectable");

    // Verify non-protectable types would fail (can't actually test at runtime)
    std::cout << YELLOW << "  ℹ " << RESET
              << "std::string would fail static_assert (non-trivially copyable)\n";
    std::cout << YELLOW << "  ℹ " << RESET
              << "Large structs (>247 bytes) would fail static_assert\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Stress Test - Multiple Values
// ═══════════════════════════════════════════════════════════════════════════
void test_stress()
{
    print_test_header("Stress Test - Multiple Protected Values");

    constexpr size_t NUM_VALUES = 1000;
    std::vector<RSStandard<float>> protected_values;
    protected_values.reserve(NUM_VALUES);

    // Create many protected values
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<float> original_values;
    for (size_t i = 0; i < NUM_VALUES; ++i) {
        float v = dist(rng);
        original_values.push_back(v);
        protected_values.emplace_back(v);
    }

    // Verify all values
    bool all_correct = true;
    for (size_t i = 0; i < NUM_VALUES; ++i) {
        if (!approx_equal(protected_values[i].get(), original_values[i], 1e-5f)) {
            all_correct = false;
            break;
        }
    }
    check(all_correct, "All 1000 values stored and retrieved correctly");

    // Inject errors in some values and correct
    size_t corrected_count = 0;
    for (size_t i = 0; i < 100; ++i) {
        protected_values[i].inject_error(0, i % 8);
        auto result = protected_values[i].try_correct();
        if (result && approx_equal(*result, original_values[i], 1e-5f)) {
            corrected_count++;
        }
    }

    std::cout << YELLOW << "  ℹ " << RESET << "Corrected " << corrected_count
              << "/100 injected errors\n";
    check(corrected_count >= 90, "At least 90% of single errors corrected");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: Edge Cases
// ═══════════════════════════════════════════════════════════════════════════
void test_edge_cases()
{
    print_test_header("Edge Cases");

    // Test with zero
    RSStandard<float> zero_value(0.0f);
    check(approx_equal(zero_value.get(), 0.0f), "Zero value preserved");

    // Test with negative zero
    RSStandard<float> neg_zero(-0.0f);
    float nz = neg_zero.get();
    check(std::signbit(nz) || nz == 0.0f, "Negative zero handled");

    // Test with infinity
    RSStandard<float> inf_value(std::numeric_limits<float>::infinity());
    check(std::isinf(inf_value.get()), "Infinity preserved");

    // Test with NaN (tricky - NaN != NaN)
    RSStandard<float> nan_value(std::numeric_limits<float>::quiet_NaN());
    check(std::isnan(nan_value.get()), "NaN preserved");

    // Test with very small values
    RSStandard<float> tiny(std::numeric_limits<float>::min());
    check(tiny.get() > 0.0f, "Smallest positive float preserved");

    // Test with very large values
    RSStandard<float> huge(std::numeric_limits<float>::max());
    check(huge.get() == std::numeric_limits<float>::max(), "Largest float preserved");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 9: Different Data Types
// ═══════════════════════════════════════════════════════════════════════════
void test_different_types()
{
    print_test_header("Different Data Types");

    // Test with double
    RSStandard<double> double_value(3.141592653589793);
    check(std::abs(double_value.get() - 3.141592653589793) < 1e-15, "double precision preserved");

    // Test with int
    RSStandard<int> int_value(-42);
    check(int_value.get() == -42, "int value preserved");

    // Test with uint64_t
    RSStandard<uint64_t> uint64_value(0xDEADBEEFCAFEBABE);
    check(uint64_value.get() == 0xDEADBEEFCAFEBABE, "uint64_t value preserved");

    // Test with small struct
    struct SmallStruct {
        float x, y, z;
        int flags;
    };
    static_assert(is_rs_protectable_v<SmallStruct>, "SmallStruct should be protectable");

    SmallStruct s{1.0f, 2.0f, 3.0f, 42};
    RSStandard<SmallStruct> struct_value(s);
    SmallStruct retrieved = struct_value.get();

    check(approx_equal(retrieved.x, 1.0f) && approx_equal(retrieved.y, 2.0f) &&
              approx_equal(retrieved.z, 3.0f) && retrieved.flags == 42,
          "Struct value preserved");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════
int main()
{
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════════╗\n"
              << "║   MultibitProtection RS Integration Test Suite            ║\n"
              << "║   Tour of C++ Philosophy Validation                       ║\n"
              << "╚═══════════════════════════════════════════════════════════╝\n"
              << RESET;

    test_rs_backend_basic();
    test_rs_backend_error_correction();
    test_rs_protected_value();
    test_multibit_protection_rs();
    test_hybrid_protection();
    test_type_traits();
    test_stress();
    test_edge_cases();
    test_different_types();

    std::cout << "\n"
              << BOLD << "═══════════════════════════════════════════════════════════\n"
              << "  RESULTS: " << GREEN << tests_passed << " passed" << RESET << ", "
              << (tests_failed > 0 ? RED : GREEN) << tests_failed << " failed" << RESET << "\n"
              << "═══════════════════════════════════════════════════════════\n"
              << RESET;

    return tests_failed > 0 ? 1 : 0;
}
