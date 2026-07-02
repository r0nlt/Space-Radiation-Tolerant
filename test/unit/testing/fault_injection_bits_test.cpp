/**
 * @file fault_injection_bits_test.cpp
 * @brief Regression test for SystematicFaultInjector::injectFault bit semantics
 *
 * injectFault previously reinterpret_cast the value as a std::bitset, which is
 * undefined behavior and relies on an unspecified internal layout. It now
 * operates on a memcpy'd unsigned integer; these tests pin down the intended
 * semantics: bit i of the injected fault corresponds to bit i of the value's
 * object representation.
 */

#include <rad_ml/testing/fault_injection.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

using rad_ml::testing::SystematicFaultInjector;

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

template <typename T>
uint64_t bitsOf(T value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

void test_single_bit_flip_int()
{
    SystematicFaultInjector injector;

    const int original = 0x12345678;
    const int corrupted = injector.injectFault(original, SystematicFaultInjector::SINGLE_BIT, 4);

    check(corrupted == (original ^ (1 << 4)), "SINGLE_BIT flips exactly the requested bit");

    const int restored = injector.injectFault(corrupted, SystematicFaultInjector::SINGLE_BIT, 4);
    check(restored == original, "flipping the same bit twice restores the value");
}

void test_single_bit_flip_float()
{
    SystematicFaultInjector injector;

    const float original = 42.125f;
    const float corrupted =
        injector.injectFault(original, SystematicFaultInjector::SINGLE_BIT, 21);

    check((bitsOf(original) ^ bitsOf(corrupted)) == (1ull << 21),
          "float SINGLE_BIT changes exactly bit 21 of the representation");

    const float restored =
        injector.injectFault(corrupted, SystematicFaultInjector::SINGLE_BIT, 21);
    check(bitsOf(restored) == bitsOf(original), "float double flip restores the representation");
}

void test_stuck_at_semantics()
{
    SystematicFaultInjector injector;

    const uint8_t all_ones = 0xFF;
    const uint8_t stuck_zero =
        injector.injectFault(all_ones, SystematicFaultInjector::STUCK_AT_ZERO, 3);
    check((stuck_zero & (1 << 3)) == 0, "STUCK_AT_ZERO clears the requested bit");

    const uint8_t all_zeros = 0x00;
    const uint8_t stuck_one =
        injector.injectFault(all_zeros, SystematicFaultInjector::STUCK_AT_ONE, 6);
    check((stuck_one & (1 << 6)) != 0, "STUCK_AT_ONE sets the requested bit");

    // Stuck-at faults are idempotent
    check(injector.injectFault(stuck_one, SystematicFaultInjector::STUCK_AT_ONE, 6) == stuck_one,
          "STUCK_AT_ONE is idempotent");
}

void test_double_precision_supported()
{
    SystematicFaultInjector injector;

    const double original = -1.5e300;
    const double corrupted =
        injector.injectFault(original, SystematicFaultInjector::SINGLE_BIT, 52);

    check((bitsOf(original) ^ bitsOf(corrupted)) == (1ull << 52),
          "double SINGLE_BIT changes exactly bit 52 of the representation");
}

}  // namespace

int main()
{
    test_single_bit_flip_int();
    test_single_bit_flip_float();
    test_stuck_at_semantics();
    test_double_precision_supported();

    if (failures == 0) {
        std::cout << "All fault_injection_bits_test checks passed.\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
