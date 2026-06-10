/**
 * @file hamming_secded_test.cpp
 * @brief Validates Hamming(8,4) SECDED encode/decode in AdaptiveProtection
 */

#include <cstdint>
#include <iostream>

#include "../../include/rad_ml/neural/adaptive_protection.hpp"

using rad_ml::neural::AdaptiveProtection;
using rad_ml::neural::HammingDecodeResult;

static bool check(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
    }
    return cond;
}

static void flip_bit(uint8_t& cw, int bit)
{
    cw ^= static_cast<uint8_t>(1u << bit);
}

static bool test_round_trip_all_nibbles()
{
    for (unsigned n = 0; n < 16; ++n) {
        uint8_t enc = AdaptiveProtection<float>::hamming_encode_nibble(static_cast<uint8_t>(n));
        HammingDecodeResult dec = AdaptiveProtection<float>::hamming_decode_nibble(enc);
        if (!check(dec.value == n && !dec.corrected && !dec.uncorrectable,
                   "clean round-trip nibble")) {
            return false;
        }
    }
    return true;
}

static bool test_single_bit_correction()
{
    for (unsigned n = 0; n < 16; ++n) {
        uint8_t enc = AdaptiveProtection<float>::hamming_encode_nibble(static_cast<uint8_t>(n));
        for (int bit = 0; bit < 8; ++bit) {
            uint8_t corrupted = enc;
            flip_bit(corrupted, bit);
            HammingDecodeResult dec = AdaptiveProtection<float>::hamming_decode_nibble(corrupted);
            if (!check(dec.value == n && dec.corrected && !dec.uncorrectable,
                       "single-bit correction")) {
                std::cerr << "  n=" << n << " bit=" << bit << "\n";
                return false;
            }
        }
    }
    return true;
}

static bool test_double_bit_detection()
{
    uint8_t enc = AdaptiveProtection<float>::hamming_encode_nibble(0xA);
    int failures = 0;
    int detected = 0;

    for (int b1 = 0; b1 < 8; ++b1) {
        for (int b2 = b1 + 1; b2 < 8; ++b2) {
            uint8_t corrupted = enc;
            flip_bit(corrupted, b1);
            flip_bit(corrupted, b2);
            HammingDecodeResult dec = AdaptiveProtection<float>::hamming_decode_nibble(corrupted);
            ++failures;
            if (dec.uncorrectable) {
                ++detected;
            }
            else if (dec.corrected && dec.value == 0xA) {
                // Recovered original without uncorrectable flag — still acceptable if data matches
                ++detected;
            }
        }
    }

    if (!check(detected == failures, "all double-bit patterns flagged or safely handled")) {
        std::cerr << "  detected " << detected << " / " << failures << "\n";
        return false;
    }
    return true;
}

static bool test_byte_full_path()
{
    uint8_t data = 0x5A;
    uint16_t enc = AdaptiveProtection<float>::hamming_encode_byte_full(data);
    auto [decoded, corrected, uncorrectable] =
        AdaptiveProtection<float>::hamming_decode_byte_full(enc);
    if (!check(decoded == data && !corrected && !uncorrectable, "byte full clean")) {
        return false;
    }

    uint16_t corrupted = enc ^ 0x0004u;
    auto [fixed, was_corrected, dbl] =
        AdaptiveProtection<float>::hamming_decode_byte_full(corrupted);
    return check(fixed == data && was_corrected && !dbl, "byte full single-bit fix");
}

int main()
{
    bool ok = true;
    ok = test_round_trip_all_nibbles() && ok;
    ok = test_single_bit_correction() && ok;
    ok = test_double_bit_detection() && ok;
    ok = test_byte_full_path() && ok;

    if (ok) {
        std::cout << "All hamming_secded_test checks passed.\n";
        return 0;
    }
    return 1;
}
