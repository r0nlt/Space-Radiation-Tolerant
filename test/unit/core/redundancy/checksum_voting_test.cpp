/**
 * @file checksum_voting_test.cpp
 * @brief Regression tests for checksum-assisted adaptive voting
 *
 * Covers the CRC-assisted selection paths added to:
 *  - EnhancedVoting::adaptiveVote(a, b, c, pattern, expected_checksum):
 *    trusts a copy whose CRC matches the write-time checksum, validates
 *    reconstruction candidates when all copies are corrupted, and falls back
 *    to plain pattern voting when nothing validates.
 *  - EnhancedTMR::performWeightedVoting(): when copies disagree and the
 *    stored per-copy CRCs discriminate between intact and corrupted copies,
 *    the intact copy wins even against two agreeing (correlated-corrupted)
 *    copies.
 */

#include <rad_ml/core/redundancy/enhanced_voting.hpp>
#include <rad_ml/tmr/enhanced_tmr.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

using rad_ml::core::redundancy::EnhancedVoting;
using rad_ml::core::redundancy::FaultPattern;

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
T flipBit(T value, int bit)
{
    using UintType = typename std::conditional<sizeof(T) <= 4, uint32_t, uint64_t>::type;
    UintType bits;
    std::memcpy(&bits, &value, sizeof(T));
    bits ^= (UintType(1) << bit);
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

template <typename T>
T adaptiveWithChecksum(const T& original, const T& a, const T& b, const T& c)
{
    const FaultPattern pattern = EnhancedVoting::detectFaultPattern(a, b, c);
    return EnhancedVoting::adaptiveVote(a, b, c, pattern, EnhancedVoting::crc32(original));
}

void test_all_agree_fast_path()
{
    const float value = 3.14159f;
    check(adaptiveWithChecksum(value, value, value, value) == value,
          "all-agree fast path returns the value");
}

void test_intact_copy_beats_correlated_majority()
{
    // Two copies corrupted with the same 4-bit pattern at adjacent offsets
    // (mirrors the CORRELATED_ERRORS Monte Carlo scenario). Bit-level
    // majority voting loses the overlapping bits, but the intact copy's CRC
    // validates.
    const uint32_t original = 0x12345678u;
    uint32_t a = original ^ (0xFu << 8);
    uint32_t b = original ^ (0xFu << 9);
    uint32_t c = original;

    check(adaptiveWithChecksum(original, a, b, c) == original,
          "intact copy is selected under correlated corruption");

    // Plain (checksum-free) adaptive voting gets this wrong: the three
    // overlapping corrupted bits outvote the intact copy. This documents the
    // gap the checksum closes; if plain voting ever starts passing, the
    // scenario is no longer exercising the fallback.
    const FaultPattern pattern = EnhancedVoting::detectFaultPattern(a, b, c);
    check(EnhancedVoting::adaptiveVote(a, b, c, pattern) != original,
          "plain adaptive voting loses this case (documents the gap)");
}

void test_identical_corruption_of_two_copies()
{
    // Both corrupted copies are bit-identical, so they form a (wrong)
    // majority. Only the checksum identifies the intact copy.
    const double original = -2.718281828;
    const double corrupted = flipBit(original, 17);

    check(adaptiveWithChecksum(original, corrupted, corrupted, original) == original,
          "CRC-validating copy beats two identically corrupted copies");
}

void test_reconstruction_candidate_validation()
{
    // All three copies corrupted at distinct bit positions: no copy
    // validates, but bit-level majority reconstructs the original and the
    // checksum confirms it.
    const uint32_t original = 0xCAFEBABEu;
    const uint32_t a = flipBit(original, 3);
    const uint32_t b = flipBit(original, 14);
    const uint32_t c = flipBit(original, 27);

    check(adaptiveWithChecksum(original, a, b, c) == original,
          "validated bit-level reconstruction recovers from all-copy corruption");
}

void test_fallback_when_nothing_validates()
{
    // With a checksum that matches nothing, the result must equal plain
    // pattern-based adaptive voting.
    const uint32_t a = flipBit(0xDEADBEEFu, 1);
    const uint32_t b = flipBit(0xDEADBEEFu, 2);
    const uint32_t c = flipBit(0xDEADBEEFu, 3);
    const FaultPattern pattern = EnhancedVoting::detectFaultPattern(a, b, c);

    const uint32_t with_bogus_crc = EnhancedVoting::adaptiveVote(a, b, c, pattern, 0u);
    const uint32_t plain = EnhancedVoting::adaptiveVote(a, b, c, pattern);
    check(with_bogus_crc == plain, "non-validating checksum falls back to plain adaptive voting");
}

void test_enhanced_tmr_crc_assisted_voting()
{
    // Radiation-style corruption (corruptCopy does not re-arm the CRC) of
    // two copies with identical values: they agree, but their CRCs fail
    // while copy 2 validates, so the intact copy must win.
    const float original = 42.5f;
    rad_ml::tmr::EnhancedTMR<float> tmr(original);

    const float corrupted = flipBit(original, 30);
    tmr.corruptCopy(0, corrupted);
    tmr.corruptCopy(1, corrupted);

    check(tmr.get() == original,
          "EnhancedTMR returns the CRC-validating copy over an agreeing corrupted pair");

    // Divergent corruption of two copies: all three disagree, only copy 1
    // validates.
    rad_ml::tmr::EnhancedTMR<float> tmr2(original);
    tmr2.corruptCopy(0, flipBit(original, 5));
    tmr2.corruptCopy(2, flipBit(original, 22));

    check(tmr2.get() == original,
          "EnhancedTMR recovers from divergent double-copy corruption via CRC");

    // setRawCopy re-arms the CRC, so the checksum carries no information and
    // the existing majority logic must still handle single-copy corruption.
    rad_ml::tmr::EnhancedTMR<float> tmr3(original);
    tmr3.setRawCopy(0, flipBit(original, 9));
    check(tmr3.get() == original, "setRawCopy corruption is still recovered by majority voting");
}

}  // namespace

int main()
{
    test_all_agree_fast_path();
    test_intact_copy_beats_correlated_majority();
    test_identical_corruption_of_two_copies();
    test_reconstruction_candidate_validation();
    test_fallback_when_nothing_validates();
    test_enhanced_tmr_crc_assisted_voting();

    if (failures == 0) {
        std::cout << "All checksum voting tests passed.\n";
        return 0;
    }
    std::cerr << failures << " checksum voting test(s) failed.\n";
    return 1;
}
