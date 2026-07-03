/**
 * @file unified_memory_test.cpp
 * @brief Regression tests for UnifiedMemoryManager protection lifecycle
 *
 * Covers the allocate -> corrupt -> verify -> deallocate path for every
 * concrete protection level (CANARY, CRC, ECC, TMR). These paths were
 * previously broken:
 *  - CANARY allocations were tracked by the raw pointer but users received an
 *    offset pointer, so deallocate() failed lookup and leaked
 *  - is_protected/protection_level were never recorded, so integrity checks
 *    never ran
 *  - verifyMemoryIntegrity() used the global default protection level instead
 *    of the allocation's own level, and deadlocked when called from
 *    deallocate()
 *  - CRC checksums were computed but never stored or compared
 */

#include <rad_ml/core/memory/unified_memory.hpp>
#include <rad_ml/utils/bit_manipulation.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

using rad_ml::memory::MemoryFlags;
using rad_ml::memory::MemoryProtectionLevel;
using rad_ml::memory::UnifiedMemoryManager;

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

UnifiedMemoryManager& manager() { return UnifiedMemoryManager::getInstance(); }

void test_unprotected_roundtrip()
{
    void* ptr = manager().allocate(64);
    check(ptr != nullptr, "unprotected allocate returns memory");
    check(manager().isAllocated(ptr), "unprotected allocation is tracked");
    check(manager().verifyMemoryIntegrity(ptr), "unprotected verify is a no-op success");
    check(manager().deallocate(ptr), "unprotected deallocate succeeds");
    check(!manager().isAllocated(ptr), "deallocated pointer is no longer tracked");
    check(!manager().deallocate(ptr), "double free is rejected");
}

void test_canary_roundtrip_and_detection()
{
    constexpr size_t size = 32;
    auto* ptr = static_cast<uint8_t*>(
        manager().allocate(size, MemoryFlags::DEFAULT, MemoryProtectionLevel::CANARY));
    check(ptr != nullptr, "CANARY allocate returns memory");
    check(manager().isAllocated(ptr), "CANARY allocation is tracked by the user pointer");

    const auto* info = manager().getAllocationInfo(ptr);
    check(info != nullptr, "CANARY allocation info is available");
    if (info) {
        check(info->is_protected.load(), "CANARY allocation is marked protected");
        check(info->protection_level == MemoryProtectionLevel::CANARY,
              "CANARY allocation records its protection level");
        check(info->original_ptr != static_cast<void*>(ptr),
              "CANARY user pointer is offset from the raw allocation");
    }

    // Writing the entire user region must not disturb the canaries
    std::memset(ptr, 0xAB, size);
    check(manager().verifyMemoryIntegrity(ptr), "CANARY verify passes after user writes");

    // Buffer overflow: clobber the trailing canary
    ptr[size] ^= 0xFF;
    check(!manager().verifyMemoryIntegrity(ptr), "CANARY verify detects overflow");
    ptr[size] ^= 0xFF;
    check(manager().verifyMemoryIntegrity(ptr), "CANARY verify passes after restore");

    // Buffer underflow: clobber the leading canary
    ptr[-1] ^= 0xFF;
    check(!manager().verifyMemoryIntegrity(ptr), "CANARY verify detects underflow");

    // This was the original bug: deallocate must find the allocation via the
    // user pointer (and must free the raw pointer, not the offset one)
    check(manager().deallocate(ptr), "CANARY deallocate succeeds despite corruption");
    check(!manager().isAllocated(ptr), "CANARY allocation is untracked after free");
}

void test_canary_aligned_allocation()
{
    constexpr size_t alignment = 64;
    void* ptr = manager().allocate(100, MemoryFlags::ALIGNED, MemoryProtectionLevel::CANARY,
                                   "aligned canary test", alignment);
    check(ptr != nullptr, "aligned CANARY allocate returns memory");
    check(reinterpret_cast<uintptr_t>(ptr) % alignment == 0,
          "aligned CANARY user pointer honors the requested alignment");
    check(manager().verifyMemoryIntegrity(ptr), "aligned CANARY verify passes");
    check(manager().deallocate(ptr), "aligned CANARY deallocate succeeds");
}

void test_invalid_alignment_is_rejected()
{
    // Zero or non-power-of-two alignments previously caused a division by
    // zero in the header-size computation (and are invalid for
    // std::aligned_alloc); they must be rejected at the API boundary
    void* ptr = manager().allocate(32, MemoryFlags::ALIGNED | MemoryFlags::NO_THROW,
                                   MemoryProtectionLevel::CANARY, "zero alignment test", 0);
    check(ptr == nullptr, "alignment 0 with NO_THROW returns nullptr");

    ptr = manager().allocate(32, MemoryFlags::ALIGNED | MemoryFlags::NO_THROW,
                             MemoryProtectionLevel::NONE, "non-pow2 alignment test", 24);
    check(ptr == nullptr, "non-power-of-two alignment with NO_THROW returns nullptr");

    bool threw = false;
    try {
        (void)manager().allocate(32, MemoryFlags::ALIGNED, MemoryProtectionLevel::NONE,
                                 "throwing alignment test", 0);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "alignment 0 without NO_THROW throws std::invalid_argument");
}

void test_crc_detects_bit_flip()
{
    constexpr size_t size = 48;
    auto* ptr = static_cast<uint8_t*>(
        manager().allocate(size, MemoryFlags::ZERO_INITIALIZED, MemoryProtectionLevel::CRC));
    check(ptr != nullptr, "CRC allocate returns memory");
    check(manager().verifyMemoryIntegrity(ptr), "CRC verify passes right after allocation");

    // Legitimate write followed by re-arming the checksum
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<uint8_t>(i * 7);
    }
    check(manager().protectMemory(ptr, MemoryProtectionLevel::CRC),
          "CRC protection can be re-armed after writes");
    check(manager().verifyMemoryIntegrity(ptr), "CRC verify passes after re-arm");

    // Simulated SEU: single bit flip must now be detected (previously the
    // stored CRC did not exist and verification always passed)
    ptr[13] ^= 0x04;
    check(!manager().verifyMemoryIntegrity(ptr), "CRC verify detects a single bit flip");
    ptr[13] ^= 0x04;
    check(manager().verifyMemoryIntegrity(ptr), "CRC verify passes after restoring the bit");

    // Changing to a different concrete protection level would overflow the
    // reserved metadata space and must be rejected
    check(!manager().protectMemory(ptr, MemoryProtectionLevel::TMR),
          "changing protection level after allocation is rejected");

    check(manager().deallocate(ptr), "CRC deallocate succeeds");
}

void test_ecc_detects_bit_flip()
{
    constexpr size_t size = 16;
    auto* ptr = static_cast<uint8_t*>(
        manager().allocate(size, MemoryFlags::ZERO_INITIALIZED, MemoryProtectionLevel::ECC));
    check(ptr != nullptr, "ECC allocate returns memory");

    for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<uint8_t>(0x30 + i);
    }
    check(manager().protectMemory(ptr, MemoryProtectionLevel::ECC),
          "ECC protection can be re-armed after writes");
    check(manager().verifyMemoryIntegrity(ptr), "ECC verify passes after re-arm");

    ptr[5] ^= 0x10;
    check(!manager().verifyMemoryIntegrity(ptr), "ECC verify detects a single bit flip");

    check(manager().deallocate(ptr), "ECC deallocate succeeds despite corruption");
}

void test_tmr_detects_divergent_copy()
{
    constexpr size_t size = 24;
    auto* ptr = static_cast<uint8_t*>(
        manager().allocate(size, MemoryFlags::ZERO_INITIALIZED, MemoryProtectionLevel::TMR));
    check(ptr != nullptr, "TMR allocate returns memory");

    for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<uint8_t>(0xC0 - i);
    }
    check(manager().protectMemory(ptr, MemoryProtectionLevel::TMR),
          "TMR protection can be re-armed after writes");
    check(manager().verifyMemoryIntegrity(ptr), "TMR verify passes after re-arm");

    // Corrupt one byte in the second copy
    ptr[size + 3] ^= 0x01;
    check(!manager().verifyMemoryIntegrity(ptr), "TMR verify detects a divergent copy");

    // Deallocation repairs via majority vote and must still succeed
    check(manager().deallocate(ptr), "TMR deallocate succeeds despite corruption");
}

void test_corruption_callback_fires()
{
    int callback_count = 0;
    size_t id = manager().registerCorruptionCallback(
        [&callback_count](void*, size_t, const std::string&) { ++callback_count; });

    auto* ptr = static_cast<uint8_t*>(
        manager().allocate(8, MemoryFlags::ZERO_INITIALIZED, MemoryProtectionLevel::CANARY));
    ptr[8] ^= 0xFF;  // Clobber trailing canary
    manager().verifyMemoryIntegrity(ptr);
    check(callback_count == 1, "corruption callback fires on detection");

    ptr[8] ^= 0xFF;
    check(manager().deallocate(ptr), "callback test allocation deallocates");
    check(manager().unregisterCorruptionCallback(id), "corruption callback unregisters");
}

void test_stats_track_protected_allocations()
{
    const auto before = manager().getStats();

    void* ptr = manager().allocate(16, MemoryFlags::DEFAULT, MemoryProtectionLevel::CANARY);
    auto during = manager().getStats();
    check(during.protected_allocations == before.protected_allocations + 1,
          "protected allocation count increments");
    check(during.current_allocations == before.current_allocations + 1,
          "current allocation count increments");

    manager().deallocate(ptr);
    auto after = manager().getStats();
    check(after.protected_allocations == before.protected_allocations,
          "protected allocation count returns to baseline");
    check(after.current_allocations == before.current_allocations,
          "current allocation count returns to baseline");
}

void test_bit_manipulation_roundtrip()
{
    using rad_ml::utils::BitManipulation;

    const float f = 3.14159f;
    const float f_flipped = BitManipulation::flipBit(f, 7);
    check(f != f_flipped, "float bit flip changes the value");
    check(BitManipulation::flipBit(f_flipped, 7) == f, "float double flip restores the value");
    check(BitManipulation::countBitDifferences(f, f_flipped) == 1,
          "float single flip differs by exactly one bit");

    const double d = -2.718281828;
    const double d_flipped = BitManipulation::flipBit(d, 55);
    check(BitManipulation::flipBit(d_flipped, 55) == d, "double double flip restores the value");
    check(BitManipulation::countBitDifferences(d, d_flipped) == 1,
          "double single flip differs by exactly one bit");

    check(BitManipulation::isBitSet(uint32_t{0b1000}, 3), "isBitSet finds a set integer bit");
    check(!BitManipulation::isBitSet(uint32_t{0b1000}, 2), "isBitSet rejects a clear integer bit");
}

}  // namespace

int main()
{
    test_unprotected_roundtrip();
    test_canary_roundtrip_and_detection();
    test_canary_aligned_allocation();
    test_invalid_alignment_is_rejected();
    test_crc_detects_bit_flip();
    test_ecc_detects_bit_flip();
    test_tmr_detects_divergent_copy();
    test_corruption_callback_fires();
    test_stats_track_protected_allocations();
    test_bit_manipulation_roundtrip();

    if (failures == 0) {
        std::cout << "All unified_memory_test checks passed.\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
