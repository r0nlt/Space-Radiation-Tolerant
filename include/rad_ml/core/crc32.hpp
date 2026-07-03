/**
 * @file crc32.hpp
 * @brief Canonical CRC-32 implementation for the rad_ml framework
 *
 * Single source of truth for CRC-32 (reflected, polynomial 0xEDB88320,
 * init 0xFFFFFFFF, final XOR 0xFFFFFFFF -- the standard "CRC-32/ISO-HDLC"
 * used by zlib/Ethernet; CRC of "123456789" is 0xCBF43926).
 *
 * Every integrity check in the framework must delegate here so that a
 * checksum computed by one subsystem (e.g. EnhancedTMR's per-copy CRCs) can
 * be verified by another (e.g. a memory scrubber) without silent divergence.
 *
 * The implementation is deliberately bitwise and branch-free rather than
 * table-driven:
 *  - no lookup table in RAM that could itself be corrupted by an upset;
 *    the computation only touches registers
 *  - deterministic execution time per byte regardless of data pattern
 *    (no data-dependent branches or cache-dependent table lookups), which
 *    real-time space applications rely on
 */

#ifndef RAD_ML_CORE_CRC32_HPP
#define RAD_ML_CORE_CRC32_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rad_ml {
namespace core {

class Crc32 {
   public:
    /**
     * @brief Compute the CRC-32 checksum of a memory region
     *
     * @param data Pointer to the data
     * @param length Length of the data in bytes
     * @return CRC-32 checksum
     */
    static std::uint32_t compute(const void* data, std::size_t length) noexcept
    {
        const std::uint8_t* bytes = static_cast<const std::uint8_t*>(data);
        std::uint32_t crc = 0xFFFFFFFFu;
        for (std::size_t i = 0; i < length; ++i) {
            crc ^= bytes[i];
            for (int j = 0; j < 8; ++j) {
                // Branch-free: (~(crc & 1) + 1) is 0xFFFFFFFF if the low bit
                // is set, 0 otherwise
                crc = (crc >> 1) ^ (kPolynomial & (~(crc & 1u) + 1u));
            }
        }
        return ~crc;
    }

    /**
     * @brief Compute the CRC-32 checksum of a trivially copyable value
     */
    template <typename T>
    static std::uint32_t compute(const T& value) noexcept
    {
        static_assert(std::is_trivially_copyable<T>::value,
                      "Crc32::compute requires a trivially copyable type");
        return compute(&value, sizeof(T));
    }

    /**
     * @brief Verify a value against a previously computed checksum
     */
    template <typename T>
    static bool verify(const T& value, std::uint32_t expected_checksum) noexcept
    {
        return compute(value) == expected_checksum;
    }

   private:
    static constexpr std::uint32_t kPolynomial = 0xEDB88320u;
};

}  // namespace core
}  // namespace rad_ml

#endif  // RAD_ML_CORE_CRC32_HPP
