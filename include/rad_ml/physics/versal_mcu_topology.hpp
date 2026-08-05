#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace rad_ml {
namespace physics {

/**
 * @brief VC1902 configuration-memory 2-bit MCU topology
 *
 * Mayo et al., IEEE TNS 2025, DOI 10.1109/TNS.2025.3531510,
 * Section IV-B recommends a simplified injection campaign in which about 90%
 * of 2-bit MCUs have a readback-index separation of 3200 and the remainder use
 * 22399 or 47998. These are separate configuration cells, not adjacent bits in
 * one software byte and not an MBU in one memory word.
 */
class VersalConfigurationMcuTopology {
   public:
    static constexpr std::array<std::uint64_t, 3> modeled_offsets = {
        3200, 22399, 47998};

    template <typename RandomEngine>
    static std::uint64_t sampleOffset(RandomEngine& engine)
    {
        std::discrete_distribution<std::size_t> distribution({0.90, 0.05, 0.05});
        return modeled_offsets[distribution(engine)];
    }

    template <typename RandomEngine>
    static std::array<std::uint64_t, 2> sampleBitIndices(
        std::uint64_t total_bits, RandomEngine& engine)
    {
        if (total_bits <= 2 * modeled_offsets.back()) {
            throw std::invalid_argument(
                "Versal MCU topology requires a configuration-bitstream-sized region");
        }

        std::uniform_int_distribution<std::uint64_t> seed_distribution(0, total_bits - 1);
        const std::uint64_t seed = seed_distribution(engine);
        const std::uint64_t offset = sampleOffset(engine);
        const std::uint64_t partner =
            seed + offset < total_bits ? seed + offset : seed - offset;
        return {seed, partner};
    }

    static bool isModeledOffset(std::uint64_t offset) noexcept
    {
        for (const auto modeled : modeled_offsets) {
            if (modeled == offset) return true;
        }
        return false;
    }
};

}  // namespace physics
}  // namespace rad_ml
