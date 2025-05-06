#pragma once

#include <vector>
#include <functional>
#include <random>
#include <memory>

#include "../common/types.hpp"
#include "environment.hpp"

namespace rad_ml {
namespace radiation {

/**
 * @brief Simulator for Single Event Upsets (SEUs) in memory
 */
class SEUSimulator {
public:
    /**
     * @brief Constructor
     * 
     * @param environment Radiation environment to simulate
     * @param seed Random seed for reproducibility (0 for random)
     */
    SEUSimulator(
        std::shared_ptr<Environment> environment = nullptr,
        uint32 seed = 0
    ) : environment_(environment) {
        // Initialize random number generator
        if (seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        } else {
            rng_.seed(seed);
        }
    }
    
    /**
     * @brief Set the radiation environment
     * 
     * @param environment Environment to simulate
     */
    void setEnvironment(std::shared_ptr<Environment> environment) {
        environment_ = environment;
    }
    
    /**
     * @brief Get the current environment
     * 
     * @return Current environment
     */
    std::shared_ptr<Environment> getEnvironment() const {
        return environment_;
    }
    
    /**
     * @brief Inject bit flips into a memory region
     * 
     * @param data Pointer to memory
     * @param size_bytes Size of memory in bytes
     * @param duration_ms Duration of exposure in milliseconds
     * @return Number of bit flips injected
     */
    uint32 injectBitFlips(void* data, size_t size_bytes, uint32 duration_ms) {
        if (!environment_ || !data || size_bytes == 0) {
            return 0;
        }
        
        // Calculate expected number of bit flips based on environment
        float flux = environment_->getSEUFlux();
        float cross_section = environment_->getSEUCrossSection();
        
        // Expected number of SEUs
        float expected_seus = flux * cross_section * (size_bytes * 8) * (duration_ms / 1000.0f);
        
        // Use Poisson distribution to determine actual number of SEUs
        std::poisson_distribution<uint32> poisson(expected_seus);
        uint32 num_seus = poisson(rng_);
        
        // Inject bit flips
        uint8* byte_data = static_cast<uint8*>(data);
        for (uint32 i = 0; i < num_seus; ++i) {
            // Random byte in the memory region
            std::uniform_int_distribution<size_t> byte_dist(0, size_bytes - 1);
            size_t byte_idx = byte_dist(rng_);
            
            // Random bit in the byte
            std::uniform_int_distribution<uint8> bit_dist(0, 7);
            uint8 bit_idx = bit_dist(rng_);
            
            // Flip the bit
            byte_data[byte_idx] ^= (1 << bit_idx);
        }
        
        return num_seus;
    }
    
    /**
     * @brief Simulate SEUs for a specific duration
     * 
     * @param memory_regions Vector of memory regions (ptr, size)
     * @param duration_ms Duration of simulation in milliseconds
     * @param callback Callback function called after each injection
     * @return Total number of SEUs injected
     */
    uint32 simulateSEUs(
        const std::vector<std::pair<void*, size_t>>& memory_regions,
        uint32 duration_ms,
        std::function<void(void* ptr, size_t size, uint32 bit_flips)> callback = nullptr
    ) {
        uint32 total_seus = 0;
        
        for (const auto& region : memory_regions) {
            uint32 seus = injectBitFlips(region.first, region.second, duration_ms);
            total_seus += seus;
            
            if (callback && seus > 0) {
                callback(region.first, region.second, seus);
            }
        }
        
        return total_seus;
    }

private:
    std::shared_ptr<Environment> environment_;
    std::mt19937 rng_;
};

} // namespace radiation
} // namespace rad_ml 