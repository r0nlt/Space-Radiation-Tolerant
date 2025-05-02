#pragma once

#include <cstdint>
#include <cstddef>
#include <random>
#include <chrono>
#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace rad_ml {
namespace testing {

/**
 * @brief Types of faults that can be injected
 */
enum class FaultType {
    SingleBitFlip,     ///< Flip a single bit
    MultiBitFlip,      ///< Flip multiple bits
    StuckAtZero,       ///< Set a byte to all zeros
    StuckAtOne,        ///< Set a byte to all ones
    RandomValue        ///< Replace with a random value
};

/**
 * @brief Result of a fault injection test
 */
struct FaultInjectionResult {
    bool detected;              ///< Whether the fault was detected
    bool corrected;             ///< Whether the fault was corrected
    std::string description;    ///< Description of what happened
};

/**
 * @brief Fault injector for radiation testing
 * 
 * This class provides mechanisms to inject faults into memory regions
 * to simulate the effects of radiation in a controlled manner.
 */
class FaultInjector {
public:
    /**
     * @brief Constructor
     */
    FaultInjector()
        : rng_(std::random_device{}()) {}
    
    /**
     * @brief Inject a single bit flip at a random location
     * 
     * @param data Pointer to the memory region
     * @param size_bytes Size of the memory region in bytes
     */
    template <typename T>
    void injectRandomBitFlip(T* data, size_t size_bytes) {
        // Pick random byte and bit
        size_t byte_index = getRandomIndex(size_bytes);
        uint8_t bit_index = getRandomIndex(8);
        
        // Flip the bit
        uint8_t* bytes = reinterpret_cast<uint8_t*>(data);
        bytes[byte_index] ^= (1 << bit_index);
    }
    
    /**
     * @brief Inject a specific fault type at a random location
     * 
     * @param data Pointer to the memory region
     * @param size_bytes Size of the memory region in bytes
     * @param fault_type Type of fault to inject
     */
    template <typename T>
    void injectFault(T* data, size_t size_bytes, FaultType fault_type) {
        uint8_t* bytes = reinterpret_cast<uint8_t*>(data);
        size_t byte_index = getRandomIndex(size_bytes);
        
        switch (fault_type) {
            case FaultType::SingleBitFlip:
                bytes[byte_index] ^= (1 << getRandomIndex(8));
                break;
            
            case FaultType::MultiBitFlip: {
                // Flip 2-4 bits in the same byte
                size_t num_bits = getRandomIndex(3) + 2; // 2 to 4 bits
                for (size_t i = 0; i < num_bits; ++i) {
                    bytes[byte_index] ^= (1 << getRandomIndex(8));
                }
                break;
            }
            
            case FaultType::StuckAtZero:
                bytes[byte_index] = 0;
                break;
            
            case FaultType::StuckAtOne:
                bytes[byte_index] = 0xFF;
                break;
            
            case FaultType::RandomValue:
                bytes[byte_index] = getRandomByte();
                break;
        }
    }
    
    /**
     * @brief Test a component's resilience to faults
     * 
     * @param component The component to test
     * @param inject_fault Function to inject a fault into the component
     * @param verify_component Function to verify if the component is working correctly
     * @param repair_component Function to repair the component
     * @param num_trials Number of fault injection trials to perform
     * @return FaultInjectionResult with statistics about the test
     */
    template <typename Component>
    FaultInjectionResult testResilience(
        Component& component,
        std::function<void(Component&)> inject_fault,
        std::function<bool(Component&)> verify_component,
        std::function<void(Component&)> repair_component,
        int num_trials = 100
    ) {
        int detected_count = 0;
        int corrected_count = 0;
        
        for (int i = 0; i < num_trials; ++i) {
            // Inject a fault
            inject_fault(component);
            
            // Check if the fault was detected
            bool detected = !verify_component(component);
            if (detected) {
                detected_count++;
                
                // Try to repair
                repair_component(component);
                
                // Check if repair was successful
                bool repaired = verify_component(component);
                if (repaired) {
                    corrected_count++;
                }
            }
        }
        
        FaultInjectionResult result;
        result.detected = detected_count > 0;
        result.corrected = corrected_count > 0;
        result.description = "Detected " + std::to_string(detected_count) + "/" + 
                             std::to_string(num_trials) + " faults, corrected " + 
                             std::to_string(corrected_count) + "/" + 
                             std::to_string(detected_count) + " detected faults";
        
        return result;
    }
    
private:
    std::mt19937 rng_;
    
    /**
     * @brief Get a random index within a range
     * 
     * @param max_value Maximum value (exclusive)
     * @return Random index between 0 and max_value-1
     */
    size_t getRandomIndex(size_t max_value) {
        return std::uniform_int_distribution<size_t>(0, max_value - 1)(rng_);
    }
    
    /**
     * @brief Get a random byte value
     * 
     * @return Random byte between 0 and 255
     */
    uint8_t getRandomByte() {
        return static_cast<uint8_t>(std::uniform_int_distribution<int>(0, 255)(rng_));
    }
};

} // namespace testing
} // namespace rad_ml 