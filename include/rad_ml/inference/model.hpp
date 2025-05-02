#pragma once

#include <rad_ml/core/memory/memory_scrubber.hpp>
#include <functional>
#include <memory>
#include <chrono>

namespace rad_ml {
namespace inference {

/**
 * @brief Interface for radiation-tolerant neural network models
 * 
 * This abstract class defines the interface that all radiation-tolerant
 * models should implement. It includes methods for inference, repair,
 * and health monitoring.
 */
class RadiationTolerantModel {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~RadiationTolerantModel() = default;
    
    /**
     * @brief Run inference on the model
     * 
     * This method should be implemented by derived classes to perform
     * the forward pass through the model.
     * 
     * @param input Pointer to the input data
     * @param input_size Size of the input data in bytes
     * @param output Pointer to where the output should be stored
     * @param output_capacity Maximum size of the output in bytes
     * @return The actual size of the output in bytes
     */
    virtual size_t runInference(
        const void* input,
        size_t input_size,
        void* output,
        size_t output_capacity
    ) = 0;
    
    /**
     * @brief Repair any corrupted parts of the model
     * 
     * This method should be implemented by derived classes to fix
     * any radiation-induced faults in the model's parameters.
     */
    virtual void repair() = 0;
    
    /**
     * @brief Check the health of the model
     * 
     * @return true if the model is healthy, false otherwise
     */
    virtual bool isHealthy() const = 0;
    
    /**
     * @brief Enable automatic scrubbing
     * 
     * @param interval_ms How often to perform scrubbing in milliseconds
     */
    void enableAutoScrubbing(unsigned long interval_ms = 1000) {
        scrubber_ = std::make_unique<core::memory::MemoryScrubber>(interval_ms);
        scrubber_->registerMemoryRegion<RadiationTolerantModel>(
            this,
            sizeof(*this),
            [this](RadiationTolerantModel*, size_t) {
                repair();
            }
        );
        scrubber_->start();
    }
    
    /**
     * @brief Disable automatic scrubbing
     */
    void disableAutoScrubbing() {
        if (scrubber_) {
            scrubber_->stop();
            scrubber_.reset();
        }
    }
    
private:
    // Memory scrubber for automatic repair
    std::unique_ptr<core::memory::MemoryScrubber> scrubber_;
};

} // namespace inference
} // namespace rad_ml 