#pragma once

#include <chrono>
#include <cmath>
#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <functional>
#include <iostream>

namespace rad_ml {
namespace core {
namespace radiation {

/**
 * @brief Adaptive radiation protection system
 * 
 * This system dynamically adjusts the level of radiation protection
 * based on detected error rates and environmental conditions. It helps
 * optimize the trade-off between performance and reliability in
 * space radiation environments.
 */
class AdaptiveProtection {
public:
    /// Protection levels that can be dynamically adjusted
    enum class ProtectionLevel {
        MINIMAL,    ///< For low-radiation environments, minimal overhead
        STANDARD,   ///< Default level for most orbit conditions
        ENHANCED,   ///< For passing through radiation belts or solar events
        MAXIMUM     ///< For extreme radiation conditions or critical operations
    };
    
    /// Current radiation environment assessment
    struct RadiationEnvironment {
        double estimated_flux;             ///< Particles/cm²/s
        uint32_t bit_flips_detected;       ///< Count since last assessment
        uint32_t computation_errors;       ///< Count since last assessment
        std::chrono::system_clock::time_point last_assessment;
    };
    
    /// Configuration parameters for each protection level
    struct ProtectionConfig {
        uint32_t redundancy_level;       ///< 1=none, 2=double, 3=triple
        uint32_t scrubbing_interval_ms;  ///< How often to scrub memory
        bool enable_temporal_redundancy; ///< Whether to repeat computations
        bool enable_checkpoint_recovery; ///< Whether to create checkpoints
    };
    
    /// Callback signature for level change notifications
    using LevelChangeCallback = std::function<void(ProtectionLevel)>;
    
    /**
     * @brief Constructor with initial configuration
     * 
     * @param initial_level Starting protection level
     */
    explicit AdaptiveProtection(ProtectionLevel initial_level = ProtectionLevel::STANDARD)
        : current_level_(initial_level) {
        // Initialize environment assessment
        environment_.estimated_flux = 0.0;
        environment_.bit_flips_detected = 0;
        environment_.computation_errors = 0;
        environment_.last_assessment = std::chrono::system_clock::now();
    }
    
    /**
     * @brief Update radiation environment assessment based on error rates
     * 
     * @param new_bit_flips Number of new bit flips detected since last update
     * @param new_computation_errors Number of new computation errors detected
     */
    void updateEnvironment(uint32_t new_bit_flips, uint32_t new_computation_errors) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - environment_.last_assessment).count();
        
        if (elapsed > 0) {
            // Update flux estimate
            double error_rate = (new_bit_flips + new_computation_errors) / 
                                static_cast<double>(elapsed);
            
            // Exponential moving average for smooth transitions
            constexpr double alpha = 0.3; // Smoothing factor
            environment_.estimated_flux = 
                alpha * error_rate + (1.0 - alpha) * environment_.estimated_flux;
            
            // Update counters
            environment_.bit_flips_detected = new_bit_flips;
            environment_.computation_errors = new_computation_errors;
            environment_.last_assessment = now;
            
            // Adjust protection level based on new assessment
            adjustProtectionLevel();
        }
    }
    
    /**
     * @brief Get current protection level
     * 
     * @return Current protection level
     */
    ProtectionLevel getProtectionLevel() const {
        return current_level_;
    }
    
    /**
     * @brief Get environment assessment
     * 
     * @return Current radiation environment assessment
     */
    const RadiationEnvironment& getEnvironment() const {
        return environment_;
    }
    
    /**
     * @brief Get configuration for current protection level
     * 
     * @return Protection configuration parameters
     */
    ProtectionConfig getConfiguration() const {
        return protection_configs_[static_cast<int>(current_level_)];
    }
    
    /**
     * @brief Temporarily increase protection level for critical operations
     * 
     * @param duration_ms How long to maintain the increased protection (ms)
     */
    void temporarilyIncreaseLevel(uint32_t duration_ms = 5000) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Save current level
        ProtectionLevel original_level = current_level_;
        
        // Increase level by one (if not already at maximum)
        if (current_level_ != ProtectionLevel::MAXIMUM) {
            current_level_ = static_cast<ProtectionLevel>(
                static_cast<int>(current_level_) + 1);
                
            // Notify subscribers about change
            notifyLevelChange();
            
            // Schedule reversion after duration
            // Note: In a real implementation, this would use a proper timer
            // For now, we'll simulate with a warning
            std::cout << "WARNING: temporarilyIncreaseLevel called - in a real implementation, "
                      << "protection would return to " << static_cast<int>(original_level)
                      << " after " << duration_ms << "ms" << std::endl;
        }
    }
    
    /**
     * @brief Set initial protection level
     * 
     * @param level The protection level to set
     */
    void setInitialLevel(ProtectionLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (current_level_ != level) {
            current_level_ = level;
            notifyLevelChange();
        }
    }
    
    /**
     * @brief Register a callback for protection level changes
     * 
     * @param callback Function to call when protection level changes
     * @return Handle that can be used to unregister the callback
     */
    int registerLevelChangeCallback(LevelChangeCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int handle = next_callback_handle_++;
        callbacks_[handle] = std::move(callback);
        return handle;
    }
    
    /**
     * @brief Unregister a previously registered callback
     * 
     * @param handle The handle returned from registerLevelChangeCallback
     * @return true if callback was found and removed, false otherwise
     */
    bool unregisterLevelChangeCallback(int handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = callbacks_.find(handle);
        if (it != callbacks_.end()) {
            callbacks_.erase(it);
            return true;
        }
        return false;
    }
    
private:
    // Static configuration for protection levels
    static constexpr ProtectionConfig protection_configs_[4] = {
        // MINIMAL
        {1, 5000,  false, false},
        
        // STANDARD
        {2, 1000,  false, true},
        
        // ENHANCED
        {3, 500,   true,  true},
        
        // MAXIMUM
        {3, 100,   true,  true}
    };
    
    // Thresholds for different protection levels
    static constexpr double STANDARD_THRESHOLD = 0.01;  // errors/sec
    static constexpr double ENHANCED_THRESHOLD = 0.1;   // errors/sec
    static constexpr double MAXIMUM_THRESHOLD = 1.0;    // errors/sec
    
    // Current state
    ProtectionLevel current_level_;
    RadiationEnvironment environment_;
    
    // Thread safety
    std::mutex mutex_;
    
    // Callback management
    std::unordered_map<int, LevelChangeCallback> callbacks_;
    int next_callback_handle_ = 0;
    
    /**
     * @brief Adjust protection level based on environment
     */
    void adjustProtectionLevel() {
        // Determine appropriate level based on flux
        ProtectionLevel recommended_level;
        
        if (environment_.estimated_flux >= MAXIMUM_THRESHOLD) {
            recommended_level = ProtectionLevel::MAXIMUM;
        } else if (environment_.estimated_flux >= ENHANCED_THRESHOLD) {
            recommended_level = ProtectionLevel::ENHANCED;
        } else if (environment_.estimated_flux >= STANDARD_THRESHOLD) {
            recommended_level = ProtectionLevel::STANDARD;
        } else {
            recommended_level = ProtectionLevel::MINIMAL;
        }
        
        // Only change level if it differs from current
        if (recommended_level != current_level_) {
            current_level_ = recommended_level;
            // Notify subscribers about level change
            notifyLevelChange();
        }
    }
    
    /**
     * @brief Notify subscribers about protection level changes
     */
    void notifyLevelChange() {
        ProtectionLevel level = current_level_;
        
        // Call all registered callbacks
        for (const auto& [_, callback] : callbacks_) {
            callback(level);
        }
    }
};

} // namespace radiation
} // namespace core
} // namespace rad_ml 