/**
 * @file adaptive_framework.hpp
 * @brief Environment-aware adaptive radiation protection
 * 
 * This file defines an adaptive framework that adjusts protection strategies
 * based on the current radiation environment and observed error patterns.
 */

#ifndef RAD_ML_ADAPTIVE_FRAMEWORK_HPP
#define RAD_ML_ADAPTIVE_FRAMEWORK_HPP

#include <chrono>
#include <unordered_map>
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include "../redundancy/enhanced_voting.hpp"
#include "../runtime/error_tracker.hpp"

namespace rad_ml {
namespace core {
namespace adaptive {

/**
 * Adaptive framework that adjusts protection strategy based on environment
 * 
 * This class monitors error rates and patterns to dynamically adjust
 * protection levels, providing optimal tradeoffs between reliability
 * and performance.
 */
class AdaptiveFramework {
public:
    /**
     * Radiation environment types with predefined characteristics
     */
    enum class EnvironmentType {
        BENIGN,       // Low radiation, standard operation
        LEO,          // Low Earth Orbit
        MEO,          // Medium Earth Orbit
        GEO,          // Geosynchronous Orbit
        SOLAR_FLARE,  // Solar flare or other transient event
        JUPITER,      // Jupiter or similar high-radiation
        EXTREME       // Maximum protection level
    };
    
    /**
     * Protection level settings
     */
    struct ProtectionSettings {
        int scrubbing_interval_ms;
        float error_threshold;
        bool use_weighted_voting;
        bool use_ecc_memory;
        int redundancy_level; // 1=none, 3=TMR, 5=5MR, etc.
        float checkpoint_frequency;
    };
    
    /**
     * Constructor initializes with default benign environment
     */
    AdaptiveFramework() 
        : environment(EnvironmentType::BENIGN),
          adaptive_mode_enabled(true),
          last_environment_check(std::chrono::steady_clock::now()),
          environment_check_interval(std::chrono::seconds(10)) {
        
        // Initialize environment-specific settings
        initializeEnvironmentSettings();
        
        // Apply initial settings
        updateProtectionStrategy();
    }
    
    /**
     * Configure environment type manually
     * 
     * @param env Environment type to set
     */
    void setEnvironment(EnvironmentType env) {
        if (environment != env) {
            environment = env;
            updateProtectionStrategy();
            
            // Notify listeners of environment change
            for (const auto& [id, callback] : environment_change_callbacks) {
                callback(env);
            }
        }
    }
    
    /**
     * Get current environment type
     * 
     * @return Current environment setting
     */
    EnvironmentType getEnvironment() const {
        return environment;
    }
    
    /**
     * Enable/disable adaptive protection
     * 
     * @param enable Whether to enable adaptive mode
     */
    void setAdaptiveMode(bool enable) {
        adaptive_mode_enabled = enable;
        if (enable) {
            // Immediately check environment when enabling
            autoDetectEnvironment();
        }
    }
    
    /**
     * Check if adaptive mode is enabled
     * 
     * @return True if adaptive mode is active
     */
    bool isAdaptiveModeEnabled() const {
        return adaptive_mode_enabled;
    }
    
    /**
     * Get current protection settings
     * 
     * @return Current protection settings
     */
    const ProtectionSettings& getCurrentSettings() const {
        return current_settings;
    }
    
    /**
     * Register callback for environment changes
     * 
     * @param callback Function to call when environment changes
     * @return ID that can be used to unregister
     */
    int registerEnvironmentChangeCallback(
        std::function<void(EnvironmentType)> callback) {
        
        static int next_id = 0;
        int id = next_id++;
        
        environment_change_callbacks[id] = std::move(callback);
        return id;
    }
    
    /**
     * Unregister environment change callback
     * 
     * @param id ID returned from registerEnvironmentChangeCallback
     * @return True if successfully unregistered
     */
    bool unregisterEnvironmentChangeCallback(int id) {
        return environment_change_callbacks.erase(id) > 0;
    }
    
    /**
     * Auto-detect environment based on error statistics
     * 
     * This analyzes recent error patterns to determine the
     * most appropriate environment setting.
     */
    void autoDetectEnvironment() {
        if (!adaptive_mode_enabled) return;
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_environment_check < environment_check_interval) {
            return; // Don't check too frequently
        }
        
        last_environment_check = now;
        
        const auto& tracker = runtime::getGlobalErrorTracker();
        float error_rate = tracker.getErrorRate();
        
        // First check extreme conditions that need immediate response
        if (error_rate > 10.0f) {
            setEnvironment(EnvironmentType::EXTREME);
            return;
        }
        
        // Analyze error pattern distribution
        auto distribution = tracker.getPatternDistribution();
        
        // Calculate a weighted score based on error rate and pattern severity
        float severity_score = 0.0f;
        
        // Weight different patterns by severity
        constexpr std::array<float, 6> pattern_weights = {
            1.0f,  // SINGLE_BIT
            2.0f,  // ADJACENT_BITS
            3.0f,  // BYTE_ERROR
            4.0f,  // WORD_ERROR
            5.0f,  // BURST_ERROR
            2.5f   // UNKNOWN
        };
        
        for (size_t i = 0; i < distribution.size(); ++i) {
            severity_score += distribution[i] * pattern_weights[i];
        }
        
        severity_score *= error_rate; // Scale by error rate
        
        // Map severity score to environment
        EnvironmentType detected_env;
        
        if (severity_score < 0.01f) {
            detected_env = EnvironmentType::BENIGN;
        } else if (severity_score < 0.1f) {
            detected_env = EnvironmentType::LEO;
        } else if (severity_score < 0.5f) {
            detected_env = EnvironmentType::MEO;
        } else if (severity_score < 1.0f) {
            detected_env = EnvironmentType::GEO;
        } else if (severity_score < 5.0f) {
            detected_env = EnvironmentType::SOLAR_FLARE;
        } else if (severity_score < 20.0f) {
            detected_env = EnvironmentType::JUPITER;
        } else {
            detected_env = EnvironmentType::EXTREME;
        }
        
        // Apply hysteresis to prevent rapid oscillation
        if (detected_env != environment) {
            // Only change if the new environment is at least two levels different,
            // or we've been in the same environment for a while
            int env_diff = std::abs(static_cast<int>(detected_env) - static_cast<int>(environment));
            
            auto time_in_current_env = now - last_environment_change;
            bool stable_duration = time_in_current_env > std::chrono::minutes(5);
            
            if (env_diff >= 2 || stable_duration) {
                setEnvironment(detected_env);
                last_environment_change = now;
            }
        }
    }
    
    /**
     * Log an error for statistics and potential environment adjustment
     * 
     * @param pattern The detected fault pattern
     * @param data Optional details about the error
     */
    void logError(redundancy::FaultPattern pattern, const std::string& data = "") {
        // Forward to global error tracker
        runtime::getGlobalErrorTracker().recordError(pattern, data);
        
        // Check if we should reanalyze environment
        error_count_since_last_check++;
        
        // If error rate suddenly spikes, check environment immediately
        if (error_count_since_last_check >= 10) {
            autoDetectEnvironment();
            error_count_since_last_check = 0;
        }
    }
    
    /**
     * Override settings for a specific environment
     * 
     * @param env The environment to customize
     * @param settings New settings for this environment
     */
    void customizeEnvironmentSettings(EnvironmentType env, const ProtectionSettings& settings) {
        environment_settings[env] = settings;
        
        // If this is the current environment, apply changes immediately
        if (env == environment) {
            updateProtectionStrategy();
        }
    }
    
    /**
     * Perform periodic maintenance tasks
     * 
     * This should be called regularly to ensure adaptation
     * to changing radiation conditions.
     */
    void performMaintenance() {
        // Check if environment needs to be updated
        autoDetectEnvironment();
    }

private:
    EnvironmentType environment;
    bool adaptive_mode_enabled;
    ProtectionSettings current_settings;
    std::unordered_map<EnvironmentType, ProtectionSettings> environment_settings;
    std::unordered_map<int, std::function<void(EnvironmentType)>> environment_change_callbacks;
    
    // Timing control
    std::chrono::steady_clock::time_point last_environment_check;
    std::chrono::steady_clock::time_point last_environment_change;
    std::chrono::steady_clock::duration environment_check_interval;
    int error_count_since_last_check = 0;
    
    /**
     * Initialize default settings for each environment
     */
    void initializeEnvironmentSettings() {
        // Benign environment - minimal protection
        environment_settings[EnvironmentType::BENIGN] = {
            .scrubbing_interval_ms = 5000,    // 5 seconds
            .error_threshold = 0.1f,
            .use_weighted_voting = false,
            .use_ecc_memory = false,
            .redundancy_level = 3,           // Basic TMR
            .checkpoint_frequency = 0.01f     // 1%
        };
        
        // Low Earth Orbit
        environment_settings[EnvironmentType::LEO] = {
            .scrubbing_interval_ms = 1000,    // 1 second
            .error_threshold = 0.05f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 3,            // TMR
            .checkpoint_frequency = 0.05f     // 5%
        };
        
        // Medium Earth Orbit
        environment_settings[EnvironmentType::MEO] = {
            .scrubbing_interval_ms = 500,     // 500ms
            .error_threshold = 0.02f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 3,            // TMR
            .checkpoint_frequency = 0.1f      // 10%
        };
        
        // Geosynchronous Orbit
        environment_settings[EnvironmentType::GEO] = {
            .scrubbing_interval_ms = 250,     // 250ms
            .error_threshold = 0.01f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 3,            // TMR
            .checkpoint_frequency = 0.2f      // 20%
        };
        
        // Solar Flare - temporary intense radiation
        environment_settings[EnvironmentType::SOLAR_FLARE] = {
            .scrubbing_interval_ms = 100,     // 100ms
            .error_threshold = 0.005f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 5,            // 5MR for critical components
            .checkpoint_frequency = 0.5f      // 50%
        };
        
        // Jupiter or similar intense environment
        environment_settings[EnvironmentType::JUPITER] = {
            .scrubbing_interval_ms = 50,      // 50ms
            .error_threshold = 0.001f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 5,            // 5MR
            .checkpoint_frequency = 0.8f      // 80%
        };
        
        // Extreme environment - maximum protection
        environment_settings[EnvironmentType::EXTREME] = {
            .scrubbing_interval_ms = 10,      // 10ms
            .error_threshold = 0.0005f,
            .use_weighted_voting = true,
            .use_ecc_memory = true,
            .redundancy_level = 7,            // 7MR for extreme cases
            .checkpoint_frequency = 1.0f      // 100% (checkpoint everything)
        };
    }
    
    /**
     * Update protection strategy based on current environment
     */
    void updateProtectionStrategy() {
        // Get settings for current environment
        auto it = environment_settings.find(environment);
        if (it != environment_settings.end()) {
            current_settings = it->second;
        } else {
            // Fallback to safest settings if not found
            current_settings = environment_settings[EnvironmentType::EXTREME];
        }
    }
};

/**
 * Get global adaptive framework instance
 * 
 * @return Reference to singleton framework
 */
inline AdaptiveFramework& getGlobalAdaptiveFramework() {
    static AdaptiveFramework framework;
    return framework;
}

} // namespace adaptive
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ADAPTIVE_FRAMEWORK_HPP 