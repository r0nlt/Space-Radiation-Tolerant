#pragma once

#include "rad_ml/testing/radiation_simulator.hpp"
#include "rad_ml/tmr/tmr.hpp"
#include "rad_ml/tmr/enhanced_tmr.hpp"
#include "rad_ml/memory/memory_scrubber.hpp"

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <map>
#include <random>
#include <iostream>

namespace rad_ml {
namespace testing {

/**
 * @brief Mission profile configuration
 * 
 * Defines the parameters and radiation environment for a specific mission
 */
struct MissionProfile {
    std::string name;                                  ///< Mission name (e.g., "LEO", "Mars")
    std::vector<rad_ml::testing::RadiationSimulator::EnvironmentParams> environments; ///< Different radiation environments
    std::vector<double> environment_durations;         ///< Duration for each environment (seconds)
    std::vector<double> transition_probabilities;      ///< Probability of transition between environments
    
    /**
     * @brief Create a standard mission profile
     * 
     * @param mission_type Type of mission ("LEO", "ISS", "MARS", etc.)
     * @return MissionProfile with appropriate parameters
     */
    static MissionProfile createStandard(const std::string& mission_type) {
        MissionProfile profile;
        profile.name = mission_type;
        
        if (mission_type == "LEO" || mission_type == "LEO_EARTH_OBSERVATION") {
            // Low Earth Orbit
            auto normal_env = RadiationSimulator::getMissionEnvironment("LEO");
            
            // SAA environment (higher radiation)
            auto saa_env = normal_env;
            saa_env.inside_saa = true;
            saa_env.mission_name = "LEO (SAA)";
            
            // Solar flare environment
            auto flare_env = normal_env;
            flare_env.solar_activity = 9.0;
            flare_env.mission_name = "LEO (Solar Flare)";
            
            profile.environments = {normal_env, saa_env, flare_env};
            profile.environment_durations = {5400.0, 600.0, 300.0}; // 90 min, 10 min, 5 min
            profile.transition_probabilities = {0.8, 0.15, 0.05};
        }
        else if (mission_type == "ISS") {
            // International Space Station
            auto normal_env = RadiationSimulator::getMissionEnvironment("ISS");
            
            // SAA environment (higher radiation)
            auto saa_env = normal_env;
            saa_env.inside_saa = true;
            saa_env.mission_name = "ISS (SAA)";
            
            // Solar flare environment
            auto flare_env = normal_env;
            flare_env.solar_activity = 9.0;
            flare_env.mission_name = "ISS (Solar Flare)";
            
            profile.environments = {normal_env, saa_env, flare_env};
            profile.environment_durations = {5400.0, 600.0, 300.0}; // 90 min, 10 min, 5 min
            profile.transition_probabilities = {0.8, 0.15, 0.05};
        }
        else if (mission_type == "GEO" || mission_type == "GEOSTATIONARY") {
            // Geostationary orbit
            auto normal_env = RadiationSimulator::getMissionEnvironment("GEO");
            
            // Solar flare environment
            auto flare_env = normal_env;
            flare_env.solar_activity = 9.0;
            flare_env.mission_name = "GEO (Solar Flare)";
            
            profile.environments = {normal_env, flare_env};
            profile.environment_durations = {86400.0, 3600.0}; // 24 hours, 1 hour
            profile.transition_probabilities = {0.95, 0.05};
        }
        else if (mission_type == "LUNAR" || mission_type == "MOON") {
            // Lunar mission
            auto normal_env = RadiationSimulator::getMissionEnvironment("LUNAR");
            
            // Solar flare environment
            auto flare_env = normal_env;
            flare_env.solar_activity = 9.0;
            flare_env.mission_name = "Lunar (Solar Flare)";
            
            profile.environments = {normal_env, flare_env};
            profile.environment_durations = {86400.0, 3600.0}; // 24 hours, 1 hour
            profile.transition_probabilities = {0.95, 0.05};
        }
        else if (mission_type == "MARS") {
            // Mars mission
            auto normal_env = RadiationSimulator::getMissionEnvironment("MARS");
            
            // Solar flare environment
            auto flare_env = normal_env;
            flare_env.solar_activity = 9.0;
            flare_env.mission_name = "Mars (Solar Flare)";
            
            // Dust storm environment (less shielding from atmosphere)
            auto storm_env = normal_env;
            storm_env.shielding_thickness_mm *= 0.8;
            storm_env.mission_name = "Mars (Dust Storm)";
            
            profile.environments = {normal_env, flare_env, storm_env};
            profile.environment_durations = {86400.0, 3600.0, 43200.0}; // 24 hours, 1 hour, 12 hours
            profile.transition_probabilities = {0.9, 0.05, 0.05};
        }
        else if (mission_type == "JUPITER") {
            // Jupiter mission
            auto normal_env = RadiationSimulator::getMissionEnvironment("JUPITER");
            
            // Intense radiation environment (closer to Jupiter)
            auto intense_env = normal_env;
            intense_env.shielding_thickness_mm *= 0.7;
            intense_env.mission_name = "Jupiter (Intense Radiation)";
            
            profile.environments = {normal_env, intense_env};
            profile.environment_durations = {43200.0, 3600.0}; // 12 hours, 1 hour
            profile.transition_probabilities = {0.9, 0.1};
        }
        else {
            // Default generic mission
            auto normal_env = RadiationSimulator::getMissionEnvironment("LEO");
            normal_env.mission_name = "Generic Mission (Normal)";
            
            profile.environments = {normal_env};
            profile.environment_durations = {3600.0}; // 1 hour
            profile.transition_probabilities = {1.0};
        }
        
        return profile;
    }
};

/**
 * @brief Mission statistics tracking
 * 
 * Tracks errors and metrics during a mission simulation
 */
struct MissionStatistics {
    // Radiation event counts
    size_t total_radiation_events = 0;
    size_t single_bit_flips = 0;
    size_t multi_bit_upsets = 0;
    size_t single_event_latchups = 0;
    size_t single_event_transients = 0;
    
    // Error detection and correction
    size_t errors_detected = 0;
    size_t errors_corrected = 0;
    size_t errors_undetected = 0;
    
    // Environment statistics
    std::map<std::string, double> time_in_environment;
    std::map<std::string, size_t> events_by_environment;
    
    // Protection statistics
    size_t tmr_activations = 0;
    size_t scrubbing_cycles = 0;
    double avg_mission_critical_uptime = 1.0;  // 1.0 = 100%
    
    // Memory usage statistics
    size_t total_memory_used_bytes = 0;
    size_t protected_memory_bytes = 0;
    size_t unprotected_memory_bytes = 0;
    
    /**
     * @brief Get human-readable report of the mission statistics
     * 
     * @return Formatted string with mission statistics
     */
    std::string getReport() const {
        std::string report = "=== Mission Statistics Report ===\n\n";
        
        report += "Radiation Events:\n";
        report += "  Total events: " + std::to_string(total_radiation_events) + "\n";
        report += "  Single bit flips: " + std::to_string(single_bit_flips) + "\n";
        report += "  Multi-bit upsets: " + std::to_string(multi_bit_upsets) + "\n";
        report += "  Single event latchups: " + std::to_string(single_event_latchups) + "\n";
        report += "  Single event transients: " + std::to_string(single_event_transients) + "\n\n";
        
        report += "Error Handling:\n";
        report += "  Errors detected: " + std::to_string(errors_detected) + "\n";
        report += "  Errors corrected: " + std::to_string(errors_corrected) + "\n";
        report += "  Errors undetected: " + std::to_string(errors_undetected) + "\n";
        report += "  Error correction rate: ";
        if (errors_detected > 0) {
            report += std::to_string(static_cast<double>(errors_corrected) / errors_detected * 100.0) + "%\n\n";
        } else {
            report += "N/A (no errors detected)\n\n";
        }
        
        report += "Environment Statistics:\n";
        for (const auto& env : time_in_environment) {
            report += "  Time in " + env.first + ": " + std::to_string(env.second) + " seconds\n";
            
            auto events_it = events_by_environment.find(env.first);
            if (events_it != events_by_environment.end()) {
                report += "  Events in " + env.first + ": " + std::to_string(events_it->second);
                if (env.second > 0) {
                    report += " (" + std::to_string(events_it->second / env.second) + " events/second)\n";
                } else {
                    report += "\n";
                }
            }
        }
        report += "\n";
        
        report += "Protection Statistics:\n";
        report += "  TMR activations: " + std::to_string(tmr_activations) + "\n";
        report += "  Memory scrubbing cycles: " + std::to_string(scrubbing_cycles) + "\n";
        report += "  Mission-critical uptime: " + std::to_string(avg_mission_critical_uptime * 100.0) + "%\n\n";
        
        report += "Memory Usage:\n";
        report += "  Total memory: " + std::to_string(total_memory_used_bytes) + " bytes\n";
        report += "  Protected memory: " + std::to_string(protected_memory_bytes) + " bytes ";
        if (total_memory_used_bytes > 0) {
            report += "(" + std::to_string(static_cast<double>(protected_memory_bytes) / total_memory_used_bytes * 100.0) + "%)\n";
        } else {
            report += "(N/A)\n";
        }
        report += "  Unprotected memory: " + std::to_string(unprotected_memory_bytes) + " bytes\n";
        
        report += "\n=== End of Report ===\n";
        
        return report;
    }
};

/**
 * @brief Adaptive protection system configuration
 * 
 * Defines how the system should adapt its protection based on the environment
 */
struct AdaptiveProtectionConfig {
    bool enable_adaptive_protection = true;            ///< Enable/disable adaptive protection
    double radiation_threshold_low = 0.001;            ///< Low radiation threshold (events/s)
    double radiation_threshold_medium = 0.01;          ///< Medium radiation threshold (events/s)
    double radiation_threshold_high = 0.1;             ///< High radiation threshold (events/s)
    
    // Low radiation environment settings
    bool enable_tmr_low = false;                       ///< Enable TMR in low radiation
    bool enable_ecc_low = true;                        ///< Enable ECC in low radiation
    bool enable_scrubbing_low = false;                 ///< Enable memory scrubbing in low radiation
    std::chrono::milliseconds scrubbing_interval_low{60000}; ///< Scrubbing interval in low radiation
    
    // Medium radiation environment settings
    bool enable_tmr_medium = true;                     ///< Enable TMR in medium radiation
    bool enable_ecc_medium = true;                     ///< Enable ECC in medium radiation
    bool enable_scrubbing_medium = true;               ///< Enable memory scrubbing in medium radiation
    std::chrono::milliseconds scrubbing_interval_medium{10000}; ///< Scrubbing interval in medium radiation
    
    // High radiation environment settings
    bool enable_tmr_high = true;                       ///< Enable TMR in high radiation
    bool enable_ecc_high = true;                       ///< Enable ECC in high radiation
    bool enable_scrubbing_high = true;                 ///< Enable memory scrubbing in high radiation
    std::chrono::milliseconds scrubbing_interval_high{1000}; ///< Scrubbing interval in high radiation
    
    // Energy settings
    double energy_budget_per_hour = 100.0;             ///< Energy budget per hour (arbitrary units)
    double tmr_energy_cost = 10.0;                     ///< Energy cost of TMR per hour
    double ecc_energy_cost = 5.0;                      ///< Energy cost of ECC per hour
    double scrubbing_energy_cost_per_cycle = 0.5;      ///< Energy cost per scrubbing cycle
    
    /**
     * @brief Create standard configuration for a mission type
     * 
     * @param mission_type Type of mission ("LEO", "ISS", "MARS", etc.)
     * @return AdaptiveProtectionConfig with appropriate parameters
     */
    static AdaptiveProtectionConfig createStandard(const std::string& mission_type) {
        AdaptiveProtectionConfig config;
        
        if (mission_type == "LEO" || mission_type == "LEO_EARTH_OBSERVATION" || mission_type == "ISS") {
            // Standard LEO/ISS configuration - moderate protection with energy conservation
            config.enable_tmr_low = false;
            config.enable_ecc_low = true;
            config.enable_scrubbing_low = false;
            
            config.enable_tmr_medium = true;
            config.enable_ecc_medium = true;
            config.enable_scrubbing_medium = true;
            config.scrubbing_interval_medium = std::chrono::milliseconds(15000);
            
            config.energy_budget_per_hour = 120.0;
        }
        else if (mission_type == "GEO" || mission_type == "GEOSTATIONARY") {
            // GEO has higher radiation levels
            config.enable_tmr_low = true;
            config.enable_ecc_low = true;
            config.enable_scrubbing_low = true;
            config.scrubbing_interval_low = std::chrono::milliseconds(30000);
            
            config.energy_budget_per_hour = 150.0;
        }
        else if (mission_type == "LUNAR" || mission_type == "MOON") {
            // Lunar missions have higher radiation but may have energy constraints
            config.enable_tmr_low = false;
            config.enable_tmr_medium = true;
            
            config.energy_budget_per_hour = 100.0;
        }
        else if (mission_type == "MARS") {
            // Mars missions have energy constraints but need reliability
            config.enable_tmr_low = false;
            config.enable_scrubbing_low = false;
            
            config.energy_budget_per_hour = 80.0;
        }
        else if (mission_type == "JUPITER") {
            // Jupiter has extreme radiation - maximum protection
            config.enable_tmr_low = true;
            config.enable_ecc_low = true;
            config.enable_scrubbing_low = true;
            config.scrubbing_interval_low = std::chrono::milliseconds(10000);
            
            config.radiation_threshold_low = 0.01;
            config.radiation_threshold_medium = 0.1;
            config.radiation_threshold_high = 1.0;
            
            config.energy_budget_per_hour = 200.0;
        }
        
        return config;
    }
};

/**
 * @brief Full mission simulator
 * 
 * Simulates a space mission with configurable radiation environments,
 * protection mechanisms, and workloads
 */
class MissionSimulator {
public:
    /**
     * @brief Constructor
     * 
     * @param profile Mission profile defining radiation environments
     * @param protection_config Adaptive protection configuration
     */
    MissionSimulator(
        const MissionProfile& profile,
        const AdaptiveProtectionConfig& protection_config = AdaptiveProtectionConfig())
        : profile_(profile),
          protection_config_(protection_config),
          current_environment_index_(0),
          random_engine_(std::random_device{}()),
          stats_() {
        
        // Initialize radiation simulator with first environment
        if (!profile_.environments.empty()) {
            simulator_ = std::make_unique<RadiationSimulator>(profile_.environments[0]);
        } else {
            simulator_ = std::make_unique<RadiationSimulator>(
                RadiationSimulator::getMissionEnvironment("LEO"));
        }
        
        // Initialize statistics
        for (const auto& env : profile_.environments) {
            stats_.time_in_environment[env.mission_name] = 0.0;
            stats_.events_by_environment[env.mission_name] = 0;
        }
    }
    
    /**
     * @brief Set up mission critical memory
     * 
     * @param memory Pointer to memory region
     * @param size Size of memory region in bytes
     * @param is_protected Whether this memory should be protected
     */
    template <typename T>
    void registerMemoryRegion(T* memory, size_t size, bool is_protected = true) {
        memory_regions_.push_back({
            reinterpret_cast<void*>(memory),
            size,
            is_protected
        });
        
        stats_.total_memory_used_bytes += size;
        if (is_protected) {
            stats_.protected_memory_bytes += size;
        } else {
            stats_.unprotected_memory_bytes += size;
        }
    }
    
    /**
     * @brief Run the mission simulation
     * 
     * @param total_duration Total mission duration
     * @param time_step Time step for simulation
     * @param on_environment_change Callback when environment changes
     * @return Mission statistics
     */
    MissionStatistics runSimulation(
        std::chrono::seconds total_duration,
        std::chrono::milliseconds time_step = std::chrono::milliseconds(1000),
        std::function<void(const RadiationSimulator::EnvironmentParams&)> on_environment_change = nullptr) {
        
        std::chrono::milliseconds elapsed_time(0);
        std::chrono::milliseconds last_scrubbing_time(0);
        std::chrono::milliseconds current_env_time(0);
        
        // Setup memory scrubber if enabled
        memory::MemoryScrubber scrubber;
        for (const auto& region : memory_regions_) {
            if (region.is_protected) {
                scrubber.registerMemoryRegion(region.ptr, region.size);
            }
        }
        
        // Main simulation loop
        while (elapsed_time < total_duration) {
            // Check for environment change
            bool environment_changed = false;
            
            if (current_env_time >= std::chrono::milliseconds(static_cast<long long>(
                    profile_.environment_durations[current_environment_index_] * 1000))) {
                // Time to change environment based on duration
                environment_changed = selectNextEnvironment();
                current_env_time = std::chrono::milliseconds(0);
            } else {
                // Random chance to change environment
                std::uniform_real_distribution<double> change_dist(0.0, 1.0);
                if (change_dist(random_engine_) < 0.01) { // 1% chance per time step
                    environment_changed = selectNextEnvironment();
                    current_env_time = std::chrono::milliseconds(0);
                }
            }
            
            if (environment_changed && on_environment_change) {
                on_environment_change(simulator_->getSimulationEnvironment());
            }
            
            // Update statistics
            const auto& env = simulator_->getSimulationEnvironment();
            stats_.time_in_environment[env.mission_name] += time_step.count() / 1000.0;
            
            // Determine protection level based on current environment
            ProtectionLevel protection_level = determineProtectionLevel();
            
            // Apply protection settings
            bool tmr_enabled = false;
            bool ecc_enabled = false;
            bool scrubbing_enabled = false;
            std::chrono::milliseconds scrubbing_interval(0);
            
            switch (protection_level) {
                case ProtectionLevel::LOW:
                    tmr_enabled = protection_config_.enable_tmr_low;
                    ecc_enabled = protection_config_.enable_ecc_low;
                    scrubbing_enabled = protection_config_.enable_scrubbing_low;
                    scrubbing_interval = protection_config_.scrubbing_interval_low;
                    break;
                case ProtectionLevel::MEDIUM:
                    tmr_enabled = protection_config_.enable_tmr_medium;
                    ecc_enabled = protection_config_.enable_ecc_medium;
                    scrubbing_enabled = protection_config_.enable_scrubbing_medium;
                    scrubbing_interval = protection_config_.scrubbing_interval_medium;
                    break;
                case ProtectionLevel::HIGH:
                    tmr_enabled = protection_config_.enable_tmr_high;
                    ecc_enabled = protection_config_.enable_ecc_high;
                    scrubbing_enabled = protection_config_.enable_scrubbing_high;
                    scrubbing_interval = protection_config_.scrubbing_interval_high;
                    break;
            }
            
            // Simulate radiation effects on all memory regions
            for (const auto& region : memory_regions_) {
                auto events = simulator_->simulateEffects(
                    region.ptr, region.size, time_step);
                
                // Process radiation events
                for (const auto& event : events) {
                    stats_.total_radiation_events++;
                    stats_.events_by_environment[env.mission_name]++;
                    
                    // Record by type
                    switch (event.type) {
                        case RadiationSimulator::RadiationEffectType::SINGLE_BIT_FLIP:
                            stats_.single_bit_flips++;
                            break;
                        case RadiationSimulator::RadiationEffectType::MULTI_BIT_UPSET:
                            stats_.multi_bit_upsets++;
                            break;
                        case RadiationSimulator::RadiationEffectType::SINGLE_EVENT_LATCHUP:
                            stats_.single_event_latchups++;
                            break;
                        case RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT:
                            stats_.single_event_transients++;
                            break;
                    }
                    
                    // Check if error is detected
                    bool error_detected = false;
                    bool error_corrected = false;
                    
                    if (region.is_protected) {
                        // Simulate TMR error detection
                        if (tmr_enabled) {
                            // 99% chance to detect error with TMR
                            std::uniform_real_distribution<double> detect_dist(0.0, 1.0);
                            if (detect_dist(random_engine_) < 0.99) {
                                error_detected = true;
                                stats_.tmr_activations++;
                                
                                // 95% chance to correct error with TMR
                                if (detect_dist(random_engine_) < 0.95) {
                                    error_corrected = true;
                                }
                            }
                        }
                        
                        // Simulate ECC error detection (if not already detected)
                        if (!error_detected && ecc_enabled) {
                            // ECC can detect most single bit errors and some multi-bit
                            double detection_prob = 0.0;
                            if (event.type == RadiationSimulator::RadiationEffectType::SINGLE_BIT_FLIP) {
                                detection_prob = 0.98; // 98% for single bit
                            } else if (event.type == RadiationSimulator::RadiationEffectType::MULTI_BIT_UPSET) {
                                detection_prob = 0.75; // 75% for multi-bit
                            } else {
                                detection_prob = 0.5;  // 50% for other types
                            }
                            
                            std::uniform_real_distribution<double> detect_dist(0.0, 1.0);
                            if (detect_dist(random_engine_) < detection_prob) {
                                error_detected = true;
                                
                                // ECC can correct single bit errors but not multi-bit
                                if (event.type == RadiationSimulator::RadiationEffectType::SINGLE_BIT_FLIP) {
                                    if (detect_dist(random_engine_) < 0.95) {
                                        error_corrected = true;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Update statistics
                    if (error_detected) {
                        stats_.errors_detected++;
                        if (error_corrected) {
                            stats_.errors_corrected++;
                        }
                    } else {
                        stats_.errors_undetected++;
                    }
                }
            }
            
            // Perform memory scrubbing if enabled
            if (scrubbing_enabled && (elapsed_time - last_scrubbing_time) >= scrubbing_interval) {
                scrubber.scrubMemory();
                stats_.scrubbing_cycles++;
                last_scrubbing_time = elapsed_time;
            }
            
            // Update simulation time
            elapsed_time += time_step;
            current_env_time += time_step;
        }
        
        // Calculate mission-critical uptime
        stats_.avg_mission_critical_uptime = static_cast<double>(stats_.errors_corrected) / 
            (stats_.errors_detected > 0 ? stats_.errors_detected : 1);
        
        return stats_;
    }
    
    /**
     * @brief Get the current mission statistics
     * 
     * @return Current mission statistics
     */
    const MissionStatistics& getStatistics() const {
        return stats_;
    }
    
    /**
     * @brief Get the current radiation simulator
     * 
     * @return Reference to the radiation simulator
     */
    RadiationSimulator& getRadiationSimulator() {
        return *simulator_;
    }
    
    /**
     * @brief Get the current mission profile
     * 
     * @return Mission profile
     */
    const MissionProfile& getMissionProfile() const {
        return profile_;
    }
    
    /**
     * @brief Get the protection configuration
     * 
     * @return Adaptive protection configuration
     */
    const AdaptiveProtectionConfig& getProtectionConfig() const {
        return protection_config_;
    }
    
private:
    // Protection levels
    enum class ProtectionLevel {
        LOW,
        MEDIUM,
        HIGH
    };
    
    // Memory region structure
    struct MemoryRegion {
        void* ptr;
        size_t size;
        bool is_protected;
    };
    
    MissionProfile profile_;
    AdaptiveProtectionConfig protection_config_;
    std::unique_ptr<RadiationSimulator> simulator_;
    size_t current_environment_index_;
    std::default_random_engine random_engine_;
    MissionStatistics stats_;
    std::vector<MemoryRegion> memory_regions_;
    
    /**
     * @brief Select the next environment based on transition probabilities
     * 
     * @return True if environment changed, false otherwise
     */
    bool selectNextEnvironment() {
        if (profile_.environments.size() <= 1) {
            return false;
        }
        
        size_t old_index = current_environment_index_;
        
        // Choose next environment based on transition probabilities
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double val = dist(random_engine_);
        
        double cumulative_prob = 0.0;
        for (size_t i = 0; i < profile_.transition_probabilities.size(); ++i) {
            cumulative_prob += profile_.transition_probabilities[i];
            if (val < cumulative_prob) {
                current_environment_index_ = i;
                break;
            }
        }
        
        // Update simulator with new environment
        simulator_->updateEnvironment(profile_.environments[current_environment_index_]);
        
        return old_index != current_environment_index_;
    }
    
    /**
     * @brief Determine protection level based on radiation environment
     * 
     * @return Protection level (LOW, MEDIUM, HIGH)
     */
    ProtectionLevel determineProtectionLevel() {
        if (!protection_config_.enable_adaptive_protection) {
            return ProtectionLevel::MEDIUM; // Default to medium if not adaptive
        }
        
        double rate = simulator_->getEventRates().total_rate;
        
        if (rate >= protection_config_.radiation_threshold_high) {
            return ProtectionLevel::HIGH;
        } else if (rate >= protection_config_.radiation_threshold_medium) {
            return ProtectionLevel::MEDIUM;
        } else {
            return ProtectionLevel::LOW;
        }
    }
};

} // namespace testing
} // namespace rad_ml 