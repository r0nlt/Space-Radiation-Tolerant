#pragma once

#include <rad_ml/core/radiation/adaptive_protection.hpp>
#include <rad_ml/testing/radiation_simulator.hpp>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace rad_ml {
namespace mission {

/**
 * @brief Mission profile configuration system
 * 
 * This class provides pre-configured settings for different space missions,
 * allowing the framework to be optimized for specific radiation environments.
 */
class MissionProfile {
public:
    /// Known mission types
    enum class MissionType {
        LEO_EARTH_OBSERVATION,   ///< Low Earth Orbit imaging satellite
        MEDIUM_EARTH_ORBIT,      ///< MEO constellation (e.g., GPS)
        GEOSTATIONARY,           ///< GEO communications satellite
        LUNAR_ORBIT,             ///< Lunar orbit mission
        LUNAR_SURFACE,           ///< Moon surface operations
        MARS_TRANSIT,            ///< Earth-Mars transit
        MARS_ORBIT,              ///< Mars orbit operations
        MARS_SURFACE,            ///< Mars surface operations
        DEEP_SPACE               ///< Beyond Mars missions
    };
    
    /// Radiation environment characteristics
    struct RadiationEnvironment {
        double total_dose_annual_krad;       ///< Expected annual dose
        double peak_flux_particles_cm2_s;    ///< Peak particle flux
        double sea_likelihood;               ///< Likelihood of South Atlantic Anomaly encounters
        double solar_event_sensitivity;      ///< Sensitivity to solar events
        double galactic_cosmic_ray_exposure; ///< Exposure to GCRs
    };
    
    /// Hardware configuration
    struct HardwareConfig {
        std::string processor_type;        ///< E.g., "RAD750", "LEON4", "Versal AI Core"
        double process_node_nm;            ///< Process technology node
        bool has_ecc_memory;               ///< Whether ECC memory is available
        bool has_hardware_tmr;             ///< Whether hardware TMR is available
        double available_memory_mb;        ///< Available memory
        double available_compute_gflops;   ///< Available compute capacity
        double power_budget_w;             ///< Power budget
    };
    
    /// Software configuration
    struct SoftwareConfig {
        uint32_t scrubbing_interval_ms;    ///< Memory scrubbing interval
        uint32_t checkpoint_interval_s;    ///< Checkpoint creation interval
        bool enable_recovery_mode;         ///< Whether recovery mode is enabled
        uint32_t redundancy_level;         ///< Level of redundancy (1-3)
        bool enable_fallback_models;       ///< Whether fallback ML models are enabled
        bool use_quantized_models;         ///< Whether to use quantized models
        uint32_t quantization_bits;        ///< Bits for quantization (if enabled)
    };
    
    /// Telemetry configuration
    struct TelemetryConfig {
        bool enable_logging;               ///< Whether to log telemetry
        std::string log_file_path;         ///< Path to log file
        uint32_t log_interval_ms;          ///< How often to log
        bool log_error_details;            ///< Whether to log detailed error info
        uint32_t log_retention_days;       ///< How long to keep logs
    };

    /**
     * @brief Constructor with mission type
     * 
     * @param mission_type Type of mission
     */
    explicit MissionProfile(MissionType mission_type) 
        : mission_type_(mission_type) {
        // Initialize with defaults for the mission type
        initializeDefaults();
    }
    
    /**
     * @brief Constructor with mission name
     * 
     * @param mission_name Name of the mission (e.g., "LEO", "MARS_SURFACE")
     */
    explicit MissionProfile(const std::string& mission_name) {
        if (mission_name == "LEO" || mission_name == "LEO_EARTH_OBSERVATION") {
            mission_type_ = MissionType::LEO_EARTH_OBSERVATION;
        } else if (mission_name == "MEO" || mission_name == "MEDIUM_EARTH_ORBIT") {
            mission_type_ = MissionType::MEDIUM_EARTH_ORBIT;
        } else if (mission_name == "GEO" || mission_name == "GEOSTATIONARY") {
            mission_type_ = MissionType::GEOSTATIONARY;
        } else if (mission_name == "LUNAR_ORBIT") {
            mission_type_ = MissionType::LUNAR_ORBIT;
        } else if (mission_name == "LUNAR_SURFACE" || mission_name == "MOON") {
            mission_type_ = MissionType::LUNAR_SURFACE;
        } else if (mission_name == "MARS_TRANSIT") {
            mission_type_ = MissionType::MARS_TRANSIT;
        } else if (mission_name == "MARS_ORBIT") {
            mission_type_ = MissionType::MARS_ORBIT;
        } else if (mission_name == "MARS_SURFACE" || mission_name == "MARS") {
            mission_type_ = MissionType::MARS_SURFACE;
        } else if (mission_name == "DEEP_SPACE" || mission_name == "JUPITER") {
            mission_type_ = MissionType::DEEP_SPACE;
        } else {
            // Default to LEO
            mission_type_ = MissionType::LEO_EARTH_OBSERVATION;
        }
        
        initializeDefaults();
    }
    
    /**
     * @brief Get the radiation environment for this mission
     * 
     * @return Radiation environment characteristics
     */
    const RadiationEnvironment& getRadiationEnvironment() const {
        return radiation_env_;
    }
    
    /**
     * @brief Get the hardware configuration for this mission
     * 
     * @return Hardware configuration
     */
    const HardwareConfig& getHardwareConfig() const {
        return hardware_config_;
    }
    
    /**
     * @brief Get the software configuration for this mission
     * 
     * @return Software configuration
     */
    const SoftwareConfig& getSoftwareConfig() const {
        return software_config_;
    }
    
    /**
     * @brief Get the telemetry configuration for this mission
     * 
     * @return Telemetry configuration
     */
    const TelemetryConfig& getTelemetryConfig() const {
        return telemetry_config_;
    }
    
    /**
     * @brief Get a simulation environment for this mission
     * 
     * @return Radiation simulation environment parameters
     */
    testing::RadiationSimulator::EnvironmentParams getSimulationEnvironment() const {
        testing::RadiationSimulator::EnvironmentParams params;
        
        switch (mission_type_) {
            case MissionType::LEO_EARTH_OBSERVATION:
                params = testing::RadiationSimulator::getMissionEnvironment("LEO");
                break;
            case MissionType::MEDIUM_EARTH_ORBIT:
                params = testing::RadiationSimulator::getMissionEnvironment("MEO");
                params.altitude_km = 20000.0;
                break;
            case MissionType::GEOSTATIONARY:
                params = testing::RadiationSimulator::getMissionEnvironment("GEO");
                break;
            case MissionType::LUNAR_ORBIT:
            case MissionType::LUNAR_SURFACE:
                params = testing::RadiationSimulator::getMissionEnvironment("LUNAR");
                break;
            case MissionType::MARS_TRANSIT:
            case MissionType::MARS_ORBIT:
            case MissionType::MARS_SURFACE:
                params = testing::RadiationSimulator::getMissionEnvironment("MARS");
                break;
            case MissionType::DEEP_SPACE:
                params = testing::RadiationSimulator::getMissionEnvironment("JUPITER");
                break;
        }
        
        return params;
    }
    
    /**
     * @brief Get the initial protection level for this mission
     * 
     * @return Appropriate protection level for this mission
     */
    core::radiation::AdaptiveProtection::ProtectionLevel getInitialProtectionLevel() const {
        using ProtectionLevel = core::radiation::AdaptiveProtection::ProtectionLevel;
        
        switch (mission_type_) {
            case MissionType::LEO_EARTH_OBSERVATION:
                return ProtectionLevel::STANDARD;
                
            case MissionType::MEDIUM_EARTH_ORBIT:
                return ProtectionLevel::STANDARD;
                
            case MissionType::GEOSTATIONARY:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::LUNAR_ORBIT:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::LUNAR_SURFACE:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::MARS_TRANSIT:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::MARS_ORBIT:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::MARS_SURFACE:
                return ProtectionLevel::ENHANCED;
                
            case MissionType::DEEP_SPACE:
                return ProtectionLevel::MAXIMUM;
                
            default:
                return ProtectionLevel::STANDARD;
        }
    }
    
    /**
     * @brief Configure a neural network based on mission profile
     * 
     * @tparam NetworkType Type of the neural network
     * @param network Network to configure
     */
    template <typename NetworkType>
    void configureNetwork(NetworkType& network) const {
        // This is a template for how a network would be configured
        // The actual implementation depends on the NetworkType interface
        
        // Set protection level based on mission
        network.setInitialProtectionLevel(getInitialProtectionLevel());
        
        // Configure scrubbing based on mission
        network.setScrubInterval(software_config_.scrubbing_interval_ms);
        
        // Set redundancy level
        network.setRedundancyLevel(software_config_.redundancy_level);
        
        // Configure checkpoint/recovery if enabled
        if (software_config_.enable_recovery_mode) {
            network.enableRecovery(software_config_.checkpoint_interval_s);
        }
        
        // Configure quantization if enabled
        if (software_config_.use_quantized_models) {
            network.setQuantizationBits(software_config_.quantization_bits);
        }
    }
    
    /**
     * @brief Get human-readable description of this mission profile
     * 
     * @return Description string
     */
    std::string getDescription() const {
        std::string desc = "Mission Profile: ";
        
        switch (mission_type_) {
            case MissionType::LEO_EARTH_OBSERVATION:
                desc += "Low Earth Orbit Earth Observation";
                break;
            case MissionType::MEDIUM_EARTH_ORBIT:
                desc += "Medium Earth Orbit";
                break;
            case MissionType::GEOSTATIONARY:
                desc += "Geostationary Orbit";
                break;
            case MissionType::LUNAR_ORBIT:
                desc += "Lunar Orbit";
                break;
            case MissionType::LUNAR_SURFACE:
                desc += "Lunar Surface";
                break;
            case MissionType::MARS_TRANSIT:
                desc += "Mars Transit";
                break;
            case MissionType::MARS_ORBIT:
                desc += "Mars Orbit";
                break;
            case MissionType::MARS_SURFACE:
                desc += "Mars Surface";
                break;
            case MissionType::DEEP_SPACE:
                desc += "Deep Space";
                break;
        }
        
        desc += "\n\nRadiation Environment:";
        desc += "\n  Annual Dose: " + std::to_string(radiation_env_.total_dose_annual_krad) + " krad";
        desc += "\n  Peak Flux: " + std::to_string(radiation_env_.peak_flux_particles_cm2_s) + " particles/cm²/s";
        desc += "\n  GCR Exposure: " + std::to_string(radiation_env_.galactic_cosmic_ray_exposure);
        
        desc += "\n\nHardware Configuration:";
        desc += "\n  Processor: " + hardware_config_.processor_type;
        desc += "\n  Process Node: " + std::to_string(hardware_config_.process_node_nm) + " nm";
        desc += "\n  ECC Memory: " + std::string(hardware_config_.has_ecc_memory ? "Yes" : "No");
        desc += "\n  Hardware TMR: " + std::string(hardware_config_.has_hardware_tmr ? "Yes" : "No");
        desc += "\n  Memory: " + std::to_string(hardware_config_.available_memory_mb) + " MB";
        desc += "\n  Compute: " + std::to_string(hardware_config_.available_compute_gflops) + " GFLOPS";
        desc += "\n  Power Budget: " + std::to_string(hardware_config_.power_budget_w) + " W";
        
        desc += "\n\nSoftware Configuration:";
        desc += "\n  Scrubbing Interval: " + std::to_string(software_config_.scrubbing_interval_ms) + " ms";
        desc += "\n  Redundancy Level: " + std::to_string(software_config_.redundancy_level);
        desc += "\n  Recovery Mode: " + std::string(software_config_.enable_recovery_mode ? "Enabled" : "Disabled");
        desc += "\n  Use Quantized Models: " + std::string(software_config_.use_quantized_models ? "Yes" : "No");
        if (software_config_.use_quantized_models) {
            desc += "\n  Quantization Bits: " + std::to_string(software_config_.quantization_bits);
        }
        
        return desc;
    }
    
private:
    MissionType mission_type_;
    RadiationEnvironment radiation_env_;
    HardwareConfig hardware_config_;
    SoftwareConfig software_config_;
    TelemetryConfig telemetry_config_;
    
    /**
     * @brief Initialize default settings based on mission type
     */
    void initializeDefaults() {
        switch (mission_type_) {
            case MissionType::LEO_EARTH_OBSERVATION:
                initializeForLEO();
                break;
            case MissionType::MEDIUM_EARTH_ORBIT:
                initializeForMEO();
                break;
            case MissionType::GEOSTATIONARY:
                initializeForGEO();
                break;
            case MissionType::LUNAR_ORBIT:
                initializeForLunarOrbit();
                break;
            case MissionType::LUNAR_SURFACE:
                initializeForLunarSurface();
                break;
            case MissionType::MARS_TRANSIT:
                initializeForMarsTransit();
                break;
            case MissionType::MARS_ORBIT:
                initializeForMarsOrbit();
                break;
            case MissionType::MARS_SURFACE:
                initializeForMarsSurface();
                break;
            case MissionType::DEEP_SPACE:
                initializeForDeepSpace();
                break;
        }
    }
    
    // Initialize for specific mission types
    void initializeForLEO() {
        // Set radiation environment for LEO
        radiation_env_.total_dose_annual_krad = 10.0;
        radiation_env_.peak_flux_particles_cm2_s = 1e3;
        radiation_env_.sea_likelihood = 0.2;
        radiation_env_.solar_event_sensitivity = 0.5;
        radiation_env_.galactic_cosmic_ray_exposure = 0.3;
        
        // Set hardware defaults for LEO
        hardware_config_.processor_type = "RAD750";
        hardware_config_.process_node_nm = 180.0;
        hardware_config_.has_ecc_memory = true;
        hardware_config_.has_hardware_tmr = false;
        hardware_config_.available_memory_mb = 256.0;
        hardware_config_.available_compute_gflops = 0.5;
        hardware_config_.power_budget_w = 15.0;
        
        // Set software defaults for LEO
        software_config_.scrubbing_interval_ms = 1000;
        software_config_.checkpoint_interval_s = 300;
        software_config_.enable_recovery_mode = true;
        software_config_.redundancy_level = 2;
        software_config_.enable_fallback_models = true;
        software_config_.use_quantized_models = false;
        software_config_.quantization_bits = 8;
        
        // Set telemetry defaults for LEO
        telemetry_config_.enable_logging = true;
        telemetry_config_.log_file_path = "leo_mission_log.txt";
        telemetry_config_.log_interval_ms = 60000;  // Every minute
        telemetry_config_.log_error_details = true;
        telemetry_config_.log_retention_days = 30;
    }
    
    void initializeForMEO() {
        // Similar to LEO but with higher radiation exposure
        initializeForLEO();
        
        radiation_env_.total_dose_annual_krad = 20.0;
        radiation_env_.peak_flux_particles_cm2_s = 2e3;
        
        software_config_.scrubbing_interval_ms = 500;
        software_config_.redundancy_level = 3;
    }
    
    void initializeForGEO() {
        // Higher radiation, different processor, etc.
        initializeForLEO();
        
        radiation_env_.total_dose_annual_krad = 30.0;
        radiation_env_.peak_flux_particles_cm2_s = 5e3;
        radiation_env_.sea_likelihood = 0.0;
        
        hardware_config_.processor_type = "LEON4";
        hardware_config_.available_memory_mb = 512.0;
        hardware_config_.available_compute_gflops = 1.0;
        
        software_config_.scrubbing_interval_ms = 200;
        software_config_.redundancy_level = 3;
        software_config_.use_quantized_models = true;
    }
    
    void initializeForLunarOrbit() {
        // Configuration for lunar orbit
        initializeForGEO();  // Similar to GEO as baseline
        
        radiation_env_.total_dose_annual_krad = 25.0;
        radiation_env_.sea_likelihood = 0.0;  // N/A for Lunar
        radiation_env_.galactic_cosmic_ray_exposure = 0.6;
        
        telemetry_config_.log_file_path = "lunar_orbit_log.txt";
    }
    
    void initializeForLunarSurface() {
        // Configuration for lunar surface
        initializeForLunarOrbit();
        
        radiation_env_.total_dose_annual_krad = 20.0;  // Slightly lower due to planet shielding
        
        hardware_config_.available_memory_mb = 1024.0;  // More memory for surface operations
        hardware_config_.available_compute_gflops = 2.0;
        hardware_config_.power_budget_w = 25.0;
        
        telemetry_config_.log_file_path = "lunar_surface_log.txt";
    }
    
    void initializeForMarsTransit() {
        // Configuration for Mars transit
        initializeForDeepSpace();  // Use Deep Space as baseline
        
        radiation_env_.total_dose_annual_krad = 50.0;
        
        telemetry_config_.log_file_path = "mars_transit_log.txt";
    }
    
    void initializeForMarsOrbit() {
        // Configuration for Mars orbit
        initializeForMarsTransit();
        
        radiation_env_.total_dose_annual_krad = 35.0;  // Lower than transit
        
        telemetry_config_.log_file_path = "mars_orbit_log.txt";
    }
    
    void initializeForMarsSurface() {
        // Configuration for Mars surface
        initializeForMarsOrbit();
        
        radiation_env_.total_dose_annual_krad = 25.0;  // Lower due to atmosphere
        
        hardware_config_.processor_type = "Versal AI Core";
        hardware_config_.process_node_nm = 20.0;
        hardware_config_.available_memory_mb = 2048.0;
        hardware_config_.available_compute_gflops = 5.0;
        
        telemetry_config_.log_file_path = "mars_surface_log.txt";
    }
    
    void initializeForDeepSpace() {
        // Configuration for deep space missions
        // Highest radiation protection
        initializeForGEO();  // Start from GEO baseline
        
        radiation_env_.total_dose_annual_krad = 100.0;
        radiation_env_.peak_flux_particles_cm2_s = 1e4;
        radiation_env_.sea_likelihood = 0.0;  // N/A for deep space
        radiation_env_.solar_event_sensitivity = 1.0;
        radiation_env_.galactic_cosmic_ray_exposure = 1.0;
        
        hardware_config_.processor_type = "LEON4";
        hardware_config_.process_node_nm = 65.0;  // More radiation-tolerant process
        hardware_config_.has_hardware_tmr = true;
        hardware_config_.available_memory_mb = 4096.0;
        hardware_config_.available_compute_gflops = 3.0;
        hardware_config_.power_budget_w = 35.0;
        
        software_config_.scrubbing_interval_ms = 100;
        software_config_.checkpoint_interval_s = 60;
        software_config_.redundancy_level = 3;
        software_config_.use_quantized_models = true;
        software_config_.quantization_bits = 4;  // Very conservative
        
        telemetry_config_.log_file_path = "deep_space_log.txt";
        telemetry_config_.log_interval_ms = 10000;  // More frequent logging
    }
};

} // namespace mission
} // namespace rad_ml 