#pragma once

#include <string>
#include <memory>
#include <unordered_map>

#include "../common/types.hpp"

namespace rad_ml {
namespace radiation {

/**
 * @brief Enumeration of pre-defined radiation environments
 */
enum class EnvironmentType {
    GROUND_LEVEL,        ///< Ground level (sea level) radiation
    AVIONIC,             ///< Commercial aircraft altitude (35,000 ft)
    LOW_EARTH_ORBIT,     ///< Low Earth orbit (400-1000 km)
    MEDIUM_EARTH_ORBIT,  ///< Medium Earth orbit (2000-35,786 km)
    GEOSTATIONARY_ORBIT, ///< Geostationary orbit (35,786 km)
    LUNAR,               ///< Lunar surface
    MARS,                ///< Mars surface
    JUPITER,             ///< Jupiter orbit
    SOLAR_FLARE,         ///< During solar flare
    CUSTOM               ///< Custom environment
};

/**
 * @brief Base class for radiation environments
 */
class Environment {
public:
    /**
     * @brief Constructor
     * 
     * @param type Environment type
     * @param name Environment name
     */
    Environment(
        EnvironmentType type = EnvironmentType::GROUND_LEVEL,
        const std::string& name = "Ground Level"
    ) : type_(type), name_(name) {}
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Environment() = default;
    
    /**
     * @brief Get the environment type
     * 
     * @return Environment type
     */
    EnvironmentType getType() const {
        return type_;
    }
    
    /**
     * @brief Get the environment name
     * 
     * @return Environment name
     */
    const std::string& getName() const {
        return name_;
    }
    
    /**
     * @brief Set the SEU flux (number of SEUs per cm² per second)
     * 
     * @param flux SEU flux value
     */
    void setSEUFlux(float flux) {
        seu_flux_ = flux;
    }
    
    /**
     * @brief Get the SEU flux
     * 
     * @return SEU flux value
     */
    float getSEUFlux() const {
        return seu_flux_;
    }
    
    /**
     * @brief Set the SEU cross-section (cm² per bit)
     * 
     * @param cross_section Cross-section value
     */
    void setSEUCrossSection(float cross_section) {
        seu_cross_section_ = cross_section;
    }
    
    /**
     * @brief Get the SEU cross-section
     * 
     * @return SEU cross-section value
     */
    float getSEUCrossSection() const {
        return seu_cross_section_;
    }
    
    /**
     * @brief Set a custom property
     * 
     * @param key Property key
     * @param value Property value
     */
    void setProperty(const std::string& key, float value) {
        properties_[key] = value;
    }
    
    /**
     * @brief Get a custom property
     * 
     * @param key Property key
     * @param default_value Default value if property doesn't exist
     * @return Property value
     */
    float getProperty(const std::string& key, float default_value = 0.0f) const {
        auto it = properties_.find(key);
        if (it != properties_.end()) {
            return it->second;
        }
        return default_value;
    }
    
    /**
     * @brief Create a pre-defined environment
     * 
     * @param type Environment type
     * @return Shared pointer to the environment
     */
    static std::shared_ptr<Environment> createEnvironment(EnvironmentType type) {
        auto env = std::make_shared<Environment>(type);
        
        // Set default values based on environment type
        switch (type) {
            case EnvironmentType::GROUND_LEVEL:
                env->name_ = "Ground Level";
                env->seu_flux_ = 1e-14f;  // Very low at ground level
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::AVIONIC:
                env->name_ = "Avionic (35,000 ft)";
                env->seu_flux_ = 1e-10f;  // Higher due to altitude
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::LOW_EARTH_ORBIT:
                env->name_ = "Low Earth Orbit";
                env->seu_flux_ = 1e-7f;  // Significant in LEO
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::MEDIUM_EARTH_ORBIT:
                env->name_ = "Medium Earth Orbit";
                env->seu_flux_ = 5e-7f;  // Higher than LEO
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::GEOSTATIONARY_ORBIT:
                env->name_ = "Geostationary Orbit";
                env->seu_flux_ = 1e-6f;  // High radiation
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::LUNAR:
                env->name_ = "Lunar Surface";
                env->seu_flux_ = 2e-7f;  // No magnetosphere protection
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::MARS:
                env->name_ = "Mars Surface";
                env->seu_flux_ = 1e-7f;  // Thin atmosphere
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::JUPITER:
                env->name_ = "Jupiter Orbit";
                env->seu_flux_ = 5e-6f;  // Very high radiation
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::SOLAR_FLARE:
                env->name_ = "Solar Flare";
                env->seu_flux_ = 1e-5f;  // Extreme conditions
                env->seu_cross_section_ = 1e-14f;
                break;
                
            case EnvironmentType::CUSTOM:
            default:
                env->name_ = "Custom Environment";
                env->seu_flux_ = 0.0f;  // Must be set by the user
                env->seu_cross_section_ = 1e-14f;
                break;
        }
        
        return env;
    }

protected:
    EnvironmentType type_;
    std::string name_;
    float seu_flux_ = 0.0f;  // SEUs per cm² per second
    float seu_cross_section_ = 1e-14f;  // cm² per bit
    std::unordered_map<std::string, float> properties_;
};

} // namespace radiation
} // namespace rad_ml 