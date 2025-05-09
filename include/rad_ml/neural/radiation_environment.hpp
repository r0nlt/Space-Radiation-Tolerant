/**
 * @file radiation_environment.hpp
 * @brief Space radiation environment model for neural network testing
 * 
 * This file defines a space radiation environment model that can be used
 * to simulate different radiation conditions in various orbits and space
 * environments for testing neural network radiation hardening techniques.
 */

#ifndef RAD_ML_NEURAL_RADIATION_ENVIRONMENT_HPP
#define RAD_ML_NEURAL_RADIATION_ENVIRONMENT_HPP

#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <array>
#include <memory>
#include <functional>
#include <limits>
#include <algorithm>

namespace rad_ml {
namespace neural {

/**
 * @brief Struct to store radiation flux measurements
 */
struct RadiationFlux {
    double electron_flux;  ///< Electron flux in particles/cm²/s
    double proton_flux;    ///< Proton flux in particles/cm²/s
    double heavy_ion_flux; ///< Heavy ion flux in particles/cm²/s
    double total_dose_rate;///< Total ionizing dose rate in rad/hour
    
    /**
     * @brief Default constructor
     */
    RadiationFlux()
        : electron_flux(0.0), proton_flux(0.0), 
          heavy_ion_flux(0.0), total_dose_rate(0.0) {}
    
    /**
     * @brief Full constructor
     */
    RadiationFlux(double e_flux, double p_flux, double hi_flux, double dose)
        : electron_flux(e_flux), proton_flux(p_flux), 
          heavy_ion_flux(hi_flux), total_dose_rate(dose) {}
          
    /**
     * @brief Get total particle flux
     * 
     * @return Total particle flux in particles/cm²/s
     */
    double getTotalFlux() const {
        return electron_flux + proton_flux + heavy_ion_flux;
    }
    
    /**
     * @brief Scale all flux values
     * 
     * @param factor Scaling factor
     */
    void scale(double factor) {
        electron_flux *= factor;
        proton_flux *= factor;
        heavy_ion_flux *= factor;
        total_dose_rate *= factor;
    }
    
    /**
     * @brief Combine with another flux measurement
     * 
     * @param other Other flux measurement
     * @return Combined flux
     */
    RadiationFlux operator+(const RadiationFlux& other) const {
        return RadiationFlux(
            electron_flux + other.electron_flux,
            proton_flux + other.proton_flux,
            heavy_ion_flux + other.heavy_ion_flux,
            total_dose_rate + other.total_dose_rate
        );
    }
};

/**
 * @brief Struct to represent orbital position
 */
struct OrbitalPosition {
    double latitude;   ///< Latitude in degrees (-90 to 90)
    double longitude;  ///< Longitude in degrees (-180 to 180)
    double altitude;   ///< Altitude in km above Earth's surface
    
    /**
     * @brief Convert altitude to radius from Earth's center
     * 
     * @return Radius in km
     */
    double getRadius() const {
        const double earth_radius = 6371.0; // Earth radius in km
        return earth_radius + altitude;
    }
    
    /**
     * @brief Calculate distance from another position
     * 
     * @param other Other orbital position
     * @return Distance in km
     */
    double distanceTo(const OrbitalPosition& other) const {
        // Haversine formula for great-circle distance
        const double earth_radius = 6371.0; // Earth radius in km
        const double lat1_rad = latitude * M_PI / 180.0;
        const double lat2_rad = other.latitude * M_PI / 180.0;
        const double delta_lat = (other.latitude - latitude) * M_PI / 180.0;
        const double delta_lon = (other.longitude - longitude) * M_PI / 180.0;
        
        const double a = std::sin(delta_lat/2) * std::sin(delta_lat/2) +
                         std::cos(lat1_rad) * std::cos(lat2_rad) *
                         std::sin(delta_lon/2) * std::sin(delta_lon/2);
        const double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
        
        // Surface distance
        double surface_distance = earth_radius * c;
        
        // Add altitude component
        double altitude_diff = std::abs(altitude - other.altitude);
        
        return std::sqrt(surface_distance*surface_distance + altitude_diff*altitude_diff);
    }
};

/**
 * @brief Enumeration of space mission profiles
 */
enum class SpaceMission {
    LEO_EQUATORIAL,    ///< Low Earth Orbit, low inclination
    LEO_POLAR,         ///< Low Earth Orbit, polar
    MEO,               ///< Medium Earth Orbit
    GEO,               ///< Geostationary Orbit
    LUNAR,             ///< Lunar mission
    MARS,              ///< Mars mission
    JUPITER,           ///< Jupiter mission
    SOLAR_PROBE        ///< Solar probe mission
};

/**
 * @brief Model of radiation environment for various space missions
 */
class RadiationEnvironment {
public:
    /**
     * @brief Constructor with mission profile
     * 
     * @param mission Space mission profile
     */
    explicit RadiationEnvironment(SpaceMission mission)
        : mission_(mission), 
          solar_activity_(0.5), // Medium solar activity by default
          saa_model_(createSAAModel()),
          van_allen_model_(createVanAllenModel()),
          galactic_cosmic_ray_model_(createGCRModel()),
          solar_particle_model_(createSPEModel())
    {
        // Initialize shielding effectiveness (default aluminum 5mm)
        shielding_thickness_ = 5.0; // mm
        updateShieldingEffectiveness();
    }
    
    /**
     * @brief Get the name of the mission
     * 
     * @return Mission name as string
     */
    std::string getMissionName() const {
        switch (mission_) {
            case SpaceMission::LEO_EQUATORIAL: return "Low Earth Orbit (Equatorial)";
            case SpaceMission::LEO_POLAR: return "Low Earth Orbit (Polar)";
            case SpaceMission::MEO: return "Medium Earth Orbit";
            case SpaceMission::GEO: return "Geostationary Orbit";
            case SpaceMission::LUNAR: return "Lunar Mission";
            case SpaceMission::MARS: return "Mars Mission";
            case SpaceMission::JUPITER: return "Jupiter Mission";
            case SpaceMission::SOLAR_PROBE: return "Solar Probe";
            default: return "Unknown Mission";
        }
    }
    
    /**
     * @brief Set the level of solar activity
     * 
     * @param activity Solar activity level (0.0-1.0, where 1.0 is solar maximum)
     */
    void setSolarActivity(double activity) {
        solar_activity_ = std::clamp(activity, 0.0, 1.0);
    }
    
    /**
     * @brief Set spacecraft shielding thickness
     * 
     * @param thickness_mm Shielding thickness in mm
     * @param material Material type (0=aluminum, 1=polyethylene, 2=tungsten)
     */
    void setShielding(double thickness_mm, int material = 0) {
        shielding_thickness_ = thickness_mm;
        shielding_material_ = material;
        updateShieldingEffectiveness();
    }
    
    /**
     * @brief Calculate radiation flux at a specific position
     * 
     * @param position Orbital position
     * @return Radiation flux measurements
     */
    RadiationFlux calculateRadiationFlux(const OrbitalPosition& position) const {
        // Initialize with base GCR level
        RadiationFlux flux = galactic_cosmic_ray_model_(position);
        
        // Add mission-specific environments
        switch (mission_) {
            case SpaceMission::LEO_EQUATORIAL:
            case SpaceMission::LEO_POLAR: {
                // Consider Van Allen belts and SAA
                flux = flux + van_allen_model_(position);
                
                // Add South Atlantic Anomaly effects if in the region
                flux = flux + saa_model_(position);
                break;
            }
            
            case SpaceMission::MEO: {
                // Higher radiation in Medium Earth Orbit (heart of Van Allen belts)
                RadiationFlux van_allen_flux = van_allen_model_(position);
                van_allen_flux.scale(3.0); // MEO has stronger Van Allen effects
                flux = flux + van_allen_flux;
                break;
            }
            
            case SpaceMission::LUNAR: {
                // Lunar missions have less shielding from Earth's magnetosphere
                flux.scale(1.5);
                break;
            }
            
            case SpaceMission::MARS: {
                // Mars has no magnetosphere
                flux.scale(2.0);
                break;
            }
            
            case SpaceMission::JUPITER: {
                // Jupiter has intense radiation belts
                flux.scale(1000.0);
                break;
            }
            
            case SpaceMission::SOLAR_PROBE: {
                // Solar missions have intense particle flux
                double solar_distance = 1.0; // AU, would be calculated from position
                if (solar_distance < 0.3) {
                    flux.scale(10.0 / (solar_distance * solar_distance));
                }
                break;
            }
            
            default:
                break;
        }
        
        // Apply solar activity factor
        applySolarActivityEffects(flux);
        
        // Apply shielding
        applyShielding(flux);
        
        return flux;
    }
    
    /**
     * @brief Calculate SEU probability based on radiation flux
     * 
     * @param position Orbital position
     * @return SEU probability per bit per day
     */
    double calculateSEUProbability(const OrbitalPosition& position) const {
        // Get radiation flux
        RadiationFlux flux = calculateRadiationFlux(position);
        
        // Convert to SEU probability
        // This is a simplified model - real SEU probability depends on many factors
        // These coefficients would be derived from actual radiation testing
        const double electron_factor = 1e-12;
        const double proton_factor = 1e-9;
        const double heavy_ion_factor = 1e-6;
        
        double seu_prob = flux.electron_flux * electron_factor +
                          flux.proton_flux * proton_factor +
                          flux.heavy_ion_flux * heavy_ion_factor;
        
        // Apply device-specific sensitivity factor
        return seu_prob * device_sensitivity_;
    }
    
    /**
     * @brief Simulate a solar flare event
     * 
     * @param intensity Flare intensity (0.0-1.0)
     * @return Flux increase factor
     */
    double simulateSolarFlare(double intensity = 1.0) {
        // Store previous solar activity
        double previous_activity = solar_activity_;
        
        // Increase solar activity temporarily
        solar_activity_ = std::min(1.0, solar_activity_ + intensity * 0.5);
        
        // Calculate impact factor
        double impact_factor = 1.0 + intensity * 100.0;
        
        // Create decay function to restore normal conditions
        // In a real simulation, this would be called over time
        solar_flare_decay_ = [this, previous_activity](double time_elapsed) {
            double decay_factor = std::exp(-time_elapsed / 24.0); // 24-hour decay
            solar_activity_ = previous_activity + 
                              (solar_activity_ - previous_activity) * decay_factor;
            return solar_activity_ - previous_activity;
        };
        
        return impact_factor;
    }
    
    /**
     * @brief Generate a radiation heatmap over a trajectory
     * 
     * @param trajectory Vector of orbital positions
     * @return Vector of SEU probabilities corresponding to each position
     */
    std::vector<double> generateHeatmap(const std::vector<OrbitalPosition>& trajectory) const {
        std::vector<double> heatmap;
        heatmap.reserve(trajectory.size());
        
        for (const auto& position : trajectory) {
            heatmap.push_back(calculateSEUProbability(position));
        }
        
        return heatmap;
    }
    
    /**
     * @brief Generate an orbital position based on orbital parameters
     * 
     * @param inclination Orbit inclination in degrees
     * @param altitude Altitude in km
     * @param time_hours Time in hours since epoch
     * @return OrbitalPosition
     */
    OrbitalPosition generateOrbitPosition(
        double inclination, 
        double altitude, 
        double time_hours
    ) const {
        // Simplified orbital mechanics
        const double orbit_period = calculateOrbitPeriod(altitude);
        const double phase = fmod(time_hours / orbit_period * 2.0 * M_PI, 2.0 * M_PI);
        
        // Calculate latitude based on inclination
        double latitude = inclination * std::sin(phase);
        
        // Calculate longitude (simplified model assuming Earth-fixed reference)
        double longitude = fmod(time_hours * 15.0, 360.0) - 180.0;
        
        return {latitude, longitude, altitude};
    }

private:
    SpaceMission mission_;
    double solar_activity_;
    double shielding_thickness_;
    int shielding_material_;
    double shielding_effectiveness_electrons_;
    double shielding_effectiveness_protons_;
    double shielding_effectiveness_heavy_ions_;
    double device_sensitivity_ = 1.0; // Device-specific SEU sensitivity
    
    // Function type for radiation models
    using RadiationModelFunction = std::function<RadiationFlux(const OrbitalPosition&)>;
    
    // Radiation environment models
    RadiationModelFunction saa_model_;
    RadiationModelFunction van_allen_model_;
    RadiationModelFunction galactic_cosmic_ray_model_;
    RadiationModelFunction solar_particle_model_;
    
    // Solar flare decay function
    std::function<double(double)> solar_flare_decay_;
    
    /**
     * @brief Update shielding effectiveness based on thickness and material
     */
    void updateShieldingEffectiveness() {
        // Base attenuation factors per mm of shielding material
        double material_factor_e = 0.1; // Aluminum base electron attenuation
        double material_factor_p = 0.05; // Aluminum base proton attenuation
        double material_factor_h = 0.01; // Aluminum base heavy ion attenuation
        
        // Adjust for different materials
        if (shielding_material_ == 1) {
            // Polyethylene (better for protons/neutrons)
            material_factor_p *= 1.5;
        } else if (shielding_material_ == 2) {
            // Tungsten (better for gamma/electrons)
            material_factor_e *= 2.0;
            material_factor_h *= 1.5;
        }
        
        // Calculate effectiveness (exponential attenuation)
        shielding_effectiveness_electrons_ = std::exp(-material_factor_e * shielding_thickness_);
        shielding_effectiveness_protons_ = std::exp(-material_factor_p * shielding_thickness_);
        shielding_effectiveness_heavy_ions_ = std::exp(-material_factor_h * shielding_thickness_);
    }
    
    /**
     * @brief Apply shielding to radiation flux
     * 
     * @param flux Radiation flux to modify
     */
    void applyShielding(RadiationFlux& flux) const {
        flux.electron_flux *= shielding_effectiveness_electrons_;
        flux.proton_flux *= shielding_effectiveness_protons_;
        flux.heavy_ion_flux *= shielding_effectiveness_heavy_ions_;
        
        // Recalculate total dose based on attenuated particle flux
        flux.total_dose_rate = 
            flux.electron_flux * 1e-3 +
            flux.proton_flux * 1e-2 +
            flux.heavy_ion_flux * 1e-1;
    }
    
    /**
     * @brief Apply solar activity effects to radiation flux
     * 
     * @param flux Radiation flux to modify
     */
    void applySolarActivityEffects(RadiationFlux& flux) const {
        // Solar activity affects different particles differently
        // High solar activity increases solar particles but can decrease GCR
        double gcr_factor = 1.0 - 0.3 * solar_activity_; // GCR decrease with solar activity
        double solar_factor = 1.0 + 10.0 * solar_activity_; // Solar particles increase
        
        // Adjust flux based on particle type and solar activity
        flux.electron_flux = flux.electron_flux * gcr_factor + flux.electron_flux * solar_factor * 0.5;
        flux.proton_flux = flux.proton_flux * gcr_factor + flux.proton_flux * solar_factor;
        flux.heavy_ion_flux = flux.heavy_ion_flux * gcr_factor;
        
        // Recalculate total dose rate
        flux.total_dose_rate = 
            flux.electron_flux * 1e-3 +
            flux.proton_flux * 1e-2 +
            flux.heavy_ion_flux * 1e-1;
    }
    
    /**
     * @brief Calculate orbital period for a given altitude
     * 
     * @param altitude Altitude in km
     * @return Orbital period in hours
     */
    double calculateOrbitPeriod(double altitude) const {
        const double earth_radius = 6371.0; // km
        const double earth_mu = 3.986e14; // m^3/s^2
        
        // Semi-major axis in meters
        double a = (earth_radius + altitude) * 1000.0;
        
        // Period in seconds
        double period_seconds = 2.0 * M_PI * std::sqrt(std::pow(a, 3) / earth_mu);
        
        // Return period in hours
        return period_seconds / 3600.0;
    }
    
    /**
     * @brief Create South Atlantic Anomaly model
     * 
     * @return Function modeling SAA radiation
     */
    RadiationModelFunction createSAAModel() const {
        return [](const OrbitalPosition& position) -> RadiationFlux {
            // SAA centered around -30° latitude, -40° longitude
            const double saa_lat = -30.0;
            const double saa_lon = -40.0;
            const double saa_radius = 20.0; // degrees
            
            // Calculate distance from SAA center (simplified)
            double lat_diff = position.latitude - saa_lat;
            double lon_diff = position.longitude - saa_lon;
            double angular_distance = std::sqrt(lat_diff*lat_diff + lon_diff*lon_diff);
            
            // Check if in SAA
            if (angular_distance > saa_radius) {
                return RadiationFlux(); // Not in SAA, return zero flux
            }
            
            // Calculate intensity based on distance from center (gaussian falloff)
            double intensity = std::exp(-angular_distance*angular_distance/(2.0*10.0*10.0));
            
            // SAA has higher proton flux
            return RadiationFlux(
                1e4 * intensity,  // Electrons
                1e6 * intensity,  // Protons (high in SAA)
                100 * intensity,  // Heavy ions
                0.1 * intensity   // Total dose rate (rad/hour)
            );
        };
    }
    
    /**
     * @brief Create Van Allen belts model
     * 
     * @return Function modeling Van Allen radiation
     */
    RadiationModelFunction createVanAllenModel() const {
        return [](const OrbitalPosition& position) -> RadiationFlux {
            // Inner belt: ~1000-6000 km, mostly protons
            // Outer belt: ~13000-60000 km, mostly electrons
            
            const double earth_radius = 6371.0; // km
            double radius = position.getRadius();
            double altitude = position.altitude;
            
            // L-shell parameter (simplified model)
            double l_value = radius / earth_radius / std::cos(position.latitude * M_PI / 180.0);
            
            // Inner belt (L~1.5)
            double inner_belt_factor = std::exp(-(l_value-1.5)*(l_value-1.5)/0.5);
            
            // Outer belt (L~4-5)
            double outer_belt_factor = std::exp(-(l_value-4.5)*(l_value-4.5)/2.0);
            
            // Scale factors for altitude below the belts
            double altitude_factor = 1.0;
            if (altitude < 1000.0) {
                altitude_factor = altitude / 1000.0;
            }
            
            // Inner belt has more protons, outer belt more electrons
            return RadiationFlux(
                1e5 * outer_belt_factor * altitude_factor,  // Electrons (high in outer belt)
                1e5 * inner_belt_factor * altitude_factor,  // Protons (high in inner belt)
                100 * (inner_belt_factor + outer_belt_factor) * altitude_factor,  // Heavy ions
                0.05 * (inner_belt_factor + outer_belt_factor) * altitude_factor  // Total dose rate
            );
        };
    }
    
    /**
     * @brief Create Galactic Cosmic Ray model
     * 
     * @return Function modeling GCR radiation
     */
    RadiationModelFunction createGCRModel() const {
        return [](const OrbitalPosition& position) -> RadiationFlux {
            // GCR intensity increases with altitude as Earth's magnetic protection decreases
            const double earth_radius = 6371.0; // km
            double altitude = position.altitude;
            
            // Magnetic latitude effect (more particles at poles)
            double mag_lat_factor = 1.0 + 0.5 * std::abs(position.latitude) / 90.0;
            
            // Altitude effect (increases with altitude)
            double altitude_factor = 1.0;
            if (altitude < 1000.0) {
                altitude_factor = 0.7 + 0.3 * altitude / 1000.0;
            } else if (altitude < 20000.0) {
                altitude_factor = 1.0 + 0.5 * (altitude - 1000.0) / 19000.0;
            } else {
                altitude_factor = 1.5;
            }
            
            // Base GCR levels
            double base_electron = 1e3;
            double base_proton = 1e2;
            double base_heavy_ion = 10.0;
            double base_dose = 0.001;
            
            // Apply factors
            double factor = mag_lat_factor * altitude_factor;
            
            return RadiationFlux(
                base_electron * factor,
                base_proton * factor,
                base_heavy_ion * factor,
                base_dose * factor
            );
        };
    }
    
    /**
     * @brief Create Solar Particle Event model
     * 
     * @return Function modeling SPE radiation
     */
    RadiationModelFunction createSPEModel() const {
        // Note: This would be used during solar flare events
        return [](const OrbitalPosition& position) -> RadiationFlux {
            // Base model - would be scaled by flare intensity
            return RadiationFlux(
                1e6,  // Electrons
                1e5,  // Protons
                1e3,  // Heavy ions
                1.0   // Total dose rate (rad/hour)
            );
        };
    }
};

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_RADIATION_ENVIRONMENT_HPP 