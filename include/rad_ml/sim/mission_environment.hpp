/**
 * Mission Environment Models
 * 
 * This file contains definitions for space mission environments.
 */

#pragma once

#include <string>
#include <vector>

namespace rad_ml {
namespace sim {

/**
 * Temperature range for mission environments
 */
struct TemperatureRange {
    double min;              // Minimum temperature in K
    double max;              // Maximum temperature in K
    double cycle_period;     // Cycle period in hours
};

/**
 * Radiation environment parameters for space missions
 */
struct RadiationEnvironment {
    // Orbit parameters
    double altitude;                 // km
    double distance_from_sun;        // AU
    
    // Radiation sources
    double gcr_intensity;            // Relative intensity
    double solar_activity;           // 0-1 scale
    double trapped_proton_flux;      // protons/cm²/s
    double trapped_electron_flux;    // electrons/cm²/s
    
    // Environmental factors
    bool saa_region;                 // South Atlantic Anomaly
    double atmosphere_depth;         // g/cm²
    double magnetic_field_strength;  // Relative to Earth
    
    // Temperature profile
    TemperatureRange temperature;
};

/**
 * Mission parameters for material optimization
 */
struct MissionParameters {
    double duration;                 // hours
    double dose_rate;                // Gy/h
    double temperature;              // K
    std::vector<double> radiation_spectrum;  // Energy spectrum
    bool is_vacuum;
    bool has_atomic_oxygen;
};

/**
 * Calculate dose rate for a mission environment
 */
double calculateDoseRate(const RadiationEnvironment& env);

/**
 * Calculate radiation dose for a mission environment
 */
double calculateDose(const RadiationEnvironment& env, double mission_duration);

/**
 * Get mission temperature (historically used before temperature ranges were added)
 */
double getMissionTemperature(const std::string& mission_profile);

/**
 * Get radiation spectrum for a mission environment
 */
std::vector<double> getRadiationSpectrum(const RadiationEnvironment& env);

/**
 * Calculate material protection score for a mission
 */
double calculateMaterialProtectionScore(
    const struct MaterialProperties& material,
    const MissionParameters& mission_params);

/**
 * Calculate stress from strain for a material
 */
double calculateStressFromStrain(
    const Eigen::Matrix3d& strain,
    const struct MaterialProperties& material);

/**
 * Calculate expected stress for a material under strain
 */
double calculateExpectedStress(
    const Eigen::Matrix3d& strain,
    const struct MaterialProperties& material);

/**
 * Set defect distribution based on radiation environment
 */
void setDefectDistribution(
    class Field3D<double>& vacancy_concentration,
    class Field3D<double>& interstitial_concentration,
    const RadiationEnvironment& env,
    const struct MaterialProperties& material,
    double temperature_factor = 1.0,
    double stress_factor = 1.0);

} // namespace sim
} // namespace rad_ml 