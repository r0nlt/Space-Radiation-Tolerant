/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <string>

namespace rad_ml {
namespace sim {

// Temperature range in Kelvin
struct TemperatureRange {
    double min = 273.0;  // Default to 0°C
    double max = 293.0;  // Default to 20°C
};

// Radiation environment characteristics
struct RadiationEnvironment {
    // Radiation parameters
    double trapped_proton_flux = 0.0;  // particles/cm²/s
    double trapped_electron_flux = 0.0; // particles/cm²/s
    double galactic_cosmic_ray_flux = 0.0; // particles/cm²/s
    double solar_particle_event_flux = 0.0; // particles/cm²/s
    
    // Environmental parameters
    TemperatureRange temperature;
    double solar_activity = 0.0;  // Scale from 0-1
    bool saa_region = false;  // South Atlantic Anomaly region flag
    
    // Simulation parameters
    bool enable_single_event_effects = true;  // Enable/disable SEE simulation
    double single_event_upset_probability = 0.0001;  // Per bit per day
    double single_event_latchup_probability = 0.00001;  // Per device per day
    
    // Calculate total radiation flux
    double getTotalFlux() const {
        return trapped_proton_flux + 
               trapped_electron_flux + 
               galactic_cosmic_ray_flux + 
               solar_particle_event_flux;
    }
    
    // Calculate average temperature
    double getAverageTemperature() const {
        return (temperature.min + temperature.max) / 2.0;
    }
    
    // Check if environment is considered "high radiation"
    bool isHighRadiation() const {
        return getTotalFlux() > 1.0e8 || saa_region || solar_activity > 0.7;
    }
};

// Mission phase definitions
struct MissionPhase {
    std::string name;
    double duration_hours;
    RadiationEnvironment environment;
    bool is_critical = false;
};

} // namespace sim
} // namespace rad_ml 
