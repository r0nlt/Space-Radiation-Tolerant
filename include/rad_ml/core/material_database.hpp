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
/**
 * Material Database
 * 
 * This file contains definitions for material properties database.
 */

#pragma once

#include <string>
#include <map>
#include <vector>

namespace rad_ml {
namespace core {

/**
 * Material properties for radiation simulation
 */
struct MaterialProperties {
    std::string name;
    double density;                   // g/cm³
    double hydrogen_content;          // wt%
    double z_effective;               // Effective atomic number
    double radiation_length;          // g/cm²
    double nuclear_interaction_length; // g/cm²
    
    // Radiation attenuation properties
    double gcr_proton_reduction;      // % reduction at 10 g/cm²
    double gcr_fe_reduction;          // % reduction at 10 g/cm²
    double neutron_production_coef;   // Relative to aluminum
    
    // Solar particle event attenuation
    double spe_proton_attenuation;    // Factor at 5 g/cm²
    double spe_electron_attenuation;  // Factor at 5 g/cm²
    
    // Physics parameters (derived or measured)
    double displacement_energy;       // eV
    double diffusion_coefficient;     // m²/s
    double migration_energy;          // eV
    double recombination_radius;      // Angstrom
    std::vector<double> defect_formation_energies; // eV
    
    // NASA model parameters
    double yield_strength;            // MPa
    double vacuum_modifier;           // Effect factor in vacuum
    double ao_modifier;               // Atomic oxygen effect factor
    double radiation_tolerance;       // Relative scale (0-100)
    
    // Temperature and mechanical sensitivity
    enum class TempSensitivity { LOW, MODERATE, HIGH, EXTREME };
    TempSensitivity temp_sensitivity;
    
    enum class MechSensitivity { LOW, MODERATE, HIGH };
    MechSensitivity mech_sensitivity;
    
    /**
     * Calculate threshold for a given temperature
     */
    double calculateThresholdForTemperature(double temperature_K) const;
};

/**
 * Load material database with standard material properties
 */
std::map<std::string, MaterialProperties> loadMaterialDatabase();

} // namespace core
} // namespace rad_ml 
