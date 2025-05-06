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
    
    // Temperature and mechanical sensitivity
    enum TempSensitivity { LOW, MODERATE, HIGH, EXTREME };
    TempSensitivity temp_sensitivity;
    
    enum MechSensitivity { LOW, MODERATE, HIGH };
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