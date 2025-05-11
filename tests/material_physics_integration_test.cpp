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
 * Material Physics Integration Test
 * 
 * This test demonstrates how to use standardized material specifications
 * as inputs for advanced physics-based radiation damage models.
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <Eigen/Dense>

// Include the rad_ml headers with proper namespaces
#include "rad_ml/physics/quantum_models.hpp"
#include "rad_ml/physics/field_theory.hpp"
#include "rad_ml/physics/stochastic_models.hpp"
#include "rad_ml/core/material_database.hpp"
#include "rad_ml/physics/transport_equation.hpp"
#include "rad_ml/sim/mission_environment.hpp"

// Use appropriate namespaces
using rad_ml::physics::CrystalLattice;
using rad_ml::physics::DFTParameters;
using rad_ml::physics::DefectDistribution;
using rad_ml::physics::Field3D;
using rad_ml::physics::FieldParameters;
using rad_ml::physics::FreeEnergyFunctional;
using rad_ml::physics::Grid3D;
using rad_ml::physics::MaterialParameters;
using rad_ml::physics::ParticleType;
using rad_ml::physics::ProtonParticle;
using rad_ml::physics::ElectronParticle;
using rad_ml::physics::CrossSectionData;
using rad_ml::physics::SimulationResults;
using rad_ml::physics::TimeEvolutionResults;
using rad_ml::physics::TransportSolution;
using rad_ml::sim::RadiationEnvironment;
using rad_ml::sim::MissionParameters;
using rad_ml::sim::TemperatureRange;

// NASA-inspired physics models
// 1. Temperature-Dependent Correction Model
double calculateTemperatureCorrectedThreshold(double baseThreshold, double temperature_K) {
    double tempFactor = 1.0;
    
    // Cryogenic enhancement of damage (below 150K)
    if (temperature_K < 150) {
        tempFactor = 0.1 + 0.9 * (temperature_K / 150.0);
    }
    // Enhanced annealing at high temperatures (above 400K)
    else if (temperature_K > 400) {
        tempFactor = 1.0 + 0.5 * ((temperature_K - 400) / 100.0);
    }
    
    return baseThreshold * tempFactor;
}

// 2. Synergistic Effects Model
double calculateSynergyFactor(double temperature_K, double mechanical_stress, double yield_strength) {
    // Default synergy factor
    double synergyFactor = 1.0;
    
    // High temperature + stress creates super-linear effect (based on NASA 2025 spec)
    if (temperature_K > 350 && mechanical_stress > 0.3 * yield_strength) {
        synergyFactor = 1.5;
    }
    
    return synergyFactor;
}

// 3. Mechanical Load Effect Model
double calculateMechanicalLoadFactor(double stress_MPa, double yield_strength_MPa, double radiation_dose) {
    // Normalized stress level (0-1)
    double normalized_stress = std::min(1.0, stress_MPa / (0.7 * yield_strength_MPa));
    
    // No effect at zero stress, exponential effect approaching yield
    double stressFactor = std::exp(2.0 * normalized_stress) - 1.0;
    
    // Higher radiation makes material more sensitive to stress
    double radiationEnhancement = 1.0 + std::log10(std::max(1.0, radiation_dose)) * 0.1;
    
    return 1.0 + stressFactor * radiationEnhancement;
}

// 4. Combined Environmental Effects Model
double calculateCombinedDamageEffect(
    const struct MaterialProperties& material,
    double radiation_dose,
    double temperature_K,
    double mechanical_stress_MPa,
    bool vacuum_condition,
    bool atomic_oxygen_present);

// Define material properties structure for testing
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
    
    // Calculate threshold for a given temperature
    double calculateThresholdForTemperature(double temperature_K) const {
        return calculateTemperatureCorrectedThreshold(this->displacement_energy, temperature_K);
    }
};

// Temperature range for missions
struct TemperatureRange {
    double min;
    double max;
    double cycle_period; // hours
};

// Calculate base damage based on material and dose
double calculateBaseDamage(const MaterialProperties& material, double radiation_dose) {
    // Simple linear model based on material properties
    double sensitivity = 1.0 / material.displacement_energy;
    return sensitivity * radiation_dose;
}

// Implementation of Combined Environmental Effects Model
double calculateCombinedDamageEffect(
    const MaterialProperties& material,
    double radiation_dose,
    double temperature_K,
    double mechanical_stress_MPa,
    bool vacuum_condition,
    bool atomic_oxygen_present) {
    
    // Base radiation damage based on material type and dose
    double baseDamage = calculateBaseDamage(material, radiation_dose);
    
    // Temperature modification
    double tempFactor = calculateTemperatureCorrectedThreshold(1.0, temperature_K);
    
    // Mechanical stress enhancement
    double stressFactor = calculateMechanicalLoadFactor(
        mechanical_stress_MPa, material.yield_strength, radiation_dose);
    
    // Environmental condition modifiers
    double envFactor = 1.0;
    if (vacuum_condition) {
        envFactor *= material.vacuum_modifier;
    }
    if (atomic_oxygen_present) {
        envFactor *= material.ao_modifier;
    }
    
    // Calculate synergistic effects
    double synergyFactor = calculateSynergyFactor(
        temperature_K, mechanical_stress_MPa, material.yield_strength);
    
    return baseDamage * tempFactor * stressFactor * envFactor * synergyFactor;
}

// Create material database from specifications
std::map<std::string, MaterialProperties> loadMaterialDatabase() {
    std::map<std::string, MaterialProperties> materials;
    
    // Aluminum properties
    MaterialProperties aluminum;
    aluminum.name = "Aluminum";
    aluminum.density = 2.70;
    aluminum.hydrogen_content = 0.0;
    aluminum.z_effective = 13.0;
    aluminum.radiation_length = 24.01;
    aluminum.nuclear_interaction_length = 107.2;
    aluminum.gcr_proton_reduction = 25.0;
    aluminum.gcr_fe_reduction = 18.0;
    aluminum.neutron_production_coef = 1.0;
    aluminum.spe_proton_attenuation = 0.42;
    aluminum.spe_electron_attenuation = 0.12;
    aluminum.displacement_energy = 16.0;
    aluminum.diffusion_coefficient = 1.4e-19;
    aluminum.migration_energy = 0.58;
    aluminum.recombination_radius = 3.2;
    aluminum.defect_formation_energies = {3.1, 2.8, 4.2};
    // NASA model parameters
    aluminum.yield_strength = 276.0; // MPa
    aluminum.vacuum_modifier = 1.2;
    aluminum.ao_modifier = 1.1;
    aluminum.temp_sensitivity = MaterialProperties::HIGH;
    aluminum.mech_sensitivity = MaterialProperties::MODERATE;
    materials["Aluminum"] = aluminum;
    
    // Polyethylene properties
    MaterialProperties polyethylene;
    polyethylene.name = "Polyethylene";
    polyethylene.density = 0.95;
    polyethylene.hydrogen_content = 14.3;
    polyethylene.z_effective = 5.2;
    polyethylene.radiation_length = 44.77;
    polyethylene.nuclear_interaction_length = 52.2;
    polyethylene.gcr_proton_reduction = 35.0;
    polyethylene.gcr_fe_reduction = 31.0;
    polyethylene.neutron_production_coef = 0.25;
    polyethylene.spe_proton_attenuation = 0.57;
    polyethylene.spe_electron_attenuation = 0.22;
    polyethylene.displacement_energy = 8.5;
    polyethylene.diffusion_coefficient = 2.8e-20;
    polyethylene.migration_energy = 0.31;
    polyethylene.recombination_radius = 4.1;
    polyethylene.defect_formation_energies = {1.8, 2.1, 2.5};
    // NASA model parameters
    polyethylene.yield_strength = 25.0; // MPa
    polyethylene.vacuum_modifier = 0.8;
    polyethylene.ao_modifier = 0.7;
    polyethylene.temp_sensitivity = MaterialProperties::EXTREME;
    polyethylene.mech_sensitivity = MaterialProperties::MODERATE;
    materials["Polyethylene"] = polyethylene;
    
    // Water properties
    MaterialProperties water;
    water.name = "Water";
    water.density = 1.00;
    water.hydrogen_content = 11.2;
    water.z_effective = 7.2;
    water.radiation_length = 36.08;
    water.nuclear_interaction_length = 83.6;
    water.gcr_proton_reduction = 33.0;
    water.gcr_fe_reduction = 29.0;
    water.neutron_production_coef = 0.28;
    water.spe_proton_attenuation = 0.54;
    water.spe_electron_attenuation = 0.20;
    water.displacement_energy = 7.2;
    water.diffusion_coefficient = 9.3e-19;
    water.migration_energy = 0.22;
    water.recombination_radius = 4.8;
    water.defect_formation_energies = {1.2, 1.9, 2.2};
    // NASA model parameters
    water.yield_strength = 0.0; // Not applicable for water
    water.vacuum_modifier = 2.0; // Significant effects in vacuum (boiling)
    water.ao_modifier = 1.0;
    water.temp_sensitivity = MaterialProperties::EXTREME;
    water.mech_sensitivity = MaterialProperties::LOW;
    materials["Water"] = water;
    
    // Lead properties
    MaterialProperties lead;
    lead.name = "Lead";
    lead.density = 11.35;
    lead.hydrogen_content = 0.0;
    lead.z_effective = 82.0;
    lead.radiation_length = 6.37;
    lead.nuclear_interaction_length = 199.6;
    lead.gcr_proton_reduction = 12.0;
    lead.gcr_fe_reduction = 10.0;
    lead.neutron_production_coef = 1.95;
    lead.spe_proton_attenuation = 0.24;
    lead.spe_electron_attenuation = 0.03;
    lead.displacement_energy = 25.0;
    lead.diffusion_coefficient = 3.2e-20;
    lead.migration_energy = 0.82;
    lead.recombination_radius = 2.9;
    lead.defect_formation_energies = {4.5, 3.9, 5.2};
    // NASA model parameters
    lead.yield_strength = 12.0; // MPa
    lead.vacuum_modifier = 1.1;
    lead.ao_modifier = 1.0;
    lead.temp_sensitivity = MaterialProperties::MODERATE;
    lead.mech_sensitivity = MaterialProperties::LOW;
    materials["Lead"] = lead;
    
    // Boron Carbide properties
    MaterialProperties boron_carbide;
    boron_carbide.name = "Boron Carbide";
    boron_carbide.density = 2.52;
    boron_carbide.hydrogen_content = 0.0;
    boron_carbide.z_effective = 7.6;
    boron_carbide.radiation_length = 42.10;
    boron_carbide.nuclear_interaction_length = 75.3;
    boron_carbide.gcr_proton_reduction = 34.0;
    boron_carbide.gcr_fe_reduction = 29.0;
    boron_carbide.neutron_production_coef = 0.27;
    boron_carbide.spe_proton_attenuation = 0.53;
    boron_carbide.spe_electron_attenuation = 0.19;
    boron_carbide.displacement_energy = 28.0;
    boron_carbide.diffusion_coefficient = 5.6e-21;
    boron_carbide.migration_energy = 1.14;
    boron_carbide.recombination_radius = 3.1;
    boron_carbide.defect_formation_energies = {4.9, 5.3, 6.1};
    // NASA model parameters
    boron_carbide.yield_strength = 350.0; // MPa
    boron_carbide.vacuum_modifier = 1.0;
    boron_carbide.ao_modifier = 1.0;
    boron_carbide.temp_sensitivity = MaterialProperties::LOW;
    boron_carbide.mech_sensitivity = MaterialProperties::HIGH;
    materials["Boron Carbide"] = boron_carbide;
    
    return materials;
}

// Calculate radiation environment parameters based on mission profile
void setupRadiationEnvironment(const std::string& mission_profile, 
                              RadiationEnvironment& env) {
    if (mission_profile == "LEO") {
        env.altitude = 400;         // km
        env.gcr_intensity = 0.3;    // relative intensity
        env.solar_activity = 0.5;   // moderate
        env.trapped_proton_flux = 1.0e6; // protons/cm²/s
        env.trapped_electron_flux = 2.0e7; // electrons/cm²/s
        env.saa_region = true;      // South Atlantic Anomaly
        // Add temperature profile
        env.temperature.min = 173.0; // -100°C
        env.temperature.max = 393.0; // +120°C
        env.temperature.cycle_period = 1.5; // 90 minutes
    } 
    else if (mission_profile == "GEO") {
        env.altitude = 35786;       // km
        env.gcr_intensity = 0.8;    // relative intensity
        env.solar_activity = 0.5;   // moderate
        env.trapped_proton_flux = 1.0e4; // protons/cm²/s
        env.trapped_electron_flux = 1.0e6; // electrons/cm²/s
        env.saa_region = false;
        // Add temperature profile
        env.temperature.min = 120.0; // -153°C
        env.temperature.max = 373.0; // +100°C
        env.temperature.cycle_period = 24.0; // 24 hours (daily cycle)
    }
    else if (mission_profile == "MARS") {
        env.distance_from_sun = 1.52; // AU
        env.gcr_intensity = 0.95;     // relative intensity
        env.solar_activity = 0.5;     // moderate
        env.atmosphere_depth = 16;    // g/cm²
        env.trapped_proton_flux = 0;  // no trapped radiation
        env.trapped_electron_flux = 0;
        // Add temperature profile
        env.temperature.min = 153.0; // -120°C
        env.temperature.max = 293.0; // +20°C
        env.temperature.cycle_period = 24.65; // Mars day
    }
    else if (mission_profile == "JUPITER") {
        env.distance_from_sun = 5.2;  // AU
        env.gcr_intensity = 1.0;      // relative intensity
        env.solar_activity = 0.5;     // moderate
        env.trapped_proton_flux = 1.0e9; // extreme radiation belts
        env.trapped_electron_flux = 1.0e10;
        env.magnetic_field_strength = 14.0; // relative to Earth
        // Add temperature profile
        env.temperature.min = 100.0; // -173°C
        env.temperature.max = 300.0; // +27°C 
        env.temperature.cycle_period = 9.8; // Jupiter day in hours
    }
}

//-------------------------------------------------------------------------------
// Test functions using material properties as inputs to physics models
//-------------------------------------------------------------------------------

// Test for NASA-inspired physics models
void testNASAPhysicsModels(const std::map<std::string, MaterialProperties>& materials) {
    std::cout << "Testing NASA-inspired Physics Models..." << std::endl;
    
    // Test 1: Temperature Correction Model
    std::cout << "Temperature Correction Model Test:" << std::endl;
    double temp_150k = calculateTemperatureCorrectedThreshold(1.0, 150.0);
    double temp_30k = calculateTemperatureCorrectedThreshold(1.0, 30.0);
    double temp_300k = calculateTemperatureCorrectedThreshold(1.0, 300.0);
    double temp_500k = calculateTemperatureCorrectedThreshold(1.0, 500.0);
    
    std::cout << "  150K: " << temp_150k << " (Expected: 1.0)" << std::endl;
    std::cout << "  30K: " << temp_30k << " (Expected: ~0.28)" << std::endl;
    std::cout << "  300K: " << temp_300k << " (Expected: 1.0)" << std::endl;
    std::cout << "  500K: " << temp_500k << " (Expected: ~1.5)" << std::endl;
    
    assert(std::abs(temp_150k - 1.0) < 0.01);
    assert(std::abs(temp_30k - 0.28) < 0.05);
    assert(std::abs(temp_300k - 1.0) < 0.01);
    assert(std::abs(temp_500k - 1.5) < 0.05);
    
    // Test 2: Mechanical Load Effect Model
    std::cout << "Mechanical Load Effect Model Test:" << std::endl;
    double yield = 400.0; // MPa
    double rad_dose = 1.0e6; // Gy
    double low_stress = calculateMechanicalLoadFactor(40.0, yield, rad_dose);
    double med_stress = calculateMechanicalLoadFactor(160.0, yield, rad_dose);
    double high_stress = calculateMechanicalLoadFactor(260.0, yield, rad_dose);
    
    std::cout << "  10% Yield: " << low_stress << std::endl;
    std::cout << "  40% Yield: " << med_stress << std::endl;
    std::cout << "  65% Yield: " << high_stress << std::endl;
    
    // Verify exponential growth pattern
    assert(high_stress > med_stress * 1.5);
    assert(med_stress > low_stress * 1.5);
    
    // Test 3: Synergistic Effects Model
    std::cout << "Synergistic Effects Model Test:" << std::endl;
    double synergy_normal = calculateSynergyFactor(300.0, 0.4 * yield, yield);
    double synergy_high = calculateSynergyFactor(400.0, 0.4 * yield, yield);
    
    std::cout << "  Normal conditions: " << synergy_normal << std::endl;
    std::cout << "  High temp + stress: " << synergy_high << std::endl;
    
    assert(std::abs(synergy_normal - 1.0) < 0.01);
    assert(std::abs(synergy_high - 1.5) < 0.01);
    
    // Test 4: Combined Environmental Effects Model
    std::cout << "Combined Environmental Effects Model Test:" << std::endl;
    const MaterialProperties& aluminum = materials.at("Aluminum");
    
    double combined_normal = calculateCombinedDamageEffect(
        aluminum, 1.0e4, 300.0, 50.0, false, false);
    
    double combined_extreme = calculateCombinedDamageEffect(
        aluminum, 1.0e4, 450.0, 0.5 * aluminum.yield_strength, true, true);
    
    std::cout << "  Normal conditions: " << combined_normal << std::endl;
    std::cout << "  Extreme conditions: " << combined_extreme << std::endl;
    
    // Verify extreme conditions result in higher damage
    assert(combined_extreme > combined_normal * 2.0);
    
    std::cout << "NASA physics models tests passed!" << std::endl;
}

// Test 1: Boltzmann Transport Model with Material Cross-Sections
void testBoltzmannTransportWithMaterials(const std::map<std::string, MaterialProperties>& materials,
                                       const RadiationEnvironment& env) {
    std::cout << "Testing Boltzmann Transport Model with different materials..." << std::endl;
    
    // Setup constants
    const int spatial_points = 100;
    const int angular_points = 24;
    const int energy_bins = 50;
    
    // For each material, test radiation transport
    for (const auto& [name, material] : materials) {
        std::cout << "  Testing material: " << name << std::endl;
        
        // Generate cross-sections from material properties
        CrossSectionData cross_sections;
        cross_sections.total = material.radiation_length;
        cross_sections.elastic = material.nuclear_interaction_length * 0.4;
        cross_sections.inelastic = material.nuclear_interaction_length * 0.6;
        cross_sections.z_effective = material.z_effective;
        
        // Initialize fluence tensor Φ(x,Ω,E)
        Eigen::Tensor<double, 3> fluence(spatial_points, angular_points, energy_bins);
        fluence.setZero();
        
        // Setup source based on environment
        Eigen::Tensor<double, 3> source = setupRadiationSource(env, spatial_points, 
                                                             angular_points, energy_bins);
        
        // Generate material-specific cross-section tensors
        Eigen::Tensor<double, 2> sigma_t = generateMaterialCrossSections(material, energy_bins);
        Eigen::Tensor<double, 4> sigma_s = generateScatteringCrossSections(material, angular_points, energy_bins);
        
        // Solve transport equation
        TransportSolution solution = solveTransportEquation(fluence, source, sigma_t, sigma_s);
        
        // Calculate dose distribution
        Eigen::Tensor<double, 1> dose = calculateDoseDistribution(solution.fluence, material.density);
        
        // Calculate average attenuation
        double proton_attenuation = calculateAverageAttenuation(solution.fluence, ProtonParticle);
        double electron_attenuation = calculateAverageAttenuation(solution.fluence, ElectronParticle);
        
        // Verify against expected attenuation properties
        double proton_expected = material.spe_proton_attenuation;
        double electron_expected = material.spe_electron_attenuation;
        
        std::cout << "    Proton Attenuation: " << proton_attenuation 
                  << " (Expected: " << proton_expected << ")" << std::endl;
        std::cout << "    Electron Attenuation: " << electron_attenuation 
                  << " (Expected: " << electron_expected << ")" << std::endl;
        
        // Verify calculations are within 10% of expected values
        assert(std::abs(proton_attenuation - proton_expected) < 0.1 * proton_expected);
        assert(std::abs(electron_attenuation - electron_expected) < 0.1 * electron_expected);
    }
    
    std::cout << "Boltzmann transport model tests passed!" << std::endl;
}

// Test 2: Quantum Defect Formation using Material Properties
void testQuantumDefectFormationWithMaterials(const std::map<std::string, MaterialProperties>& materials,
                                          const RadiationEnvironment& env) {
    std::cout << "Testing Quantum Defect Formation with different materials..." << std::endl;
    
    // Use mid-point of temperature cycle
    double mission_temperature = (env.temperature.min + env.temperature.max) / 2.0;
    
    for (const auto& [name, material] : materials) {
        std::cout << "  Testing material: " << name << std::endl;
        
        // Setup Density Functional Theory parameters based on material
        DFTParameters params;
        params.kpoint_mesh = {6, 6, 6};
        params.energy_cutoff = 400.0; // eV
        params.temperature = mission_temperature;   // K
        
        // Use material density to create appropriate lattice
        double lattice_constant = calculateLatticeConstant(material.density, material.z_effective);
        CrystalLattice crystal = createAppropriateLatticetype(material.z_effective, lattice_constant);
        
        // Apply temperature-dependent displacement energy
        double temperature_adjusted_energy = material.calculateThresholdForTemperature(mission_temperature);
        std::cout << "    Temperature-adjusted displacement energy: " << temperature_adjusted_energy 
                  << " eV (base: " << material.displacement_energy << " eV)" << std::endl;
        
        // Primary knock-on atom energy
        double pka_energy = 1.0e6; // eV
        
        // Simulate cascade with quantum molecular dynamics
        DefectDistribution defects = simulateDisplacementCascade(
            crystal, pka_energy, params, temperature_adjusted_energy);
        
        // Kinchin-Pease prediction with temperature correction
        int predicted_defects = kinchinPeaseModel(pka_energy, temperature_adjusted_energy);
        int actual_defects = defects.interstitials.size();
        
        std::cout << "    Predicted defects: " << predicted_defects << std::endl;
        std::cout << "    Actual defects: " << actual_defects << std::endl;
        
        // Verify Kinchin-Pease model prediction (within 30%)
        assert(std::abs(actual_defects - predicted_defects) < 0.3 * predicted_defects);
        
        // Additional verification with NRT model
        double dpa = calculateDisplacementsPerAtom(defects, crystal);
        std::cout << "    Displacements per atom: " << dpa << std::endl;
    }
    
    std::cout << "Quantum defect formation tests passed!" << std::endl;
}

// Test 3: Stochastic Damage Evolution with Material Parameters
void testStochasticDamageEvolutionWithMaterials(
        const std::map<std::string, MaterialProperties>& materials,
        const RadiationEnvironment& env) {
    std::cout << "Testing Stochastic Damage Evolution with different materials..." << std::endl;
    
    // Setup parameters
    const int time_steps = 1000;
    const double simulation_time = 3600.0; // 1 hour in seconds
    const double temperature = (env.temperature.min + env.temperature.max) / 2.0; // K
    const double applied_stress = 20.0; // MPa (moderate stress)
    
    // Calculate radiation dose from environment
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5; // Gy
    
    for (const auto& [name, material] : materials) {
        std::cout << "  Testing material: " << name << std::endl;
        
        // Create material parameters from specifications
        MaterialParameters mat_params;
        mat_params.diffusion_coefficient = material.diffusion_coefficient;
        mat_params.recombination_radius = material.recombination_radius;
        mat_params.migration_energy = material.migration_energy;
        mat_params.displacement_energy = material.calculateThresholdForTemperature(temperature);
        
        // Apply mechanical load and synergistic effects
        double stress_factor = calculateMechanicalLoadFactor(applied_stress, material.yield_strength, radiation_dose);
        double synergy_factor = calculateSynergyFactor(temperature, applied_stress, material.yield_strength);
        
        std::cout << "    Temperature-adjusted threshold: " << mat_params.displacement_energy << " eV" << std::endl;
        std::cout << "    Mechanical stress factor: " << stress_factor << std::endl;
        std::cout << "    Synergy factor: " << synergy_factor << std::endl;
        
        // Calculate radiation-induced generation rate with combined effects
        double base_generation_rate = calculateGenerationRate(env, material);
        double modified_generation_rate = base_generation_rate * stress_factor * synergy_factor;
        
        std::cout << "    Base generation rate: " << base_generation_rate << " defects/cm³/s" << std::endl;
        std::cout << "    Modified generation rate: " << modified_generation_rate << " defects/cm³/s" << std::endl;
        
        // Initial defect concentrations (5 defect species)
        Eigen::VectorXd concentrations = Eigen::VectorXd::Zero(5);
        
        // Define drift term (deterministic part of SDE)
        auto driftTerm = createDriftTerm(mat_params, modified_generation_rate);
        
        // Define diffusion term (stochastic part of SDE)
        auto diffusionTerm = createDiffusionTerm(mat_params, temperature);
        
        // Solve stochastic differential equation
        SimulationResults results = solveStochasticDE(
            concentrations, driftTerm, diffusionTerm, 
            time_steps, simulation_time, temperature, applied_stress);
        
        // Calculate steady-state concentration with modified generation rate
        double theoretical_ss_concentration = modified_generation_rate / 
            (4.0 * M_PI * mat_params.diffusion_coefficient * mat_params.recombination_radius);
        
        std::cout << "    Final concentration: " << results.final_concentration.sum() << " defects/cm³" << std::endl;
        std::cout << "    Theoretical steady-state: " << theoretical_ss_concentration << " defects/cm³" << std::endl;
        
        // Verify convergence of Monte Carlo ensemble and comparison to theory
        assert(results.statistical_error < 0.05);
        assert(std::abs(results.final_concentration.sum() - theoretical_ss_concentration) 
               < 0.2 * theoretical_ss_concentration);
    }
    
    std::cout << "Stochastic damage evolution tests passed!" << std::endl;
}

// Test 4: Field Theory with Material-Specific Free Energy
void testFieldTheoryWithMaterials(const std::map<std::string, MaterialProperties>& materials,
                                const RadiationEnvironment& env) {
    std::cout << "Testing Field Theory with different materials..." << std::endl;
    
    // Setup 3D simulation grid
    Grid3D grid(50, 50, 50, 1.0); // 50³ grid with 1Å spacing
    
    for (const auto& [name, material] : materials) {
        std::cout << "  Testing material: " << name << std::endl;
        
        // Create field parameters from material properties
        FieldParameters params;
        
        // Gradient energy coefficient depends on material properties
        params.kappa = calculateGradientEnergyCoefficient(material);
        
        // Interaction matrix depends on defect formation energies
        params.gamma = createInteractionMatrix(material.defect_formation_energies);
        
        // Initialize concentration fields for different defect types
        std::vector<Field3D> fields(material.defect_formation_energies.size(), Field3D(grid));
        
        // Set initial conditions based on radiation environment
        initializeDefectFields(fields, env, material);
        
        // Create free energy functional
        FreeEnergyFunctional F(params);
        
        // Calculate functional derivative δF/δC_i
        std::vector<Field3D> derivatives = F.calculateDerivatives(fields);
        
        // Calculate radiation dose from environment
        double radiation_dose = calculateDose(env, simulation_time);
        
        // Time evolution solver
        TimeEvolutionResults results = solveFieldEquations(
            fields, derivatives, radiation_dose, temperature, applied_stress);
        
        // Verify conservation laws
        double defect_difference = std::abs(results.total_defects_final - results.total_defects_initial);
        double relative_difference = defect_difference / results.total_defects_initial;
        
        std::cout << "    Initial defects: " << results.total_defects_initial << std::endl;
        std::cout << "    Final defects: " << results.total_defects_final << std::endl;
        std::cout << "    Conservation error: " << relative_difference << std::endl;
        
        // Conservation of defects within small numerical error
        assert(relative_difference < 1e-6);
        
        // Verify defect distribution pattern
        double clustering_ratio = calculateClusteringRatio(results.final_fields);
        std::cout << "    Defect clustering ratio: " << clustering_ratio << std::endl;
    }
    
    std::cout << "Field theory tests passed!" << std::endl;
}

// Test 5: Micromechanical Model with Material Properties
void testMicromechanicalModelWithMaterials(const std::map<std::string, MaterialProperties>& materials,
                                        const RadiationEnvironment& env) {
    std::cout << "Testing Micromechanical Model with different materials..." << std::endl;
    
    // Setup environmental conditions
    double temperature = (env.temperature.min + env.temperature.max) / 2.0;
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    
    for (const auto& [name, material] : materials) {
        std::cout << "  Testing material: " << name << std::endl;
        
        // Create crystal lattice based on material
        CrystalLattice crystal = createAppropriateLatticetype(material.z_effective, 
                                                           calculateLatticeConstant(material.density, material.z_effective));
        
        // Initialize strain and stress tensors
        Eigen::Matrix3d strain_applied = Eigen::Matrix3d::Zero();
        strain_applied(0, 0) = 0.001; // uniaxial strain
        
        // Calculate elasticity tensor from material properties
        Eigen::Tensor<double, 4> elasticity_tensor = calculateElasticityTensor(material);
        
        // Create defect strain tensors based on material
        std::vector<Eigen::Matrix3d> defect_strain_tensors = createDefectStrainTensors(material);
        
        // Initialize defect concentration field
        Field3D<double> vacancy_concentration(grid);
        Field3D<double> interstitial_concentration(grid);
        
        // Apply NASA model factors to defect distribution
        double temp_factor = calculateTemperatureCorrectedThreshold(1.0, temperature);
        double stress_factor = calculateMechanicalLoadFactor(
            calculateStressFromStrain(strain_applied, material), material.yield_strength, radiation_dose);
        
        std::cout << "    Temperature factor: " << temp_factor << std::endl;
        std::cout << "    Stress factor: " << stress_factor << std::endl;
        
        // Set defect distributions with NASA model factors
        setDefectDistribution(vacancy_concentration, interstitial_concentration, env, material, 
                            temp_factor, stress_factor);
        
        // Calculate radiation-induced strain field
        Field3D<Eigen::Matrix3d> radiation_strain = 
            calculateRadiationInducedStrain(vacancy_concentration, interstitial_concentration, 
                                          defect_strain_tensors);
        
        // Solve for stress field with elasticity tensor
        Field3D<Eigen::Matrix3d> stress_field = 
            calculateStressField(crystal, strain_applied, radiation_strain, elasticity_tensor);
        
        // Verify mechanical equilibrium
        double divergence = calculateStressDivergence(stress_field);
        std::cout << "    Stress divergence: " << divergence << std::endl;
        
        // Check stress-strain relationship
        double average_stress = calculateAverageStress(stress_field);
        double expected_stress = calculateExpectedStress(strain_applied, material);
        
        std::cout << "    Average stress: " << average_stress << " MPa" << std::endl;
        std::cout << "    Expected stress: " << expected_stress << " MPa" << std::endl;
        
        // Verify mechanical equilibrium and stress calculation
        assert(divergence < 1e-6);
        assert(std::abs(average_stress - expected_stress) < 0.1 * expected_stress);
    }
    
    std::cout << "Micromechanical model tests passed!" << std::endl;
}

// Test 6: Mission Profile Optimization using Material Properties
void testMissionOptimizationWithMaterials(
        const std::map<std::string, MaterialProperties>& materials,
        const std::vector<std::string>& mission_profiles) {
    std::cout << "Testing Mission Profile Optimization with materials..." << std::endl;
    
    // For each mission, find optimal material
    for (const auto& mission : mission_profiles) {
        std::cout << "  Testing mission: " << mission << std::endl;
        
        // Setup radiation environment for this mission
        RadiationEnvironment env;
        setupRadiationEnvironment(mission, env);
        
        // Calculate radiation dose
        double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
        
        // Calculate mission parameters
        MissionParameters mission_params;
        mission_params.duration = 8760; // 1 year in hours
        mission_params.dose_rate = calculateDoseRate(env); // Gy/h
        mission_params.temperature = (env.temperature.min + env.temperature.max) / 2.0;
        mission_params.radiation_spectrum = getRadiationSpectrum(env);
        mission_params.is_vacuum = true;
        mission_params.has_atomic_oxygen = (mission == "LEO");
        
        // Find best material for this mission
        std::string best_material;
        double best_score = -1.0;
        
        for (const auto& [name, material] : materials) {
            // Calculate combined damage effect for this material and mission
            double stress_level = 0.2 * material.yield_strength; // 20% of yield strength
            
            double damage_factor = calculateCombinedDamageEffect(
                material, radiation_dose, mission_params.temperature, 
                stress_level, mission_params.is_vacuum, mission_params.has_atomic_oxygen);
            
            // Calculate material effectiveness score (inverse of damage)
            double protection_score = 1.0 / (1.0 + damage_factor);
            double weight_penalty = material.density / 10.0; // Normalize by 10 g/cm³
            double final_score = protection_score / weight_penalty;
            
            std::cout << "    Material: " << name 
                      << ", Damage: " << damage_factor 
                      << ", Score: " << final_score << std::endl;
            
            if (final_score > best_score) {
                best_score = final_score;
                best_material = name;
            }
        }
        
        std::cout << "  Optimal material for " << mission << ": " << best_material 
                  << " (Score: " << best_score << ")" << std::endl;
        
        // Verify material selection against expected optimal choices
        if (mission == "LEO") {
            assert(best_material == "Aluminum" || best_material == "Polyethylene");
        }
        else if (mission == "MARS") {
            assert(best_material == "Polyethylene" || best_material == "Water");
        }
        else if (mission == "JUPITER") {
            assert(best_material == "Polyethylene" || best_material == "Boron Carbide");
        }
    }
    
    std::cout << "Mission optimization tests passed!" << std::endl;
}

// Main test function
int main() {
    std::cout << "Starting Material Physics Integration Tests..." << std::endl;
    
    // Load material database with properties
    std::map<std::string, MaterialProperties> materials = loadMaterialDatabase();
    
    // Define mission profiles to test
    std::vector<std::string> mission_profiles = {"LEO", "GEO", "MARS", "JUPITER"};
    
    // Test NASA physics models
    testNASAPhysicsModels(materials);
    
    // Run tests for each mission
    for (const auto& mission : mission_profiles) {
        std::cout << "\nTesting with mission profile: " << mission << std::endl;
        
        // Setup radiation environment
        RadiationEnvironment env;
        setupRadiationEnvironment(mission, env);
        
        // Run physics tests with material properties
        testBoltzmannTransportWithMaterials(materials, env);
        testQuantumDefectFormationWithMaterials(materials, env);
        testStochasticDamageEvolutionWithMaterials(materials, env);
        testFieldTheoryWithMaterials(materials, env);
        testMicromechanicalModelWithMaterials(materials, env);
    }
    
    // Test mission optimization
    testMissionOptimizationWithMaterials(materials, mission_profiles);
    
    std::cout << "\nAll material physics integration tests passed successfully!" << std::endl;
    return 0;
} 
