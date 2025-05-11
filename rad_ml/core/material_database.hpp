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
#include <vector>
#include <unordered_map>

namespace rad_ml {
namespace core {

// Material properties relevant for radiation protection
struct MaterialProperties {
    std::string name;
    double density = 0.0;             // g/cm³
    double yield_strength = 0.0;      // MPa
    double thermal_conductivity = 0.0; // W/(m·K)
    double radiation_tolerance = 0.0;  // Relative scale 0-100
    double neutron_cross_section = 0.0; // barns
    double proton_stopping_power = 0.0; // MeV·cm²/g
    double electron_stopping_power = 0.0; // MeV·cm²/g
    
    // Calculate radiation protection factor based on properties
    double getRadiationProtectionFactor(double thickness_mm) const {
        if (density <= 0.0) return 1.0; // No protection
        
        // Simple model for radiation attenuation
        double mass_thickness = density * thickness_mm / 10.0; // g/cm²
        double attenuation = 1.0 - std::exp(-mass_thickness / 10.0);
        
        // Scale by material quality
        return 1.0 + (radiation_tolerance / 100.0) * attenuation * 5.0;
    }
};

// Database of common materials used in spacecraft
class MaterialDatabase {
public:
    MaterialDatabase() {
        initializeMaterials();
    }
    
    // Get material properties by name
    const MaterialProperties& getMaterial(const std::string& name) const {
        auto it = materials_.find(name);
        if (it != materials_.end()) {
            return it->second;
        }
        return default_material_;
    }
    
    // Get all available material names
    std::vector<std::string> getAvailableMaterials() const {
        std::vector<std::string> names;
        for (const auto& pair : materials_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    // Initialize database with common spacecraft materials
    void initializeMaterials() {
        // Aluminum (common spacecraft structural material)
        MaterialProperties aluminum;
        aluminum.name = "Aluminum";
        aluminum.density = 2.7;
        aluminum.yield_strength = 270.0;
        aluminum.thermal_conductivity = 237.0;
        aluminum.radiation_tolerance = 50.0;
        aluminum.neutron_cross_section = 1.4;
        aluminum.proton_stopping_power = 2.1;
        aluminum.electron_stopping_power = 1.6;
        materials_["Aluminum"] = aluminum;
        
        // Titanium (high-strength structural material)
        MaterialProperties titanium;
        titanium.name = "Titanium";
        titanium.density = 4.5;
        titanium.yield_strength = 880.0;
        titanium.thermal_conductivity = 21.9;
        titanium.radiation_tolerance = 70.0;
        titanium.neutron_cross_section = 4.0;
        titanium.proton_stopping_power = 2.8;
        titanium.electron_stopping_power = 1.9;
        materials_["Titanium"] = titanium;
        
        // Polyethylene (radiation shielding material)
        MaterialProperties polyethylene;
        polyethylene.name = "Polyethylene";
        polyethylene.density = 0.94;
        polyethylene.yield_strength = 25.0;
        polyethylene.thermal_conductivity = 0.5;
        polyethylene.radiation_tolerance = 30.0;
        polyethylene.neutron_cross_section = 1.1;
        polyethylene.proton_stopping_power = 2.2;
        polyethylene.electron_stopping_power = 1.7;
        materials_["Polyethylene"] = polyethylene;
        
        // Silicon (semiconductor material for electronics)
        MaterialProperties silicon;
        silicon.name = "Silicon";
        silicon.density = 2.33;
        silicon.yield_strength = 130.0;
        silicon.thermal_conductivity = 149.0;
        silicon.radiation_tolerance = 20.0;
        silicon.neutron_cross_section = 2.2;
        silicon.proton_stopping_power = 1.8;
        silicon.electron_stopping_power = 1.5;
        materials_["Silicon"] = silicon;
        
        // Tantalum (used in capacitors, high radiation tolerance)
        MaterialProperties tantalum;
        tantalum.name = "Tantalum";
        tantalum.density = 16.6;
        tantalum.yield_strength = 170.0;
        tantalum.thermal_conductivity = 57.0;
        tantalum.radiation_tolerance = 85.0;
        tantalum.neutron_cross_section = 21.0;
        tantalum.proton_stopping_power = 4.5;
        tantalum.electron_stopping_power = 2.8;
        materials_["Tantalum"] = tantalum;
        
        // Tungsten (high-density shielding material)
        MaterialProperties tungsten;
        tungsten.name = "Tungsten";
        tungsten.density = 19.3;
        tungsten.yield_strength = 750.0;
        tungsten.thermal_conductivity = 173.0;
        tungsten.radiation_tolerance = 90.0;
        tungsten.neutron_cross_section = 18.4;
        tungsten.proton_stopping_power = 5.1;
        tungsten.electron_stopping_power = 3.2;
        materials_["Tungsten"] = tungsten;
        
        // Carbon Fiber Composite (lightweight structural material)
        MaterialProperties carbon_fiber;
        carbon_fiber.name = "Carbon Fiber Composite";
        carbon_fiber.density = 1.6;
        carbon_fiber.yield_strength = 600.0;
        carbon_fiber.thermal_conductivity = 10.0;
        carbon_fiber.radiation_tolerance = 40.0;
        carbon_fiber.neutron_cross_section = 0.7;
        carbon_fiber.proton_stopping_power = 1.7;
        carbon_fiber.electron_stopping_power = 1.3;
        materials_["Carbon Fiber"] = carbon_fiber;
        
        // Radiation-Hardened Electronics Package
        MaterialProperties rad_hard_electronics;
        rad_hard_electronics.name = "Rad-Hard Electronics Package";
        rad_hard_electronics.density = 3.5;  // Mixed materials
        rad_hard_electronics.yield_strength = 200.0;
        rad_hard_electronics.thermal_conductivity = 80.0;
        rad_hard_electronics.radiation_tolerance = 95.0;
        rad_hard_electronics.neutron_cross_section = 5.0;
        rad_hard_electronics.proton_stopping_power = 3.0;
        rad_hard_electronics.electron_stopping_power = 2.0;
        materials_["Rad-Hard Electronics"] = rad_hard_electronics;
        
        // Default material (properties of air, essentially no protection)
        default_material_.name = "Default";
        default_material_.density = 0.001;
        default_material_.yield_strength = 0.0;
        default_material_.thermal_conductivity = 0.026;
        default_material_.radiation_tolerance = 0.0;
        default_material_.neutron_cross_section = 0.0;
        default_material_.proton_stopping_power = 0.0;
        default_material_.electron_stopping_power = 0.0;
    }
    
    std::unordered_map<std::string, MaterialProperties> materials_;
    MaterialProperties default_material_;
};

} // namespace core
} // namespace rad_ml 
