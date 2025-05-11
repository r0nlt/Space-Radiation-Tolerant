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
 * Physics-Driven Protection System
 * 
 * This file integrates the NASA physics models with the TMR protection strategies
 * to create a radiation-tolerant machine learning framework.
 */

#pragma once

#include "rad_ml/tmr/adaptive_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace rad_ml {
namespace tmr {

/**
 * NASA Physics Model implementations for radiation effects
 */
class PhysicsModels {
public:
    /**
     * Calculate temperature corrected threshold for radiation effects
     * Based on NASA radiation effects models
     * 
     * @param base_threshold Base threshold at reference temperature (295K)
     * @param temperature_K Temperature in Kelvin
     * @return Corrected threshold
     */
    static double calculateTemperatureCorrectedThreshold(double base_threshold, double temperature_K) {
        // NASA model for temperature effect on SEU threshold
        const double reference_temp_K = 295.0;  // Room temperature reference
        const double activation_energy_eV = 0.04;  // Typical activation energy for silicon
        const double boltzmann_constant_eV_K = 8.617e-5;  // Boltzmann constant in eV/K
        
        // Calculate correction factor using Arrhenius equation
        double correction_factor = exp(
            -activation_energy_eV / boltzmann_constant_eV_K * 
            (1.0 / temperature_K - 1.0 / reference_temp_K)
        );
        
        // Higher temperature = lower threshold (more susceptible to radiation)
        return base_threshold * correction_factor;
    }
    
    /**
     * Calculate mechanical stress factor for radiation effects
     * Based on NASA combined radiation/stress models
     * 
     * @param stress_MPa Mechanical stress in MPa
     * @param yield_strength_MPa Material yield strength in MPa
     * @param radiation_dose Radiation dose in rad
     * @return Mechanical load factor (higher = more protection needed)
     */
    static double calculateMechanicalLoadFactor(
        double stress_MPa, 
        double yield_strength_MPa, 
        double radiation_dose
    ) {
        // Normalized stress relative to yield strength (0-1)
        double normalized_stress = std::min(1.0, stress_MPa / yield_strength_MPa);
        
        // NASA model parameters
        const double stress_exponent = 2.0;
        const double dose_sensitivity = 0.01;
        
        // Calculate stress factor (quadratic response to stress level)
        double stress_factor = 1.0 + stress_exponent * pow(normalized_stress, 2);
        
        // Radiation can enhance stress effects (synergistic effect)
        double radiation_factor = 1.0 + dose_sensitivity * radiation_dose;
        
        return stress_factor * radiation_factor;
    }
    
    /**
     * Calculate synergy factor between temperature and mechanical stress
     * Based on NASA synergistic effects model
     * 
     * @param temperature_K Temperature in Kelvin
     * @param stress_MPa Mechanical stress in MPa
     * @param yield_strength_MPa Material yield strength in MPa
     * @return Synergy factor (higher = more protection needed)
     */
    static double calculateSynergyFactor(
        double temperature_K,
        double stress_MPa,
        double yield_strength_MPa
    ) {
        // Normalized stress relative to yield strength (0-1)
        double normalized_stress = std::min(1.0, stress_MPa / yield_strength_MPa);
        
        // Base synergy factor
        double synergy_factor = 1.0;
        
        // Apply NASA synergy factor for high temperature and stress
        // 1.5× when temperature > 350K and stress > 0.3×yield_strength
        if (temperature_K > 350.0 && normalized_stress > 0.3) {
            synergy_factor = 1.5;
        }
        
        // Scale synergy factor continuously based on temperature and stress
        double temp_scale = std::max(0.0, (temperature_K - 300.0) / 50.0); // 0 at 300K, 1 at 350K
        double stress_scale = std::max(0.0, (normalized_stress - 0.1) / 0.2); // 0 at 0.1, 1 at 0.3
        
        // Combined continuous synergy factor
        double continuous_factor = 1.0 + 0.5 * std::min(1.0, temp_scale * stress_scale);
        
        // Use the higher of the two factors
        return std::max(synergy_factor, continuous_factor);
    }
};

/**
 * Sensitivity-based resource allocation for protection
 */
class SensitivityBasedResourceAllocator {
private:
    // Available resources (computational overhead allowed)
    double total_protection_resources = 1.0;
    std::vector<double> allocated_resources;
    
public:
    explicit SensitivityBasedResourceAllocator(int num_layers, double resources = 1.0)
        : total_protection_resources(resources), allocated_resources(num_layers, resources / num_layers)
    {
    }
    
    /**
     * Optimize resource allocation based on layer sensitivities and environmental factors
     * Implements a constrained optimization algorithm
     * 
     * @param layer_sensitivities Sensitivity of each layer to radiation effects
     * @param environment Current radiation environment
     * @param material Material properties
     * @return Vector of allocated protection resources per layer
     */
    std::vector<double> optimizeResourceAllocation(
        const std::vector<double>& layer_sensitivities,
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material
    ) {
        const int num_layers = layer_sensitivities.size();
        std::vector<double> optimal_allocation(num_layers, 0.0);
        
        // Calculate environment severity factor
        double temperature = (env.temperature.min + env.temperature.max) / 2.0;
        double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
        double env_severity = calculateEnvironmentSeverity(env, temperature, radiation_dose);
        
        // Calculate sensitivity-environment products (importance factors)
        std::vector<double> importance_factors(num_layers);
        double sum_importance = 0.0;
        
        for (int i = 0; i < num_layers; i++) {
            // Layer importance is proportional to its sensitivity and position in network
            double position_factor = 1.0 - (0.5 * i / num_layers);  // Earlier layers more important
            importance_factors[i] = layer_sensitivities[i] * env_severity * position_factor;
            sum_importance += importance_factors[i];
        }
        
        // Normalize and allocate resources proportionally to importance
        for (int i = 0; i < num_layers; i++) {
            if (sum_importance > 0.0) {
                optimal_allocation[i] = (importance_factors[i] / sum_importance) * total_protection_resources;
            } else {
                // Equal distribution if all importance factors are zero
                optimal_allocation[i] = total_protection_resources / num_layers;
            }
        }
        
        allocated_resources = optimal_allocation;
        return optimal_allocation;
    }
    
    /**
     * Get protection level based on allocated resources
     * @param layer_index Layer index
     * @return Appropriate protection level based on allocated resources
     */
    ProtectionLevel getProtectionLevel(int layer_index) const {
        if (layer_index < 0 || layer_index >= allocated_resources.size()) {
            return ProtectionLevel::BASIC_TMR;
        }
        
        double resource_level = allocated_resources[layer_index];
        
        // Map resource level to protection level
        if (resource_level < 0.3) {
            return ProtectionLevel::BASIC_TMR;
        } else if (resource_level < 0.6) {
            return ProtectionLevel::ENHANCED_TMR;
        } else if (resource_level < 0.8) {
            return ProtectionLevel::HEALTH_WEIGHTED_TMR;
        } else {
            return ProtectionLevel::HYBRID_REDUNDANCY;
        }
    }
    
private:
    /**
     * Calculate environment severity factor
     */
    double calculateEnvironmentSeverity(
        const sim::RadiationEnvironment& env,
        double temperature_K,
        double radiation_dose
    ) {
        // Base severity from radiation dose
        double severity = radiation_dose * 1.0e-3;
        
        // Apply temperature correction
        double temp_factor = PhysicsModels::calculateTemperatureCorrectedThreshold(1.0, temperature_K);
        severity *= temp_factor;
        
        // Apply solar activity factor
        severity *= (1.0 + env.solar_activity);
        
        // SAA region has higher severity
        if (env.saa_region) {
            severity *= 2.0;
        }
        
        return severity;
    }
};

/**
 * Multi-scale time protection manager
 */
class MultiScaleProtectionManager {
private:
    // Time scales for different protection mechanisms
    enum class TimeScale {
        MICROSECOND,  // Individual computation protection
        SECOND,       // Layer-level validation
        MINUTE,       // Mission phase adaptation
        HOUR,         // System health monitoring
        DAY           // Long-term trend analysis
    };
    
    // Last update timestamps for each scale
    std::map<TimeScale, std::chrono::time_point<std::chrono::steady_clock>> last_updates;
    
    // Update intervals for each scale
    std::map<TimeScale, std::chrono::milliseconds> update_intervals;
    
    // Protection state at each scale
    std::map<TimeScale, int> protection_states;
    
public:
    MultiScaleProtectionManager() {
        // Initialize default update intervals
        auto now = std::chrono::steady_clock::now();
        
        // Initialize last update times
        for (int scale = static_cast<int>(TimeScale::MICROSECOND); 
             scale <= static_cast<int>(TimeScale::DAY); 
             scale++) {
            TimeScale ts = static_cast<TimeScale>(scale);
            last_updates[ts] = now;
            protection_states[ts] = 0;
        }
        
        // Set default update intervals
        update_intervals[TimeScale::MICROSECOND] = std::chrono::milliseconds(1);
        update_intervals[TimeScale::SECOND] = std::chrono::milliseconds(1000);
        update_intervals[TimeScale::MINUTE] = std::chrono::milliseconds(60000);
        update_intervals[TimeScale::HOUR] = std::chrono::milliseconds(3600000);
        update_intervals[TimeScale::DAY] = std::chrono::milliseconds(86400000);
    }
    
    /**
     * Update protection at appropriate time scales
     * @param env Current radiation environment
     * @param material Material properties
     * @return Highest time scale that was updated
     */
    TimeScale updateProtection(
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material
    ) {
        auto now = std::chrono::steady_clock::now();
        TimeScale highest_updated = TimeScale::MICROSECOND;
        
        // Check each time scale
        for (int scale = static_cast<int>(TimeScale::MICROSECOND); 
             scale <= static_cast<int>(TimeScale::DAY); 
             scale++) {
            
            TimeScale ts = static_cast<TimeScale>(scale);
            auto elapsed = now - last_updates[ts];
            
            // If enough time has passed for this scale
            if (elapsed >= update_intervals[ts]) {
                // Update protection at this scale
                updateProtectionAtScale(ts, env, material);
                last_updates[ts] = now;
                highest_updated = ts;
            }
        }
        
        return highest_updated;
    }
    
    /**
     * Get protection adjustment factor based on all time scales
     * @return Combined protection adjustment factor
     */
    double getProtectionFactor() const {
        // Base protection factor
        double factor = 1.0;
        
        // Apply adjustments from each time scale
        for (int scale = static_cast<int>(TimeScale::MICROSECOND); 
             scale <= static_cast<int>(TimeScale::DAY); 
             scale++) {
            
            TimeScale ts = static_cast<TimeScale>(scale);
            
            // Different scales have different weights
            double scale_weight = 0.1;
            switch (ts) {
                case TimeScale::MICROSECOND: scale_weight = 0.2; break;
                case TimeScale::SECOND: scale_weight = 0.3; break;
                case TimeScale::MINUTE: scale_weight = 0.2; break;
                case TimeScale::HOUR: scale_weight = 0.2; break;
                case TimeScale::DAY: scale_weight = 0.1; break;
            }
            
            // Apply protection state at this scale
            factor *= (1.0 + scale_weight * protection_states.at(ts) / 10.0);
        }
        
        return factor;
    }
    
private:
    /**
     * Update protection at a specific time scale
     */
    void updateProtectionAtScale(
        TimeScale scale,
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material
    ) {
        // Different logic based on time scale
        switch (scale) {
            case TimeScale::MICROSECOND:
                // Microsecond scale - individual computation protection
                // (typically handled directly in TMR strategies)
                protection_states[scale] = static_cast<int>(env.trapped_proton_flux / 1.0e7);
                break;
                
            case TimeScale::SECOND:
                // Second scale - layer-level validation
                protection_states[scale] = env.saa_region ? 10 : 
                                          (env.solar_activity > 0.7 ? 8 : 5);
                break;
                
            case TimeScale::MINUTE:
                // Minute scale - mission phase adaptation
                protection_states[scale] = calculateMissionPhaseState(env);
                break;
                
            case TimeScale::HOUR:
                // Hour scale - system health monitoring
                protection_states[scale] = calculateSystemHealthState(env, material);
                break;
                
            case TimeScale::DAY:
                // Day scale - long-term trends
                protection_states[scale] = calculateLongTermState(env);
                break;
        }
    }
    
    /**
     * Calculate mission phase state based on environment
     */
    int calculateMissionPhaseState(const sim::RadiationEnvironment& env) {
        if (env.saa_region) return 10;  // South Atlantic Anomaly - highest protection
        if (env.solar_activity > 0.8) return 9;  // Solar storm
        if (env.solar_activity > 0.5) return 7;  // Elevated solar activity
        
        double radiation_level = (env.trapped_proton_flux + env.trapped_electron_flux) / 1.0e8;
        return std::min(10, static_cast<int>(radiation_level * 10));
    }
    
    /**
     * Calculate system health state
     */
    int calculateSystemHealthState(
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material
    ) {
        // This would normally incorporate actual system health metrics
        // For now, use a simplified model based on environment and material
        double temperature = (env.temperature.min + env.temperature.max) / 2.0;
        double temp_factor = PhysicsModels::calculateTemperatureCorrectedThreshold(1.0, temperature);
        
        // More resilient materials maintain better health
        double material_factor = material.radiation_tolerance / 100.0;
        
        // Simple health model (0-10 scale, 10 = needs most protection)
        double health_metric = 5.0 * temp_factor / material_factor;
        return std::min(10, static_cast<int>(health_metric));
    }
    
    /**
     * Calculate long-term state based on environment trends
     */
    int calculateLongTermState(const sim::RadiationEnvironment& env) {
        // This would normally use historical data
        // For now, use solar cycle position as proxy
        return std::min(10, static_cast<int>(env.solar_activity * 10));
    }
};

/**
 * Physics-driven protection system that adapts to space radiation environments
 */
class PhysicsDrivenProtection {
private:
    sim::RadiationEnvironment current_env;
    core::MaterialProperties material;
    MissionAwareProtectionController mission_controller;
    LayerProtectionPolicy layer_policy;
    CheckpointManager checkpoint_mgr;
    
    // Enhanced components
    SensitivityBasedResourceAllocator resource_allocator;
    MultiScaleProtectionManager multi_scale_manager;
    std::vector<double> layer_sensitivities;
    
    // Internal state
    double last_temperature_factor = 1.0;
    double last_stress_factor = 1.0;
    double last_synergy_factor = 1.0;
    
public:
    /**
     * Create physics-driven protection system
     * @param material Material properties for radiation protection
     * @param num_layers Number of neural network layers (if applicable)
     */
    PhysicsDrivenProtection(
        const core::MaterialProperties& mat, 
        int num_layers = 1
    ) : material(mat),
        mission_controller(mat),
        layer_policy(num_layers),
        checkpoint_mgr(300.0),  // 5 minutes default
        resource_allocator(num_layers),
        layer_sensitivities(num_layers, 0.5)  // Default medium sensitivity
    {
    }
    
    /**
     * Set layer sensitivity based on analysis or empirical measurements
     * @param layer_index Layer index
     * @param sensitivity Sensitivity value (0-1)
     */
    void setLayerSensitivity(int layer_index, double sensitivity) {
        if (layer_index >= 0 && layer_index < layer_sensitivities.size()) {
            layer_sensitivities[layer_index] = sensitivity;
            layer_policy.setLayerSensitivity(layer_index, sensitivity);
        }
    }
    
    /**
     * Update environment and adjust protection accordingly
     * @param env Updated radiation environment
     * @param mechanical_stress_MPa Current mechanical stress
     */
    void updateEnvironment(
        const sim::RadiationEnvironment& env,
        double mechanical_stress_MPa = 0.0
    ) {
        current_env = env;
        
        // Calculate average temperature
        double temperature = (env.temperature.min + env.temperature.max) / 2.0;
        
        // Calculate radiation dose
        double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
        
        // Calculate physics model factors using NASA models
        last_temperature_factor = PhysicsModels::calculateTemperatureCorrectedThreshold(1.0, temperature);
        last_stress_factor = PhysicsModels::calculateMechanicalLoadFactor(
            mechanical_stress_MPa, material.yield_strength, radiation_dose);
        last_synergy_factor = PhysicsModels::calculateSynergyFactor(
            temperature, mechanical_stress_MPa, material.yield_strength);
        
        // Update mission controller
        mission_controller.updateEnvironment(env);
        
        // Optimize layer protection based on new environment
        layer_policy.optimizeProtection(env, material);
        
        // Adjust checkpoint intervals
        checkpoint_mgr.adjustCheckpointInterval(
            env, 
            last_temperature_factor,
            last_stress_factor,
            last_synergy_factor
        );
        
        // Enhanced protection: Sensitivity-based resource allocation
        resource_allocator.optimizeResourceAllocation(
            layer_sensitivities, env, material);
            
        // Enhanced protection: Multi-scale time protection
        multi_scale_manager.updateProtection(env, material);
    }
    
    /**
     * Execute operation with physics-driven protection
     * @param operation Operation to execute
     * @param layer_index Neural network layer index (if applicable)
     * @param criticality Component criticality (0-1)
     * @return Protected result
     */
    template<typename T>
    TMRResult<T> executeProtected(
        const std::function<T()>& operation,
        int layer_index = 0,
        double criticality = 1.0
    ) {
        // Get protection level from resource allocator (sensitivity-based)
        ProtectionLevel allocated_level = resource_allocator.getProtectionLevel(layer_index);
        
        // Get protection level from layer policy (environment-based)
        ProtectionLevel layer_level = layer_policy.getLayerProtection(layer_index);
        
        // Use the higher protection level from the two approaches
        ProtectionLevel level = (allocated_level > layer_level) ? allocated_level : layer_level;
        
        // Create appropriate strategy
        std::unique_ptr<TMRStrategy<T>> strategy;
        
        // Pre-calculate any needed values for strategy initialization
        double protection_factor = multi_scale_manager.getProtectionFactor();
        
        // Use layer-specific protection if available, otherwise use global
        if (level != ProtectionLevel::NONE) {
            if (level == ProtectionLevel::BASIC_TMR) {
                strategy = std::make_unique<BasicTMR<T>>();
            } else if (level == ProtectionLevel::ENHANCED_TMR) {
                strategy = std::make_unique<EnhancedTMR<T>>();
            } else if (level == ProtectionLevel::STUCK_BIT_TMR) {
                strategy = std::make_unique<StuckBitTMR<T>>();
            } else if (level == ProtectionLevel::HEALTH_WEIGHTED_TMR) {
                strategy = std::make_unique<HealthWeightedTMR<T>>();
            } else if (level == ProtectionLevel::HYBRID_REDUNDANCY) {
                double delay = 50.0 * protection_factor;
                strategy = std::make_unique<HybridRedundancy<T>>(delay);
            } else {
                // Default case
                strategy = std::make_unique<BasicTMR<T>>();
            }
        } else {
            // Fall back to global protection
            strategy = mission_controller.createCurrentStrategy<T>(criticality);
        }
        
        // Execute with chosen protection
        return strategy->execute(operation);
    }
    
    /**
     * Get current checkpoint interval
     */
    double getCheckpointInterval() const {
        return checkpoint_mgr.getCurrentInterval();
    }
    
    /**
     * Enter a specific mission phase
     */
    void enterMissionPhase(const std::string& phase_name) {
        mission_controller.enterMissionPhase(phase_name);
    }
    
    /**
     * Get the current global protection level
     */
    ProtectionLevel getCurrentGlobalProtection() const {
        return mission_controller.getCurrentProtectionLevel();
    }
    
    /**
     * Get layer-specific protection level
     */
    ProtectionLevel getLayerProtection(int layer_index) const {
        return layer_policy.getLayerProtection(layer_index);
    }
    
    /**
     * Get the current physics model factors
     */
    void getCurrentFactors(
        double& temp_factor,
        double& stress_factor,
        double& synergy_factor
    ) const {
        temp_factor = last_temperature_factor;
        stress_factor = last_stress_factor;
        synergy_factor = last_synergy_factor;
    }
};

/**
 * Neural network layer with physics-driven protection
 * This is a simplified example layer to demonstrate integration
 */
template<typename T>
class ProtectedNeuralLayer {
public:
    int layer_index;
    PhysicsDrivenProtection& protection;
    double criticality;
    std::vector<T> weights;
    std::vector<T> biases;
    
    ProtectedNeuralLayer(
        int idx,
        PhysicsDrivenProtection& prot,
        double critical = 1.0
    ) : layer_index(idx),
        protection(prot),
        criticality(critical)
    {
    }
    
    /**
     * Set weights with protection
     */
    void setWeights(const std::vector<T>& new_weights) {
        // Execute weight update with protection
        auto update_op = [&]() {
            weights = new_weights;
            return true;
        };
        
        protection.executeProtected<bool>(update_op, layer_index, criticality);
    }
    
    /**
     * Set biases with protection
     */
    void setBiases(const std::vector<T>& new_biases) {
        // Execute bias update with protection
        auto update_op = [&]() {
            biases = new_biases;
            return true;
        };
        
        protection.executeProtected<bool>(update_op, layer_index, criticality);
    }
    
    /**
     * Forward pass with protection
     */
    std::vector<T> forward(const std::vector<T>& inputs) {
        // Create operation that computes forward pass
        auto forward_op = [&]() {
            // Check if weights and biases are properly initialized
            if (weights.empty() || biases.empty()) {
                // Return empty vector if not initialized
                return std::vector<T>{};
            }
            
            // Calculate expected output size
            size_t output_size = weights.size() / inputs.size();
            if (output_size == 0 || output_size != biases.size()) {
                // Dimensions mismatch, return empty result
                return std::vector<T>{};
            }
            
            std::vector<T> outputs(output_size);
            
            // Simple matrix multiplication (in reality would be more complex)
            for (size_t i = 0; i < outputs.size(); ++i) {
                outputs[i] = biases[i];
                for (size_t j = 0; j < inputs.size(); ++j) {
                    outputs[i] += weights[i * inputs.size() + j] * inputs[j];
                }
                // Apply activation function (ReLU for simplicity)
                outputs[i] = outputs[i] > 0 ? outputs[i] : 0;
            }
            
            return outputs;
        };
        
        // Execute with protection and return result
        TMRResult<std::vector<T>> result = 
            protection.executeProtected<std::vector<T>>(forward_op, layer_index, criticality);
        
        return result.value;
    }
};

/**
 * Demonstrate usage of physics-driven protection for a simple neural network
 */
void demonstratePhysicsDrivenProtection() {
    // Material properties (aluminum for example)
    core::MaterialProperties aluminum;
    aluminum.yield_strength = 270.0; // MPa
    aluminum.radiation_tolerance = 50.0; // Relative scale
    
    // Create physics-driven protection with 3 network layers
    PhysicsDrivenProtection protection(aluminum, 3);
    
    // Set different sensitivities based on analysis
    protection.setLayerSensitivity(0, 0.9);  // Input layer (high sensitivity)
    protection.setLayerSensitivity(1, 0.6);  // Hidden layer (medium sensitivity)
    protection.setLayerSensitivity(2, 0.3);  // Output layer (lower sensitivity)
    
    // Set different criticality for each layer
    ProtectedNeuralLayer<float> layer1(0, protection, 1.0);   // Input layer (high criticality)
    ProtectedNeuralLayer<float> layer2(1, protection, 0.7);   // Hidden layer (medium criticality)
    ProtectedNeuralLayer<float> layer3(2, protection, 0.5);   // Output layer (lower criticality)
    
    // Example LEO environment
    sim::RadiationEnvironment leo_env;
    leo_env.trapped_proton_flux = 1.0e8;
    leo_env.trapped_electron_flux = 5.0e7;
    leo_env.temperature.min = 270.0; // K
    leo_env.temperature.max = 290.0; // K
    leo_env.solar_activity = 0.4;
    leo_env.saa_region = false;
    
    // Update protection with current environment
    protection.updateEnvironment(leo_env, 20.0); // 20 MPa mechanical stress
    
    // Example input
    std::vector<float> input = {0.1f, 0.2f, 0.3f};
    
    // Forward pass through the network, with TMR protection
    std::vector<float> hidden = layer1.forward(input);
    std::vector<float> output = layer2.forward(hidden);
    std::vector<float> result = layer3.forward(output);
    
    // Enter high-radiation region (South Atlantic Anomaly)
    leo_env.saa_region = true;
    leo_env.trapped_proton_flux = 5.0e9;
    leo_env.temperature.min = 310.0; // K
    leo_env.temperature.max = 350.0; // K
    
    // Update protection for new environment
    protection.enterMissionPhase("SAA_CROSSING");
    protection.updateEnvironment(leo_env, 50.0); // Increased mechanical stress
    
    // Continue processing with enhanced protection
    std::vector<float> protected_result = layer3.forward(output);
}

} // namespace tmr
} // namespace rad_ml 
