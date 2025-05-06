/**
 * Adaptive Radiation Protection Strategies Implementation
 * 
 * This file contains implementation details for the adaptive protection strategies.
 */

#pragma once

#include "rad_ml/tmr/adaptive_protection.hpp"

namespace rad_ml {
namespace tmr {

//------------------------------------------------------------------------------
// BasicTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> BasicTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Voting
    TMRResult<T> result;
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
        result.error_detected = false;
        result.error_corrected = false;
    } else if (result1 == result2) {
        // Majority result1/result2
        result.value = result1;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result1 == result3) {
        // Majority result1/result3
        result.value = result1;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result2 == result3) {
        // Majority result2/result3
        result.value = result2;
        result.confidence = 0.67;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No majority, use first result but with low confidence
        result.value = result1;
        result.confidence = 0.33;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// HealthWeightedTMR Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> HealthWeightedTMR<T>::execute(const std::function<T()>& operation) {
    // Execute operation three times
    T result1 = operation();
    T result2 = operation();
    T result3 = operation();
    
    // Calculate weighted voting based on health scores
    double total_health = health_scores[0] + health_scores[1] + health_scores[2];
    double weight1 = health_scores[0] / total_health;
    double weight2 = health_scores[1] / total_health;
    double weight3 = health_scores[2] / total_health;
    
    // Initialize result
    TMRResult<T> result;
    result.error_detected = false;
    result.error_corrected = false;
    
    // Voting with health scores
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
    } else if (result1 == result2) {
        // Check if combined weight is high enough
        double combined_weight = weight1 + weight2;
        if (combined_weight > 0.6) {
            result.value = result1;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            // Update health scores
            updateHealthScores(2, true); // Component 3 had error
        } else {
            // Use highest health component
            int highest_idx = 0;
            double highest_weight = weight1;
            if (weight2 > highest_weight) {
                highest_idx = 1;
                highest_weight = weight2;
            }
            if (weight3 > highest_weight) {
                highest_idx = 2;
                highest_weight = weight3;
            }
            
            result.value = (highest_idx == 0) ? result1 : (highest_idx == 1) ? result2 : result3;
            result.confidence = highest_weight;
            result.error_detected = true;
            result.error_corrected = false;
        }
    } else if (result1 == result3) {
        // Similar logic as above
        double combined_weight = weight1 + weight3;
        if (combined_weight > 0.6) {
            result.value = result1;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            updateHealthScores(1, true);
        } else {
            // Use highest health component
            // ... (similar logic)
        }
    } else if (result2 == result3) {
        // Similar logic as above
        double combined_weight = weight2 + weight3;
        if (combined_weight > 0.6) {
            result.value = result2;
            result.confidence = combined_weight;
            result.error_detected = true;
            result.error_corrected = true;
            updateHealthScores(0, true);
        } else {
            // Use highest health component
            // ... (similar logic)
        }
    } else {
        // No majority, use component with highest health
        if (weight1 >= weight2 && weight1 >= weight3) {
            result.value = result1;
            result.confidence = weight1;
        } else if (weight2 >= weight1 && weight2 >= weight3) {
            result.value = result2;
            result.confidence = weight2;
        } else {
            result.value = result3;
            result.confidence = weight3;
        }
        result.error_detected = true;
        result.error_corrected = false;
        
        // Update all health scores
        updateHealthScores(0, true);
        updateHealthScores(1, true);
        updateHealthScores(2, true);
    }
    
    return result;
}

template<typename T>
void HealthWeightedTMR<T>::updateHealthScores(int component_index, bool had_error) {
    // Update health score based on error
    if (had_error) {
        // Decrease health score, but keep it above 0.1
        health_scores[component_index] = std::max(0.1, health_scores[component_index] * 0.8);
    } else {
        // Increase health score, but keep it below 1.0
        health_scores[component_index] = std::min(1.0, health_scores[component_index] * 1.1);
    }
}

//------------------------------------------------------------------------------
// HybridRedundancy Implementation
//------------------------------------------------------------------------------

template<typename T>
TMRResult<T> HybridRedundancy<T>::execute(const std::function<T()>& operation) {
    // Spatial redundancy
    T result1 = operation();
    
    // Add temporal delay to avoid correlated errors
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(time_delay_ms)));
    
    // More spatial redundancy
    T result2 = operation();
    
    // Another temporal delay
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(time_delay_ms)));
    
    // Final redundant operation
    T result3 = operation();
    
    // Simple voting for now, can be extended with more complex strategies
    TMRResult<T> result;
    if (result1 == result2 && result1 == result3) {
        // All agree
        result.value = result1;
        result.confidence = 1.0;
        result.error_detected = false;
        result.error_corrected = false;
    } else if (result1 == result2) {
        // Majority result1/result2
        result.value = result1;
        result.confidence = 0.8;  // Higher confidence due to temporal separation
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result1 == result3) {
        // Majority result1/result3
        result.value = result1;
        result.confidence = 0.8;
        result.error_detected = true;
        result.error_corrected = true;
    } else if (result2 == result3) {
        // Majority result2/result3
        result.value = result2;
        result.confidence = 0.8;
        result.error_detected = true;
        result.error_corrected = true;
    } else {
        // No majority, use first result but with low confidence
        result.value = result1;
        result.confidence = 0.33;
        result.error_detected = true;
        result.error_corrected = false;
    }
    
    return result;
}

//------------------------------------------------------------------------------
// TMRStrategyFactory Implementation
//------------------------------------------------------------------------------

template<typename T>
std::unique_ptr<TMRStrategy<T>> TMRStrategyFactory::createStrategy(
    const sim::RadiationEnvironment& env,
    const core::MaterialProperties& material,
    double criticality
) {
    // Calculate temperature and stress factors
    double temperature = (env.temperature.min + env.temperature.max) / 2.0;
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    
    // Use a moderate stress level for protection calculation
    double mechanical_stress = 0.1 * material.yield_strength;
    
    // Calculate optimal protection level based on environment
    ProtectionLevel level = calculateOptimalProtectionLevel(
        env, material, criticality, temperature, mechanical_stress
    );
    
    // Create appropriate strategy
    switch (level) {
        case ProtectionLevel::NONE:
            // Just a single computation with no redundancy
            return std::make_unique<BasicTMR<T>>();
            
        case ProtectionLevel::BASIC_TMR:
            return std::make_unique<BasicTMR<T>>();
            
        case ProtectionLevel::ENHANCED_TMR:
            return std::make_unique<EnhancedTMR<T>>();
            
        case ProtectionLevel::STUCK_BIT_TMR:
            return std::make_unique<StuckBitTMR<T>>();
            
        case ProtectionLevel::HEALTH_WEIGHTED_TMR:
            return std::make_unique<HealthWeightedTMR<T>>();
            
        case ProtectionLevel::HYBRID_REDUNDANCY:
            // Calculate appropriate delay based on environment
            double delay = 50.0;  // Base delay in ms
            if (env.trapped_proton_flux > 1.0e8) {
                delay = 200.0;  // Longer delay for high radiation
            }
            return std::make_unique<HybridRedundancy<T>>(delay);
    }
    
    // Default
    return std::make_unique<BasicTMR<T>>();
}

ProtectionLevel TMRStrategyFactory::calculateOptimalProtectionLevel(
    const sim::RadiationEnvironment& env,
    const core::MaterialProperties& material,
    double criticality,
    double temperature_K,
    double mechanical_stress_MPa
) {
    // Calculate radiation environment factors
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    
    // Calculate physics model factors
    double temp_factor = calculateTemperatureCorrectedThreshold(1.0, temperature_K);
    double stress_factor = calculateMechanicalLoadFactor(mechanical_stress_MPa, material.yield_strength, radiation_dose);
    double synergy_factor = calculateSynergyFactor(temperature_K, mechanical_stress_MPa, material.yield_strength);
    
    // Combined environmental effect
    double environmental_factor = temp_factor * stress_factor * synergy_factor;
    
    // Protection needed increases with environment factor and criticality
    double protection_need = environmental_factor * criticality;
    
    // Select appropriate protection level based on protection need
    if (protection_need < 0.2) {
        return ProtectionLevel::NONE;
    } else if (protection_need < 0.5) {
        return ProtectionLevel::BASIC_TMR;
    } else if (protection_need < 1.0) {
        return ProtectionLevel::ENHANCED_TMR;
    } else if (protection_need < 1.5) {
        return ProtectionLevel::STUCK_BIT_TMR;
    } else if (protection_need < 2.0) {
        return ProtectionLevel::HEALTH_WEIGHTED_TMR;
    } else {
        return ProtectionLevel::HYBRID_REDUNDANCY;
    }
}

//------------------------------------------------------------------------------
// CheckpointManager Implementation
//------------------------------------------------------------------------------

CheckpointManager::CheckpointManager(double base_interval_s)
    : base_checkpoint_interval_s(base_interval_s),
      current_interval_s(base_interval_s)
{
}

void CheckpointManager::adjustCheckpointInterval(
    const sim::RadiationEnvironment& env,
    double temp_factor,
    double stress_factor,
    double synergy_factor
) {
    // Calculate radiation intensity
    double radiation_intensity = (env.trapped_proton_flux + env.trapped_electron_flux) / 1.0e6;
    
    // Combined factor affects checkpoint frequency
    double combined_factor = temp_factor * stress_factor * synergy_factor;
    
    // Calculate new interval
    // Higher radiation + higher environmental factors = more frequent checkpoints
    double adjusted_interval = base_checkpoint_interval_s / 
                              (1.0 + radiation_intensity * combined_factor);
    
    // Clamp to reasonable range (10% to 200% of base interval)
    current_interval_s = std::min(
        std::max(adjusted_interval, 0.1 * base_checkpoint_interval_s),
        2.0 * base_checkpoint_interval_s
    );
}

//------------------------------------------------------------------------------
// LayerProtectionPolicy Implementation
//------------------------------------------------------------------------------

LayerProtectionPolicy::LayerProtectionPolicy(int num_layers)
    : layer_protection_levels(num_layers, ProtectionLevel::BASIC_TMR),
      layer_sensitivities(num_layers, 0.5)  // Default medium sensitivity
{
}

void LayerProtectionPolicy::setLayerProtection(int layer_index, ProtectionLevel level) {
    if (layer_index >= 0 && layer_index < layer_protection_levels.size()) {
        layer_protection_levels[layer_index] = level;
    }
}

ProtectionLevel LayerProtectionPolicy::getLayerProtection(int layer_index) const {
    if (layer_index >= 0 && layer_index < layer_protection_levels.size()) {
        return layer_protection_levels[layer_index];
    }
    return ProtectionLevel::BASIC_TMR;  // Default
}

void LayerProtectionPolicy::setLayerSensitivity(int layer_index, double sensitivity) {
    if (layer_index >= 0 && layer_index < layer_sensitivities.size()) {
        layer_sensitivities[layer_index] = sensitivity;
    }
}

void LayerProtectionPolicy::optimizeProtection(
    const sim::RadiationEnvironment& env,
    const core::MaterialProperties& material
) {
    // Calculate environment factors
    double temperature = (env.temperature.min + env.temperature.max) / 2.0;
    double radiation_dose = (env.trapped_proton_flux + env.trapped_electron_flux) * 1.0e-5;
    
    // For each layer, adjust protection based on sensitivity and environment
    for (int i = 0; i < layer_protection_levels.size(); ++i) {
        double sensitivity = layer_sensitivities[i];
        
        // Early layers in neural networks are often more critical
        // Apply higher protection for early layers
        double layer_position_factor = 1.0 - (static_cast<double>(i) / layer_protection_levels.size()) * 0.5;
        
        // Calculate protection need for this layer
        double protection_need = sensitivity * layer_position_factor;
        
        // Determine protection level based on environment and layer sensitivity
        ProtectionLevel level = TMRStrategyFactory::calculateOptimalProtectionLevel(
            env, material, protection_need, temperature, 0.1 * material.yield_strength
        );
        
        // Set protection level for this layer
        layer_protection_levels[i] = level;
    }
}

//------------------------------------------------------------------------------
// MissionAwareProtectionController Implementation
//------------------------------------------------------------------------------

MissionAwareProtectionController::MissionAwareProtectionController(const core::MaterialProperties& mat)
    : material(mat)
{
    // Initialize default mission phase levels
    mission_phase_levels["NOMINAL"] = ProtectionLevel::BASIC_TMR;
    mission_phase_levels["SAA_CROSSING"] = ProtectionLevel::HEALTH_WEIGHTED_TMR;
    mission_phase_levels["SOLAR_STORM"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["LOW_POWER"] = ProtectionLevel::BASIC_TMR;
}

void MissionAwareProtectionController::updateEnvironment(const sim::RadiationEnvironment& env) {
    current_env = env;
    
    // Automatically adjust protection based on radiation levels
    double radiation_intensity = (env.trapped_proton_flux + env.trapped_electron_flux) / 1.0e6;
    
    // Detect South Atlantic Anomaly
    if (env.saa_region) {
        current_global_level = mission_phase_levels["SAA_CROSSING"];
    }
    // Detect high solar activity
    else if (env.solar_activity > 0.7) {
        current_global_level = mission_phase_levels["SOLAR_STORM"];
    }
    // Normal operation with radiation-based adjustment
    else {
        if (radiation_intensity < 1.0) {
            current_global_level = ProtectionLevel::BASIC_TMR;
        } else if (radiation_intensity < 10.0) {
            current_global_level = ProtectionLevel::ENHANCED_TMR;
        } else if (radiation_intensity < 100.0) {
            current_global_level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
        } else {
            current_global_level = ProtectionLevel::HYBRID_REDUNDANCY;
        }
    }
}

void MissionAwareProtectionController::setMissionPhaseProtection(const std::string& phase_name, ProtectionLevel level) {
    mission_phase_levels[phase_name] = level;
}

void MissionAwareProtectionController::enterMissionPhase(const std::string& phase_name) {
    auto it = mission_phase_levels.find(phase_name);
    if (it != mission_phase_levels.end()) {
        current_global_level = it->second;
    }
}

template<typename T>
std::unique_ptr<TMRStrategy<T>> MissionAwareProtectionController::createCurrentStrategy(double criticality) {
    // Create strategy based on current global level
    switch (current_global_level) {
        case ProtectionLevel::NONE:
            return std::make_unique<BasicTMR<T>>();
            
        case ProtectionLevel::BASIC_TMR:
            return std::make_unique<BasicTMR<T>>();
            
        case ProtectionLevel::ENHANCED_TMR:
            return std::make_unique<EnhancedTMR<T>>();
            
        case ProtectionLevel::STUCK_BIT_TMR:
            return std::make_unique<StuckBitTMR<T>>();
            
        case ProtectionLevel::HEALTH_WEIGHTED_TMR:
            return std::make_unique<HealthWeightedTMR<T>>();
            
        case ProtectionLevel::HYBRID_REDUNDANCY:
            return std::make_unique<HybridRedundancy<T>>();
    }
    
    // Default
    return std::make_unique<BasicTMR<T>>();
}

} // namespace tmr
} // namespace rad_ml 