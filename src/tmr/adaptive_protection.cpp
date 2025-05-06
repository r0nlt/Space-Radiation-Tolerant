/**
 * Adaptive Radiation Protection Strategies Implementation
 * 
 * This file contains implementations for the non-template functions in adaptive_protection.hpp
 */

#include "rad_ml/tmr/adaptive_protection.hpp"
#include "rad_ml/tmr/adaptive_protection_impl.hpp" // Include template implementations
#include <thread>
#include <algorithm>

namespace rad_ml {
namespace tmr {

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
    // Start with base interval
    double adjusted_interval = base_checkpoint_interval_s;
    
    // Apply temperature factor (higher temperature = more frequent checkpoints)
    adjusted_interval /= temp_factor;
    
    // Apply stress factor (higher stress = more frequent checkpoints)
    adjusted_interval /= stress_factor;
    
    // Apply synergy factor (higher synergy = more frequent checkpoints)
    adjusted_interval /= synergy_factor;
    
    // Special case for SAA region (more frequent checkpoints)
    if (env.saa_region) {
        adjusted_interval /= 2.0;
    }
    
    // Apply solar activity factor (higher activity = more frequent checkpoints)
    adjusted_interval /= (1.0 + env.solar_activity);
    
    // Ensure reasonable bounds (between 10s and 1 hour)
    adjusted_interval = std::max(10.0, std::min(3600.0, adjusted_interval));
    
    // Update current interval
    current_interval_s = adjusted_interval;
}

//------------------------------------------------------------------------------
// LayerProtectionPolicy Implementation
//------------------------------------------------------------------------------

LayerProtectionPolicy::LayerProtectionPolicy(int num_layers)
    : layer_protection_levels(num_layers, ProtectionLevel::BASIC_TMR),
      layer_sensitivities(num_layers, 0.5)
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
    return ProtectionLevel::BASIC_TMR; // Default protection level
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
        
        // Calculate effective protection level based on radiation environment
        ProtectionLevel level;
        
        // Higher protection for high radiation or critical layers
        if (env.saa_region || env.solar_activity > 0.7 || protection_need > 0.8) {
            level = ProtectionLevel::HYBRID_REDUNDANCY;
        } else if (env.solar_activity > 0.5 || protection_need > 0.6) {
            level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
        } else if (env.solar_activity > 0.3 || protection_need > 0.4) {
            level = ProtectionLevel::ENHANCED_TMR;
        } else {
            level = ProtectionLevel::BASIC_TMR;
        }
        
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
    // Initialize mission phase protection levels
    mission_phase_levels["LAUNCH"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["ORBIT_INSERTION"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["NORMAL_OPERATION"] = ProtectionLevel::BASIC_TMR;
    mission_phase_levels["SAA_CROSSING"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["SOLAR_STORM"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["CRITICAL_OPERATION"] = ProtectionLevel::HYBRID_REDUNDANCY;
    mission_phase_levels["LOW_POWER"] = ProtectionLevel::ENHANCED_TMR;
    mission_phase_levels["SAFE_MODE"] = ProtectionLevel::HYBRID_REDUNDANCY;
}

void MissionAwareProtectionController::updateEnvironment(const sim::RadiationEnvironment& env) {
    current_env = env;
    
    // Determine appropriate protection level based on environment
    if (env.saa_region) {
        current_global_level = ProtectionLevel::HYBRID_REDUNDANCY;
    } else if (env.solar_activity > 0.7) {
        current_global_level = ProtectionLevel::HYBRID_REDUNDANCY;
    } else if (env.solar_activity > 0.5) {
        current_global_level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
    } else if (env.solar_activity > 0.3) {
        current_global_level = ProtectionLevel::ENHANCED_TMR;
    } else {
        current_global_level = ProtectionLevel::BASIC_TMR;
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

// Explicit instantiations for commonly used types
template class BasicTMR<float>;
template class BasicTMR<int>;
template class BasicTMR<std::vector<float>>;

template class EnhancedTMR<float>;
template class EnhancedTMR<int>;
template class EnhancedTMR<std::vector<float>>;

template class StuckBitTMR<float>;
template class StuckBitTMR<int>;
template class StuckBitTMR<std::vector<float>>;

template class HealthWeightedTMR<float>;
template class HealthWeightedTMR<int>;
template class HealthWeightedTMR<std::vector<float>>;

template class HybridRedundancy<float>;
template class HybridRedundancy<int>;
template class HybridRedundancy<std::vector<float>>;

template std::unique_ptr<TMRStrategy<float>> MissionAwareProtectionController::createCurrentStrategy<float>(double);
template std::unique_ptr<TMRStrategy<int>> MissionAwareProtectionController::createCurrentStrategy<int>(double);
template std::unique_ptr<TMRStrategy<std::vector<float>>> MissionAwareProtectionController::createCurrentStrategy<std::vector<float>>(double);

// Additional explicit instantiations for NoProtection
template class NoProtection<float>;
template class NoProtection<int>;
template class NoProtection<std::vector<float>>;

// Add instantiations for the bool type
template class BasicTMR<bool>;
template class EnhancedTMR<bool>;
template class StuckBitTMR<bool>;
template class HealthWeightedTMR<bool>;
template class HybridRedundancy<bool>;
template class NoProtection<bool>;
template std::unique_ptr<TMRStrategy<bool>> MissionAwareProtectionController::createCurrentStrategy<bool>(double);

} // namespace tmr
} // namespace rad_ml 