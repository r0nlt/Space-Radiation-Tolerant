/**
 * Adaptive Radiation Protection Strategies
 * 
 * This file contains strategies that dynamically adapt protection levels
 * based on physics models and mission environment.
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>

#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

namespace rad_ml {
namespace tmr {

/**
 * Protection level enumeration with increasing levels of protection
 */
enum class ProtectionLevel {
    NONE,               // No protection (for non-critical operations)
    BASIC_TMR,          // Basic Triple Modular Redundancy
    ENHANCED_TMR,       // Enhanced TMR with CRC checksums
    STUCK_BIT_TMR,      // TMR with stuck-bit detection
    HEALTH_WEIGHTED_TMR, // TMR with health tracking
    HYBRID_REDUNDANCY   // Combined spatial and temporal redundancy
};

/**
 * Result type for TMR operations containing value and confidence
 */
template<typename T>
struct TMRResult {
    T value;                  // The computed value
    double confidence;        // Confidence level (0-1)
    bool error_detected;      // Whether an error was detected
    bool error_corrected;     // Whether an error was corrected
};

/**
 * Base class for all TMR strategies
 */
template<typename T>
class TMRStrategy {
public:
    virtual ~TMRStrategy() = default;
    
    /**
     * Execute operation with TMR protection
     * @param operation The function to execute with protection
     * @return TMR result with value and confidence
     */
    virtual TMRResult<T> execute(const std::function<T()>& operation) = 0;
    
    /**
     * Get the protection level of this strategy
     * @return Protection level enum
     */
    virtual ProtectionLevel getProtectionLevel() const = 0;
};

/**
 * Basic TMR with majority voting
 */
template<typename T>
class BasicTMR : public TMRStrategy<T> {
public:
    TMRResult<T> execute(const std::function<T()>& operation) override;
    ProtectionLevel getProtectionLevel() const override { return ProtectionLevel::BASIC_TMR; }
};

/**
 * Enhanced TMR with CRC checksums
 */
template<typename T>
class EnhancedTMR : public TMRStrategy<T> {
public:
    TMRResult<T> execute(const std::function<T()>& operation) override;
    ProtectionLevel getProtectionLevel() const override { return ProtectionLevel::ENHANCED_TMR; }
};

/**
 * Stuck-Bit TMR with specialized bit-level protection
 */
template<typename T>
class StuckBitTMR : public TMRStrategy<T> {
public:
    TMRResult<T> execute(const std::function<T()>& operation) override;
    ProtectionLevel getProtectionLevel() const override { return ProtectionLevel::STUCK_BIT_TMR; }
};

/**
 * Health-Weighted TMR with component health tracking
 */
template<typename T>
class HealthWeightedTMR : public TMRStrategy<T> {
private:
    std::vector<double> health_scores = {1.0, 1.0, 1.0};
    
public:
    TMRResult<T> execute(const std::function<T()>& operation) override;
    ProtectionLevel getProtectionLevel() const override { return ProtectionLevel::HEALTH_WEIGHTED_TMR; }
    
    /**
     * Update health scores based on error history
     */
    void updateHealthScores(int component_index, bool had_error);
};

/**
 * Hybrid Redundancy combining spatial and temporal approaches
 */
template<typename T>
class HybridRedundancy : public TMRStrategy<T> {
private:
    double time_delay_ms = 50.0;  // Time delay between redundant operations
    
public:
    explicit HybridRedundancy(double delay_ms = 50.0) : time_delay_ms(delay_ms) {}
    
    TMRResult<T> execute(const std::function<T()>& operation) override;
    ProtectionLevel getProtectionLevel() const override { return ProtectionLevel::HYBRID_REDUNDANCY; }
    
    /**
     * Set time delay for temporal redundancy
     */
    void setTimeDelay(double delay_ms) { time_delay_ms = delay_ms; }
};

/**
 * Factory for creating appropriate TMR strategy based on environment
 */
class TMRStrategyFactory {
public:
    /**
     * Create appropriate TMR strategy based on environment
     * @param env Radiation environment
     * @param material Material properties
     * @param criticality Component criticality (0-1)
     * @return TMR strategy instance
     */
    template<typename T>
    static std::unique_ptr<TMRStrategy<T>> createStrategy(
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material,
        double criticality
    );
    
    /**
     * Calculate optimal protection level based on environment
     */
    static ProtectionLevel calculateOptimalProtectionLevel(
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material,
        double criticality,
        double temperature_K,
        double mechanical_stress_MPa
    );
};

/**
 * Manager for checkpoint-recovery with physics-based intervals
 */
class CheckpointManager {
private:
    double base_checkpoint_interval_s;
    double current_interval_s;
    
public:
    explicit CheckpointManager(double base_interval_s = 300.0);
    
    /**
     * Adjust checkpoint interval based on physics models
     * @param env Radiation environment
     * @param temp_factor Temperature factor from physics model
     * @param stress_factor Mechanical stress factor from physics model
     * @param synergy_factor Synergy factor from physics model
     */
    void adjustCheckpointInterval(
        const sim::RadiationEnvironment& env,
        double temp_factor,
        double stress_factor,
        double synergy_factor
    );
    
    /**
     * Get current checkpoint interval
     */
    double getCurrentInterval() const { return current_interval_s; }
};

/**
 * Layer-specific protection for neural networks
 */
class LayerProtectionPolicy {
private:
    std::vector<ProtectionLevel> layer_protection_levels;
    std::vector<double> layer_sensitivities;
    
public:
    /**
     * Initialize with default protection levels
     * @param num_layers Number of layers in the network
     */
    explicit LayerProtectionPolicy(int num_layers);
    
    /**
     * Set protection level for a specific layer
     */
    void setLayerProtection(int layer_index, ProtectionLevel level);
    
    /**
     * Get protection level for a specific layer
     */
    ProtectionLevel getLayerProtection(int layer_index) const;
    
    /**
     * Set layer sensitivity based on analysis
     */
    void setLayerSensitivity(int layer_index, double sensitivity);
    
    /**
     * Optimize protection levels based on environment and sensitivities
     */
    void optimizeProtection(
        const sim::RadiationEnvironment& env,
        const core::MaterialProperties& material
    );
};

/**
 * Bit-specific protection strategies
 */
class BitProtectionStrategy {
public:
    /**
     * Apply Hamming code to protect data
     * @param data Data to protect
     * @return Protected data with Hamming codes
     */
    template<typename T>
    static std::vector<T> applyHammingCode(const std::vector<T>& data);
    
    /**
     * Decode and correct Hamming-protected data
     * @param protected_data Protected data with Hamming codes
     * @return Decoded and corrected data
     */
    template<typename T>
    static std::vector<T> decodeHammingCode(const std::vector<T>& protected_data);
    
    /**
     * Detect stuck bits in memory
     * @param data Data to check
     * @param reference Reference data (if available)
     * @return Indices of detected stuck bits
     */
    template<typename T>
    static std::vector<int> detectStuckBits(const std::vector<T>& data, const std::vector<T>* reference = nullptr);
};

/**
 * Mission-aware protection controller that adapts protection based on mission phase
 */
class MissionAwareProtectionController {
private:
    ProtectionLevel current_global_level = ProtectionLevel::BASIC_TMR;
    sim::RadiationEnvironment current_env;
    core::MaterialProperties material;
    std::map<std::string, ProtectionLevel> mission_phase_levels;
    
public:
    /**
     * Initialize with material and default environment
     */
    explicit MissionAwareProtectionController(const core::MaterialProperties& mat);
    
    /**
     * Update environment and adjust protection
     */
    void updateEnvironment(const sim::RadiationEnvironment& env);
    
    /**
     * Set protection level for a specific mission phase
     */
    void setMissionPhaseProtection(const std::string& phase_name, ProtectionLevel level);
    
    /**
     * Enter mission phase with predefined protection level
     */
    void enterMissionPhase(const std::string& phase_name);
    
    /**
     * Get current global protection level
     */
    ProtectionLevel getCurrentProtectionLevel() const { return current_global_level; }
    
    /**
     * Create appropriate TMR strategy based on current protection level
     */
    template<typename T>
    std::unique_ptr<TMRStrategy<T>> createCurrentStrategy(double criticality = 1.0);
};

} // namespace tmr
} // namespace rad_ml 