#pragma once

#include "enhanced_stuck_bit_tmr.hpp"
#include "temporal_redundancy.hpp"
#include "../core/recovery/checkpoint_manager.hpp"

namespace rad_ml {
namespace tmr {

/**
 * @brief Hybrid redundancy combining spatial and temporal approaches
 * 
 * Uses both TMR (spatial) and temporal redundancy techniques
 * to provide maximum protection against radiation effects
 */
template <typename T>
class HybridRedundancy {
public:
    /**
     * @brief Constructor for hybrid redundancy
     * 
     * @param initial_value Initial value to protect
     */
    explicit HybridRedundancy(const T& initial_value = T())
        : tmr_(initial_value),
          temporal_redundancy_(3, std::chrono::milliseconds(10)),
          checkpoint_mgr_(std::make_unique<core::recovery::CheckpointManager<T>>(5, std::chrono::seconds(30))),
          radiation_level_(1.0),
          checkpoint_version_(0) {}
    
    /**
     * @brief Get value with combined protections
     * 
     * First uses temporal redundancy to execute TMR get multiple times,
     * providing protection against both spatial and temporal faults
     * 
     * @return Protected value
     */
    T get() const {
        // First use temporal redundancy to execute TMR get multiple times
        return temporal_redundancy_.execute(
            tmr_,
            [](const EnhancedStuckBitTMR<T>& tmr) {
                return tmr.get();
            });
    }
    
    /**
     * @brief Set value with TMR protection
     * 
     * @param value New value to protect
     */
    void set(const T& value) {
        tmr_.set(value);
        
        // Create checkpoint after setting new value
        checkpoint();
    }
    
    /**
     * @brief Enhanced repair combining TMR repair and temporal validation
     * 
     * @return true if repair was successful
     */
    bool repair() {
        // First attempt standard TMR repair
        tmr_.repair();
        
        // Verify if repair was successful
        T before_temporal = tmr_.get();
        
        // Use temporal redundancy as additional validation
        T after_temporal = temporal_redundancy_.execute(
            tmr_,
            [](const EnhancedStuckBitTMR<T>& tmr) {
                return tmr.get();
            });
            
        // If values match, repair was successful
        if (before_temporal == after_temporal) {
            return true;
        }
        
        // If temporal and TMR values disagree, attempt rollback
        return rollback();
    }
    
    /**
     * @brief Configure radiation sensitivity
     * 
     * @param radiation_level Radiation level factor (1.0 = normal)
     */
    void updateRadiationEnvironment(double radiation_level) {
        radiation_level_ = radiation_level;
        
        // Adjust temporal redundancy settings based on radiation
        if (radiation_level > 5.0) {
            // Increase executions and delay in high radiation
            temporal_redundancy_.reconfigure(5, 20);
        } else if (radiation_level > 2.0) {
            // Moderate settings for medium radiation
            temporal_redundancy_.reconfigure(4, 15);
        } else {
            // Default settings for low radiation
            temporal_redundancy_.reconfigure(3, 10);
        }
        
        // Also adjust voting threshold based on radiation level
        adjustVotingThresholds();
    }
    
    /**
     * @brief Create a checkpoint of current state
     */
    void checkpoint() {
        if (checkpoint_mgr_) {
            checkpoint_mgr_->createCheckpoint(tmr_.get(), ++checkpoint_version_);
        }
    }
    
    /**
     * @brief Rollback to last known good state
     * 
     * @return true if rollback was successful
     */
    bool rollback() {
        if (!checkpoint_mgr_) {
            return false;
        }
        
        T checkpoint_value;
        if (checkpoint_mgr_->getLatestCheckpoint(checkpoint_value)) {
            tmr_.set(checkpoint_value);
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Enable checkpoint/rollback capabilities
     * 
     * @param max_checkpoints Maximum number of checkpoints to maintain
     * @param interval_seconds Time between checkpoints in seconds
     */
    void enableCheckpointing(size_t max_checkpoints = 5, 
                             uint64_t interval_seconds = 30) {
        checkpoint_mgr_ = std::make_unique<core::recovery::CheckpointManager<T>>(
            max_checkpoints, std::chrono::seconds(interval_seconds));
    }
    
    /**
     * @brief Get access to underlying TMR for diagnostics
     * 
     * @return Reference to TMR implementation
     */
    const EnhancedStuckBitTMR<T>& getTMR() const {
        return tmr_;
    }
    
    /**
     * @brief Get mutable access to underlying TMR for advanced operations
     * 
     * @return Reference to TMR implementation
     */
    EnhancedStuckBitTMR<T>& getTMR() {
        return tmr_;
    }
    
private:
    EnhancedStuckBitTMR<T> tmr_;
    TemporalRedundancy<EnhancedStuckBitTMR<T>, T> temporal_redundancy_;
    std::unique_ptr<core::recovery::CheckpointManager<T>> checkpoint_mgr_;
    double radiation_level_;
    uint64_t checkpoint_version_;
    double voting_confidence_threshold_{0.6};
    
    /**
     * @brief Adjust voting thresholds based on radiation level
     */
    void adjustVotingThresholds() {
        // In high radiation, be more conservative with voting
        if (radiation_level_ > 5.0) {
            // Require higher confidence for high radiation
            voting_confidence_threshold_ = 0.9;
        } else if (radiation_level_ > 2.0) {
            voting_confidence_threshold_ = 0.8;
        } else {
            // Normal radiation environment
            voting_confidence_threshold_ = 0.6;
        }
    }
};

} // namespace tmr
} // namespace rad_ml 