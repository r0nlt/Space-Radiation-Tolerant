#pragma once

#include "rad_ml/tmr/tmr.hpp"
#include <functional>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>
#include <map>
#include <string>

namespace rad_ml {
namespace tmr {

/**
 * @brief CRC-32 calculator
 * 
 * Implements CRC-32 calculation for data integrity verification
 */
class CRC32 {
public:
    /**
     * @brief Initialize the CRC lookup table
     */
    CRC32() {
        uint32_t polynomial = 0xEDB88320;
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (size_t j = 0; j < 8; j++) {
                if (c & 1) {
                    c = polynomial ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            table_[i] = c;
        }
    }

    /**
     * @brief Calculate the CRC-32 checksum of data
     * 
     * @param data Pointer to the data
     * @param length Length of the data in bytes
     * @return CRC-32 checksum
     */
    uint32_t calculate(const void* data, size_t length) const {
        uint32_t crc = 0xFFFFFFFF;
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        
        for (size_t i = 0; i < length; i++) {
            uint8_t index = (crc ^ bytes[i]) & 0xFF;
            crc = (crc >> 8) ^ table_[index];
        }
        
        return ~crc;
    }

private:
    uint32_t table_[256];
};

/**
 * @brief Enhanced Triple Modular Redundancy with CRC checking
 * 
 * Extends the basic TMR class with additional features:
 * - CRC-32 integrity checking for each copy
 * - Adaptive voting threshold
 * - Health monitoring of copies
 * - Logging and error statistics
 * 
 * @tparam T Type of the data to be protected
 */
template <typename T>
class EnhancedTMR {
public:
    /**
     * @brief Constructor
     * 
     * @param initial_value Initial value
     * @param on_error_callback Optional callback when error is detected
     */
    explicit EnhancedTMR(const T& initial_value = T(), 
                        std::function<void(const T&, const T&)> on_error_callback = nullptr)
        : on_error_callback_(on_error_callback), 
          last_verification_time_(std::chrono::steady_clock::now()) {
        // Initialize all copies with the same value
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = initial_value;
        }
        
        // Calculate initial CRC values
        recalculateCRCs();
        
        // Initialize health scores
        for (size_t i = 0; i < num_copies_; i++) {
            health_scores_[i] = 1.0;
        }
    }
    
    /**
     * @brief Get the current value with majority voting
     * 
     * @return Current value based on majority voting with health weighting
     */
    T get() {
        // Verify CRCs before performing operation
        verifyCRCs();
        
        // Use weighted voting for enhanced resilience
        return performWeightedVoting();
    }
    
    /**
     * @brief Set a new value to all copies
     * 
     * @param value New value
     */
    void set(const T& value) {
        for (size_t i = 0; i < num_copies_; i++) {
            copies_[i] = value;
            health_scores_[i] = 1.0; // Reset health on explicit set
        }
        
        recalculateCRCs();
        total_set_operations_++;
    }
    
    /**
     * @brief Get the raw value of a specific copy (for testing)
     * 
     * @param index Index of the copy (0-2)
     * @return Raw value of the specified copy
     */
    T getRawCopy(size_t index) const {
        if (index < num_copies_) {
            return copies_[index];
        }
        // Return the most trusted copy if index out of bounds
        size_t best_idx = 0;
        double best_health = health_scores_[0];
        for (size_t i = 1; i < num_copies_; i++) {
            if (health_scores_[i] > best_health) {
                best_health = health_scores_[i];
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    /**
     * @brief Set the raw value of a specific copy (for testing)
     * 
     * @param index Index of the copy (0-2)
     * @param value New value
     */
    void setRawCopy(size_t index, const T& value) {
        if (index < num_copies_) {
            copies_[index] = value;
            // Recalculate CRC for the changed copy
            crcs_[index] = crc_calculator_.calculate(&copies_[index], sizeof(T));
            // Update health score
            health_scores_[index] *= 0.9; // Penalize for external modification
        }
    }
    
    /**
     * @brief Force verification of all copies
     * 
     * @return True if verification passed, false if errors detected
     */
    bool verify() {
        return verifyCRCs();
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Human-readable string with error statistics
     */
    std::string getErrorStats() const {
        std::string stats = "Enhanced TMR Error Statistics:\n";
        stats += "  Total operations: " + std::to_string(total_set_operations_ + total_get_operations_) + "\n";
        stats += "  Set operations: " + std::to_string(total_set_operations_) + "\n";
        stats += "  Get operations: " + std::to_string(total_get_operations_) + "\n";
        stats += "  CRC validation failures: " + std::to_string(crc_validation_failures_) + "\n";
        stats += "  Voting disagreements: " + std::to_string(voting_disagreements_) + "\n";
        stats += "  Copy health scores: [";
        for (size_t i = 0; i < num_copies_; i++) {
            stats += std::to_string(health_scores_[i]);
            if (i < num_copies_ - 1) {
                stats += ", ";
            }
        }
        stats += "]\n";
        
        return stats;
    }
    
    /**
     * @brief Set verification interval
     * 
     * @param interval Interval between automatic verifications
     */
    void setVerificationInterval(std::chrono::milliseconds interval) {
        verification_interval_ = interval;
    }
    
    /**
     * @brief Get the current verification interval
     * 
     * @return Current verification interval
     */
    std::chrono::milliseconds getVerificationInterval() const {
        return verification_interval_;
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        total_set_operations_ = 0;
        total_get_operations_ = 0;
        crc_validation_failures_ = 0;
        voting_disagreements_ = 0;
        
        for (size_t i = 0; i < num_copies_; i++) {
            health_scores_[i] = 1.0;
        }
    }
    
    /**
     * @brief Enable or disable health-weighted voting
     * 
     * @param enable Whether to enable health-weighted voting
     */
    void enableHealthWeightedVoting(bool enable) {
        use_health_weighted_voting_ = enable;
    }
    
    /**
     * @brief Regenerate copies from the most trusted copy
     * 
     * Call this periodically to refresh copies from the most trusted one
     * 
     * @return True if regeneration was performed
     */
    bool regenerateCopies() {
        // Find the most trusted copy
        size_t best_idx = 0;
        double best_health = health_scores_[0];
        for (size_t i = 1; i < num_copies_; i++) {
            if (health_scores_[i] > best_health) {
                best_health = health_scores_[i];
                best_idx = i;
            }
        }
        
        // Skip regeneration if all copies are equally healthy
        bool all_same = true;
        for (size_t i = 0; i < num_copies_; i++) {
            if (health_scores_[i] != best_health) {
                all_same = false;
                break;
            }
        }
        
        if (all_same) {
            return false;
        }
        
        // Regenerate other copies from the most trusted one
        for (size_t i = 0; i < num_copies_; i++) {
            if (i != best_idx) {
                copies_[i] = copies_[best_idx];
                health_scores_[i] = best_health * 0.95; // Slightly reduce health of regenerated copies
            }
        }
        
        recalculateCRCs();
        return true;
    }
    
private:
    static constexpr size_t num_copies_ = 3;
    T copies_[num_copies_];
    uint32_t crcs_[num_copies_];
    mutable double health_scores_[num_copies_];
    CRC32 crc_calculator_;
    
    bool use_health_weighted_voting_ = true;
    std::function<void(const T&, const T&)> on_error_callback_;
    
    // Error statistics
    mutable size_t total_set_operations_ = 0;
    mutable size_t total_get_operations_ = 0;
    mutable size_t crc_validation_failures_ = 0;
    mutable size_t voting_disagreements_ = 0;
    
    // Timing for periodic verification
    mutable std::chrono::steady_clock::time_point last_verification_time_;
    std::chrono::milliseconds verification_interval_{5000}; // 5 seconds by default
    
    /**
     * @brief Recalculate CRCs for all copies
     */
    void recalculateCRCs() {
        for (size_t i = 0; i < num_copies_; i++) {
            crcs_[i] = crc_calculator_.calculate(&copies_[i], sizeof(T));
        }
    }
    
    /**
     * @brief Verify CRC values for all copies
     * 
     * @return True if all CRCs match, false otherwise
     */
    bool verifyCRCs() {
        auto now = std::chrono::steady_clock::now();
        
        // Only perform periodic verification if enough time has passed
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_verification_time_) < verification_interval_) {
            return true;
        }
        
        last_verification_time_ = now;
        
        bool all_valid = true;
        
        for (size_t i = 0; i < num_copies_; i++) {
            uint32_t current_crc = crc_calculator_.calculate(&copies_[i], sizeof(T));
            if (current_crc != crcs_[i]) {
                all_valid = false;
                health_scores_[i] *= 0.7; // Reduce health score for corrupted copy
                crc_validation_failures_++;
            }
        }
        
        return all_valid;
    }
    
    /**
     * @brief Perform majority voting considering health scores
     * 
     * @return Voted value
     */
    T performWeightedVoting() {
        total_get_operations_++;
        
        // Skip voting if health scores are very skewed - use most trusted copy
        double max_health = 0.0;
        double min_health = 1.0;
        size_t best_index = 0;
        
        for (size_t i = 0; i < num_copies_; i++) {
            if (health_scores_[i] > max_health) {
                max_health = health_scores_[i];
                best_index = i;
            }
            if (health_scores_[i] < min_health) {
                min_health = health_scores_[i];
            }
        }
        
        // If one copy is much more trusted than others, just use it
        if (max_health > 3.0 * min_health) {
            return copies_[best_index];
        }
        
        // Normal case: perform voting
        bool disagreement = false;
        
        // Check if there's a disagreement
        for (size_t i = 1; i < num_copies_; i++) {
            if (copies_[i] != copies_[0]) {
                disagreement = true;
                break;
            }
        }
        
        // Fast path when all copies agree
        if (!disagreement) {
            return copies_[0];
        }
        
        // There's a disagreement - perform weighted voting
        voting_disagreements_++;
        
        if (use_health_weighted_voting_) {
            // Count occurrences of each value, weighted by health
            std::map<T, double> value_scores;
            for (size_t i = 0; i < num_copies_; i++) {
                value_scores[copies_[i]] += health_scores_[i];
            }
            
            // Find most common value
            T best_value = copies_[0];
            double best_score = value_scores[copies_[0]];
            
            for (const auto& pair : value_scores) {
                if (pair.second > best_score) {
                    best_score = pair.second;
                    best_value = pair.first;
                }
            }
            
            // Penalize health scores of incorrect copies
            for (size_t i = 0; i < num_copies_; i++) {
                if (copies_[i] != best_value) {
                    health_scores_[i] *= 0.8;
                }
            }
            
            // Call error callback if provided
            if (on_error_callback_) {
                // Find the corrupted value that differs from voted value
                for (size_t i = 0; i < num_copies_; i++) {
                    if (copies_[i] != best_value) {
                        on_error_callback_(best_value, copies_[i]);
                        break;
                    }
                }
            }
            
            return best_value;
        } else {
            // Simple majority voting without health weighting
            std::map<T, int> value_counts;
            for (size_t i = 0; i < num_copies_; i++) {
                value_counts[copies_[i]]++;
            }
            
            // Find most common value
            T best_value = copies_[0];
            int best_count = value_counts[copies_[0]];
            
            for (const auto& pair : value_counts) {
                if (pair.second > best_count) {
                    best_count = pair.second;
                    best_value = pair.first;
                }
            }
            
            // Penalize health scores of incorrect copies
            for (size_t i = 0; i < num_copies_; i++) {
                if (copies_[i] != best_value) {
                    health_scores_[i] *= 0.8;
                }
            }
            
            // Call error callback if provided
            if (on_error_callback_) {
                // Find the corrupted value that differs from voted value
                for (size_t i = 0; i < num_copies_; i++) {
                    if (copies_[i] != best_value) {
                        on_error_callback_(best_value, copies_[i]);
                        break;
                    }
                }
            }
            
            return best_value;
        }
    }
};

} // namespace tmr
} // namespace rad_ml 