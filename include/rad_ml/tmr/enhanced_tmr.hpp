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

#include "rad_ml/tmr/tmr.hpp"
#include <functional>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>
#include <map>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <type_traits>

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

// Forward declare for the factory
template <typename T>
class EnhancedTMR;

/**
 * @brief Factory for creating TMR instances
 * 
 * This factory ensures proper memory management and consistent initialization
 */
class TMRFactory {
public:
    /**
     * @brief Create an EnhancedTMR instance
     * 
     * @tparam T Type of the data to be protected
     * @param initial_value Initial value
     * @param on_error_callback Optional callback for error detection
     * @return std::shared_ptr to EnhancedTMR instance
     */
    template <typename T>
    static std::shared_ptr<EnhancedTMR<T>> createEnhancedTMR(
        const T& initial_value = T(),
        std::function<void(const T&, const T&)> on_error_callback = nullptr) {
        
        return std::make_shared<EnhancedTMR<T>>(initial_value, on_error_callback);
    }
    
    /**
     * @brief Create a basic TMR instance
     * 
     * @tparam T Type of the data to be protected
     * @param initial_value Initial value
     * @param on_error_callback Optional callback for error detection
     * @return std::shared_ptr to TMR instance
     */
    template <typename T>
    static std::shared_ptr<TMR<T>> createTMR(
        const T& initial_value = T(),
        std::function<void(const T&, const T&)> on_error_callback = nullptr) {
        
        return std::make_shared<TMR<T>>(initial_value, on_error_callback);
    }
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
        
        // Static validation for types that can be safely copied
        static_assert(std::is_copy_constructible_v<T>, 
            "EnhancedTMR requires copy-constructible types");
        
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
    T get() const {
        // Thread safety for statistics and health scores
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if we need to verify CRCs based on time interval
        const auto now = std::chrono::steady_clock::now();
        if (now - last_verification_time_ >= verification_interval_) {
            verifyCRCs();
            last_verification_time_ = now;
        }
        
        // Use weighted voting for enhanced resilience
        T result = performWeightedVoting();
        
        total_get_operations_++;
        return result;
    }
    
    /**
     * @brief Try to get value with error handling
     * 
     * @return std::optional<T> containing value if successful, or std::nullopt if error
     */
    std::optional<T> tryGet() const {
        try {
            return get();
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    
    /**
     * @brief Set a new value to all copies
     * 
     * @param value New value
     */
    void set(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
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
        std::lock_guard<std::mutex> lock(mutex_);
        
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
        std::lock_guard<std::mutex> lock(mutex_);
        
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
        std::lock_guard<std::mutex> lock(mutex_);
        return verifyCRCs();
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Human-readable string with error statistics
     */
    std::string getErrorStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
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
        std::lock_guard<std::mutex> lock(mutex_);
        verification_interval_ = interval;
    }
    
    /**
     * @brief Get the current verification interval
     * 
     * @return Current verification interval
     */
    std::chrono::milliseconds getVerificationInterval() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return verification_interval_;
    }
    
    /**
     * @brief Reset error statistics
     */
    void resetErrorStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        
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
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the most trusted copy
        size_t best_idx = 0;
        double best_health = health_scores_[0];
        
        for (size_t i = 1; i < num_copies_; i++) {
            if (health_scores_[i] > best_health) {
                best_health = health_scores_[i];
                best_idx = i;
            }
        }
        
        // Only regenerate if trust is high enough
        if (best_health < 0.6) {
            return false; // No copy is trustworthy enough
        }
        
        // Regenerate all copies from the most trusted one
        T trusted_value = copies_[best_idx];
        
        for (size_t i = 0; i < num_copies_; i++) {
            if (i != best_idx) {
                copies_[i] = trusted_value;
                health_scores_[i] = 0.9; // Slightly less than perfect
            }
        }
        
        // Recalculate CRCs
        recalculateCRCs();
        
        return true;
    }
    
    /**
     * @brief Create and return a deep copy of this TMR instance
     * 
     * @return std::unique_ptr to a new EnhancedTMR instance
     */
    std::unique_ptr<EnhancedTMR<T>> clone() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Create a new instance with the current value
        auto result = std::make_unique<EnhancedTMR<T>>(get(), on_error_callback_);
        
        // Copy health scores
        for (size_t i = 0; i < num_copies_; i++) {
            result->health_scores_[i] = health_scores_[i];
        }
        
        // Copy other settings
        result->verification_interval_ = verification_interval_;
        result->use_health_weighted_voting_ = use_health_weighted_voting_;
        
        return result;
    }
    
private:
    static constexpr size_t num_copies_ = 3;
    T copies_[num_copies_];
    uint32_t crcs_[num_copies_] = {0};
    
    // Thread synchronization
    mutable std::mutex mutex_;
    
    // CRC calculator
    CRC32 crc_calculator_;
    
    // Voting configuration
    bool use_health_weighted_voting_ = true;
    std::function<void(const T&, const T&)> on_error_callback_;
    
    // Health tracking
    mutable double health_scores_[num_copies_] = {1.0, 1.0, 1.0};
    
    // Performance tracking
    mutable size_t total_set_operations_ = 0;
    mutable size_t total_get_operations_ = 0;
    mutable size_t crc_validation_failures_ = 0;
    mutable size_t voting_disagreements_ = 0;
    
    // Verification timing
    mutable std::chrono::steady_clock::time_point last_verification_time_;
    std::chrono::milliseconds verification_interval_{5000}; // 5 seconds by default
    
    /**
     * @brief Recalculate CRC values for all copies
     * 
     * Must be called with mutex_ held
     */
    void recalculateCRCs() {
        for (size_t i = 0; i < num_copies_; i++) {
            crcs_[i] = crc_calculator_.calculate(&copies_[i], sizeof(T));
        }
    }
    
    /**
     * @brief Verify integrity of all copies using CRC
     * 
     * Must be called with mutex_ held
     * 
     * @return True if all checks passed, false if errors detected
     */
    bool verifyCRCs() const {
        bool all_valid = true;
        
        for (size_t i = 0; i < num_copies_; i++) {
            uint32_t current_crc = crc_calculator_.calculate(&copies_[i], sizeof(T));
            if (current_crc != crcs_[i]) {
                // CRC validation failed
                all_valid = false;
                crc_validation_failures_++;
                
                // Penalize health score for CRC failures
                health_scores_[i] = std::max(0.1, health_scores_[i] - 0.3);
                
                // Log or notify about the error
                if (on_error_callback_) {
                    // Get the most likely correct value from the other copies
                    T likely_correct = (i == 0) ? copies_[1] : copies_[0];
                    on_error_callback_(likely_correct, copies_[i]);
                }
            }
        }
        
        return all_valid;
    }
    
    /**
     * @brief Perform weighted voting to determine correct value
     * 
     * Must be called with mutex_ held
     * 
     * @return Result of voting
     */
    T performWeightedVoting() const {
        // Special case: if all values are the same, return any copy
        if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
            // Update all health scores positively
            for (size_t i = 0; i < num_copies_; i++) {
                health_scores_[i] = std::min(1.0, health_scores_[i] + 0.05);
            }
            return copies_[0];
        }
        
        // We have a disagreement
        voting_disagreements_++;
        
        if (use_health_weighted_voting_) {
            // Health-weighted voting
            
            // Check if any two copies agree
            bool copy_0_1_agree = (copies_[0] == copies_[1]);
            bool copy_0_2_agree = (copies_[0] == copies_[2]);
            bool copy_1_2_agree = (copies_[1] == copies_[2]);
            
            if (copy_0_1_agree || copy_0_2_agree || copy_1_2_agree) {
                // Two copies agree, use that value
                if (copy_0_1_agree) {
                    // First and second copies agree
                    health_scores_[0] = std::min(1.0, health_scores_[0] + 0.05);
                    health_scores_[1] = std::min(1.0, health_scores_[1] + 0.05);
                    health_scores_[2] = std::max(0.1, health_scores_[2] - 0.2);
                    return copies_[0];
                } 
                else if (copy_0_2_agree) {
                    // First and third copies agree
                    health_scores_[0] = std::min(1.0, health_scores_[0] + 0.05);
                    health_scores_[2] = std::min(1.0, health_scores_[2] + 0.05);
                    health_scores_[1] = std::max(0.1, health_scores_[1] - 0.2);
                    return copies_[0];
                } 
                else {
                    // Second and third copies agree
                    health_scores_[1] = std::min(1.0, health_scores_[1] + 0.05);
                    health_scores_[2] = std::min(1.0, health_scores_[2] + 0.05);
                    health_scores_[0] = std::max(0.1, health_scores_[0] - 0.2);
                    return copies_[1];
                }
            } else {
                // All three copies disagree - use health-weighted selection
                double total_health = 0.0;
                for (size_t i = 0; i < num_copies_; i++) {
                    total_health += health_scores_[i];
                }
                
                // If we somehow have zero total health, just return first copy
                if (total_health <= 0.0) {
                    return copies_[0];
                }
                
                // Use health scores to pick most trusted copy
                double r = std::uniform_real_distribution<double>(0.0, total_health)(
                    random_engine_);
                
                double cumulative = 0.0;
                for (size_t i = 0; i < num_copies_; i++) {
                    cumulative += health_scores_[i];
                    if (r <= cumulative) {
                        // Slightly boost the selected copy's health
                        health_scores_[i] = std::min(1.0, health_scores_[i] + 0.02);
                        return copies_[i];
                    }
                }
                
                // Fallback (should not reach here)
                return copies_[0];
            }
        } else {
            // Standard TMR voting (no health weighting)
            if (copies_[0] == copies_[1]) {
                return copies_[0];
            } else if (copies_[0] == copies_[2]) {
                return copies_[0];
            } else if (copies_[1] == copies_[2]) {
                return copies_[1];
            } else {
                // All three disagree - return the first copy as fallback
                return copies_[0];
            }
        }
    }
    
    // Random engine for weighted voting
    inline static thread_local std::mt19937 random_engine_{std::random_device{}()};
};

} // namespace tmr
} // namespace rad_ml 
