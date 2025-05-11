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

#include "rad_ml/core/redundancy/enhanced_tmr.hpp"
#include <array>
#include <deque>
#include <vector>
#include <cmath>
#include <map>

namespace rad_ml {
namespace tmr {

/**
 * @brief Health-Weighted Triple Modular Redundancy
 * 
 * Extends EnhancedTMR with a history-based health tracking system
 * that dynamically adjusts voting weights based on past reliability.
 * 
 * @tparam T The type of the value to protect with TMR
 */
template <typename T>
class HealthWeightedTMR {
public:
    static constexpr size_t num_copies_ = 3;
    static constexpr size_t history_length_ = 10; // Remember last N votes
    
    /**
     * @brief Constructor with initial value
     * 
     * @param initial_value Initial value for all copies
     */
    explicit HealthWeightedTMR(const T& initial_value = T()) 
        : copies_{initial_value, initial_value, initial_value},
          checksums_{0, 0, 0},
          health_scores_{1.0, 1.0, 1.0},
          error_counters_{0, 0, 0} {
        recalculateChecksums();
    }
    
    /**
     * @brief Get the current value using health-weighted voting
     * 
     * @return The correct value based on health-weighted voting
     */
    T get() const {
        // First, verify CRC checksums
        bool checksum_valid[num_copies_] = {
            verifyChecksum(0),
            verifyChecksum(1),
            verifyChecksum(2)
        };
        
        // Count valid copies and update health metrics
        int valid_count = 0;
        for (int i = 0; i < num_copies_; ++i) {
            if (checksum_valid[i]) {
                valid_count++;
            } else {
                // Penalize health score for invalid checksum
                updateHealthScore(i, false);
            }
        }
        
        // If all checksums valid, use weighted voting
        if (valid_count >= 2) {
            return performWeightedVoting(checksum_valid);
        }
        
        // Recovery from severe corruption - try to salvage a valid copy
        for (int i = 0; i < num_copies_; ++i) {
            if (checksum_valid[i]) {
                return copies_[i];
            }
        }
        
        // Last resort - use highest health score
        size_t best_idx = 0;
        for (size_t i = 1; i < num_copies_; ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        
        // Increase error count and update health scores
        error_counters_[best_idx]++;
        
        return copies_[best_idx];
    }
    
    /**
     * @brief Set the value in all three copies
     * 
     * @param value The value to set
     */
    void set(const T& value) {
        for (size_t i = 0; i < num_copies_; ++i) {
            copies_[i] = value;
        }
        
        // Reset health scores on explicit set
        std::fill(health_scores_.begin(), health_scores_.end(), 1.0);
        recalculateChecksums();
    }
    
    /**
     * @brief Get the health scores for diagnostics
     * 
     * @return Array of health scores
     */
    std::array<double, num_copies_> getHealthScores() const {
        return health_scores_;
    }
    
    /**
     * @brief Get error statistics for each module
     * 
     * @return Array of error counters
     */
    std::array<size_t, num_copies_> getErrorCounters() const {
        return error_counters_;
    }
    
    /**
     * @brief Repair any corrupted values
     */
    void repair() {
        T value = get(); // Get correct value via voting
        
        // Update all copies with correct value
        for (size_t i = 0; i < num_copies_; ++i) {
            copies_[i] = value;
        }
        
        recalculateChecksums();
    }
    
    /**
     * @brief Manually regenerate copies from most reliable copy
     */
    void regenerateCopies() {
        // Find most reliable copy
        size_t best_idx = 0;
        for (size_t i = 1; i < num_copies_; ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        
        // Copy best value to all copies
        T value = copies_[best_idx];
        for (size_t i = 0; i < num_copies_; ++i) {
            copies_[i] = value;
        }
        
        recalculateChecksums();
    }
    
private:
    // The redundant copies of the value
    std::array<T, num_copies_> copies_;
    
    // Checksums for each value
    mutable std::array<uint32_t, num_copies_> checksums_;
    
    // Dynamic health scores for each copy (0.0 to 1.0)
    mutable std::array<double, num_copies_> health_scores_;
    
    // Error counters for each copy
    mutable std::array<size_t, num_copies_> error_counters_;
    
    // Vote history for each copy (for trend analysis)
    mutable std::array<std::deque<bool>, num_copies_> vote_history_;
    
    /**
     * @brief Calculate checksums for all values
     */
    void recalculateChecksums() {
        for (size_t i = 0; i < num_copies_; ++i) {
            checksums_[i] = calculateChecksum(copies_[i]);
        }
    }
    
    /**
     * @brief Verify checksum for a specific value
     * 
     * @param index Index of the value (0-2)
     * @return true if checksum is valid
     */
    bool verifyChecksum(size_t index) const {
        if (index >= num_copies_) return false;
        return checksums_[index] == calculateChecksum(copies_[index]);
    }
    
    /**
     * @brief Calculate checksum for a value
     * 
     * @param value Value to calculate checksum for
     * @return Checksum value
     */
    uint32_t calculateChecksum(const T& value) const {
        // CRC-32 implementation (simplified for example)
        const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
        uint32_t crc = 0xFFFFFFFF;
        
        for (size_t i = 0; i < sizeof(T); ++i) {
            uint8_t byte = data[i];
            crc ^= byte;
            for (size_t j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        
        return ~crc;
    }
    
    /**
     * @brief Update health score for a copy based on vote correctness
     * 
     * @param index Index of the copy (0-2)
     * @param correct Whether this copy's value was correct
     */
    void updateHealthScore(size_t index, bool correct) const {
        const double reward_factor = 0.05;  // How much to increase for correct vote
        const double penalty_factor = 0.2;  // How much to decrease for incorrect vote
        
        // Update vote history
        if (vote_history_[index].size() >= history_length_) {
            vote_history_[index].pop_front();
        }
        vote_history_[index].push_back(correct);
        
        // Update health score
        if (correct) {
            // Reward correct vote (limited to 1.0)
            health_scores_[index] = std::min(1.0, health_scores_[index] + reward_factor);
        } else {
            // Penalize incorrect vote (with lower bound of 0.1)
            health_scores_[index] = std::max(0.1, health_scores_[index] - penalty_factor);
            error_counters_[index]++;
        }
        
        // Apply additional penalty for trending errors
        if (vote_history_[index].size() >= 3) {
            size_t recent_error_count = 0;
            for (size_t i = vote_history_[index].size() - 3; i < vote_history_[index].size(); ++i) {
                if (!vote_history_[index][i]) {
                    recent_error_count++;
                }
            }
            
            // Apply additional penalty for consistent errors
            if (recent_error_count >= 2) {
                health_scores_[index] = std::max(0.05, health_scores_[index] - 0.1);
            }
        }
    }
    
    /**
     * @brief Perform weighted voting based on health scores
     * 
     * @param checksum_valid Array indicating which checksums are valid
     * @return The value with highest weighted vote
     */
    T performWeightedVoting(const bool checksum_valid[]) const {
        // Special case: if all values are the same, return that value
        if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
            // Update all health scores positively
            for (size_t i = 0; i < num_copies_; ++i) {
                updateHealthScore(i, true);
            }
            return copies_[0];
        }
        
        // Map of values and their weighted votes
        std::map<T, double> weighted_votes;
        
        // Calculate total weight of valid copies
        double total_valid_weight = 0.0;
        for (size_t i = 0; i < num_copies_; ++i) {
            if (checksum_valid[i]) {
                total_valid_weight += health_scores_[i];
            }
        }
        
        // Assign weighted votes
        for (size_t i = 0; i < num_copies_; ++i) {
            if (checksum_valid[i]) {
                weighted_votes[copies_[i]] += 
                    health_scores_[i] / total_valid_weight;
            }
        }
        
        // Find value with highest weighted vote
        T best_value = copies_[0];
        double best_score = 0.0;
        
        for (const auto& entry : weighted_votes) {
            if (entry.second > best_score) {
                best_value = entry.first;
                best_score = entry.second;
            }
        }
        
        // Update health scores based on voting results
        for (size_t i = 0; i < num_copies_; ++i) {
            if (checksum_valid[i]) {
                updateHealthScore(i, copies_[i] == best_value);
            }
        }
        
        return best_value;
    }
};

} // namespace tmr
} // namespace rad_ml 
