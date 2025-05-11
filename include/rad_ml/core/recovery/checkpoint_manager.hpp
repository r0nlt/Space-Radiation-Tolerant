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

#include <map>
#include <chrono>
#include <vector>
#include <memory>
#include <functional>

namespace rad_ml {
namespace core {
namespace recovery {

/**
 * @brief Checkpoint manager for radiation-tolerant computing
 * 
 * Manages periodic checkpoints and rollback capabilities for critical data
 */
template <typename T>
class CheckpointManager {
public:
    /**
     * @brief Construct a new Checkpoint Manager
     * 
     * @param max_checkpoints Maximum number of checkpoints to store
     * @param checkpoint_interval Time between automatic checkpoints
     */
    CheckpointManager(size_t max_checkpoints = 3, 
                     std::chrono::seconds checkpoint_interval = std::chrono::seconds(60))
        : max_checkpoints_(max_checkpoints), 
          checkpoint_interval_(checkpoint_interval),
          last_checkpoint_time_(std::chrono::steady_clock::now()) {}
    
    /**
     * @brief Create a checkpoint of the current data
     * 
     * @param data Data to checkpoint
     * @param version_id Unique identifier for this checkpoint
     */
    void createCheckpoint(const T& data, uint64_t version_id) {
        auto now = std::chrono::steady_clock::now();
        
        // Only checkpoint at specified intervals
        if (now - last_checkpoint_time_ < checkpoint_interval_) {
            return;
        }
        
        checkpoints_.push_back({version_id, data, now});
        last_checkpoint_time_ = now;
        
        // Remove oldest checkpoint if we exceed the limit
        if (checkpoints_.size() > max_checkpoints_) {
            checkpoints_.erase(checkpoints_.begin());
        }
    }
    
    /**
     * @brief Retrieve the latest valid checkpoint
     * 
     * @param data Reference to store retrieved data
     * @return true if a checkpoint was retrieved, false otherwise
     */
    bool getLatestCheckpoint(T& data) {
        if (checkpoints_.empty()) {
            return false;
        }
        
        data = checkpoints_.back().data;
        return true;
    }
    
    /**
     * @brief Rollback to a specific version if available
     * 
     * @param version_id Version to rollback to
     * @param data Reference to store retrieved data
     * @return true if rollback was successful, false otherwise
     */
    bool rollbackToVersion(uint64_t version_id, T& data) {
        for (auto it = checkpoints_.rbegin(); it != checkpoints_.rend(); ++it) {
            if (it->version_id == version_id) {
                data = it->data;
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Rollback to the most recent checkpoint that passes validation
     * 
     * @param data Reference to store retrieved data
     * @param validator Function to validate checkpoint integrity
     * @return true if rollback was successful, false otherwise
     */
    bool rollbackToValid(T& data, std::function<bool(const T&)> validator) {
        for (auto it = checkpoints_.rbegin(); it != checkpoints_.rend(); ++it) {
            if (validator(it->data)) {
                data = it->data;
                return true;
            }
        }
        return false;
    }
    
private:
    /**
     * @brief Structure to hold checkpoint data
     */
    struct Checkpoint {
        uint64_t version_id;
        T data;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<Checkpoint> checkpoints_;
    size_t max_checkpoints_;
    std::chrono::seconds checkpoint_interval_;
    std::chrono::steady_clock::time_point last_checkpoint_time_;
};

} // namespace recovery
} // namespace core
} // namespace rad_ml 
