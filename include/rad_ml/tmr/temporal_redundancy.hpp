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

#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <map>
#include <thread>

namespace rad_ml {
namespace tmr {

/**
 * @brief Temporal redundancy implementation
 * 
 * Executes operations multiple times and compares results
 * to detect and correct transient faults
 */
template <typename T, typename ResultType>
class TemporalRedundancy {
public:
    /**
     * @brief Constructor for temporal redundancy
     * 
     * @param num_executions Number of times to execute the operation
     * @param delay_between Delay between executions to avoid correlated errors
     */
    TemporalRedundancy(
        size_t num_executions = 3,
        std::chrono::milliseconds delay_between = std::chrono::milliseconds(10))
        : num_executions_(num_executions),
          delay_between_(delay_between) {}
    
    /**
     * @brief Execute operation multiple times with time-based voting
     * 
     * @param data Input data for the operation
     * @param operation Function to execute
     * @return Result determined by temporal voting
     */
    ResultType execute(const T& data, std::function<ResultType(const T&)> operation) const {
        std::vector<ResultType> results;
        results.reserve(num_executions_);
        
        // Execute multiple times with delay between
        for (size_t i = 0; i < num_executions_; ++i) {
            results.push_back(operation(data));
            
            // Add delay between executions to avoid correlated errors
            if (i < num_executions_ - 1) {
                std::this_thread::sleep_for(delay_between_);
            }
        }
        
        // Find most common result (similar to spatial voting)
        return findMostCommonResult(results);
    }
    
    /**
     * @brief Change configuration based on radiation environment
     * 
     * @param num_executions New number of executions
     * @param delay_ms New delay in milliseconds
     */
    void reconfigure(size_t num_executions, uint64_t delay_ms) {
        num_executions_ = num_executions;
        delay_between_ = std::chrono::milliseconds(delay_ms);
    }
    
private:
    size_t num_executions_;
    std::chrono::milliseconds delay_between_;
    
    /**
     * @brief Find most common result through voting
     * 
     * @param results Vector of results from multiple executions
     * @return Most common result (temporal majority)
     */
    ResultType findMostCommonResult(const std::vector<ResultType>& results) const {
        // Count occurrences of each result
        std::map<ResultType, size_t> result_counts;
        for (const auto& result : results) {
            result_counts[result]++;
        }
        
        // Find result with highest count
        auto max_element = std::max_element(
            result_counts.begin(), result_counts.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            
        return max_element->first;
    }
};

} // namespace tmr
} // namespace rad_ml 
