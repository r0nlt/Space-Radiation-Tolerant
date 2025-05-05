#pragma once

#include <vector>
#include <functional>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rad_ml {
namespace advanced {

/**
 * @brief Algorithmic diversity for radiation tolerance
 * 
 * Implements multiple different algorithms for the same task
 * to provide stronger protection against radiation effects
 */
template <typename T, typename ResultType>
class AlgorithmicDiversity {
public:
    /**
     * @brief Constructor for algorithmic diversity
     */
    AlgorithmicDiversity() : reliability_history_size_(10) {}
    
    /**
     * @brief Add a new implementation approach
     * 
     * @param name Unique name for this implementation
     * @param impl Implementation function
     * @param initial_reliability Initial reliability score (0.0-1.0)
     */
    void addImplementation(const std::string& name, 
                          std::function<ResultType(const T&)> impl,
                          double initial_reliability = 1.0) {
        implementations_[name] = impl;
        reliability_scores_[name] = initial_reliability;
        
        // Initialize reliability history
        reliability_history_[name] = std::vector<bool>(reliability_history_size_, true);
    }
    
    /**
     * @brief Execute all implementations and determine result
     * 
     * @param data Input data for the operation
     * @return Result determined by consensus with reliability weighting
     */
    ResultType execute(const T& data) {
        std::map<ResultType, double> weighted_votes;
        
        // Execute each implementation and collect results
        for (const auto& impl_pair : implementations_) {
            const std::string& name = impl_pair.first;
            const auto& impl = impl_pair.second;
            
            // Skip implementations with very low reliability
            if (reliability_scores_[name] < 0.2) {
                continue;
            }
            
            // Execute the implementation
            ResultType result;
            try {
                result = impl(data);
                
                // Add weighted vote for this result
                weighted_votes[result] += reliability_scores_[name];
            } catch (...) {
                // Implementation failed - update reliability
                updateReliability(name, false);
                continue;
            }
        }
        
        if (weighted_votes.empty()) {
            // Fallback to most reliable implementation if no consensus
            const std::string& most_reliable = getMostReliableImplementation();
            if (!most_reliable.empty()) {
                return implementations_[most_reliable](data);
            }
            
            // If no implementations available, throw exception
            throw std::runtime_error("No reliable implementations available");
        }
        
        // Find result with highest weighted vote
        auto max_element = std::max_element(
            weighted_votes.begin(), weighted_votes.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            
        return max_element->first;
    }
    
    /**
     * @brief Update reliability score for an implementation
     * 
     * @param name Implementation name
     * @param success Whether the implementation produced correct results
     */
    void updateReliability(const std::string& name, bool success) {
        auto it = reliability_scores_.find(name);
        if (it == reliability_scores_.end()) {
            return;
        }
        
        // Update reliability history
        auto& history = reliability_history_[name];
        history.erase(history.begin());
        history.push_back(success);
        
        // Calculate new reliability score based on recent history
        size_t success_count = std::count(history.begin(), history.end(), true);
        double new_reliability = static_cast<double>(success_count) / history.size();
        
        // Update reliability score with decay factor
        const double decay_factor = 0.9; // Favor recent performance over historical
        it->second = (it->second * decay_factor) + (new_reliability * (1.0 - decay_factor));
    }
    
    /**
     * @brief Get the current reliability score for an implementation
     * 
     * @param name Implementation name
     * @return Reliability score (0.0-1.0)
     */
    double getReliability(const std::string& name) const {
        auto it = reliability_scores_.find(name);
        return (it != reliability_scores_.end()) ? it->second : 0.0;
    }
    
    /**
     * @brief Set the reliability history size
     * 
     * @param size Number of past executions to track
     */
    void setReliabilityHistorySize(size_t size) {
        reliability_history_size_ = size;
        
        // Resize all histories
        for (auto& hist_pair : reliability_history_) {
            auto& history = hist_pair.second;
            bool fill_value = true;
            if (!history.empty()) {
                // Use the most recent value for filling
                fill_value = history.back();
            }
            history.resize(reliability_history_size_, fill_value);
        }
    }
    
    /**
     * @brief Get the name of the most reliable implementation
     * 
     * @return Name of most reliable implementation
     */
    std::string getMostReliableImplementation() const {
        if (reliability_scores_.empty()) {
            return "";
        }
        
        auto max_element = std::max_element(
            reliability_scores_.begin(), reliability_scores_.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            
        return max_element->first;
    }
    
    /**
     * @brief Check if a specific implementation is available
     * 
     * @param name Implementation name
     * @return true if implementation exists
     */
    bool hasImplementation(const std::string& name) const {
        return implementations_.find(name) != implementations_.end();
    }
    
private:
    std::map<std::string, std::function<ResultType(const T&)>> implementations_;
    std::map<std::string, double> reliability_scores_;
    std::map<std::string, std::vector<bool>> reliability_history_;
    size_t reliability_history_size_;
};

} // namespace advanced
} // namespace rad_ml 