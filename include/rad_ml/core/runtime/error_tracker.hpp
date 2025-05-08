/**
 * @file error_tracker.hpp
 * @brief Thread-safe radiation error tracking and analysis
 * 
 * This file defines a lock-free error tracking system that monitors
 * radiation-induced errors and provides statistical analysis.
 */

#ifndef RAD_ML_ERROR_TRACKER_HPP
#define RAD_ML_ERROR_TRACKER_HPP

#include <atomic>
#include <chrono>
#include <array>
#include <deque>
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include "../redundancy/enhanced_voting.hpp"

namespace rad_ml {
namespace core {
namespace runtime {

/**
 * Thread-safe, lock-free radiation error tracker
 * 
 * This class provides mechanisms to record, analyze, and respond to
 * radiation-induced errors in a multi-threaded environment.
 */
class RadiationErrorTracker {
public:
    /**
     * Default constructor initializes counters
     */
    RadiationErrorTracker() 
        : error_count(0), 
          current_error_rate(0.0f),
          pattern_counts{} {}
    
    /**
     * Record a new error with lock-free atomic update
     * 
     * @param pattern Detected fault pattern
     * @param data Optional additional data about the error
     */
    void recordError(redundancy::FaultPattern pattern, const std::string& data = "") {
        // Increment total count with relaxed ordering
        error_count.fetch_add(1, std::memory_order_relaxed);
        
        // Update pattern-specific counts
        size_t pattern_idx = static_cast<size_t>(pattern);
        if (pattern_idx < pattern_counts.size()) {
            pattern_counts[pattern_idx].fetch_add(1, std::memory_order_relaxed);
        }
        
        // Record timestamp with release ordering to ensure visibility
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        last_error_time.store(now, std::memory_order_release);
        
        // Store detailed information if provided
        if (!data.empty()) {
            std::lock_guard<std::mutex> lock(history_mutex);
            error_history.push_back({
                std::chrono::steady_clock::now(),
                pattern,
                data
            });
            
            // Limit history size
            if (error_history.size() > max_history_size) {
                error_history.pop_front();
            }
        }
        
        // Update error rate periodically
        updateErrorRate();
    }
    
    /**
     * Get current error rate (errors per second)
     * 
     * @return Current error rate
     */
    float getErrorRate() const {
        return current_error_rate.load(std::memory_order_acquire);
    }
    
    /**
     * Get total error count
     * 
     * @return Total number of errors recorded
     */
    uint64_t getTotalErrorCount() const {
        return error_count.load(std::memory_order_acquire);
    }
    
    /**
     * Get pattern distribution for adaptive strategies
     * 
     * @return Distribution of errors by pattern type
     */
    std::array<float, 6> getPatternDistribution() const {
        std::array<float, 6> distribution{};
        
        // Safely read atomic values
        uint64_t total = error_count.load(std::memory_order_acquire);
        if (total == 0) return distribution;
        
        // Calculate percentages
        for (size_t i = 0; i < pattern_counts.size(); ++i) {
            uint64_t count = pattern_counts[i].load(std::memory_order_relaxed);
            distribution[i] = static_cast<float>(count) / total;
        }
        
        return distribution;
    }
    
    /**
     * Get error count for a specific pattern
     * 
     * @param pattern The pattern to get count for
     * @return Number of errors with this pattern
     */
    uint64_t getPatternCount(redundancy::FaultPattern pattern) const {
        size_t pattern_idx = static_cast<size_t>(pattern);
        if (pattern_idx < pattern_counts.size()) {
            return pattern_counts[pattern_idx].load(std::memory_order_relaxed);
        }
        return 0;
    }
    
    /**
     * Check if error rate exceeds a threshold
     * 
     * @param threshold Threshold in errors per second
     * @return True if current rate exceeds threshold
     */
    bool isErrorRateExceeded(float threshold) const {
        return getErrorRate() > threshold;
    }
    
    /**
     * Get time since last error
     * 
     * @return Duration since last error
     */
    std::chrono::milliseconds getTimeSinceLastError() const {
        auto last = last_error_time.load(std::memory_order_acquire);
        if (last == 0) {
            return std::chrono::milliseconds::max(); // No errors yet
        }
        
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::chrono::milliseconds(
            (now - last) / (std::chrono::steady_clock::period::den / 1000)
        );
    }
    
    /**
     * Reset all error statistics
     */
    void reset() {
        error_count.store(0, std::memory_order_release);
        current_error_rate.store(0.0f, std::memory_order_release);
        last_error_time.store(0, std::memory_order_release);
        
        for (auto& count : pattern_counts) {
            count.store(0, std::memory_order_relaxed);
        }
        
        std::lock_guard<std::mutex> lock(history_mutex);
        error_history.clear();
    }
    
    /**
     * Get recent error history
     * 
     * @param max_entries Maximum number of entries to return
     * @return Vector of recent error records
     */
    struct ErrorRecord {
        std::chrono::steady_clock::time_point timestamp;
        redundancy::FaultPattern pattern;
        std::string data;
    };
    
    std::vector<ErrorRecord> getRecentErrors(size_t max_entries = 100) const {
        std::lock_guard<std::mutex> lock(history_mutex);
        
        size_t count = std::min(max_entries, error_history.size());
        std::vector<ErrorRecord> result(count);
        
        // Copy most recent entries
        std::copy(
            error_history.end() - count,
            error_history.end(),
            result.begin()
        );
        
        return result;
    }
    
private:
    // Total error count
    std::atomic<uint64_t> error_count;
    
    // Pattern-specific counts (using enum ordinal values as index)
    std::array<std::atomic<uint64_t>, 6> pattern_counts;
    
    // Timestamps for rate calculation
    std::atomic<int64_t> last_error_time{0};
    std::atomic<float> current_error_rate{0.0f};
    
    // Error history with mutex protection (not lock-free)
    mutable std::mutex history_mutex;
    std::deque<ErrorRecord> error_history;
    static constexpr size_t max_history_size = 1000;
    
    // Error rate calculation
    void updateErrorRate() {
        static std::atomic<int64_t> last_calculation_time{0};
        static std::atomic<uint64_t> last_calculation_count{0};
        
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last_time = last_calculation_time.load(std::memory_order_relaxed);
        
        // Update rate at most once per second
        if (now - last_time > std::chrono::seconds(1).count()) {
            // Try to update with compare_exchange
            if (last_calculation_time.compare_exchange_strong(
                    last_time, now, std::memory_order_acq_rel)) {
                
                uint64_t current_count = error_count.load(std::memory_order_relaxed);
                uint64_t last_count = last_calculation_count.exchange(
                    current_count, std::memory_order_acq_rel);
                
                int64_t time_diff = now - last_time;
                uint64_t count_diff = current_count - last_count;
                
                // Calculate new rate (errors per second)
                float new_rate = static_cast<float>(count_diff) * 
                    std::chrono::seconds(1).count() / time_diff;
                
                // Update rate with exponential smoothing
                float old_rate = current_error_rate.load(std::memory_order_relaxed);
                current_error_rate.store(
                    0.7f * new_rate + 0.3f * old_rate, 
                    std::memory_order_release);
            }
        }
    }
};

// Global error tracker singleton
inline RadiationErrorTracker& getGlobalErrorTracker() {
    static RadiationErrorTracker tracker;
    return tracker;
}

} // namespace runtime
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ERROR_TRACKER_HPP 