#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <thread>
#include <vector>

// PERFORMANCE: Cross-platform CPU pause
#if defined(_MSC_VER)
#include <intrin.h>
#define CPU_PAUSE() _mm_pause()
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
#define CPU_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_PAUSE() asm volatile("yield" ::: "memory")
#else
#define CPU_PAUSE()         \
    do {                    \
        volatile int x = 0; \
        (void)x;            \
    } while (0)
#endif
#else
#define CPU_PAUSE()         \
    do {                    \
        volatile int x = 0; \
        (void)x;            \
    } while (0)
#endif

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
    TemporalRedundancy(size_t num_executions = 3,
                       std::chrono::milliseconds delay_between =
                           std::chrono::milliseconds(1))  // PERFORMANCE: Reduced from 10ms to 1ms
        : num_executions_(num_executions), delay_between_(delay_between), fast_mode_(false)
    {
    }

    /**
     * @brief Execute operation multiple times with time-based voting
     *
     * @param data Input data for the operation
     * @param operation Function to execute
     * @return Result determined by temporal voting
     */
    ResultType execute(const T& data, std::function<ResultType(const T&)> operation) const
    {
        std::vector<ResultType> results;
        results.reserve(num_executions_);

        // PERFORMANCE OPTIMIZATION: Fast mode with no delays for low-radiation environments
        if (fast_mode_) {
            for (size_t i = 0; i < num_executions_; ++i) {
                results.push_back(operation(data));
            }
        }
        else {
            // Execute multiple times with minimal delay between
            for (size_t i = 0; i < num_executions_; ++i) {
                results.push_back(operation(data));

                // Add minimal delay between executions to avoid correlated errors
                if (i < num_executions_ - 1) {
                    // PERFORMANCE: Use CPU pause instead of sleep for sub-millisecond delays
                    if (delay_between_.count() < 1) {
                        for (int pause = 0; pause < 100; ++pause) {
                            CPU_PAUSE();
                        }
                    }
                    else {
                        std::this_thread::sleep_for(delay_between_);
                    }
                }
            }
        }

        // Find most common result (similar to spatial voting)
        return findMostCommonResult(results);
    }

    /**
     * @brief Enable fast mode for performance-critical operations
     *
     * @param enable Whether to enable fast mode (no delays)
     */
    void setFastMode(bool enable) { fast_mode_ = enable; }

    /**
     * @brief Change configuration based on radiation environment
     *
     * @param num_executions New number of executions
     * @param delay_ms New delay in milliseconds
     */
    void reconfigure(size_t num_executions, uint64_t delay_ms)
    {
        num_executions_ = num_executions;
        delay_between_ = std::chrono::milliseconds(delay_ms);

        // PERFORMANCE: Auto-enable fast mode for very low delays
        if (delay_ms == 0) {
            fast_mode_ = true;
        }
    }

   private:
    size_t num_executions_;
    std::chrono::milliseconds delay_between_;
    bool fast_mode_;

    /**
     * @brief Find most common result through voting
     *
     * @param results Vector of results from multiple executions
     * @return Most common result (temporal majority)
     */
    ResultType findMostCommonResult(const std::vector<ResultType>& results) const
    {
        // Count occurrences of each result
        std::map<ResultType, size_t> result_counts;
        for (const auto& result : results) {
            result_counts[result]++;
        }

        // Find result with highest count
        auto max_element =
            std::max_element(result_counts.begin(), result_counts.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });

        return max_element->first;
    }
};

}  // namespace tmr
}  // namespace rad_ml
