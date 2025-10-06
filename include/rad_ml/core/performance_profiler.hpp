/**
 * @file performance_profiler.hpp
 * @brief Performance profiling and monitoring system
 *
 * This file provides a comprehensive performance profiling system
 * for monitoring and optimizing the radiation-tolerant ML framework.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad_ml {
namespace profiling {

/**
 * @brief Performance metric types
 */
enum class MetricType {
    TIME,            // Execution time
    MEMORY,          // Memory usage
    CACHE_MISSES,    // Cache miss rate
    CPU_USAGE,       // CPU utilization
    SIMD_EFFICIENCY  // SIMD instruction efficiency
};

/**
 * @brief Performance measurement point
 */
struct PerformancePoint {
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    size_t memory_start;
    size_t memory_end;
    double cpu_usage;
    double cache_miss_rate;
};

/**
 * @brief Performance profiler class
 */
class PerformanceProfiler {
   public:
    static PerformanceProfiler& getInstance()
    {
        static PerformanceProfiler instance;
        return instance;
    }

    /**
     * @brief Start profiling a section of code
     */
    void startProfiling(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::high_resolution_clock::now();
        current_points_[name] = {name, now, now, 0, 0, 0.0, 0.0};
    }

    /**
     * @brief End profiling a section of code
     */
    void endProfiling(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::high_resolution_clock::now();
        if (current_points_.count(name)) {
            current_points_[name].end_time = now;
            completed_points_.push_back(current_points_[name]);
            current_points_.erase(name);
        }
    }

    /**
     * @brief Record memory usage
     */
    void recordMemoryUsage(const std::string& name, size_t memory_usage)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (current_points_.count(name)) {
            current_points_[name].memory_end = memory_usage;
        }
    }

    /**
     * @brief Record CPU usage
     */
    void recordCPUUsage(const std::string& name, double cpu_usage)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (current_points_.count(name)) {
            current_points_[name].cpu_usage = cpu_usage;
        }
    }

    /**
     * @brief Get performance statistics
     */
    void generateReport(const std::string& filename = "performance_report.txt")
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream report(filename);
        if (!report.is_open()) {
            std::cerr << "Failed to open performance report file: " << filename << std::endl;
            return;
        }

        report << "=== Radiation-Tolerant ML Framework Performance Report ===\n\n";

        for (const auto& point : completed_points_) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(point.end_time -
                                                                                  point.start_time);

            report << "Section: " << point.name << "\n";
            report << "  Duration: " << duration.count() << " μs\n";
            report << "  Memory Usage: " << point.memory_end << " bytes\n";
            report << "  CPU Usage: " << point.cpu_usage << "%\n";
            report << "  Cache Miss Rate: " << point.cache_miss_rate << "%\n";
            report << "\n";
        }

        report.close();
        std::cout << "Performance report generated: " << filename << std::endl;
    }

    /**
     * @brief Clear all performance data
     */
    void clearData()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_points_.clear();
        completed_points_.clear();
    }

   private:
    PerformanceProfiler() = default;
    ~PerformanceProfiler() = default;

    std::mutex mutex_;
    std::unordered_map<std::string, PerformancePoint> current_points_;
    std::vector<PerformancePoint> completed_points_;
};

/**
 * @brief Performance monitoring scope guard
 *
 * RAII wrapper for automatic profiling of code sections
 */
class ProfileScope {
   public:
    ProfileScope(const std::string& name) : name_(name)
    {
        PerformanceProfiler::getInstance().startProfiling(name_);
    }

    ~ProfileScope() { PerformanceProfiler::getInstance().endProfiling(name_); }

   private:
    std::string name_;
};

// Macros for easy profiling
#define PROFILE_FUNCTION() ProfileScope profile_scope(__FUNCTION__)
#define PROFILE_SCOPE(name) ProfileScope profile_scope(name)
#define START_PROFILING(name) PerformanceProfiler::getInstance().startProfiling(name)
#define END_PROFILING(name) PerformanceProfiler::getInstance().endProfiling(name)
#define GENERATE_REPORT(filename) PerformanceProfiler::getInstance().generateReport(filename)

/**
 * @brief SIMD performance monitor
 *
 * Monitors SIMD instruction usage and efficiency
 */
class SIMDPerformanceMonitor {
   public:
    static SIMDPerformanceMonitor& getInstance()
    {
        static SIMDPerformanceMonitor instance;
        return instance;
    }

    /**
     * @brief Record SIMD operation
     */
    void recordSIMDOperation(const std::string& operation, size_t vector_size,
                             size_t total_elements)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        simd_stats_[operation].vector_size += vector_size;
        simd_stats_[operation].total_elements += total_elements;
        simd_stats_[operation].call_count++;
    }

    /**
     * @brief Get SIMD efficiency for an operation
     */
    double getSIMDEfficiency(const std::string& operation)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (simd_stats_.count(operation) == 0) {
            return 0.0;
        }

        const auto& stats = simd_stats_[operation];
        if (stats.total_elements == 0) {
            return 0.0;
        }

        return static_cast<double>(stats.vector_size) / static_cast<double>(stats.total_elements);
    }

   private:
    struct SIMDStats {
        size_t vector_size = 0;
        size_t total_elements = 0;
        size_t call_count = 0;
    };

    std::mutex mutex_;
    std::unordered_map<std::string, SIMDStats> simd_stats_;
};

}  // namespace profiling
}  // namespace rad_ml
