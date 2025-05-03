#pragma once

#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <memory>
#include <functional>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <any>
#include <set>
#include <typeinfo>

#include "../tmr/enhanced_tmr.hpp"
#include "../tmr/health_weighted_tmr.hpp"
#include "../tmr/approximate_tmr.hpp"
#include "../neural/selective_hardening.hpp"
#include "physics_radiation_simulator.hpp"

namespace rad_ml {
namespace testing {

/**
 * @brief Performance metrics for benchmarking
 */
struct PerformanceMetrics {
    double execution_time_ms;     ///< Average execution time in milliseconds
    double memory_usage_bytes;    ///< Memory usage in bytes
    double energy_consumption;    ///< Estimated energy consumption (relative units)
    double throughput;            ///< Operations per second
    size_t circuit_resources;     ///< Estimated FPGA/ASIC resources (LUTs, DSPs, etc.)
};

/**
 * @brief Reliability metrics
 */
struct ReliabilityMetrics {
    size_t total_errors;          ///< Total number of errors injected
    size_t detected_errors;       ///< Number of errors detected
    size_t corrected_errors;      ///< Number of errors successfully corrected
    double error_detection_rate;  ///< Percentage of errors detected
    double error_correction_rate; ///< Percentage of errors corrected
    double silent_data_corruption_rate; ///< Rate of undetected errors
    double mean_time_to_failure;  ///< Mean time to failure in seconds
    double recovery_time;         ///< Average time to recover from an error in milliseconds
};

/**
 * @brief Test scenario for benchmarking
 */
struct TestScenario {
    std::string name;             ///< Scenario name
    double radiation_intensity;   ///< Radiation intensity (0-10)
    std::string mission_profile;  ///< Mission profile (e.g., "LEO", "JUPITER")
    double duration_seconds;      ///< Test duration in seconds
    size_t num_iterations;        ///< Number of iterations to run
    size_t data_size_bytes;       ///< Size of test data in bytes
    bool simulate_power_glitches; ///< Whether to simulate power glitches
    bool simulate_memory_errors;  ///< Whether to simulate memory errors
    bool simulate_cpu_errors;     ///< Whether to simulate CPU errors
};

/**
 * @brief Protection mechanism configuration
 */
struct ProtectionConfig {
    bool use_tmr;                 ///< Whether to use TMR
    bool use_enhanced_tmr;        ///< Whether to use enhanced TMR with CRC
    bool use_health_weighted_tmr; ///< Whether to use health-weighted TMR
    bool use_approximate_tmr;     ///< Whether to use approximate TMR
    bool use_selective_hardening; ///< Whether to use selective hardening
    double resource_budget;       ///< Resource budget for selective hardening (0-1)
    
    /**
     * @brief Create default configuration
     */
    static ProtectionConfig defaultConfig() {
        ProtectionConfig config;
        config.use_tmr = true;
        config.use_enhanced_tmr = true;
        config.use_health_weighted_tmr = false;
        config.use_approximate_tmr = false;
        config.use_selective_hardening = false;
        config.resource_budget = 0.3;
        return config;
    }
    
    /**
     * @brief Create configuration with all features enabled
     */
    static ProtectionConfig allFeaturesConfig() {
        ProtectionConfig config;
        config.use_tmr = true;
        config.use_enhanced_tmr = true;
        config.use_health_weighted_tmr = true;
        config.use_approximate_tmr = true;
        config.use_selective_hardening = true;
        config.resource_budget = 0.5;
        return config;
    }
    
    /**
     * @brief Create resource-constrained configuration
     */
    static ProtectionConfig resourceConstrainedConfig() {
        ProtectionConfig config;
        config.use_tmr = true;
        config.use_enhanced_tmr = false;
        config.use_health_weighted_tmr = false;
        config.use_approximate_tmr = true;
        config.use_selective_hardening = true;
        config.resource_budget = 0.2;
        return config;
    }
};

/**
 * @brief Benchmark results
 */
struct BenchmarkResults {
    std::string test_name;                        ///< Name of the benchmark test
    std::map<std::string, PerformanceMetrics> performance_metrics; ///< Performance metrics for each component
    std::map<std::string, ReliabilityMetrics> reliability_metrics; ///< Reliability metrics for each component
    std::map<std::string, std::vector<double>> error_rates_over_time; ///< Error rates over time for plotting
    std::map<std::string, std::vector<double>> correction_rates_over_time; ///< Correction rates over time for plotting
    std::map<std::string, std::map<std::string, double>> comparative_metrics; ///< For comparing different configurations
    
    /**
     * @brief Export results to a CSV file
     * 
     * @param filename Output file name
     * @return True if successful, false otherwise
     */
    bool exportToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        // Write header
        file << "Test: " << test_name << "\n\n";
        
        // Write performance metrics
        file << "Performance Metrics\n";
        file << "Component,Execution Time (ms),Memory Usage (bytes),Energy Consumption,Throughput,Circuit Resources\n";
        for (const auto& [component, metrics] : performance_metrics) {
            file << component << ","
                 << metrics.execution_time_ms << ","
                 << metrics.memory_usage_bytes << ","
                 << metrics.energy_consumption << ","
                 << metrics.throughput << ","
                 << metrics.circuit_resources << "\n";
        }
        file << "\n";
        
        // Write reliability metrics
        file << "Reliability Metrics\n";
        file << "Component,Total Errors,Detected Errors,Corrected Errors,Detection Rate,Correction Rate,SDC Rate,MTTF,Recovery Time\n";
        for (const auto& [component, metrics] : reliability_metrics) {
            file << component << ","
                 << metrics.total_errors << ","
                 << metrics.detected_errors << ","
                 << metrics.corrected_errors << ","
                 << metrics.error_detection_rate << ","
                 << metrics.error_correction_rate << ","
                 << metrics.silent_data_corruption_rate << ","
                 << metrics.mean_time_to_failure << ","
                 << metrics.recovery_time << "\n";
        }
        file << "\n";
        
        // Write error rates over time for the first component (if available)
        if (!error_rates_over_time.empty()) {
            file << "Error Rates Over Time\n";
            file << "Time";
            for (const auto& [component, _] : error_rates_over_time) {
                file << "," << component;
            }
            file << "\n";
            
            // Assume all components have the same number of time points
            size_t num_points = error_rates_over_time.begin()->second.size();
            for (size_t i = 0; i < num_points; ++i) {
                file << i;
                for (const auto& [component, rates] : error_rates_over_time) {
                    file << "," << rates[i];
                }
                file << "\n";
            }
            file << "\n";
        }
        
        // Write comparative metrics
        if (!comparative_metrics.empty()) {
            file << "Comparative Metrics\n";
            
            // Collect all metric names
            std::set<std::string> metric_names;
            for (const auto& [config, metrics] : comparative_metrics) {
                for (const auto& [name, _] : metrics) {
                    metric_names.insert(name);
                }
            }
            
            // Write header with metric names
            file << "Configuration";
            for (const auto& name : metric_names) {
                file << "," << name;
            }
            file << "\n";
            
            // Write metrics for each configuration
            for (const auto& [config, metrics] : comparative_metrics) {
                file << config;
                for (const auto& name : metric_names) {
                    if (metrics.count(name)) {
                        file << "," << metrics.at(name);
                    } else {
                        file << ",";
                    }
                }
                file << "\n";
            }
        }
        
        file.close();
        return true;
    }
    
    /**
     * @brief Get a human-readable summary of the results
     * 
     * @return Formatted summary string
     */
    std::string getSummary() const {
        std::stringstream ss;
        ss << "Benchmark Results: " << test_name << "\n";
        ss << "========================================\n\n";
        
        // Performance summary
        ss << "Performance Metrics:\n";
        ss << "------------------\n";
        for (const auto& [component, metrics] : performance_metrics) {
            ss << component << ":\n";
            ss << "  Execution Time: " << std::fixed << std::setprecision(3) << metrics.execution_time_ms << " ms\n";
            ss << "  Memory Usage: " << metrics.memory_usage_bytes << " bytes\n";
            ss << "  Throughput: " << metrics.throughput << " ops/sec\n\n";
        }
        
        // Reliability summary
        ss << "Reliability Metrics:\n";
        ss << "------------------\n";
        for (const auto& [component, metrics] : reliability_metrics) {
            ss << component << ":\n";
            ss << "  Error Detection Rate: " << std::fixed << std::setprecision(2) 
               << (metrics.error_detection_rate * 100.0) << "%\n";
            ss << "  Error Correction Rate: " << std::fixed << std::setprecision(2) 
               << (metrics.error_correction_rate * 100.0) << "%\n";
            ss << "  Silent Data Corruption: " << std::fixed << std::setprecision(4) 
               << (metrics.silent_data_corruption_rate * 100.0) << "%\n";
            ss << "  Mean Time to Failure: " << std::fixed << std::setprecision(2) 
               << metrics.mean_time_to_failure << " seconds\n\n";
        }
        
        // Comparative summary (if available)
        if (!comparative_metrics.empty()) {
            ss << "Comparative Analysis:\n";
            ss << "------------------\n";
            
            // Find the best configuration for each metric
            std::map<std::string, std::pair<std::string, double>> best_configs;
            
            for (const auto& [config, metrics] : comparative_metrics) {
                for (const auto& [metric, value] : metrics) {
                    if (!best_configs.count(metric) || 
                        (metric.find("error") != std::string::npos && value < best_configs[metric].second) ||
                        (metric.find("error") == std::string::npos && value > best_configs[metric].second)) {
                        best_configs[metric] = {config, value};
                    }
                }
            }
            
            // Report the 3 most important metrics
            std::vector<std::string> key_metrics = {
                "error_correction_rate",
                "execution_time_ms",
                "memory_usage_bytes"
            };
            
            for (const auto& metric : key_metrics) {
                if (best_configs.count(metric)) {
                    const auto& [best_config, value] = best_configs[metric];
                    ss << "Best " << metric << ": " << best_config 
                       << " (" << std::fixed << std::setprecision(2) << value << ")\n";
                }
            }
            
            ss << "\nRecommendation: ";
            
            // Make a simple recommendation based on error correction rate
            if (best_configs.count("error_correction_rate")) {
                ss << "Based on error correction performance, the best configuration is: " 
                   << best_configs["error_correction_rate"].first << "\n";
            } else {
                ss << "Insufficient data for recommendation.\n";
            }
        }
        
        return ss.str();
    }
};

/**
 * @brief Test data generation function type
 */
using TestDataGenerator = std::function<std::vector<uint8_t>(size_t)>;

/**
 * @brief Framework for benchmarking radiation-tolerant ML components
 */
class BenchmarkFramework {
public:
    /**
     * @brief Constructor
     */
    BenchmarkFramework() : random_engine_(std::random_device{}()) {}
    
    /**
     * @brief Run a complete benchmark suite with multiple configurations
     * 
     * @param test_name Name of the benchmark suite
     * @param test_scenario Test scenario to use
     * @param test_data_generator Function to generate test data
     * @param configs Map of configuration names to protection configs
     * @return Benchmark results
     */
    BenchmarkResults runBenchmarkSuite(
        const std::string& test_name,
        const TestScenario& test_scenario,
        TestDataGenerator test_data_generator,
        const std::map<std::string, ProtectionConfig>& configs) {
        
        BenchmarkResults results;
        results.test_name = test_name;
        
        // Create radiation simulator
        PhysicsRadiationSimulator::EnvironmentParams rad_params = 
            PhysicsRadiationSimulator::getMissionEnvironment(test_scenario.mission_profile);
        
        // Set radiation intensity
        rad_params.solar_activity = test_scenario.radiation_intensity;
        
        PhysicsRadiationSimulator radiation_sim(rad_params);
        
        // Generate test data once
        std::vector<uint8_t> test_data = test_data_generator(test_scenario.data_size_bytes);
        
        // Process each configuration
        for (const auto& [config_name, protection_config] : configs) {
            std::cout << "Testing configuration: " << config_name << "\n";
            
            // Clone test data for this configuration
            std::vector<uint8_t> config_test_data = test_data;
            
            // Set up protected test data based on configuration
            std::vector<std::any> protected_data;
            protected_data.reserve(config_test_data.size());
            
            for (size_t i = 0; i < config_test_data.size(); ++i) {
                if (protection_config.use_selective_hardening) {
                    // Use selective hardening based on position in array
                    double criticality = static_cast<double>(i) / config_test_data.size();
                    
                    // Create network component for selective hardening analysis
                    rad_ml::neural::NetworkComponent comp;
                    comp.id = "data_" + std::to_string(i);
                    comp.type = "memory";
                    comp.layer_name = "data_layer";
                    comp.layer_index = 0;
                    comp.index = i;
                    comp.value = static_cast<double>(config_test_data[i]);
                    
                    // Set criticality metrics
                    comp.criticality.sensitivity = criticality;
                    comp.criticality.activation_frequency = criticality;
                    comp.criticality.output_influence = criticality;
                    comp.criticality.complexity = 0.1;
                    comp.criticality.memory_usage = 0.1;
                    
                    // Store in vector for analysis
                    std::vector<rad_ml::neural::NetworkComponent> components = {comp};
                    
                    // Create selective hardening object
                    rad_ml::neural::HardeningConfig hardening_config = 
                        rad_ml::neural::HardeningConfig::defaultConfig();
                    hardening_config.resource_budget = protection_config.resource_budget;
                    
                    rad_ml::neural::SelectiveHardening selective_hardening(hardening_config);
                    auto analysis = selective_hardening.analyzeAndProtect(components);
                    
                    // Apply protection based on analysis
                    protected_data.push_back(selective_hardening.applyProtection(
                        config_test_data[i], comp.id, analysis));
                }
                else if (protection_config.use_enhanced_tmr) {
                    protected_data.push_back(rad_ml::tmr::EnhancedTMR<uint8_t>(config_test_data[i]));
                }
                else if (protection_config.use_health_weighted_tmr) {
                    protected_data.push_back(rad_ml::tmr::HealthWeightedTMR<uint8_t>(config_test_data[i]));
                }
                else if (protection_config.use_approximate_tmr) {
                    protected_data.push_back(rad_ml::tmr::ApproximateTMR<uint8_t>(
                        config_test_data[i], 
                        {rad_ml::tmr::ApproximationType::EXACT,
                         rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
                         rad_ml::tmr::ApproximationType::REDUCED_PRECISION}));
                }
                else if (protection_config.use_tmr) {
                    protected_data.push_back(rad_ml::tmr::EnhancedTMR<uint8_t>(config_test_data[i]));
                }
                else {
                    // No protection
                    protected_data.push_back(config_test_data[i]);
                }
            }
            
            // Run the benchmark
            auto [perf_metrics, reliability_metrics, error_rates, correction_rates] = 
                runBenchmark(config_name, test_scenario, radiation_sim, protected_data, config_test_data);
            
            // Store results
            results.performance_metrics[config_name] = perf_metrics;
            results.reliability_metrics[config_name] = reliability_metrics;
            results.error_rates_over_time[config_name] = error_rates;
            results.correction_rates_over_time[config_name] = correction_rates;
            
            // Add to comparative metrics
            results.comparative_metrics[config_name]["error_detection_rate"] = reliability_metrics.error_detection_rate;
            results.comparative_metrics[config_name]["error_correction_rate"] = reliability_metrics.error_correction_rate;
            results.comparative_metrics[config_name]["silent_data_corruption_rate"] = reliability_metrics.silent_data_corruption_rate;
            results.comparative_metrics[config_name]["execution_time_ms"] = perf_metrics.execution_time_ms;
            results.comparative_metrics[config_name]["memory_usage_bytes"] = perf_metrics.memory_usage_bytes;
            results.comparative_metrics[config_name]["energy_consumption"] = perf_metrics.energy_consumption;
            results.comparative_metrics[config_name]["throughput"] = perf_metrics.throughput;
            results.comparative_metrics[config_name]["circuit_resources"] = static_cast<double>(perf_metrics.circuit_resources);
        }
        
        return results;
    }
    
    /**
     * @brief Run a single benchmark test
     * 
     * @param config_name Configuration name
     * @param test_scenario Test scenario parameters
     * @param radiation_sim Radiation simulator
     * @param protected_data Protected test data
     * @param original_data Original unprotected data for comparison
     * @return Tuple of performance metrics, reliability metrics, error rates over time, and correction rates over time
     */
    std::tuple<PerformanceMetrics, ReliabilityMetrics, 
               std::vector<double>, std::vector<double>> 
    runBenchmark(
        const std::string& config_name,
        const TestScenario& test_scenario,
        PhysicsRadiationSimulator& radiation_sim,
        std::vector<std::any>& protected_data,
        const std::vector<uint8_t>& original_data) {
        
        // Initialize metrics
        PerformanceMetrics perf_metrics = {};
        ReliabilityMetrics reliability_metrics = {};
        
        // Vectors to track metrics over time
        std::vector<double> error_rates;
        std::vector<double> correction_rates;
        
        // Initialize timers
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Set up measurement intervals
        size_t num_intervals = 10;
        double interval_duration = test_scenario.duration_seconds / num_intervals;
        
        // Track errors per interval
        std::vector<size_t> interval_errors(num_intervals, 0);
        std::vector<size_t> interval_detections(num_intervals, 0);
        std::vector<size_t> interval_corrections(num_intervals, 0);
        
        // Run the benchmark for specified duration
        for (size_t iteration = 0; iteration < test_scenario.num_iterations; ++iteration) {
            // Reset test data for each iteration if needed
            std::vector<uint8_t> current_data = original_data;
            
            // Simulate radiation effects for the duration
            auto start_iter = std::chrono::high_resolution_clock::now();
            
            for (size_t interval = 0; interval < num_intervals; ++interval) {
                // Apply radiation effects for this interval
                auto events = radiation_sim.simulateEffects(
                    current_data.data(), 
                    current_data.size(), 
                    std::chrono::milliseconds(static_cast<int>(interval_duration * 1000)));
                
                // Count total errors
                reliability_metrics.total_errors += events.size();
                interval_errors[interval] += events.size();
                
                // Process protected data
                size_t detected = 0;
                size_t corrected = 0;
                
                for (size_t i = 0; i < protected_data.size(); ++i) {
                    // Check for errors and attempt correction
                    if (current_data[i] != original_data[i]) {
                        // Try to detect and correct based on protection type
                        if (protected_data[i].type() == typeid(rad_ml::tmr::EnhancedTMR<uint8_t>)) {
                            auto& tmr = std::any_cast<rad_ml::tmr::EnhancedTMR<uint8_t>&>(protected_data[i]);
                            // Use simple check - when get() is called, TMR performs voting
                            uint8_t current_value = tmr.get();
                            
                            // If value is different from original, assume error is detected
                            if (current_value != original_data[i]) {
                                detected++;
                                
                                // Attempt repair
                                tmr.set(original_data[i]); // Simple repair by setting to original value
                                
                                // Check if correction was successful
                                if (tmr.get() == original_data[i]) {
                                    corrected++;
                                    current_data[i] = original_data[i]; // Fix the data
                                }
                            }
                        }
                        else if (protected_data[i].type() == typeid(rad_ml::tmr::HealthWeightedTMR<uint8_t>)) {
                            auto& tmr = std::any_cast<rad_ml::tmr::HealthWeightedTMR<uint8_t>&>(protected_data[i]);
                            // Use simple check
                            uint8_t current_value = tmr.get();
                            
                            // If value is different from original, assume error is detected
                            if (current_value != original_data[i]) {
                                detected++;
                                
                                // Attempt repair
                                tmr.repair();
                                
                                // Check if correction was successful
                                if (tmr.get() == original_data[i]) {
                                    corrected++;
                                    current_data[i] = original_data[i]; // Fix the data
                                }
                            }
                        }
                        else if (protected_data[i].type() == typeid(rad_ml::tmr::ApproximateTMR<uint8_t>)) {
                            auto& tmr = std::any_cast<rad_ml::tmr::ApproximateTMR<uint8_t>&>(protected_data[i]);
                            // Use simple check
                            uint8_t current_value = tmr.get();
                            
                            // If value is different from original, assume error is detected
                            if (current_value != original_data[i]) {
                                detected++;
                                
                                // Attempt repair
                                tmr.repair();
                                
                                // Check if correction was successful
                                if (tmr.get() == original_data[i]) {
                                    corrected++;
                                    current_data[i] = original_data[i]; // Fix the data
                                }
                            }
                        }
                        // Add more types as needed
                    }
                }
                
                // Update metrics for this interval
                reliability_metrics.detected_errors += detected;
                reliability_metrics.corrected_errors += corrected;
                
                interval_detections[interval] += detected;
                interval_corrections[interval] += corrected;
                
                // Simulate CPU processing for this interval
                // This is a simplistic model - in reality, you would run actual ML operations
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            // Measure iteration time
            auto end_iter = std::chrono::high_resolution_clock::now();
            auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_iter - start_iter).count();
            perf_metrics.execution_time_ms += iter_duration;
        }
        
        // Calculate average execution time
        perf_metrics.execution_time_ms /= test_scenario.num_iterations;
        
        // Calculate throughput (operations per second)
        perf_metrics.throughput = 1000.0 * protected_data.size() / perf_metrics.execution_time_ms;
        
        // Estimate memory usage (very rough approximation)
        size_t tmr_overhead_bytes = protected_data.size() * 2; // TMR typically uses 3x memory
        size_t checksum_overhead_bytes = protected_data.size() * sizeof(uint32_t); // 4 bytes per checksum
        perf_metrics.memory_usage_bytes = original_data.size() + tmr_overhead_bytes + checksum_overhead_bytes;
        
        // Estimate circuit resources (very rough approximation)
        perf_metrics.circuit_resources = protected_data.size() * 10; // Arbitrary units
        
        // Estimate energy consumption (very rough approximation)
        perf_metrics.energy_consumption = perf_metrics.execution_time_ms * 0.1; // Arbitrary units
        
        // Calculate reliability metrics
        if (reliability_metrics.total_errors > 0) {
            reliability_metrics.error_detection_rate = 
                static_cast<double>(reliability_metrics.detected_errors) / reliability_metrics.total_errors;
            
            reliability_metrics.error_correction_rate = 
                static_cast<double>(reliability_metrics.corrected_errors) / reliability_metrics.total_errors;
        } else {
            reliability_metrics.error_detection_rate = 1.0;
            reliability_metrics.error_correction_rate = 1.0;
        }
        
        reliability_metrics.silent_data_corruption_rate = 
            reliability_metrics.total_errors > 0 ? 
            1.0 - reliability_metrics.error_detection_rate : 0.0;
        
        // Calculate MTTF and recovery time (simplified models)
        reliability_metrics.mean_time_to_failure = 
            reliability_metrics.total_errors > 0 ? 
            test_scenario.duration_seconds / reliability_metrics.total_errors : 
            test_scenario.duration_seconds * 10.0; // If no errors, assume 10x test duration
        
        reliability_metrics.recovery_time = 
            reliability_metrics.detected_errors > 0 ? 
            perf_metrics.execution_time_ms / reliability_metrics.detected_errors : 0.0;
        
        // Calculate error and correction rates over time
        for (size_t i = 0; i < num_intervals; ++i) {
            double error_rate = interval_errors[i] > 0 ? 
                static_cast<double>(interval_detections[i]) / interval_errors[i] : 1.0;
            
            double correction_rate = interval_detections[i] > 0 ? 
                static_cast<double>(interval_corrections[i]) / interval_detections[i] : 1.0;
            
            error_rates.push_back(error_rate);
            correction_rates.push_back(correction_rate);
        }
        
        return {perf_metrics, reliability_metrics, error_rates, correction_rates};
    }
    
    /**
     * @brief Generate random test data
     * 
     * @param size Size in bytes
     * @return Random data vector
     */
    static std::vector<uint8_t> generateRandomTestData(size_t size) {
        std::vector<uint8_t> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        return data;
    }
    
    /**
     * @brief Generate test data with a pattern
     * 
     * @param size Size in bytes
     * @return Patterned data vector
     */
    static std::vector<uint8_t> generatePatternedTestData(size_t size) {
        std::vector<uint8_t> data(size);
        
        for (size_t i = 0; i < size; ++i) {
            // Some pattern, e.g., incrementing values
            data[i] = static_cast<uint8_t>(i % 256);
        }
        
        return data;
    }
    
    /**
     * @brief Create default test scenarios
     * 
     * @return Map of scenario names to test scenarios
     */
    static std::map<std::string, TestScenario> createDefaultTestScenarios() {
        std::map<std::string, TestScenario> scenarios;
        
        // Low Earth Orbit - Moderate radiation
        TestScenario leo;
        leo.name = "Low Earth Orbit";
        leo.radiation_intensity = 3.0;
        leo.mission_profile = "LEO";
        leo.duration_seconds = 60.0;
        leo.num_iterations = 5;
        leo.data_size_bytes = 1024;
        leo.simulate_power_glitches = false;
        leo.simulate_memory_errors = true;
        leo.simulate_cpu_errors = false;
        scenarios["LEO"] = leo;
        
        // Mars mission - Higher radiation
        TestScenario mars;
        mars.name = "Mars Mission";
        mars.radiation_intensity = 5.0;
        mars.mission_profile = "MARS";
        mars.duration_seconds = 120.0;
        mars.num_iterations = 3;
        mars.data_size_bytes = 1024;
        mars.simulate_power_glitches = true;
        mars.simulate_memory_errors = true;
        mars.simulate_cpu_errors = true;
        scenarios["MARS"] = mars;
        
        // Jupiter mission - Extreme radiation
        TestScenario jupiter;
        jupiter.name = "Jupiter Mission";
        jupiter.radiation_intensity = 10.0;
        jupiter.mission_profile = "JUPITER";
        jupiter.duration_seconds = 180.0;
        jupiter.num_iterations = 2;
        jupiter.data_size_bytes = 1024;
        jupiter.simulate_power_glitches = true;
        jupiter.simulate_memory_errors = true;
        jupiter.simulate_cpu_errors = true;
        scenarios["JUPITER"] = jupiter;
        
        return scenarios;
    }
    
private:
    std::mt19937 random_engine_;
};

} // namespace testing
} // namespace rad_ml 