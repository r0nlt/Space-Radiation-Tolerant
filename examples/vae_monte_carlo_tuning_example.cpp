/**
 * @file vae_monte_carlo_tuning_example.cpp
 * @brief Comprehensive VAE tuning with Monte Carlo testing for AI Native Database optimization
 *
 * This example demonstrates how to use the VAEAutoTuner to systematically find
 * the best VAE configurations for different use cases:
 * 1. Data compression for database storage
 * 2. Anomaly detection in telemetry data
 * 3. Balanced performance for both use cases
 *
 * Uses Monte Carlo testing for statistical reliability and confidence intervals.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/rad_ml/core/logger.hpp"
#include "../include/rad_ml/research/vae_auto_tuner.hpp"
#include "../include/rad_ml/storage/ai_native_database.hpp"

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::storage;

/**
 * @brief Generate realistic satellite telemetry data for VAE tuning
 */
struct SatelliteTelemetry {
    float power_voltage;        // Battery voltage (V)
    float solar_current;        // Solar panel current (A)
    float temperature_cpu;      // CPU temperature (°C)
    float temperature_battery;  // Battery temperature (°C)
    float attitude_roll;        // Roll angle (degrees)
    float attitude_pitch;       // Pitch angle (degrees)
    float attitude_yaw;         // Yaw angle (degrees)
    float orbit_altitude;       // Altitude (km)
    float orbit_velocity;       // Orbital velocity (km/s)
    float communication_rssi;   // Signal strength (dBm)
    float thruster_fuel;        // Remaining fuel (%)
    float memory_usage;         // Memory utilization (%)

    std::vector<float> toVector() const
    {
        return {power_voltage,  solar_current,      temperature_cpu, temperature_battery,
                attitude_roll,  attitude_pitch,     attitude_yaw,    orbit_altitude,
                orbit_velocity, communication_rssi, thruster_fuel,   memory_usage};
    }
};

/**
 * @brief Generate comprehensive training dataset with orbital dynamics
 */
std::vector<std::vector<float>> generateTrainingData(size_t num_samples, uint64_t seed)
{
    std::mt19937 gen(seed);
    std::vector<std::vector<float>> data;
    data.reserve(num_samples);

    // Normal operating ranges with realistic distributions
    std::normal_distribution<float> voltage_dist(28.0f, 2.0f);     // 24-32V typical
    std::normal_distribution<float> current_dist(15.0f, 3.0f);     // 10-20A solar
    std::normal_distribution<float> temp_cpu_dist(25.0f, 10.0f);   // 0-50°C
    std::normal_distribution<float> temp_batt_dist(20.0f, 15.0f);  // -10-50°C
    std::uniform_real_distribution<float> attitude_dist(-180.0f, 180.0f);
    std::normal_distribution<float> altitude_dist(400.0f, 50.0f);     // LEO orbit
    std::normal_distribution<float> velocity_dist(7.8f, 0.1f);        // ~7.8 km/s
    std::normal_distribution<float> rssi_dist(-85.0f, 5.0f);          // Signal strength
    std::uniform_real_distribution<float> fuel_dist(50.0f, 100.0f);   // Fuel remaining
    std::uniform_real_distribution<float> memory_dist(30.0f, 80.0f);  // Memory usage

    for (size_t i = 0; i < num_samples; ++i) {
        SatelliteTelemetry telem;

        // Add orbital dynamics (sinusoidal patterns for realistic behavior)
        float orbit_phase = (float)i / 100.0f;                         // ~100 samples per orbit
        float eclipse_factor = std::max(0.0f, std::sin(orbit_phase));  // Solar eclipse simulation

        telem.power_voltage = voltage_dist(gen) + 2.0f * std::sin(orbit_phase);
        telem.solar_current =
            current_dist(gen) * eclipse_factor + 5.0f * std::sin(orbit_phase + M_PI / 2);
        telem.temperature_cpu = temp_cpu_dist(gen) + 10.0f * std::sin(orbit_phase);
        telem.temperature_battery = temp_batt_dist(gen) + 8.0f * std::sin(orbit_phase + M_PI / 4);
        telem.attitude_roll = attitude_dist(gen);
        telem.attitude_pitch = attitude_dist(gen);
        telem.attitude_yaw = attitude_dist(gen);
        telem.orbit_altitude = altitude_dist(gen) + 10.0f * std::sin(orbit_phase * 2);
        telem.orbit_velocity = velocity_dist(gen);
        telem.communication_rssi = rssi_dist(gen) + 5.0f * eclipse_factor;
        telem.thruster_fuel = fuel_dist(gen) - (float)i * 0.001f;  // Gradual consumption
        telem.memory_usage = memory_dist(gen) + 10.0f * std::sin(orbit_phase * 3);

        data.push_back(telem.toVector());
    }

    return data;
}

/**
 * @brief Generate validation data with different characteristics
 */
std::vector<std::vector<float>> generateValidationData(size_t num_samples, uint64_t seed)
{
    // Use different seed and slightly different parameters for validation
    return generateTrainingData(num_samples, seed + 12345);
}

/**
 * @brief Generate test data with anomalies for comprehensive evaluation
 */
std::vector<std::vector<float>> generateTestData(size_t num_samples, uint64_t seed)
{
    auto test_data = generateTrainingData(num_samples, seed + 54321);

    // Inject some anomalies for testing anomaly detection capabilities
    std::mt19937 gen(seed + 99999);
    std::uniform_real_distribution<float> anomaly_prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> anomaly_type(0, 3);

    for (auto& sample : test_data) {
        if (anomaly_prob(gen) < 0.05f) {  // 5% anomaly rate
            switch (anomaly_type(gen)) {
                case 0:                 // Power system failure
                    sample[0] *= 0.5f;  // Voltage drop
                    sample[1] *= 0.3f;  // Current drop
                    break;
                case 1:                  // Thermal anomaly
                    sample[2] += 40.0f;  // CPU overheat
                    break;
                case 2:                  // Attitude control failure
                    sample[4] += 90.0f;  // Roll deviation
                    sample[5] += 60.0f;  // Pitch deviation
                    break;
                case 3:                                             // Memory leak
                    sample[11] = 95.0f + anomaly_prob(gen) * 5.0f;  // High memory usage
                    break;
            }
        }
    }

    return test_data;
}

/**
 * @brief Run comprehensive VAE tuning with Monte Carlo testing
 */
void runComprehensiveVAETuning()
{
    std::cout << "\n=== COMPREHENSIVE VAE TUNING WITH MONTE CARLO TESTING ===" << std::endl;

    // Generate datasets
    std::cout << "Generating training, validation, and test datasets..." << std::endl;
    auto training_data = generateTrainingData(2000, 12345);
    auto validation_data = generateValidationData(500, 67890);
    auto test_data = generateTestData(300, 11111);

    std::cout << "Training samples: " << training_data.size() << std::endl;
    std::cout << "Validation samples: " << validation_data.size() << std::endl;
    std::cout << "Test samples: " << test_data.size() << std::endl;

    // Create VAE auto tuner
    VAEAutoTuner tuner(training_data, validation_data, test_data,
                       "comprehensive_vae_tuning_results.csv");

    // Set random seed for reproducible results
    tuner.setSeed(42);

    // Define search parameters
    std::vector<size_t> latent_dims = {2, 4, 6, 8, 12, 16};
    std::vector<float> beta_values = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f};
    std::vector<std::vector<size_t>> architectures = {
        {32},            // Minimal
        {64},            // Small
        {128},           // Medium
        {64, 32},        // Two-layer funnel
        {128, 64},       // Medium funnel
        {256, 128},      // Large funnel
        {128, 64, 32},   // Three-layer funnel
        {256, 128, 64},  // Large three-layer
        {512, 256, 128}  // Very large
    };
    std::vector<float> learning_rates = {0.0001f, 0.001f, 0.01f};

    // Test different use cases with different Monte Carlo trial counts
    std::vector<std::pair<std::string, size_t>> use_cases = {
        {"compression", 15},        // More trials for compression (critical for database)
        {"anomaly_detection", 12},  // Good trials for anomaly detection
        {"balanced", 10}            // Standard trials for balanced approach
    };

    std::map<std::string, VAESearchResult> best_results;

    for (const auto& [use_case, monte_carlo_trials] : use_cases) {
        std::cout << "\n--- OPTIMIZING FOR: " << use_case << " ---" << std::endl;
        std::cout << "Monte Carlo trials per configuration: " << monte_carlo_trials << std::endl;

        // Start with random search for quick exploration
        std::cout << "\nPhase 1: Random Search (Quick Exploration)" << std::endl;
        auto random_result = tuner.randomSearch(30, monte_carlo_trials, use_case);

        std::cout << "Random search completed. Best score: "
                  << (use_case == "compression"         ? random_result.result.compression_score
                      : use_case == "anomaly_detection" ? random_result.result.anomaly_score
                                                        : random_result.result.balanced_score)
                  << std::endl;

        // Follow up with focused grid search around promising regions
        std::cout << "\nPhase 2: Focused Grid Search" << std::endl;

        // Narrow down search space based on random search results
        std::vector<size_t> focused_latent_dims;
        std::vector<float> focused_betas;

        // Focus around the best configuration found
        size_t best_latent = random_result.config.latent_dim;
        float best_beta = random_result.config.beta;

        // Create focused ranges
        for (auto dim : latent_dims) {
            if (std::abs(static_cast<int>(dim) - static_cast<int>(best_latent)) <= 4) {
                focused_latent_dims.push_back(dim);
            }
        }

        for (auto beta : beta_values) {
            if (std::abs(beta - best_beta) <= 1.0f) {
                focused_betas.push_back(beta);
            }
        }

        // Ensure we have at least some options
        if (focused_latent_dims.empty()) focused_latent_dims = {best_latent};
        if (focused_betas.empty()) focused_betas = {best_beta};

        auto grid_result = tuner.gridSearch(focused_latent_dims, focused_betas, architectures,
                                            learning_rates, monte_carlo_trials + 5,
                                            use_case);  // Extra trials for final optimization

        std::cout << "Grid search completed. Best score: "
                  << (use_case == "compression"         ? grid_result.result.compression_score
                      : use_case == "anomaly_detection" ? grid_result.result.anomaly_score
                                                        : grid_result.result.balanced_score)
                  << std::endl;

        // Keep the better result
        best_results[use_case] =
            (grid_result.result.balanced_score > random_result.result.balanced_score)
                ? grid_result
                : random_result;
    }

    // Export comprehensive results
    std::cout << "\n=== EXPORTING RESULTS ===" << std::endl;
    tuner.exportResults("final_vae_tuning_results.csv");
    tuner.generateReport("vae_tuning_comprehensive_report.md");

    // Display final recommendations
    std::cout << "\n=== FINAL RECOMMENDATIONS ===" << std::endl;

    for (const auto& [use_case, result] : best_results) {
        std::cout << "\n--- BEST CONFIGURATION FOR: " << use_case << " ---" << std::endl;
        std::cout << "Latent Dimension: " << result.config.latent_dim << std::endl;
        std::cout << "Beta Parameter: " << result.config.beta << std::endl;
        std::cout << "Learning Rate: " << result.config.learning_rate << std::endl;
        std::cout << "Epochs: " << result.config.epochs << std::endl;
        std::cout << "Architecture: ";
        for (size_t i = 0; i < result.config.hidden_dims.size(); ++i) {
            std::cout << result.config.hidden_dims[i];
            if (i < result.config.hidden_dims.size() - 1) std::cout << "-";
        }
        std::cout << std::endl;

        std::cout << "\nPerformance Metrics:" << std::endl;
        std::cout << "  Compression Ratio: " << std::fixed << std::setprecision(2)
                  << result.result.compression_ratio_mean << " ± "
                  << result.result.compression_ratio_stddev << std::endl;
        std::cout << "  Reconstruction Error: " << std::fixed << std::setprecision(4)
                  << result.result.reconstruction_error_mean << " ± "
                  << result.result.reconstruction_error_stddev << std::endl;
        std::cout << "  Training Time: " << std::fixed << std::setprecision(1)
                  << result.result.training_time_ms_mean << " ± "
                  << result.result.training_time_ms_stddev << " ms" << std::endl;
        std::cout << "  Convergence Rate: " << std::fixed << std::setprecision(1)
                  << result.result.convergence_rate * 100 << "%" << std::endl;
        std::cout << "  Monte Carlo Trials: " << result.result.monte_carlo_trials << std::endl;

        std::cout << "\nComposite Scores:" << std::endl;
        std::cout << "  Compression Score: " << std::fixed << std::setprecision(3)
                  << result.result.compression_score << std::endl;
        std::cout << "  Anomaly Score: " << std::fixed << std::setprecision(3)
                  << result.result.anomaly_score << std::endl;
        std::cout << "  Balanced Score: " << std::fixed << std::setprecision(3)
                  << result.result.balanced_score << std::endl;
    }
}

/**
 * @brief Test the optimized VAE configurations with the AI Native Database
 */
void testOptimizedVAEWithDatabase(const VAESearchResult& compression_config,
                                  const VAESearchResult& anomaly_config)
{
    std::cout << "\n=== TESTING OPTIMIZED VAE WITH AI NATIVE DATABASE ===" << std::endl;

    // Test compression-optimized configuration
    std::cout << "\n--- Testing Compression-Optimized VAE ---" << std::endl;

    AINativeDatabase::Config db_config;
    db_config.db_path = "./optimized_compression_test_db";
    db_config.default_latent_dim = compression_config.config.latent_dim;
    db_config.vae_hidden_dims = compression_config.config.hidden_dims;
    db_config.max_reconstruction_error = 0.01f;  // Tight error tolerance for compression

    AINativeDatabase compression_db(db_config);
    auto init_result = compression_db.initialize({{"telemetry", 12}});

    if (init_result) {
        // Generate test data
        auto test_data = generateTrainingData(100, 99999);

        std::vector<double> compression_ratios;
        std::vector<double> reconstruction_errors;
        std::vector<double> storage_times;
        std::vector<double> retrieval_times;

        for (size_t i = 0; i < test_data.size(); ++i) {
            std::string key = "telemetry_" + std::to_string(i);

            // Store data
            auto store_start = std::chrono::high_resolution_clock::now();
            auto store_result = compression_db.store(key, test_data[i], "telemetry");
            auto store_end = std::chrono::high_resolution_clock::now();

            if (store_result) {
                storage_times.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(store_end - store_start)
                        .count());
                compression_ratios.push_back(store_result->ratio);

                // Retrieve data
                auto retrieve_start = std::chrono::high_resolution_clock::now();
                auto retrieve_result = compression_db.retrieve<float>(key);
                auto retrieve_end = std::chrono::high_resolution_clock::now();

                if (retrieve_result) {
                    retrieval_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                                                  retrieve_end - retrieve_start)
                                                  .count());
                    reconstruction_errors.push_back(retrieve_result->second.error);
                }
            }
        }

        // Calculate statistics
        auto calculate_stats = [](const std::vector<double>& values) {
            double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double variance = 0.0;
            for (double val : values) {
                variance += (val - mean) * (val - mean);
            }
            variance /= values.size();
            return std::make_pair(mean, std::sqrt(variance));
        };

        auto [comp_mean, comp_std] = calculate_stats(compression_ratios);
        auto [error_mean, error_std] = calculate_stats(reconstruction_errors);
        auto [store_mean, store_std] = calculate_stats(storage_times);
        auto [retrieve_mean, retrieve_std] = calculate_stats(retrieval_times);

        std::cout << "Database Integration Results (Compression-Optimized):" << std::endl;
        std::cout << "  Compression Ratio: " << std::fixed << std::setprecision(2) << comp_mean
                  << " ± " << comp_std << ":1" << std::endl;
        std::cout << "  Reconstruction Error: " << std::fixed << std::setprecision(4) << error_mean
                  << " ± " << error_std << std::endl;
        std::cout << "  Storage Time: " << std::fixed << std::setprecision(1) << store_mean << " ± "
                  << store_std << " μs" << std::endl;
        std::cout << "  Retrieval Time: " << std::fixed << std::setprecision(1) << retrieve_mean
                  << " ± " << retrieve_std << " μs" << std::endl;

        // Get database statistics
        auto stats = compression_db.get_statistics();
        std::cout << "  Total Entries: " << stats.total_entries << std::endl;
        std::cout << "  Storage Efficiency: " << std::fixed << std::setprecision(1)
                  << (100.0 * stats.total_compressed_bytes / stats.total_original_bytes) << "%"
                  << std::endl;
    }

    std::cout << "\n--- Testing Anomaly-Detection-Optimized VAE ---" << std::endl;
    // Similar testing for anomaly detection configuration...
    // (Implementation would follow similar pattern)
}

/**
 * @brief Run evolutionary search for advanced optimization
 */
void runEvolutionaryOptimization()
{
    std::cout << "\n=== EVOLUTIONARY VAE OPTIMIZATION ===" << std::endl;

    // Generate smaller but high-quality dataset for evolutionary search
    auto training_data = generateTrainingData(1000, 77777);
    auto validation_data = generateValidationData(200, 88888);

    VAEAutoTuner evolutionary_tuner(training_data, validation_data, {},
                                    "evolutionary_vae_results.csv");
    evolutionary_tuner.setSeed(123);

    // Run evolutionary search with different parameters
    std::cout << "Running evolutionary search..." << std::endl;
    auto evo_result = evolutionary_tuner.evolutionarySearch(25,    // population_size
                                                            15,    // generations
                                                            0.15,  // mutation_rate
                                                            8,     // monte_carlo_trials
                                                            "balanced");

    std::cout << "Evolutionary optimization completed!" << std::endl;
    std::cout << "Best configuration found:" << std::endl;
    std::cout << "  Latent Dim: " << evo_result.config.latent_dim << std::endl;
    std::cout << "  Beta: " << evo_result.config.beta << std::endl;
    std::cout << "  Balanced Score: " << evo_result.result.balanced_score << std::endl;
    std::cout << "  Convergence Rate: " << evo_result.result.convergence_rate * 100 << "%"
              << std::endl;
}

/**
 * @brief Main function demonstrating comprehensive VAE tuning
 */
int main()
{
    try {
        core::Logger::info("Starting comprehensive VAE tuning with Monte Carlo testing");

        // Run comprehensive tuning
        runComprehensiveVAETuning();

        // Run evolutionary optimization
        runEvolutionaryOptimization();

        std::cout << "\n=== VAE TUNING COMPLETED SUCCESSFULLY ===" << std::endl;
        std::cout << "Check the following files for detailed results:" << std::endl;
        std::cout << "  - comprehensive_vae_tuning_results.csv" << std::endl;
        std::cout << "  - final_vae_tuning_results.csv" << std::endl;
        std::cout << "  - vae_tuning_comprehensive_report.md" << std::endl;
        std::cout << "  - evolutionary_vae_results.csv" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during VAE tuning: " << e.what() << std::endl;
        return 1;
    }
}
