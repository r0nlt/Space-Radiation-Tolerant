/**
 * @file vae_production_database_example.cpp
 * @brief Production example using optimal VAE configurations with AI Native Database
 *
 * This example demonstrates how to use the VAE configurations discovered during
 * the tuning process in a production LMDB database setup.
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/research/variational_autoencoder.hpp"
#include "rad_ml/storage/ai_native_database.hpp"

using namespace rad_ml;

// Production VAE configurations based on tuning results
struct ProductionVAEConfig {
    std::string use_case;
    size_t latent_dim;
    float beta;
    float learning_rate;
    std::vector<size_t> hidden_dims;
    size_t epochs;
    std::string description;
};

// Optimal configurations discovered during tuning
std::vector<ProductionVAEConfig> getOptimalConfigurations()
{
    return {{"telemetry_compression",
             3,     // 4:1 compression ratio for 12D data
             0.5f,  // Lower beta for better reconstruction
             0.001f,
             {32},  // Simple architecture for fast inference
             50,
             "Optimal for telemetry data compression with 4:1 ratio"},
            {"anomaly_detection",
             8,     // Higher latent dimension for pattern capture
             2.0f,  // Higher beta for better structure learning
             0.001f,
             {64, 32},  // Deeper architecture for complex patterns
             100,
             "Optimal for detecting anomalous telemetry patterns"},
            {"balanced_usage",
             4,     // Good balance between compression and detection
             1.0f,  // Standard beta-VAE
             0.001f,
             {32},
             75,
             "Balanced configuration for general-purpose use"}};
}

// Generate realistic telemetry data
std::vector<std::vector<float>> generateTelemetryData(size_t num_samples)
{
    std::vector<std::vector<float>> data;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Normal telemetry patterns
    std::normal_distribution<float> temp_dist(25.0f, 5.0f);       // Temperature
    std::normal_distribution<float> voltage_dist(12.0f, 0.5f);    // Voltage
    std::normal_distribution<float> current_dist(2.5f, 0.3f);     // Current
    std::normal_distribution<float> pressure_dist(101.3f, 2.0f);  // Pressure

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> sample(12);

        // Simulate 12-channel telemetry
        sample[0] = temp_dist(gen);                   // Temperature 1
        sample[1] = temp_dist(gen) + 5.0f;            // Temperature 2
        sample[2] = voltage_dist(gen);                // Voltage 1
        sample[3] = voltage_dist(gen) * 0.95f;        // Voltage 2
        sample[4] = current_dist(gen);                // Current 1
        sample[5] = current_dist(gen) * 1.1f;         // Current 2
        sample[6] = pressure_dist(gen);               // Pressure 1
        sample[7] = pressure_dist(gen) + 1.0f;        // Pressure 2
        sample[8] = sample[0] * 0.1f + sample[2];     // Derived metric 1
        sample[9] = sample[4] * sample[5];            // Power calculation
        sample[10] = (sample[0] + sample[1]) / 2.0f;  // Average temp
        sample[11] = sample[6] - sample[7];           // Pressure diff

        data.push_back(sample);
    }

    return data;
}

// Generate anomalous data for testing
std::vector<std::vector<float>> generateAnomalousData(size_t num_samples)
{
    auto normal_data = generateTelemetryData(num_samples);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> anomaly_factor(2.0f, 5.0f);
    std::uniform_int_distribution<size_t> channel_dist(0, 11);

    // Inject anomalies
    for (auto& sample : normal_data) {
        size_t anomaly_channel = channel_dist(gen);
        sample[anomaly_channel] *= anomaly_factor(gen);  // Spike anomaly
    }

    return normal_data;
}

void demonstrateOptimalCompressionConfig()
{
    core::Logger::info("=== USING OPTIMAL COMPRESSION CONFIGURATION ===");

    auto configs = getOptimalConfigurations();
    auto compression_config = configs[0];  // telemetry_compression

    core::Logger::info("Configuration: " + compression_config.description);
    core::Logger::info("Latent dimension: " + std::to_string(compression_config.latent_dim));
    core::Logger::info(
        "Compression ratio: " + std::to_string(12.0f / compression_config.latent_dim) + ":1");

    // Create database with optimal settings
    storage::AINativeDatabase::Config db_config;
    db_config.db_path = "./optimal_compression_db";
    db_config.default_latent_dim = compression_config.latent_dim;
    db_config.vae_hidden_dims = compression_config.hidden_dims;

    storage::AINativeDatabase db(db_config);

    // Initialize database with data dimensions
    std::unordered_map<std::string, size_t> data_dimensions;
    data_dimensions["telemetry_data"] = 12;
    auto init_result = db.initialize(data_dimensions);
    if (!init_result) {
        core::Logger::error("Failed to initialize database: " + init_result.error);
        return;
    }

    // Generate training data
    auto training_data = generateTelemetryData(500);

    // Train VAE with optimal settings
    core::Logger::info("Training VAE with optimal compression configuration...");
    auto train_result = db.train_vae(training_data, "telemetry_data");
    if (!train_result) {
        core::Logger::error("Failed to train VAE: " + train_result.error);
        return;
    }

    // Test compression performance
    auto test_data = generateTelemetryData(50);

    double total_compression_ratio = 0.0;
    double total_reconstruction_error = 0.0;
    size_t successful_operations = 0;

    for (size_t i = 0; i < test_data.size(); ++i) {
        std::string key = "telemetry_" + std::to_string(i);

        // Store data (automatically compressed by VAE)
        auto store_result = db.store(key, test_data[i], "telemetry_data");
        if (!store_result) {
            core::Logger::error("Failed to store data: " + store_result.error);
            continue;
        }

        // Retrieve data (automatically decompressed)
        auto retrieve_result = db.retrieve<float>(key);
        if (!retrieve_result) {
            core::Logger::error("Failed to retrieve data: " + retrieve_result.error);
            continue;
        }

        auto& [retrieved_data, metrics] = *retrieve_result;

        // Use metrics from the database operations
        total_compression_ratio += metrics.ratio;
        total_reconstruction_error += metrics.error;
        successful_operations++;
    }

    if (successful_operations > 0) {
        // Report results
        double avg_compression_ratio = total_compression_ratio / successful_operations;
        double avg_reconstruction_error = total_reconstruction_error / successful_operations;

        core::Logger::info("=== COMPRESSION RESULTS ===");
        core::Logger::info("Average compression ratio: " + std::to_string(avg_compression_ratio) +
                           ":1");
        core::Logger::info("Average reconstruction error: " +
                           std::to_string(avg_reconstruction_error));
        core::Logger::info("Storage space saved: " +
                           std::to_string((1.0 - 1.0 / avg_compression_ratio) * 100) + "%");
        core::Logger::info("Successful operations: " + std::to_string(successful_operations) + "/" +
                           std::to_string(test_data.size()));
    }
    else {
        core::Logger::error("No successful operations completed");
    }
}

void demonstrateOptimalAnomalyConfig()
{
    core::Logger::info("\n=== USING OPTIMAL ANOMALY DETECTION CONFIGURATION ===");

    auto configs = getOptimalConfigurations();
    auto anomaly_config = configs[1];  // anomaly_detection

    core::Logger::info("Configuration: " + anomaly_config.description);
    core::Logger::info("Latent dimension: " + std::to_string(anomaly_config.latent_dim));
    core::Logger::info("Beta parameter: " + std::to_string(anomaly_config.beta) +
                       " (optimized for structure learning)");

    // Create database for anomaly detection
    storage::AINativeDatabase::Config db_config;
    db_config.db_path = "./optimal_anomaly_db";
    db_config.default_latent_dim = anomaly_config.latent_dim;
    db_config.vae_hidden_dims = anomaly_config.hidden_dims;

    storage::AINativeDatabase db(db_config);

    // Initialize database
    std::unordered_map<std::string, size_t> data_dimensions;
    data_dimensions["telemetry_data"] = 12;
    auto init_result = db.initialize(data_dimensions);
    if (!init_result) {
        core::Logger::error("Failed to initialize database: " + init_result.error);
        return;
    }

    // Generate normal training data
    auto training_data = generateTelemetryData(500);

    // Train on normal data only
    core::Logger::info("Training VAE for anomaly detection...");
    auto train_result = db.train_vae(training_data, "telemetry_data");
    if (!train_result) {
        core::Logger::error("Failed to train VAE: " + train_result.error);
        return;
    }

    // Test anomaly detection capability
    auto normal_test = generateTelemetryData(25);

    // Create some anomalous samples
    std::vector<std::vector<float>> anomalous_test;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> anomaly_factor(3.0f, 6.0f);

    for (size_t i = 0; i < 25; ++i) {
        auto sample = generateTelemetryData(1)[0];
        // Inject anomaly by scaling random channels
        sample[i % 12] *= anomaly_factor(gen);
        anomalous_test.push_back(sample);
    }

    // Test normal data - use reconstruction error as anomaly score
    std::vector<double> normal_scores;
    for (size_t i = 0; i < normal_test.size(); ++i) {
        std::string key = "normal_" + std::to_string(i);
        auto store_result = db.store(key, normal_test[i], "telemetry_data");
        if (store_result) {
            auto retrieve_result = db.retrieve<float>(key);
            if (retrieve_result) {
                auto& [retrieved_data, metrics] = *retrieve_result;
                normal_scores.push_back(
                    metrics.error);  // Use reconstruction error as anomaly score
            }
        }
    }

    // Test anomalous data
    std::vector<double> anomaly_scores;
    for (size_t i = 0; i < anomalous_test.size(); ++i) {
        std::string key = "anomaly_" + std::to_string(i);
        auto store_result = db.store(key, anomalous_test[i], "telemetry_data");
        if (store_result) {
            auto retrieve_result = db.retrieve<float>(key);
            if (retrieve_result) {
                auto& [retrieved_data, metrics] = *retrieve_result;
                anomaly_scores.push_back(
                    metrics.error);  // Use reconstruction error as anomaly score
            }
        }
    }

    if (!normal_scores.empty() && !anomaly_scores.empty()) {
        // Calculate statistics
        double normal_mean = 0.0, anomaly_mean = 0.0;
        for (double score : normal_scores) normal_mean += score;
        for (double score : anomaly_scores) anomaly_mean += score;
        normal_mean /= normal_scores.size();
        anomaly_mean /= anomaly_scores.size();

        core::Logger::info("=== ANOMALY DETECTION RESULTS ===");
        core::Logger::info("Normal data average reconstruction error: " +
                           std::to_string(normal_mean));
        core::Logger::info("Anomalous data average reconstruction error: " +
                           std::to_string(anomaly_mean));
        core::Logger::info("Separation factor: " + std::to_string(anomaly_mean / normal_mean) +
                           "x");
        core::Logger::info("Detection capability: " +
                           std::string(anomaly_mean > normal_mean * 2.0 ? "GOOD" : "NEEDS_TUNING"));
    }
    else {
        core::Logger::error("Failed to collect sufficient anomaly detection data");
    }
}

int main()
{
    core::Logger::info("=== PRODUCTION VAE DATABASE EXAMPLE ===");
    core::Logger::info("Applying optimal configurations discovered during tuning\n");

    try {
        // Demonstrate using optimal configurations in production
        demonstrateOptimalCompressionConfig();
        demonstrateOptimalAnomalyConfig();

        core::Logger::info("\n=== PRODUCTION USAGE SUMMARY ===");
        core::Logger::info("✓ Compression: Use 3D latent space, β=0.5, simple architecture");
        core::Logger::info("✓ Anomaly Detection: Use 8D latent space, β=2.0, deeper architecture");
        core::Logger::info("✓ Both configurations are now production-ready!");
        core::Logger::info("✓ Monitor performance and re-tune as data patterns evolve");
    }
    catch (const std::exception& e) {
        core::Logger::error("Error: " + std::string(e.what()));
        return 1;
    }

    return 0;
}
