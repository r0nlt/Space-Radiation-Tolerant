/**
 * @file vae_production_example.cpp
 * @brief Production-ready VAE example demonstrating advanced features
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/rad_ml/core/logger.hpp"
#include "../include/rad_ml/research/variational_autoencoder.hpp"

using namespace rad_ml::research;
using namespace rad_ml;

/**
 * @brief Generate realistic spacecraft telemetry data for production training
 */
std::vector<std::vector<float>> generateProductionData(size_t num_samples, size_t data_dim)
{
    std::vector<std::vector<float>> data;
    data.reserve(num_samples);

    std::random_device rd;
    std::mt19937 gen(rd());

    // More sophisticated data generation for production
    std::normal_distribution<float> altitude_dist(400.0f, 50.0f);   // km
    std::normal_distribution<float> velocity_dist(7.8f, 0.1f);      // km/s
    std::normal_distribution<float> power_dist(28.0f, 2.0f);        // V
    std::normal_distribution<float> temp_dist(20.0f, 15.0f);        // C
    std::normal_distribution<float> pressure_dist(1013.0f, 50.0f);  // mbar
    std::normal_distribution<float> solar_dist(1360.0f, 100.0f);    // W/m²

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> sample(data_dim);

        // Physics-based correlations
        float altitude = std::max(300.0f, altitude_dist(gen));
        float velocity = 7.8f * std::sqrt(398600.0f / (6371.0f + altitude));  // Orbital mechanics
        float power = power_dist(gen);
        float temperature = temp_dist(gen);

        // Add realistic noise and correlations
        sample[0] = altitude;
        sample[1] = velocity;
        sample[2] = power;
        sample[3] = temperature + (power - 28.0f) * 0.5f;  // Power affects temperature
        sample[4] = pressure_dist(gen);
        sample[5] = solar_dist(gen) * (1.0f + 0.1f * std::sin(i * 0.1f));  // Solar cycle

        // Fill remaining dimensions with correlated noise
        for (size_t j = 6; j < data_dim; ++j) {
            sample[j] = 0.1f * (sample[j - 1] + sample[j - 2]) + 0.01f * gen() / gen.max();
        }

        data.push_back(sample);
    }

    return data;
}

/**
 * @brief Demonstrate production VAE training and evaluation
 */
void demonstrateProductionVAE()
{
    core::Logger::info("=== Production VAE Demonstration ===");

    // Create production configuration
    VAEConfig config;
    config.epochs = 50;
    config.batch_size = 16;
    config.learning_rate = 0.001f;
    config.beta = 1.0f;
    config.optimizer = OptimizerType::ADAM;
    config.adam_beta1 = 0.9f;
    config.adam_beta2 = 0.999f;
    config.adam_epsilon = 1e-8f;
    config.early_stopping_patience = 10;
    config.early_stopping_min_delta = 1e-4f;
    config.validation_split = 0.2f;
    config.use_interpolation = true;
    config.interpolation_weight = 0.1f;

    // Create VAE
    core::Logger::info("Creating production VAE...");
    VariationalAutoencoder<float> vae(12, 8, {64, 32, 16}, neural::ProtectionLevel::NONE, config);

    // Generate training data
    core::Logger::info("Generating realistic spacecraft telemetry data...");
    auto training_data = generateProductionData(1000, 12);

    // Train the VAE
    core::Logger::info("Starting production training...");
    auto start_time = std::chrono::high_resolution_clock::now();

    TrainingMetrics metrics = vae.trainProduction(training_data);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // Display training results
    core::Logger::info("Training completed in " + std::to_string(duration.count()) + " seconds");
    core::Logger::info("Best validation loss: " + std::to_string(metrics.best_val_loss));
    core::Logger::info("Best epoch: " + std::to_string(metrics.best_epoch));
    core::Logger::info("Final training loss: " + std::to_string(metrics.train_losses.back()));
    core::Logger::info("Final validation loss: " + std::to_string(metrics.val_losses.back()));

    // Comprehensive evaluation
    core::Logger::info("=== Comprehensive Evaluation ===");
    auto eval_data = generateProductionData(200, 12);
    auto eval_metrics = vae.evaluateComprehensive(eval_data);

    for (const auto& [metric, value] : eval_metrics) {
        core::Logger::info(metric + ": " + std::to_string(value));
    }

    // Test generative capabilities
    core::Logger::info("=== Generative Testing ===");
    auto generated_samples = vae.generate(10);
    core::Logger::info("Generated " + std::to_string(generated_samples.size()) + " samples");

    // Analyze generated samples
    if (!generated_samples.empty()) {
        std::vector<float> means(12, 0.0f);
        for (const auto& sample : generated_samples) {
            for (size_t i = 0; i < sample.size(); ++i) {
                means[i] += sample[i];
            }
        }
        for (auto& mean : means) {
            mean /= generated_samples.size();
        }

        core::Logger::info("Generated sample means:");
        for (size_t i = 0; i < means.size(); ++i) {
            core::Logger::info("  Dimension " + std::to_string(i) + ": " +
                               std::to_string(means[i]));
        }
    }

    // Test radiation tolerance
    core::Logger::info("=== Radiation Tolerance Testing ===");
    std::vector<double> radiation_levels = {0.0, 0.1, 0.5, 1.0, 2.0};
    std::vector<float> sample = training_data[0];

    for (double rad_level : radiation_levels) {
        auto reconstruction = vae.forward(sample, rad_level);
        float error = 0.0f;
        for (size_t i = 0; i < sample.size(); ++i) {
            float diff = sample[i] - reconstruction[i];
            error += diff * diff;
        }
        error = std::sqrt(error / sample.size());

        core::Logger::info("Radiation level " + std::to_string(rad_level) +
                           " - Reconstruction error: " + std::to_string(error));
    }

    // Test interpolation
    core::Logger::info("=== Latent Space Interpolation ===");
    if (training_data.size() >= 2) {
        auto [mean1, log_var1] = vae.encode(training_data[0]);
        auto [mean2, log_var2] = vae.encode(training_data[1]);

        auto latent1 = vae.sample(mean1, log_var1);
        auto latent2 = vae.sample(mean2, log_var2);

        std::vector<float> alphas = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        for (float alpha : alphas) {
            auto interpolated = vae.interpolate(latent1, latent2, alpha);
            core::Logger::info("Interpolation α=" + std::to_string(alpha) +
                               " - First element: " + std::to_string(interpolated[0]));
        }
    }

    // Model persistence test
    core::Logger::info("=== Model Persistence Testing ===");
    std::string model_path = "production_vae_model.bin";

    if (vae.saveModel(model_path)) {
        core::Logger::info("Model saved successfully to " + model_path);

        // Create new VAE and load
        VariationalAutoencoder<float> loaded_vae(12, 8, {64, 32, 16}, neural::ProtectionLevel::NONE,
                                                 config);
        if (loaded_vae.loadModel(model_path)) {
            core::Logger::info("Model loaded successfully");

            // Compare outputs
            auto original_output = vae.forward(sample);
            auto loaded_output = loaded_vae.forward(sample);

            float diff = 0.0f;
            for (size_t i = 0; i < original_output.size(); ++i) {
                float d = original_output[i] - loaded_output[i];
                diff += d * d;
            }
            diff = std::sqrt(diff / original_output.size());

            core::Logger::info("Output difference after reload: " + std::to_string(diff));
        }
        else {
            core::Logger::error("Failed to load model");
        }
    }
    else {
        core::Logger::error("Failed to save model");
    }

    core::Logger::info("=== Production VAE demonstration completed successfully! ===");
}

int main()
{
    // Initialize logging
    core::Logger::init(core::LogLevel::INFO);
    core::Logger::info("Starting production VAE example");

    try {
        demonstrateProductionVAE();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
