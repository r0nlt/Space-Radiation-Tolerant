/**
 * @brief Quick test to compare VAE configurations for reconstruction quality
 */

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/research/vae_optimal_configs.hpp"

using namespace rad_ml;

std::vector<std::vector<float>> generateTestData(size_t num_samples)
{
    std::vector<std::vector<float>> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> sample(12);
        for (size_t j = 0; j < 12; ++j) {
            sample[j] = dist(gen);
        }
        data.push_back(sample);
    }
    return data;
}

template <typename VAEType>
double testReconstructionError(VAEType& vae, const std::vector<std::vector<float>>& test_data)
{
    double total_error = 0.0;
    for (const auto& sample : test_data) {
        auto [mean, log_var] = vae.encode(sample);
        auto latent = vae.sample(mean, log_var);
        auto reconstructed = vae.decode(latent);

        double sample_error = 0.0;
        for (size_t i = 0; i < sample.size(); ++i) {
            sample_error += std::abs(sample[i] - reconstructed[i]);
        }
        total_error += sample_error / sample.size();
    }
    return total_error / test_data.size();
}

int main()
{
    core::Logger::info("=== QUICK RECONSTRUCTION TEST ===");

    try {
        auto train_data = generateTestData(200);
        auto test_data = generateTestData(50);

        // Test 1: Original config
        core::Logger::info("Testing Original Config (3D, β=0.5)...");
        auto original_vae = research::OptimalConfigs::createCompressionVAE<float>(12);
        original_vae.train(train_data, 30, 32, 0.001f);
        double original_error = testReconstructionError(original_vae, test_data);
        core::Logger::info("Original error: " + std::to_string(original_error));

        // Test 2: High-quality config
        core::Logger::info("Testing High-Quality Config (6D, β=0.1)...");
        auto hq_vae =
            research::OptimalConfigs::ImprovedConfigs::createHighQualityCompressionVAE<float>(12);
        hq_vae.train(train_data, 30, 16, 0.0005f);
        double hq_error = testReconstructionError(hq_vae, test_data);
        core::Logger::info("High-quality error: " + std::to_string(hq_error));

        // Test 3: Minimal regularization
        core::Logger::info("Testing Minimal Regularization (4D, β=0.01)...");
        auto min_reg_vae =
            research::OptimalConfigs::ImprovedConfigs::createMinimalRegularizationVAE<float>(12);
        min_reg_vae.train(train_data, 30, 32, 0.001f);
        double min_reg_error = testReconstructionError(min_reg_vae, test_data);
        core::Logger::info("Minimal reg error: " + std::to_string(min_reg_error));

        // Find best
        std::vector<std::pair<std::string, double>> results = {
            {"Original (3D, β=0.5)", original_error},
            {"High-Quality (6D, β=0.1)", hq_error},
            {"Minimal Reg (4D, β=0.01)", min_reg_error}};

        auto best =
            std::min_element(results.begin(), results.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });

        core::Logger::info("\n=== RESULTS ===");
        for (const auto& [name, error] : results) {
            core::Logger::info(name + ": " + std::to_string(error));
        }

        core::Logger::info("\n🏆 Best: " + best->first);
        double improvement = (original_error - best->second) / original_error * 100.0;
        core::Logger::info("Improvement: " + std::to_string(improvement) + "%");

        if (improvement > 20) {
            core::Logger::info("✅ Significant improvement! Use this config for production.");
        }
    }
    catch (const std::exception& e) {
        core::Logger::error("Error: " + std::string(e.what()));
        return 1;
    }

    return 0;
}
