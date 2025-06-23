/**
 * @file vae_quick_reconstruction_test.cpp
 * @brief Quick test to compare different VAE configurations for reconstruction quality
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/research/vae_optimal_configs.hpp"

using namespace rad_ml;

// Generate realistic test data
std::vector<std::vector<float>> generateTestData(size_t num_samples, unsigned int seed = 42)
{
    std::vector<std::vector<float>> data;
    std::mt19937 gen(seed);

    std::normal_distribution<float> temp_dist(25.0f, 5.0f);
    std::normal_distribution<float> voltage_dist(12.0f, 0.5f);
    std::normal_distribution<float> current_dist(2.5f, 0.3f);
    std::normal_distribution<float> pressure_dist(101.3f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> sample(12);

        float temp1 = temp_dist(gen);
        float temp2 = temp1 + 3.0f + noise_dist(gen);
        float voltage1 = voltage_dist(gen);
        float voltage2 = voltage1 * 0.95f + noise_dist(gen);
        float current1 = current_dist(gen);
        float current2 = current1 * 1.05f + noise_dist(gen);
        float pressure1 = pressure_dist(gen);
        float pressure2 = pressure1 + 0.5f + noise_dist(gen);

        sample[0] = temp1;
        sample[1] = temp2;
        sample[2] = voltage1;
        sample[3] = voltage2;
        sample[4] = current1;
        sample[5] = current2;
        sample[6] = pressure1;
        sample[7] = pressure2;
        sample[8] = temp1 * 0.1f + voltage1 + noise_dist(gen);
        sample[9] = current1 * voltage1 + noise_dist(gen);
        sample[10] = (temp1 + temp2) / 2.0f + noise_dist(gen);
        sample[11] = pressure1 - pressure2 + noise_dist(gen);

        data.push_back(sample);
    }

    return data;
}

// Test configuration performance
struct TestResults {
    std::string config_name;
    double reconstruction_error;
    double compression_ratio;
    double training_time_ms;
    size_t latent_dim;
    float beta;
    size_t epochs;
    std::vector<size_t> architecture;

    void print() const
    {
        core::Logger::info("=== " + config_name + " ===");
        core::Logger::info("  Reconstruction Error: " + std::to_string(reconstruction_error));
        core::Logger::info("  Compression Ratio: " + std::to_string(compression_ratio) + ":1");
        core::Logger::info("  Training Time: " + std::to_string(training_time_ms) + " ms");
        core::Logger::info("  Latent Dim: " + std::to_string(latent_dim));
        core::Logger::info("  Beta: " + std::to_string(beta));
        core::Logger::info("  Epochs: " + std::to_string(epochs));
        core::Logger::info("  Architecture: [" + architectureToString() + "]");

        std::string quality_rating;
        if (reconstruction_error < 1.0)
            quality_rating = "EXCELLENT";
        else if (reconstruction_error < 2.0)
            quality_rating = "GOOD";
        else if (reconstruction_error < 5.0)
            quality_rating = "FAIR";
        else
            quality_rating = "POOR";

        core::Logger::info("  Quality Rating: " + quality_rating);
        core::Logger::info("");
    }

   private:
    std::string architectureToString() const
    {
        std::string result;
        for (size_t i = 0; i < architecture.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(architecture[i]);
        }
        return result;
    }
};

template <typename VAEType>
TestResults testConfiguration(const std::string& name, VAEType& vae,
                              const std::vector<std::vector<float>>& train_data,
                              const std::vector<std::vector<float>>& test_data,
                              const research::VAEConfig& config,
                              const std::vector<size_t>& architecture)
{
    TestResults results;
    results.config_name = name;
    results.latent_dim = config.latent_dim;
    results.beta = config.beta;
    results.epochs = config.epochs;
    results.architecture = architecture;
    results.compression_ratio = static_cast<double>(12) / config.latent_dim;

    // Train and measure time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use reduced training for quick testing
    size_t quick_epochs = std::min(static_cast<size_t>(50), config.epochs);
    vae.train(train_data, quick_epochs, config.batch_size, config.learning_rate);

    auto end_time = std::chrono::high_resolution_clock::now();
    results.training_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Test reconstruction quality
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

    results.reconstruction_error = total_error / test_data.size();

    return results;
}

int main()
{
    core::Logger::info("=== QUICK RECONSTRUCTION QUALITY TEST ===");
    core::Logger::info("Testing different configurations for better reconstruction...\n");

    try {
        // Generate test data
        auto train_data = generateTestData(300, 42);
        auto test_data = generateTestData(50, 123);

        std::vector<TestResults> all_results;

        // Test 1: Original optimal compression config
        core::Logger::info("Testing Original Optimal Compression Config...");
        auto original_vae = research::OptimalConfigs::createCompressionVAE<float>(12);
        auto original_config = research::OptimalConfigs::getCompressionConfig();
        auto original_arch = research::OptimalConfigs::getCompressionArchitecture();
        auto result1 = testConfiguration("Original Optimal (3D, β=0.5)", original_vae, train_data,
                                         test_data, original_config, original_arch);
        all_results.push_back(result1);

        // Test 2: High-quality compression config
        core::Logger::info("Testing High-Quality Compression Config...");
        auto hq_vae =
            research::OptimalConfigs::ImprovedConfigs::createHighQualityCompressionVAE<float>(12);
        auto hq_config =
            research::OptimalConfigs::ImprovedConfigs::getHighQualityCompressionConfig();
        auto hq_arch =
            research::OptimalConfigs::ImprovedConfigs::getHighQualityCompressionArchitecture();
        auto result2 = testConfiguration("High-Quality (6D, β=0.1)", hq_vae, train_data, test_data,
                                         hq_config, hq_arch);
        all_results.push_back(result2);

        // Test 3: Minimal regularization config
        core::Logger::info("Testing Minimal Regularization Config...");
        auto min_reg_vae =
            research::OptimalConfigs::ImprovedConfigs::createMinimalRegularizationVAE<float>(12);
        auto min_reg_config =
            research::OptimalConfigs::ImprovedConfigs::getMinimalRegularizationConfig();
        auto min_reg_arch =
            research::OptimalConfigs::ImprovedConfigs::getMinimalRegularizationArchitecture();
        auto result3 = testConfiguration("Minimal Regularization (4D, β=0.01)", min_reg_vae,
                                         train_data, test_data, min_reg_config, min_reg_arch);
        all_results.push_back(result3);

        // Test 4: Balanced quality config
        core::Logger::info("Testing Balanced Quality Config...");
        auto balanced_vae =
            research::OptimalConfigs::ImprovedConfigs::createBalancedQualityVAE<float>(12);
        auto balanced_config =
            research::OptimalConfigs::ImprovedConfigs::getBalancedQualityConfig();
        auto balanced_arch =
            research::OptimalConfigs::ImprovedConfigs::getBalancedQualityArchitecture();
        auto result4 = testConfiguration("Balanced Quality (4D, β=0.2)", balanced_vae, train_data,
                                         test_data, balanced_config, balanced_arch);
        all_results.push_back(result4);

        // Print all results
        core::Logger::info("=== COMPARISON RESULTS ===\n");
        for (const auto& result : all_results) {
            result.print();
        }

        // Find best reconstruction
        auto best_reconstruction = std::min_element(
            all_results.begin(), all_results.end(), [](const TestResults& a, const TestResults& b) {
                return a.reconstruction_error < b.reconstruction_error;
            });

        // Find best compression
        auto best_compression = std::max_element(
            all_results.begin(), all_results.end(), [](const TestResults& a, const TestResults& b) {
                return a.compression_ratio < b.compression_ratio;
            });

        // Find best balance (reconstruction * compression trade-off)
        auto best_balance = std::min_element(
            all_results.begin(), all_results.end(), [](const TestResults& a, const TestResults& b) {
                double score_a = a.reconstruction_error / a.compression_ratio;
                double score_b = b.reconstruction_error / b.compression_ratio;
                return score_a < score_b;
            });

        core::Logger::info("=== RECOMMENDATIONS ===");
        core::Logger::info("🏆 Best Reconstruction: " + best_reconstruction->config_name +
                           " (Error: " + std::to_string(best_reconstruction->reconstruction_error) +
                           ")");
        core::Logger::info("🗜️  Best Compression: " + best_compression->config_name + " (Ratio: " +
                           std::to_string(best_compression->compression_ratio) + ":1)");
        core::Logger::info(
            "⚖️  Best Balance: " + best_balance->config_name + " (Score: " +
            std::to_string(best_balance->reconstruction_error / best_balance->compression_ratio) +
            ")");

        // Improvement analysis
        double improvement =
            (all_results[0].reconstruction_error - best_reconstruction->reconstruction_error) /
            all_results[0].reconstruction_error * 100.0;

        core::Logger::info("\n=== IMPROVEMENT ANALYSIS ===");
        core::Logger::info("Original reconstruction error: " +
                           std::to_string(all_results[0].reconstruction_error));
        core::Logger::info("Best reconstruction error: " +
                           std::to_string(best_reconstruction->reconstruction_error));
        core::Logger::info("Improvement: " + std::to_string(improvement) + "%");

        if (improvement > 50) {
            core::Logger::info("✅ SIGNIFICANT IMPROVEMENT! Use the " +
                               best_reconstruction->config_name + " config.");
        }
        else if (improvement > 20) {
            core::Logger::info("✅ Good improvement. Consider using " +
                               best_reconstruction->config_name + " config.");
        }
        else {
            core::Logger::info("⚠️  Modest improvement. Original config may be sufficient.");
        }

        // Actionable tuning suggestions
        core::Logger::info("\n=== TUNING INSIGHTS ===");

        if (best_reconstruction->beta < 0.1) {
            core::Logger::info(
                "💡 Lower beta values (β < 0.1) significantly improve reconstruction");
        }

        if (best_reconstruction->latent_dim > 3) {
            core::Logger::info(
                "💡 Higher latent dimensions improve reconstruction at cost of compression");
        }

        bool deep_arch_wins = false;
        for (const auto& result : all_results) {
            if (result.architecture.size() > 1 &&
                result.reconstruction_error < all_results[0].reconstruction_error) {
                deep_arch_wins = true;
                break;
            }
        }

        if (deep_arch_wins) {
            core::Logger::info(
                "💡 Deeper architectures (multiple layers) improve reconstruction quality");
        }

        core::Logger::info("\n=== NEXT STEPS ===");
        core::Logger::info("1. Use '" + best_reconstruction->config_name +
                           "' for production if reconstruction quality is priority");
        core::Logger::info("2. Use '" + best_balance->config_name +
                           "' for balanced compression/quality");
        core::Logger::info(
            "3. Consider data preprocessing (normalization) for further improvements");
        core::Logger::info("4. Test with more training epochs for final tuning");
    }
    catch (const std::exception& e) {
        core::Logger::error("Error: " + std::string(e.what()));
        return 1;
    }

    return 0;
}
