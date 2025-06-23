/**
 * @file vae_quick_tuning_demo.cpp
 * @brief Quick demonstration of VAE auto tuning with Monte Carlo testing
 *
 * This is a streamlined example showing how to use your existing VAEAutoTuner
 * to find optimal configurations for your AI Native Database quickly.
 */

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../include/rad_ml/core/logger.hpp"
#include "../include/rad_ml/research/vae_auto_tuner.hpp"
#include "../include/rad_ml/storage/ai_native_database.hpp"

using namespace rad_ml;

/**
 * @brief Generate simple telemetry data for quick testing
 */
std::vector<std::vector<float>> generateQuickTestData(size_t samples, size_t dimensions,
                                                      uint64_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data;
    data.reserve(samples);

    for (size_t i = 0; i < samples; ++i) {
        std::vector<float> sample;
        sample.reserve(dimensions);

        for (size_t j = 0; j < dimensions; ++j) {
            // Add some structure to the data
            float base_value = std::sin(i * 0.1f + j) + std::cos(i * 0.05f);
            sample.push_back(base_value + dist(gen) * 0.1f);
        }
        data.push_back(sample);
    }

    return data;
}

/**
 * @brief Quick VAE tuning for compression optimization
 */
void quickCompressionTuning()
{
    std::cout << "\n=== QUICK VAE TUNING FOR COMPRESSION ===" << std::endl;

    // Generate test data (12D telemetry-like data)
    auto training_data = generateQuickTestData(500, 12, 12345);
    auto validation_data = generateQuickTestData(100, 12, 67890);

    std::cout << "Generated " << training_data.size() << " training samples" << std::endl;
    std::cout << "Generated " << validation_data.size() << " validation samples" << std::endl;

    // Create tuner
    research::VAEAutoTuner tuner(training_data, validation_data, {},
                                 "quick_compression_results.csv");
    tuner.setSeed(42);

    // Quick search parameters (small for fast demo)
    std::vector<size_t> latent_dims = {3, 4, 6, 8};  // Test compression ratios 4:1, 3:1, 2:1, 1.5:1
    std::vector<float> beta_values = {0.5f, 1.0f};   // Lower beta for better reconstruction
    std::vector<std::vector<size_t>> architectures = {
        {32},      // Simple
        {64, 32},  // Two-layer
        {128, 64}  // Larger
    };
    std::vector<float> learning_rates = {0.001f, 0.01f};

    std::cout << "\nRunning quick grid search..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use fewer Monte Carlo trials for speed
    auto result = tuner.gridSearch(latent_dims, beta_values, architectures, learning_rates, 5,
                                   "compression");  // 5 Monte Carlo trials

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Tuning completed in " << duration.count() << " seconds" << std::endl;

    // Display results
    std::cout << "\n=== BEST COMPRESSION CONFIGURATION ===" << std::endl;
    std::cout << "Latent Dimension: " << result.config.latent_dim << std::endl;
    std::cout << "Compression Ratio: " << (12.0 / result.config.latent_dim) << ":1" << std::endl;
    std::cout << "Beta Parameter: " << result.config.beta << std::endl;
    std::cout << "Learning Rate: " << result.config.learning_rate << std::endl;
    std::cout << "Architecture: ";
    for (size_t i = 0; i < result.config.hidden_dims.size(); ++i) {
        std::cout << result.config.hidden_dims[i];
        if (i < result.config.hidden_dims.size() - 1) std::cout << "-";
    }
    std::cout << std::endl;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Reconstruction Error: " << std::fixed << std::setprecision(4)
              << result.result.reconstruction_error_mean << " ± "
              << result.result.reconstruction_error_stddev << std::endl;
    std::cout << "  Compression Score: " << std::fixed << std::setprecision(3)
              << result.result.compression_score << std::endl;
    std::cout << "  Convergence Rate: " << std::fixed << std::setprecision(1)
              << result.result.convergence_rate * 100 << "%" << std::endl;
}

/**
 * @brief Quick VAE tuning for anomaly detection
 */
void quickAnomalyTuning()
{
    std::cout << "\n=== QUICK VAE TUNING FOR ANOMALY DETECTION ===" << std::endl;

    // Generate test data
    auto training_data = generateQuickTestData(400, 12, 11111);
    auto validation_data = generateQuickTestData(80, 12, 22222);

    // Create tuner
    research::VAEAutoTuner tuner(training_data, validation_data, {}, "quick_anomaly_results.csv");
    tuner.setSeed(123);

    // Parameters optimized for anomaly detection
    std::vector<size_t> latent_dims = {4, 6, 8};          // Moderate compression
    std::vector<float> beta_values = {1.0f, 2.0f, 3.0f};  // Higher beta for better structure
    std::vector<std::vector<size_t>> architectures = {{64}, {64, 32}, {128, 64}};

    std::cout << "\nRunning random search for anomaly detection..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use random search for efficiency
    auto result = tuner.randomSearch(20, 5, "anomaly_detection");  // 20 iterations, 5 MC trials

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Tuning completed in " << duration.count() << " seconds" << std::endl;

    // Display results
    std::cout << "\n=== BEST ANOMALY DETECTION CONFIGURATION ===" << std::endl;
    std::cout << "Latent Dimension: " << result.config.latent_dim << std::endl;
    std::cout << "Beta Parameter: " << result.config.beta << " (higher = better structure)"
              << std::endl;
    std::cout << "Learning Rate: " << result.config.learning_rate << std::endl;
    std::cout << "Architecture: ";
    for (size_t i = 0; i < result.config.hidden_dims.size(); ++i) {
        std::cout << result.config.hidden_dims[i];
        if (i < result.config.hidden_dims.size() - 1) std::cout << "-";
    }
    std::cout << std::endl;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Anomaly Score: " << std::fixed << std::setprecision(3)
              << result.result.anomaly_score << std::endl;
    std::cout << "  KL Divergence: " << std::fixed << std::setprecision(4)
              << result.result.kl_divergence_mean << " ± " << result.result.kl_divergence_stddev
              << std::endl;
    std::cout << "  Convergence Rate: " << std::fixed << std::setprecision(1)
              << result.result.convergence_rate * 100 << "%" << std::endl;
}

/**
 * @brief Test the tuned configuration with the AI Native Database
 */
void testWithDatabase()
{
    std::cout << "\n=== TESTING WITH AI NATIVE DATABASE ===" << std::endl;

    // Generate test data
    auto test_data = generateQuickTestData(50, 12, 99999);

    // Create database with optimized configuration
    // (In practice, you'd use the results from the tuning above)
    storage::AINativeDatabase::Config db_config;
    db_config.db_path = "./quick_test_db";
    db_config.default_latent_dim = 4;  // 3:1 compression
    db_config.vae_hidden_dims = {64, 32};
    db_config.max_reconstruction_error = 0.01f;

    storage::AINativeDatabase db(db_config);
    auto init_result = db.initialize({{"test_data", 12}});

    if (!init_result) {
        std::cout << "Failed to initialize database: " << init_result.error << std::endl;
        return;
    }

    std::cout << "Database initialized successfully" << std::endl;

    // Test storage and retrieval
    std::vector<double> compression_ratios;
    std::vector<double> errors;

    // Configuration: Maximum number of samples to test for quick demo
    constexpr size_t MAX_TEST_SAMPLES = 10;
    const size_t samples_to_test = std::min(test_data.size(), MAX_TEST_SAMPLES);

    for (size_t i = 0; i < samples_to_test; ++i) {
        std::string key = "sample_" + std::to_string(i);

        // Store
        auto store_result = db.store(key, test_data[i], "test_data");
        if (store_result) {
            compression_ratios.push_back(store_result->ratio);

            // Retrieve
            auto retrieve_result = db.retrieve<float>(key);
            if (retrieve_result) {
                errors.push_back(retrieve_result->second.error);
            }
        }
    }

    // Calculate averages
    double avg_compression = 0.0, avg_error = 0.0;
    if (!compression_ratios.empty()) {
        avg_compression =
            std::accumulate(compression_ratios.begin(), compression_ratios.end(), 0.0) /
            compression_ratios.size();
    }
    if (!errors.empty()) {
        avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    }

    std::cout << "Database Test Results (tested " << samples_to_test << " samples):" << std::endl;
    std::cout << "  Average Compression Ratio: " << std::fixed << std::setprecision(2)
              << avg_compression << ":1" << std::endl;
    std::cout << "  Average Reconstruction Error: " << std::fixed << std::setprecision(4)
              << avg_error << std::endl;

    // Get database statistics
    auto stats = db.get_statistics();
    std::cout << "  Total Entries: " << stats.total_entries << std::endl;
    std::cout << "  VAE Models: " << stats.vae_models_count << std::endl;
}

/**
 * @brief Main function
 */
int main()
{
    try {
        std::cout << "=== VAE QUICK TUNING DEMONSTRATION ===" << std::endl;
        std::cout << "This demo shows how to quickly optimize VAE configurations" << std::endl;
        std::cout << "for your AI Native Database using Monte Carlo testing." << std::endl;

        // Run quick compression tuning
        quickCompressionTuning();

        // Run quick anomaly detection tuning
        quickAnomalyTuning();

        // Test with database
        testWithDatabase();

        std::cout << "\n=== DEMO COMPLETED ===" << std::endl;
        std::cout << "Check these files for detailed results:" << std::endl;
        std::cout << "  - quick_compression_results.csv" << std::endl;
        std::cout << "  - quick_anomaly_results.csv" << std::endl;

        std::cout << "\nNext steps:" << std::endl;
        std::cout << "1. Run the full comprehensive tuning: ./vae_monte_carlo_tuning_example"
                  << std::endl;
        std::cout << "2. Use the optimal configurations in your database" << std::endl;
        std::cout << "3. Monitor performance and re-tune as needed" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
