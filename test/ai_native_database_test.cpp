/**
 * @file ai_native_database_test.cpp
 * @brief Comprehensive test for AI-Native Database
 */

#include "rad_ml/storage/ai_native_database.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/core/logger.hpp"

using namespace rad_ml::storage;

/**
 * @brief Generate test data for AI database
 */
std::vector<float> generate_test_data(size_t size, float noise_level = 0.1f)
{
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, noise_level);

    for (size_t i = 0; i < size; ++i) {
        // Generate synthetic sensor-like data with patterns
        float base_value = std::sin(i * 0.1f) * 100.0f + 200.0f;  // Temperature-like data
        data[i] = base_value + noise(gen);
    }

    return data;
}

/**
 * @brief Test basic database operations
 */
bool test_basic_operations()
{
    std::cout << "\n=== Testing Basic AI-Native Database Operations ===" << std::endl;

    try {
        // Create database configuration for datacenter
        AINativeDatabase::Config config;
        config.db_path = "./test_ai_db";
        config.max_db_size = 100 * 1024 * 1024;         // 100MB for testing
        config.default_latent_dim = 16;                 // Smaller for testing
        config.vae_hidden_dims = {64, 32};              // Smaller architecture
        config.enable_background_optimization = false;  // Disable for testing

        // Clean up any existing test database
        std::filesystem::remove_all(config.db_path);

        // Create database
        AINativeDatabase db(std::move(config));

        // Initialize with data schema
        std::unordered_map<std::string, size_t> schema = {
            {"telemetry", 32},   // 32-element telemetry vectors
            {"scientific", 64},  // 64-element scientific data
            {"sensors", 16}      // 16-element sensor data
        };

        auto init_result = db.initialize(schema);
        if (!init_result) {
            std::cout << "✗ Failed to initialize database: " << init_result.error << std::endl;
            return false;
        }
        std::cout << "✓ Database initialized successfully" << std::endl;

        // Test storing data
        std::cout << "Testing data storage..." << std::endl;
        auto telemetry_data = generate_test_data(32);

        auto store_result = db.store("test_telemetry_1", telemetry_data, "telemetry");
        if (!store_result) {
            std::cout << "✗ Failed to store telemetry data: " << store_result.error << std::endl;
            return false;
        }

        auto& metrics = *store_result;
        std::cout << "✓ Stored telemetry data:" << std::endl;
        std::cout << "  - Original size: " << metrics.original_bytes << " bytes" << std::endl;
        std::cout << "  - Compressed size: " << metrics.compressed_bytes << " bytes" << std::endl;
        std::cout << "  - Compression ratio: " << std::fixed << std::setprecision(2)
                  << metrics.ratio << "x" << std::endl;
        std::cout << "  - Reconstruction error: " << std::scientific << std::setprecision(3)
                  << metrics.error << std::endl;
        std::cout << "  - Encode time: " << metrics.encode_time.count() << " ms" << std::endl;

        // Test checking if key exists
        if (!db.contains("test_telemetry_1")) {
            std::cout << "✗ Key existence check failed" << std::endl;
            return false;
        }
        std::cout << "✓ Key existence check passed" << std::endl;

        // Test retrieving data
        std::cout << "Testing data retrieval..." << std::endl;
        auto retrieve_result = db.retrieve<float>("test_telemetry_1");
        if (!retrieve_result) {
            std::cout << "✗ Failed to retrieve data: " << retrieve_result.error << std::endl;
            return false;
        }

        auto& [retrieved_data, retrieve_metrics] = *retrieve_result;
        std::cout << "✓ Retrieved data with " << retrieved_data.size() << " elements" << std::endl;
        std::cout << "  - Decode time: " << retrieve_metrics.decode_time.count() << " ms"
                  << std::endl;

        // Verify data similarity (allowing for compression loss)
        if (retrieved_data.size() != telemetry_data.size()) {
            std::cout << "✗ Retrieved data size mismatch" << std::endl;
            return false;
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < telemetry_data.size(); ++i) {
            float diff = std::abs(telemetry_data[i] - retrieved_data[i]);
            max_diff = std::max(max_diff, diff);
        }

        if (max_diff > 10.0f) {  // Reasonable threshold for lossy compression
            std::cout << "✗ Retrieved data differs too much from original. Max diff: " << max_diff
                      << std::endl;
            return false;
        }
        std::cout << "✓ Data similarity verified (max diff: " << std::fixed << std::setprecision(3)
                  << max_diff << ")" << std::endl;

        // Test storing multiple data types
        std::cout << "Testing multiple data types..." << std::endl;
        auto scientific_data = generate_test_data(64, 0.05f);  // Lower noise for scientific data
        auto sensor_data = generate_test_data(16, 0.2f);       // Higher noise for sensors

        auto sci_store = db.store("experiment_1", scientific_data, "scientific");
        auto sensor_store = db.store("sensor_array_1", sensor_data, "sensors");

        if (!sci_store || !sensor_store) {
            std::cout << "✗ Failed to store multiple data types" << std::endl;
            return false;
        }
        std::cout << "✓ Multiple data types stored successfully" << std::endl;

        // Test getting all keys
        auto keys = db.keys();
        std::cout << "✓ Retrieved " << keys.size() << " keys from database" << std::endl;
        for (const auto& key : keys) {
            std::cout << "  - " << key << std::endl;
        }

        // Test database statistics
        auto stats = db.get_statistics();
        std::cout << "✓ Database Statistics:" << std::endl;
        std::cout << "  - Total entries: " << stats.total_entries << std::endl;
        std::cout << "  - Total original bytes: " << stats.total_original_bytes << std::endl;
        std::cout << "  - Total compressed bytes: " << stats.total_compressed_bytes << std::endl;
        std::cout << "  - Average compression ratio: " << std::fixed << std::setprecision(2)
                  << stats.average_compression_ratio << "x" << std::endl;
        std::cout << "  - Average reconstruction error: " << std::scientific << std::setprecision(3)
                  << stats.average_reconstruction_error << std::endl;
        std::cout << "  - VAE models count: " << stats.vae_models_count << std::endl;

        // Test removal
        auto remove_result = db.remove("sensor_array_1");
        if (!remove_result) {
            std::cout << "✗ Failed to remove data: " << remove_result.error << std::endl;
            return false;
        }

        if (db.contains("sensor_array_1")) {
            std::cout << "✗ Data was not properly removed" << std::endl;
            return false;
        }
        std::cout << "✓ Data removal successful" << std::endl;

        std::cout << "\n✓ All basic operations completed successfully!" << std::endl;

        // Clean up test database
        std::filesystem::remove_all("./test_ai_db");
        std::cout << "✓ Test database cleaned up" << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test async operations
 */
bool test_async_operations()
{
    std::cout << "\n=== Testing Async Operations ===" << std::endl;

    try {
        AINativeDatabase::Config config;
        config.db_path = "./test_async_db";
        config.enable_background_optimization = false;

        std::filesystem::remove_all(config.db_path);

        AINativeDatabase db(std::move(config));

        std::unordered_map<std::string, size_t> schema = {{"async_test", 32}};
        auto init_result = db.initialize(schema);
        if (!init_result) {
            std::cout << "✗ Failed to initialize async test database" << std::endl;
            return false;
        }

        // Test async store
        auto test_data = generate_test_data(32);
        auto store_future = db.store_async("async_data_1", test_data, "async_test");

        // Do some other work while storing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto store_result = store_future.get();
        if (!store_result) {
            std::cout << "✗ Async store failed: " << store_result.error << std::endl;
            return false;
        }
        std::cout << "✓ Async store completed successfully" << std::endl;

        // Test async retrieve
        auto retrieve_future = db.retrieve_async<float>("async_data_1");
        auto retrieve_result = retrieve_future.get();

        if (!retrieve_result) {
            std::cout << "✗ Async retrieve failed: " << retrieve_result.error << std::endl;
            return false;
        }

        auto& [retrieved_data, _] = *retrieve_result;
        if (retrieved_data.size() != test_data.size()) {
            std::cout << "✗ Async retrieved data size mismatch" << std::endl;
            return false;
        }
        std::cout << "✓ Async retrieve completed successfully" << std::endl;

        std::filesystem::remove_all(config.db_path);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Async test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test type safety
 */
bool test_type_safety()
{
    std::cout << "\n=== Testing Type Safety ===" << std::endl;

    try {
        AINativeDatabase::Config config;
        config.db_path = "./test_types_db";

        std::filesystem::remove_all(config.db_path);

        AINativeDatabase db(std::move(config));

        // Test different numeric types
        std::unordered_map<std::string, size_t> schema = {
            {"float_data", 16}, {"double_data", 16}, {"int_data", 16}};

        auto init_result = db.initialize(schema);
        if (!init_result) {
            std::cout << "✗ Failed to initialize type safety test" << std::endl;
            return false;
        }

        // Test float data
        std::vector<float> float_data = {1.1f, 2.2f,  3.3f,  4.4f,  5.5f,  6.6f,  7.7f,  8.8f,
                                         9.9f, 10.1f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f, 16.6f};
        auto float_result = db.store("float_test", float_data, "float_data");
        if (!float_result) {
            std::cout << "✗ Failed to store float data" << std::endl;
            return false;
        }

        // Test double data
        std::vector<double> double_data = {1.1, 2.2,  3.3,  4.4,  5.5,  6.6,  7.7,  8.8,
                                           9.9, 10.1, 11.1, 12.2, 13.3, 14.4, 15.5, 16.6};
        auto double_result = db.store("double_test", double_data, "double_data");
        if (!double_result) {
            std::cout << "✗ Failed to store double data" << std::endl;
            return false;
        }

        // Test int data
        std::vector<int> int_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        auto int_result = db.store("int_test", int_data, "int_data");
        if (!int_result) {
            std::cout << "✗ Failed to store int data" << std::endl;
            return false;
        }

        // Test retrieval with correct types
        auto float_retrieve = db.retrieve<float>("float_test");
        auto double_retrieve = db.retrieve<double>("double_test");
        auto int_retrieve = db.retrieve<int>("int_test");

        if (!float_retrieve || !double_retrieve || !int_retrieve) {
            std::cout << "✗ Failed to retrieve typed data" << std::endl;
            return false;
        }

        std::cout << "✓ Type safety test passed for float, double, and int types" << std::endl;

        std::filesystem::remove_all(config.db_path);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Type safety test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    std::cout << "AI-Native Database Test - Modern C++ Implementation" << std::endl;
    std::cout << "===================================================" << std::endl;

    bool all_tests_passed = true;

    // Run basic operations test
    if (!test_basic_operations()) {
        all_tests_passed = false;
    }

    // Run async operations test
    if (!test_async_operations()) {
        all_tests_passed = false;
    }

    // Run type safety test
    if (!test_type_safety()) {
        all_tests_passed = false;
    }

    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_tests_passed) {
        std::cout << "🎉 All tests passed! AI-Native Database is working correctly!" << std::endl;
    }
    else {
        std::cout << "❌ Some tests failed. Please check the implementation." << std::endl;
        return 1;
    }

    return 0;
}
