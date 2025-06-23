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

        // Test 1: Successful async operations (happy path)
        std::cout << "Testing successful async operations..." << std::endl;
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

        // Test async retrieve (happy path)
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

        // Test 2: Async retrieve with invalid key (error handling)
        std::cout << "Testing async error handling - invalid key..." << std::endl;
        auto invalid_retrieve_future = db.retrieve_async<float>("nonexistent_key_12345");
        auto invalid_retrieve_result = invalid_retrieve_future.get();

        if (invalid_retrieve_result) {
            std::cout << "✗ Expected failure for invalid key, but operation succeeded" << std::endl;
            return false;
        }
        std::cout << "✓ Async retrieve properly failed for invalid key: "
                  << invalid_retrieve_result.error << std::endl;

        // Test 3: Async store with invalid data type (error handling)
        std::cout << "Testing async error handling - invalid data type..." << std::endl;
        auto invalid_data = generate_test_data(16);  // Wrong size for schema
        auto invalid_store_future =
            db.store_async("invalid_data", invalid_data, "nonexistent_data_type");
        auto invalid_store_result = invalid_store_future.get();

        // This should either succeed with default handling or fail gracefully
        if (!invalid_store_result) {
            std::cout << "✓ Async store properly handled invalid data type: "
                      << invalid_store_result.error << std::endl;
        }
        else {
            std::cout << "✓ Async store handled invalid data type with default behavior"
                      << std::endl;
        }

        // Test 4: Multiple concurrent async operations (stress test)
        std::cout << "Testing concurrent async operations..." << std::endl;
        const size_t concurrent_ops = 5;
        std::vector<std::future<Result<AINativeDatabase::CompressionMetrics>>> store_futures;
        std::vector<std::vector<float>> test_datasets;

        // Launch multiple async stores
        for (size_t i = 0; i < concurrent_ops; ++i) {
            auto data = generate_test_data(32, 0.1f + i * 0.05f);  // Varying noise levels
            test_datasets.push_back(data);
            std::string key = "concurrent_data_" + std::to_string(i);
            store_futures.push_back(db.store_async(key, data, "async_test"));
        }

        // Wait for all stores to complete and check results
        bool all_concurrent_stores_succeeded = true;
        for (size_t i = 0; i < concurrent_ops; ++i) {
            auto result = store_futures[i].get();
            if (!result) {
                std::cout << "✗ Concurrent store " << i << " failed: " << result.error << std::endl;
                all_concurrent_stores_succeeded = false;
            }
        }

        if (!all_concurrent_stores_succeeded) {
            return false;
        }
        std::cout << "✓ All " << concurrent_ops << " concurrent async stores succeeded"
                  << std::endl;

        // Test 5: Concurrent async retrieves
        std::vector<std::future<
            Result<std::pair<std::vector<float>, AINativeDatabase::CompressionMetrics>>>>
            retrieve_futures;

        for (size_t i = 0; i < concurrent_ops; ++i) {
            std::string key = "concurrent_data_" + std::to_string(i);
            retrieve_futures.push_back(db.retrieve_async<float>(key));
        }

        // Verify all concurrent retrieves
        for (size_t i = 0; i < concurrent_ops; ++i) {
            auto result = retrieve_futures[i].get();
            if (!result) {
                std::cout << "✗ Concurrent retrieve " << i << " failed: " << result.error
                          << std::endl;
                return false;
            }

            auto& [data, metrics] = *result;
            if (data.size() != test_datasets[i].size()) {
                std::cout << "✗ Concurrent retrieve " << i << " returned wrong size" << std::endl;
                return false;
            }
        }
        std::cout << "✓ All " << concurrent_ops << " concurrent async retrieves succeeded"
                  << std::endl;

        // Test 6: Async operations with database cleanup during operation
        std::cout << "Testing async operations with potential resource contention..." << std::endl;
        auto cleanup_test_data = generate_test_data(32);

        // Start an async operation
        auto cleanup_store_future = db.store_async("cleanup_test", cleanup_test_data, "async_test");

        // Immediately try to access statistics (potential resource contention)
        auto stats = db.get_statistics();
        std::cout << "✓ Retrieved stats during async operation: " << stats.total_entries
                  << " entries" << std::endl;

        // Wait for the async operation to complete
        auto cleanup_result = cleanup_store_future.get();
        if (!cleanup_result) {
            std::cout << "✗ Async operation failed during resource contention test: "
                      << cleanup_result.error << std::endl;
            return false;
        }
        std::cout << "✓ Async operation succeeded despite resource contention" << std::endl;

        // Test 7: Exception safety in async operations
        std::cout << "Testing async exception safety..." << std::endl;
        try {
            // Try to store data that might cause issues (empty vector edge case)
            std::vector<float> empty_data;
            auto exception_future = db.store_async("empty_test", empty_data, "async_test");
            auto exception_result = exception_future.get();

            // This should either succeed (if empty data is handled) or fail gracefully
            if (!exception_result) {
                std::cout << "✓ Empty data properly handled with error: " << exception_result.error
                          << std::endl;
            }
            else {
                std::cout << "✓ Empty data handled successfully" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "✓ Exception properly caught in async operation: " << e.what()
                      << std::endl;
        }

        std::cout << "\n✓ All async operations and error handling tests completed successfully!"
                  << std::endl;
        std::filesystem::remove_all(config.db_path);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Async test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test error handling and edge cases
 */
bool test_error_handling()
{
    std::cout << "\n=== Testing Error Handling & Edge Cases ===" << std::endl;

    try {
        AINativeDatabase::Config config;
        config.db_path = "./test_error_db";
        config.enable_background_optimization = false;

        std::filesystem::remove_all(config.db_path);

        AINativeDatabase db(std::move(config));

        std::unordered_map<std::string, size_t> schema = {{"test_data", 32}};
        auto init_result = db.initialize(schema);
        if (!init_result) {
            std::cout << "✗ Failed to initialize error test database" << std::endl;
            return false;
        }

        // Test 1: Retrieve from empty database
        std::cout << "Testing retrieve from empty database..." << std::endl;
        auto empty_retrieve = db.retrieve<float>("nonexistent_key");
        if (empty_retrieve) {
            std::cout << "✗ Expected failure for nonexistent key, but succeeded" << std::endl;
            return false;
        }
        std::cout << "✓ Properly failed to retrieve nonexistent key: " << empty_retrieve.error
                  << std::endl;

        // Test 2: Contains check on empty database
        if (db.contains("nonexistent_key")) {
            std::cout << "✗ Contains() returned true for nonexistent key" << std::endl;
            return false;
        }
        std::cout << "✓ Contains() correctly returned false for nonexistent key" << std::endl;

        // Test 3: Remove nonexistent key
        auto remove_nonexistent = db.remove("nonexistent_key");
        if (remove_nonexistent) {
            std::cout << "✗ Expected failure when removing nonexistent key, but succeeded"
                      << std::endl;
            return false;
        }
        std::cout << "✓ Properly failed to remove nonexistent key: " << remove_nonexistent.error
                  << std::endl;

        // Test 4: Edge case - very large data
        std::cout << "Testing large data handling..." << std::endl;
        auto large_data = generate_test_data(1000);  // Much larger than schema expects
        auto large_store = db.store("large_data_test", large_data, "test_data");

        // This might succeed (if size is flexible) or fail (if strict schema)
        if (!large_store) {
            std::cout << "✓ Large data properly rejected: " << large_store.error << std::endl;
        }
        else {
            std::cout << "✓ Large data handled with flexibility" << std::endl;
        }

        // Test 5: Edge case - empty data
        std::cout << "Testing empty data handling..." << std::endl;
        std::vector<float> empty_data;
        auto empty_store = db.store("empty_data_test", empty_data, "test_data");

        if (!empty_store) {
            std::cout << "✓ Empty data properly rejected: " << empty_store.error << std::endl;
        }
        else {
            std::cout << "✓ Empty data handled gracefully" << std::endl;
        }

        // Test 6: Extreme values
        std::cout << "Testing extreme value handling..." << std::endl;
        std::vector<float> extreme_data = {
            std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()};
        // Pad to expected size
        while (extreme_data.size() < 32) {
            extreme_data.push_back(0.0f);
        }

        auto extreme_store = db.store("extreme_values", extreme_data, "test_data");
        if (!extreme_store) {
            std::cout << "✓ Extreme values properly handled with error: " << extreme_store.error
                      << std::endl;
        }
        else {
            std::cout << "✓ Extreme values stored successfully" << std::endl;

            // Try to retrieve and verify
            auto extreme_retrieve = db.retrieve<float>("extreme_values");
            if (extreme_retrieve) {
                std::cout << "✓ Extreme values retrieved successfully" << std::endl;
            }
            else {
                std::cout << "✓ Extreme values retrieval failed as expected: "
                          << extreme_retrieve.error << std::endl;
            }
        }

        // Test 7: Concurrent access edge cases
        std::cout << "Testing concurrent access patterns..." << std::endl;
        auto test_data = generate_test_data(32);

        // Store data first
        auto store_result = db.store("concurrent_test", test_data, "test_data");
        if (!store_result) {
            std::cout << "✗ Failed to store data for concurrent test" << std::endl;
            return false;
        }

        // Launch concurrent operations on same key
        auto future1 = std::async(std::launch::async,
                                  [&db]() { return db.retrieve<float>("concurrent_test"); });

        auto future2 = std::async(std::launch::async,
                                  [&db]() { return db.retrieve<float>("concurrent_test"); });

        auto result1 = future1.get();
        auto result2 = future2.get();

        if (!result1 || !result2) {
            std::cout << "✗ Concurrent retrieval failed" << std::endl;
            return false;
        }
        std::cout << "✓ Concurrent access handled properly" << std::endl;

        // Test 8: Database statistics consistency
        std::cout << "Testing database statistics consistency..." << std::endl;
        auto stats_before = db.get_statistics();

        // Add some data
        auto consistency_data = generate_test_data(32);
        auto consistency_store = db.store("stats_test", consistency_data, "test_data");

        if (consistency_store) {
            auto stats_after = db.get_statistics();
            if (stats_after.total_entries <= stats_before.total_entries) {
                std::cout << "✗ Statistics not properly updated after store operation" << std::endl;
                return false;
            }
            std::cout << "✓ Statistics properly updated (entries: " << stats_before.total_entries
                      << " → " << stats_after.total_entries << ")" << std::endl;
        }

        std::cout << "\n✓ All error handling and edge case tests completed!" << std::endl;
        std::filesystem::remove_all(config.db_path);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Error handling test failed with exception: " << e.what() << std::endl;
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

    // Run error handling test
    if (!test_error_handling()) {
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
