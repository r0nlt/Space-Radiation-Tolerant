/**
 * @file simple_ai_database_test.cpp
 * @brief Simple test for AI-Native Database
 */

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rad_ml/storage/ai_native_database_simple.hpp"

using namespace rad_ml::storage;

bool test_simple_operations()
{
    std::cout << "\n=== Testing Simple AI-Native Database ===" << std::endl;

    try {
        // Create database
        SimpleAINativeDatabase::Config config;
        config.db_path = "./test_simple_db";
        config.max_db_size = 10 * 1024 * 1024;  // 10MB for testing

        // Clean up any existing database
        std::filesystem::remove_all(config.db_path);

        SimpleAINativeDatabase db(std::move(config));

        // Initialize database
        auto init_result = db.initialize();
        if (!init_result) {
            std::cout << "✗ Failed to initialize database: " << init_result.error << std::endl;
            return false;
        }
        std::cout << "✓ Database initialized successfully" << std::endl;

        // Test storing float data
        std::vector<float> test_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
        auto store_result = db.store("test_float", test_data);
        if (!store_result) {
            std::cout << "✗ Failed to store data: " << store_result.error << std::endl;
            return false;
        }

        auto& metrics = *store_result;
        std::cout << "✓ Stored float data:" << std::endl;
        std::cout << "  - Original bytes: " << metrics.original_bytes << std::endl;
        std::cout << "  - Compressed bytes: " << metrics.compressed_bytes << std::endl;
        std::cout << "  - Compression ratio: " << std::fixed << std::setprecision(2)
                  << metrics.ratio << "x" << std::endl;
        std::cout << "  - Encode time: " << metrics.encode_time.count() << " ms" << std::endl;

        // Test checking if key exists
        if (!db.contains("test_float")) {
            std::cout << "✗ Key existence check failed" << std::endl;
            return false;
        }
        std::cout << "✓ Key existence check passed" << std::endl;

        // Test retrieving data
        auto retrieve_result = db.retrieve<float>("test_float");
        if (!retrieve_result) {
            std::cout << "✗ Failed to retrieve data: " << retrieve_result.error << std::endl;
            return false;
        }

        auto& [retrieved_data, retrieve_metrics] = *retrieve_result;
        std::cout << "✓ Retrieved data with " << retrieved_data.size() << " elements" << std::endl;
        std::cout << "  - Decode time: " << retrieve_metrics.decode_time.count() << " ms"
                  << std::endl;

        // Verify data integrity
        if (retrieved_data.size() != test_data.size()) {
            std::cout << "✗ Data size mismatch" << std::endl;
            return false;
        }

        for (size_t i = 0; i < test_data.size(); ++i) {
            if (std::abs(test_data[i] - retrieved_data[i]) > 0.001f) {
                std::cout << "✗ Data integrity check failed at index " << i << std::endl;
                return false;
            }
        }
        std::cout << "✓ Data integrity verified" << std::endl;

        // Test different data types
        std::vector<int> int_data = {10, 20, 30, 40, 50};
        auto int_store = db.store("test_int", int_data);
        if (!int_store) {
            std::cout << "✗ Failed to store int data" << std::endl;
            return false;
        }

        std::vector<double> double_data = {1.1, 2.2, 3.3, 4.4, 5.5};
        auto double_store = db.store("test_double", double_data);
        if (!double_store) {
            std::cout << "✗ Failed to store double data" << std::endl;
            return false;
        }
        std::cout << "✓ Multiple data types stored successfully" << std::endl;

        // Test getting all keys
        auto keys = db.keys();
        std::cout << "✓ Retrieved " << keys.size() << " keys from database:" << std::endl;
        for (const auto& key : keys) {
            std::cout << "  - " << key << std::endl;
        }

        // Test statistics
        auto stats = db.get_statistics();
        std::cout << "✓ Database Statistics:" << std::endl;
        std::cout << "  - Total entries: " << stats.total_entries << std::endl;
        std::cout << "  - Total original bytes: " << stats.total_original_bytes << std::endl;
        std::cout << "  - Total compressed bytes: " << stats.total_compressed_bytes << std::endl;
        std::cout << "  - Average compression ratio: " << std::fixed << std::setprecision(2)
                  << stats.average_compression_ratio << "x" << std::endl;

        // Test removal
        auto remove_result = db.remove("test_int");
        if (!remove_result) {
            std::cout << "✗ Failed to remove data: " << remove_result.error << std::endl;
            return false;
        }

        if (db.contains("test_int")) {
            std::cout << "✗ Data was not properly removed" << std::endl;
            return false;
        }
        std::cout << "✓ Data removal successful" << std::endl;

        std::cout << "\n✓ All simple database tests passed!" << std::endl;

        // Clean up
        std::filesystem::remove_all("./test_simple_db");
        std::cout << "✓ Test database cleaned up" << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    std::cout << "Simple AI-Native Database Test" << std::endl;
    std::cout << "==============================" << std::endl;

    if (test_simple_operations()) {
        std::cout << "\n🎉 All tests passed! Simple AI-Native Database is working!" << std::endl;
        return 0;
    }
    else {
        std::cout << "\n❌ Tests failed." << std::endl;
        return 1;
    }
}
