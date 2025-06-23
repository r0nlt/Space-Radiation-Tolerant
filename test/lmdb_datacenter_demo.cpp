/**
 * @file lmdb_datacenter_demo.cpp
 * @brief Datacenter LMDB Demo with AI Compression Concepts
 */

#include <lmdb.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace datacenter_demo {

/**
 * @brief Modern C++ RAII wrapper for LMDB in datacenter environment
 */
class DatacenterLMDB {
   public:
    explicit DatacenterLMDB(const std::string& db_path,
                            size_t max_size = 1024 * 1024 * 1024)  // 1GB default
        : db_path_(db_path)
    {
        // Create directory if needed
        std::filesystem::create_directories(db_path_);

        // Initialize LMDB with datacenter-optimized settings
        int rc = mdb_env_create(&env_);
        if (rc != 0) throw std::runtime_error("Failed to create LMDB environment");

        // Set large map size for datacenter workloads
        rc = mdb_env_set_mapsize(env_, max_size);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to set LMDB map size");
        }

        // Use multiple databases for better organization
        rc = mdb_env_set_maxdbs(env_, 10);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to set max databases");
        }

        // Open with datacenter-appropriate flags
        rc = mdb_env_open(env_, db_path_.c_str(), MDB_NOTLS,
                          0664);  // MDB_NOTLS for better performance in threaded environments
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to open LMDB environment");
        }

        // Open default database
        MDB_txn* txn;
        rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to begin transaction");
        }

        rc = mdb_dbi_open(txn, nullptr, 0, &dbi_);
        if (rc != 0) {
            mdb_txn_abort(txn);
            mdb_env_close(env_);
            throw std::runtime_error("Failed to open database");
        }

        // Open compression metadata database
        rc = mdb_dbi_open(txn, "compression_meta", MDB_CREATE, &meta_dbi_);
        if (rc != 0) {
            mdb_txn_abort(txn);
            mdb_env_close(env_);
            throw std::runtime_error("Failed to open compression metadata database");
        }

        rc = mdb_txn_commit(txn);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to commit transaction");
        }

        std::cout << "✓ Datacenter LMDB initialized at: " << db_path_ << std::endl;
        std::cout << "  - Max size: " << (max_size / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  - Multiple databases enabled for AI metadata" << std::endl;
    }

    ~DatacenterLMDB()
    {
        if (env_) {
            mdb_dbi_close(env_, dbi_);
            mdb_dbi_close(env_, meta_dbi_);
            mdb_env_close(env_);
            std::cout << "✓ Datacenter LMDB closed cleanly" << std::endl;
        }
    }

    // Delete copy operations (datacenter resources are unique)
    DatacenterLMDB(const DatacenterLMDB&) = delete;
    DatacenterLMDB& operator=(const DatacenterLMDB&) = delete;

    /**
     * @brief Store data with compression metadata (simulating VAE compression)
     */
    bool store_with_compression_meta(const std::string& key, const std::vector<float>& data,
                                     const std::string& data_type = "sensor_data")
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Simulate VAE compression (in real implementation, this would use your VAE)
        std::vector<float> compressed_data = simulate_vae_compression(data);

        auto compress_time = std::chrono::high_resolution_clock::now();

        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) return false;

        // Store compressed data
        MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
        MDB_val mdb_value{compressed_data.size() * sizeof(float),
                          const_cast<float*>(compressed_data.data())};

        rc = mdb_put(txn, dbi_, &mdb_key, &mdb_value, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            return false;
        }

        // Store compression metadata
        CompressionMetadata meta;
        meta.original_size = data.size();
        meta.compressed_size = compressed_data.size();
        meta.compression_ratio = static_cast<double>(data.size()) / compressed_data.size();
        std::strncpy(meta.data_type, data_type.c_str(), sizeof(meta.data_type) - 1);
        meta.data_type[sizeof(meta.data_type) - 1] = '\0';
        meta.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        meta.compression_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(compress_time - start_time)
                .count();

        std::string meta_key = key + "_meta";
        MDB_val meta_mdb_key{meta_key.size(), const_cast<char*>(meta_key.data())};
        MDB_val meta_mdb_value{sizeof(CompressionMetadata), &meta};

        rc = mdb_put(txn, meta_dbi_, &meta_mdb_key, &meta_mdb_value, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            return false;
        }

        rc = mdb_txn_commit(txn);
        if (rc == 0) {
            std::cout << "✓ Stored " << key << " (" << data_type << "):" << std::endl;
            std::cout << "  - Original size: " << meta.original_size << " elements" << std::endl;
            std::cout << "  - Compressed size: " << meta.compressed_size << " elements"
                      << std::endl;
            std::cout << "  - Compression ratio: " << std::fixed << std::setprecision(2)
                      << meta.compression_ratio << "x" << std::endl;
            std::cout << "  - Compression time: " << meta.compression_time_ms << " ms" << std::endl;
        }

        return rc == 0;
    }

    /**
     * @brief Retrieve data with decompression
     */
    std::pair<std::vector<float>, bool> retrieve_with_decompression(const std::string& key)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) return {{}, false};

        // Retrieve compressed data
        MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
        MDB_val mdb_value;

        rc = mdb_get(txn, dbi_, &mdb_key, &mdb_value);
        if (rc != 0) {
            mdb_txn_abort(txn);
            return {{}, false};
        }

        // Extract compressed data
        std::vector<float> compressed_data(
            static_cast<float*>(mdb_value.mv_data),
            static_cast<float*>(mdb_value.mv_data) + (mdb_value.mv_size / sizeof(float)));

        // Get metadata
        std::string meta_key = key + "_meta";
        MDB_val meta_mdb_key{meta_key.size(), const_cast<char*>(meta_key.data())};
        MDB_val meta_mdb_value;

        CompressionMetadata meta{};
        rc = mdb_get(txn, meta_dbi_, &meta_mdb_key, &meta_mdb_value);
        if (rc == 0) {
            std::memcpy(&meta, meta_mdb_value.mv_data, sizeof(CompressionMetadata));
        }

        mdb_txn_abort(txn);

        // Simulate VAE decompression
        std::vector<float> decompressed_data =
            simulate_vae_decompression(compressed_data, meta.original_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto decompress_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "✓ Retrieved " << key << ":" << std::endl;
        std::cout << "  - Decompressed size: " << decompressed_data.size() << " elements"
                  << std::endl;
        std::cout << "  - Data type: " << meta.data_type << std::endl;
        std::cout << "  - Decompression time: " << decompress_time.count() << " ms" << std::endl;

        return {decompressed_data, true};
    }

    /**
     * @brief Get datacenter statistics
     */
    void print_datacenter_stats() const
    {
        MDB_stat stat;
        MDB_txn* txn;

        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) return;

        rc = mdb_stat(txn, dbi_, &stat);
        if (rc == 0) {
            std::cout << "\n=== Datacenter LMDB Statistics ===" << std::endl;
            std::cout << "Data entries: " << stat.ms_entries << std::endl;
            std::cout << "Page size: " << stat.ms_psize << " bytes" << std::endl;
            std::cout << "Tree depth: " << stat.ms_depth << std::endl;
            std::cout << "Branch pages: " << stat.ms_branch_pages << std::endl;
            std::cout << "Leaf pages: " << stat.ms_leaf_pages << std::endl;
            std::cout << "Overflow pages: " << stat.ms_overflow_pages << std::endl;
        }

        rc = mdb_stat(txn, meta_dbi_, &stat);
        if (rc == 0) {
            std::cout << "Metadata entries: " << stat.ms_entries << std::endl;
        }

        mdb_txn_abort(txn);
    }

   private:
    struct CompressionMetadata {
        size_t original_size;
        size_t compressed_size;
        double compression_ratio;
        int64_t timestamp;
        int64_t compression_time_ms;
        char data_type[32];
    };

    MDB_env* env_ = nullptr;
    MDB_dbi dbi_ = 0;
    MDB_dbi meta_dbi_ = 0;  // For compression metadata
    std::string db_path_;

    /**
     * @brief Simulate VAE compression (placeholder for real VAE integration)
     */
    std::vector<float> simulate_vae_compression(const std::vector<float>& data) const
    {
        // In a real implementation, this would use your VAE encoder
        // For demo purposes, we simulate compression by taking every other element + some noise
        std::vector<float> compressed;
        compressed.reserve(data.size() / 2 + 10);  // Simulated latent space + metadata

        // Add some "latent space" values (simulation)
        for (size_t i = 0; i < data.size(); i += 2) {
            compressed.push_back(data[i] * 0.8f + 0.1f);  // Simulate lossy compression
        }

        // Add some "VAE metadata" (simulation)
        compressed.insert(compressed.end(), {1.0f, 2.0f, 3.0f});  // Simulated encoder parameters

        return compressed;
    }

    /**
     * @brief Simulate VAE decompression (placeholder for real VAE integration)
     */
    std::vector<float> simulate_vae_decompression(const std::vector<float>& compressed_data,
                                                  size_t target_size) const
    {
        // In a real implementation, this would use your VAE decoder
        std::vector<float> decompressed;
        decompressed.reserve(target_size);

        // Remove metadata (last 3 elements in our simulation)
        size_t data_end = compressed_data.size() - 3;

        // Reconstruct data (simulation)
        for (size_t i = 0; i < data_end && decompressed.size() < target_size; ++i) {
            float value =
                compressed_data[i] / 0.8f - 0.125f;  // Reverse the compression transformation
            decompressed.push_back(value);

            // Interpolate missing values (simulate VAE reconstruction)
            if (decompressed.size() < target_size) {
                float next_val =
                    (i + 1 < data_end) ? compressed_data[i + 1] / 0.8f - 0.125f : value;
                decompressed.push_back((value + next_val) / 2.0f +
                                       0.01f);  // Small reconstruction error
            }
        }

        // Ensure we have the right size
        decompressed.resize(target_size);

        return decompressed;
    }
};

/**
 * @brief Generate realistic datacenter sensor data
 */
std::vector<float> generate_datacenter_sensor_data(const std::string& sensor_type, size_t size)
{
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());

    if (sensor_type == "temperature") {
        std::normal_distribution<float> dist(22.5f, 2.0f);  // Data center temperature
        for (auto& val : data) val = dist(gen);
    }
    else if (sensor_type == "power") {
        std::normal_distribution<float> dist(150.0f, 20.0f);  // Power consumption in watts
        for (auto& val : data) val = std::max(0.0f, dist(gen));
    }
    else if (sensor_type == "network") {
        std::exponential_distribution<float> dist(0.01f);  // Network throughput
        for (auto& val : data) val = dist(gen);
    }
    else {
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        for (auto& val : data) val = dist(gen);
    }

    return data;
}

}  // namespace datacenter_demo

int main()
{
    std::cout << "Datacenter LMDB Demo with AI Compression Concepts" << std::endl;
    std::cout << "=================================================" << std::endl;

    try {
        // Create datacenter database
        datacenter_demo::DatacenterLMDB db("./datacenter_ai_db", 500 * 1024 * 1024);  // 500MB

        // Generate and store different types of datacenter data
        std::cout << "\nGenerating and storing datacenter sensor data..." << std::endl;

        // Temperature sensors
        auto temp_data = datacenter_demo::generate_datacenter_sensor_data("temperature", 100);
        db.store_with_compression_meta("rack_1_temp_sensors", temp_data, "temperature");

        // Power consumption data
        auto power_data = datacenter_demo::generate_datacenter_sensor_data("power", 50);
        db.store_with_compression_meta("server_power_consumption", power_data, "power");

        // Network throughput data
        auto network_data = datacenter_demo::generate_datacenter_sensor_data("network", 200);
        db.store_with_compression_meta("network_throughput_data", network_data, "network");

        // CPU utilization data
        auto cpu_data = datacenter_demo::generate_datacenter_sensor_data("cpu", 75);
        db.store_with_compression_meta("cpu_utilization_metrics", cpu_data, "cpu");

        std::cout << "\nRetrieving and verifying stored data..." << std::endl;

        // Retrieve and verify data
        auto [retrieved_temp, temp_success] = db.retrieve_with_decompression("rack_1_temp_sensors");
        auto [retrieved_power, power_success] =
            db.retrieve_with_decompression("server_power_consumption");
        auto [retrieved_network, network_success] =
            db.retrieve_with_decompression("network_throughput_data");
        auto [retrieved_cpu, cpu_success] =
            db.retrieve_with_decompression("cpu_utilization_metrics");

        if (temp_success && power_success && network_success && cpu_success) {
            std::cout << "\n✓ All data retrieved successfully!" << std::endl;

            // Show some sample data verification
            std::cout << "\nSample data verification:" << std::endl;
            std::cout << "Temperature data points: " << retrieved_temp.size() << std::endl;
            std::cout << "Power data points: " << retrieved_power.size() << std::endl;
            std::cout << "Network data points: " << retrieved_network.size() << std::endl;
            std::cout << "CPU data points: " << retrieved_cpu.size() << std::endl;

            // Calculate and show compression effectiveness
            float temp_error = 0.0f;
            for (size_t i = 0; i < std::min(temp_data.size(), retrieved_temp.size()); ++i) {
                temp_error += std::abs(temp_data[i] - retrieved_temp[i]);
            }
            temp_error /= temp_data.size();

            std::cout << "\nCompression Quality Analysis:" << std::endl;
            std::cout << "Average reconstruction error (temperature): " << std::fixed
                      << std::setprecision(4) << temp_error << "°C" << std::endl;
        }
        else {
            std::cout << "✗ Some data retrieval failed" << std::endl;
        }

        // Show datacenter statistics
        db.print_datacenter_stats();

        std::cout << "\n✓ Datacenter LMDB Demo completed successfully!" << std::endl;
        std::cout << "\nThis demo shows how LMDB can be integrated with AI compression"
                  << std::endl;
        std::cout << "in a datacenter environment. In a real implementation, the" << std::endl;
        std::cout << "VAE compression would use your actual neural network models." << std::endl;

        // Clean up
        std::filesystem::remove_all("./datacenter_ai_db");
        std::cout << "\n✓ Demo database cleaned up" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "✗ Demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
