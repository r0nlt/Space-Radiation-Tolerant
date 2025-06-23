#pragma once

#include <lmdb.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/research/variational_autoencoder.hpp"

namespace rad_ml::storage {

/**
 * @brief Result type for operations that can fail
 */
template <typename T>
struct Result {
    std::optional<T> value;
    std::string error;

    explicit operator bool() const noexcept { return value.has_value(); }
    T& operator*() { return *value; }
    const T& operator*() const { return *value; }
    T* operator->() { return value.operator->(); }
    const T* operator->() const { return value.operator->(); }

    static Result success(T val) { return Result{std::move(val), ""}; }
    static Result failure(std::string err) { return Result{std::nullopt, std::move(err)}; }
};

// Specialization for void
template <>
struct Result<void> {
    bool is_success;
    std::string error;

    explicit operator bool() const noexcept { return is_success; }

    static Result success() { return Result{true, ""}; }
    static Result failure(std::string err) { return Result{false, std::move(err)}; }
};

/**
 * @brief Modern C++ AI-Native Database for Datacenter Applications
 *
 * This class provides a high-performance database that uses Variational Autoencoders
 * for intelligent data compression with LMDB for persistent storage.
 * Designed with modern C++ principles: RAII, move semantics, concepts, and type safety.
 */
class AINativeDatabase {
   public:
    // Strong types for better API design
    using Key = std::string;
    using CompressionRatio = double;
    using ReconstructionError = double;
    using LatentDimension = size_t;

    /**
     * @brief Configuration for the AI-native database
     */
    struct Config {
        std::filesystem::path db_path = "./ai_native_db";
        size_t max_db_size = 10ULL * 1024 * 1024 * 1024;  // 10GB for datacenter
        LatentDimension default_latent_dim = 32;
        std::vector<size_t> vae_hidden_dims = {256, 128, 64};
        float max_reconstruction_error = 0.005f;  // Higher precision for datacenter
        bool enable_background_optimization = true;
        std::chrono::seconds optimization_interval{300};  // 5 minutes
    };

    /**
     * @brief Compression metrics for monitoring and optimization
     */
    struct CompressionMetrics {
        CompressionRatio ratio = 0.0;
        ReconstructionError error = 0.0;
        std::chrono::milliseconds encode_time{0};
        std::chrono::milliseconds decode_time{0};
        size_t original_bytes = 0;
        size_t compressed_bytes = 0;
        bool success = false;
    };

    /**
     * @brief Data type concept for storable data (C++17 compatible)
     */
    template <typename T>
    static constexpr bool is_storable_data_v =
        std::is_arithmetic_v<T> && std::is_trivially_copyable_v<T>;

    /**
     * @brief Constructor with configuration
     * @param config Database configuration
     */
    explicit AINativeDatabase(Config config);

    /**
     * @brief Default constructor with default configuration
     */
    AINativeDatabase();

    /**
     * @brief Destructor - RAII cleanup
     */
    ~AINativeDatabase();

    // Delete copy operations to prevent accidental copying of large database state
    AINativeDatabase(const AINativeDatabase&) = delete;
    AINativeDatabase& operator=(const AINativeDatabase&) = delete;

    // Enable move operations for efficient resource transfer
    AINativeDatabase(AINativeDatabase&&) noexcept;
    AINativeDatabase& operator=(AINativeDatabase&&) noexcept;

    /**
     * @brief Initialize the database system
     * @param data_dimensions Map of data type names to their dimensions
     * @return Result for error handling without exceptions
     */
    Result<void> initialize(const std::unordered_map<std::string, size_t>& data_dimensions);

    /**
     * @brief Store data with intelligent compression (async)
     * @param key Unique identifier for the data
     * @param data Data to store (supports any arithmetic type)
     * @param data_type Optional data type for specialized compression
     * @return Future with compression metrics
     */
    template <typename T>
    std::future<Result<CompressionMetrics>> store_async(const Key& key, const std::vector<T>& data,
                                                        const std::string& data_type = "default");

    /**
     * @brief Retrieve and decompress data (async)
     * @param key Data identifier
     * @return Future with decompressed data and metrics
     */
    template <typename T>
    std::future<Result<std::pair<std::vector<T>, CompressionMetrics>>> retrieve_async(
        const Key& key);

    /**
     * @brief Synchronous store operation
     */
    template <typename T>
    Result<CompressionMetrics> store(const Key& key, const std::vector<T>& data,
                                     const std::string& data_type = "default");

    /**
     * @brief Synchronous retrieve operation
     */
    template <typename T>
    Result<std::pair<std::vector<T>, CompressionMetrics>> retrieve(const Key& key);

    /**
     * @brief Check if a key exists
     */
    bool contains(const Key& key) const noexcept;

    /**
     * @brief Remove data from database
     */
    Result<void> remove(const Key& key);

    /**
     * @brief Get all keys in the database
     */
    std::vector<Key> keys() const;

    /**
     * @brief Train VAE model on provided dataset
     * @param training_data Dataset for training
     * @param data_type Data type identifier
     * @return Training result
     */
    template <typename T>
    Result<void> train_vae(const std::vector<std::vector<T>>& training_data,
                           const std::string& data_type = "default");

    /**
     * @brief Database statistics
     */
    struct Statistics {
        size_t total_entries = 0;
        size_t total_original_bytes = 0;
        size_t total_compressed_bytes = 0;
        CompressionRatio average_compression_ratio = 0.0;
        ReconstructionError average_reconstruction_error = 0.0;
        std::chrono::steady_clock::time_point last_optimization;
        size_t vae_models_count = 0;
    };

    /**
     * @brief Get current database statistics
     */
    Statistics get_statistics() const;

    /**
     * @brief Start background optimization tasks
     */
    void start_background_optimization();

    /**
     * @brief Stop background optimization tasks
     */
    void stop_background_optimization();

    /**
     * @brief Perform manual optimization
     */
    Result<void> optimize_now();

   private:
    // Preprocessing statistics for data normalization
    struct PreprocessingStats {
        std::vector<float> means;
        std::vector<float> stds;
    };

    // Compressed data package structure
    struct CompressedDataPackage {
        std::vector<float> latent_data;
        size_t original_size;
        std::string data_type;
        PreprocessingStats preprocessing_stats;
    };

    // Configuration
    Config config_;

    // LMDB environment (RAII managed)
    struct LMDBEnvironment {
        MDB_env* env = nullptr;
        MDB_dbi dbi = 0;

        LMDBEnvironment() = default;
        ~LMDBEnvironment();

        // Delete copy operations
        LMDBEnvironment(const LMDBEnvironment&) = delete;
        LMDBEnvironment& operator=(const LMDBEnvironment&) = delete;

        // Enable move operations
        LMDBEnvironment(LMDBEnvironment&& other) noexcept;
        LMDBEnvironment& operator=(LMDBEnvironment&& other) noexcept;
    };

    std::unique_ptr<LMDBEnvironment> lmdb_;

    // VAE models for different data types
    std::unordered_map<std::string, std::unique_ptr<research::VariationalAutoencoder<float>>>
        vae_models_;

    // Thread safety
    mutable std::mutex data_mutex_;          // For thread-safe access
    mutable std::mutex stats_mutex_;         // For statistics updates
    mutable std::mutex vae_mutex_;           // For VAE model access
    mutable std::mutex optimization_mutex_;  // For background optimization control

    // Background optimization
    std::atomic<bool> optimization_running_{false};
    std::unique_ptr<std::thread> optimization_thread_;

    // Statistics
    mutable Statistics stats_;

    // Internal methods
    Result<void> initialize_lmdb();
    Result<void> store_raw(const Key& key, const std::vector<uint8_t>& data);
    Result<std::vector<uint8_t>> retrieve_raw(const Key& key) const;

    template <typename T>
    std::vector<uint8_t> serialize_data(const std::vector<T>& data) const;

    template <typename T>
    Result<std::vector<T>> deserialize_data(const std::vector<uint8_t>& data) const;

    research::VariationalAutoencoder<float>* get_or_create_vae(size_t input_dim,
                                                               const std::string& data_type);
    void update_statistics(const CompressionMetrics& metrics);
    void background_optimization_loop();

    // VAE integration helper methods
    std::vector<float> preprocess_data(const std::vector<float>& data) const;
    std::vector<float> denormalize_data(const std::vector<float>& data,
                                        const PreprocessingStats& stats) const;
    void calculate_preprocessing_stats(const std::vector<float>& data,
                                       PreprocessingStats& stats) const;
    float calculate_reconstruction_error(const std::vector<float>& original,
                                         const std::vector<float>& reconstructed) const;

    // Compressed package serialization
    std::vector<uint8_t> serialize_compressed_package(const CompressedDataPackage& package) const;
    Result<CompressedDataPackage> deserialize_compressed_package(
        const std::vector<uint8_t>& data) const;

    // Error handling helpers
    std::string lmdb_error_string(int error_code) const;
};

}  // namespace rad_ml::storage
