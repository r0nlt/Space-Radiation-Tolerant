/**
 * @file ai_native_database.cpp
 * @brief Implementation of AI-Native Database for Datacenter Applications
 */

#include "rad_ml/storage/ai_native_database.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/research/vae_optimal_configs.hpp"

namespace rad_ml::storage {

// LMDBEnvironment implementation
AINativeDatabase::LMDBEnvironment::~LMDBEnvironment()
{
    if (env) {
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        std::cout << "LMDB environment closed cleanly" << std::endl;
    }
}

AINativeDatabase::LMDBEnvironment::LMDBEnvironment(LMDBEnvironment&& other) noexcept
    : env(other.env), dbi(other.dbi)
{
    other.env = nullptr;
    other.dbi = 0;
}

AINativeDatabase::LMDBEnvironment& AINativeDatabase::LMDBEnvironment::operator=(
    LMDBEnvironment&& other) noexcept
{
    if (this != &other) {
        // Clean up current resources
        if (env) {
            mdb_dbi_close(env, dbi);
            mdb_env_close(env);
        }

        // Move resources
        env = other.env;
        dbi = other.dbi;

        // Clear other object
        other.env = nullptr;
        other.dbi = 0;
    }
    return *this;
}

// AINativeDatabase implementation
AINativeDatabase::AINativeDatabase(Config config)
    : config_(std::move(config)),
      lmdb_(std::make_unique<LMDBEnvironment>()),
      optimization_running_(false)
{
    std::cout << "AI-Native Database created with path: " << config_.db_path.string() << std::endl;
}

AINativeDatabase::AINativeDatabase() : AINativeDatabase(Config{}) {}

AINativeDatabase::~AINativeDatabase()
{
    stop_background_optimization();
    std::cout << "AI-Native Database destroyed" << std::endl;
}

AINativeDatabase::AINativeDatabase(AINativeDatabase&& other) noexcept
    : config_(std::move(other.config_)),
      lmdb_(std::move(other.lmdb_)),
      vae_models_(std::move(other.vae_models_)),
      optimization_running_(other.optimization_running_.load()),
      optimization_thread_(std::move(other.optimization_thread_)),
      stats_(other.stats_)
{
}

AINativeDatabase& AINativeDatabase::operator=(AINativeDatabase&& other) noexcept
{
    if (this != &other) {
        // Stop current operations
        stop_background_optimization();

        // Move resources
        config_ = std::move(other.config_);
        lmdb_ = std::move(other.lmdb_);
        vae_models_ = std::move(other.vae_models_);
        optimization_running_ = other.optimization_running_.load();
        optimization_thread_ = std::move(other.optimization_thread_);
        stats_ = other.stats_;
    }
    return *this;
}

Result<void> AINativeDatabase::initialize(
    const std::unordered_map<std::string, size_t>& data_dimensions)
{
    std::cout << "Initializing AI-Native Database..." << std::endl;

    // Initialize LMDB
    auto lmdb_result = initialize_lmdb();
    if (!lmdb_result) {
        return Result<void>::failure("Failed to initialize LMDB: " + lmdb_result.error);
    }

    // Initialize VAE models for each data type using BREAKTHROUGH optimal configurations
    std::lock_guard<std::mutex> vae_lock(vae_mutex_);
    for (const auto& [data_type, dimension] : data_dimensions) {
        try {
            // Use the SAME optimal configuration logic as get_or_create_vae
            research::VAEConfig vae_config;
            std::vector<size_t> hidden_dims;

            if (data_type == "telemetry" || data_type == "default") {
                // Use BREAKTHROUGH compression config: theoretical 4:1 latent, effective 2.5-3.7:1
                // with metadata
                vae_config = research::OptimalConfigs::getCompressionConfig();
                hidden_dims = research::OptimalConfigs::getCompressionArchitecture();

                // Scale latent dimension proportionally for non-12D telemetry
                if (dimension != 12) {
                    vae_config.latent_dim =
                        std::max(size_t(2), dimension / 4);  // Maintain 4:1 ratio
                }

                std::cout << "Using BREAKTHROUGH telemetry compression config: " << dimension
                          << "D→" << vae_config.latent_dim
                          << "D (effective 2.5-3.7:1, β=" << vae_config.beta << ")" << std::endl;
            }
            else if (data_type == "anomaly_detection" || data_type == "monitoring" ||
                     data_type == "anomaly") {
                // Use VALIDATED anomaly detection config
                vae_config = research::OptimalConfigs::getAnomalyDetectionConfig();
                hidden_dims = research::OptimalConfigs::getAnomalyDetectionArchitecture();

                if (dimension != 12) {
                    vae_config.latent_dim = std::min(dimension / 2, size_t(16));
                }

                std::cout << "Using VALIDATED anomaly detection config: " << dimension << "D→"
                          << vae_config.latent_dim << "D (β=" << vae_config.beta << ")"
                          << std::endl;
            }
            else if (data_type == "sensors" || data_type.find("sensor") != std::string::npos) {
                // Use high-quality compression for sensor data
                vae_config =
                    research::OptimalConfigs::ImprovedConfigs::getHighQualityCompressionConfig();
                hidden_dims = research::OptimalConfigs::ImprovedConfigs::
                    getHighQualityCompressionArchitecture();

                if (dimension != 12) {
                    vae_config.latent_dim = std::max(size_t(3), dimension / 3);
                }

                std::cout << "Using HIGH-QUALITY sensor compression config: " << dimension << "D→"
                          << vae_config.latent_dim << "D (β=" << vae_config.beta << ")"
                          << std::endl;
            }
            else {
                // Use balanced configuration for unknown data types
                vae_config = research::OptimalConfigs::getBalancedConfig();
                hidden_dims = research::OptimalConfigs::getBalancedArchitecture();

                if (dimension != 12) {
                    vae_config.latent_dim =
                        std::max(size_t(2), std::min(dimension / 3, size_t(16)));

                    if (dimension >= 64) {
                        hidden_dims = {128, 64, 32};
                    }
                    else if (dimension >= 32) {
                        hidden_dims = {64, 32};
                    }
                    else {
                        hidden_dims = {32};
                    }
                }

                std::cout << "Using BALANCED config for '" << data_type << "': " << dimension
                          << "D→" << vae_config.latent_dim << "D (β=" << vae_config.beta << ")"
                          << std::endl;
            }

            vae_models_[data_type] = std::make_unique<research::VariationalAutoencoder<float>>(
                dimension, vae_config.latent_dim, hidden_dims, neural::ProtectionLevel::NONE,
                vae_config);

            std::cout << "✅ Initialized OPTIMAL VAE for '" << data_type << "' with " << dimension
                      << "D→" << vae_config.latent_dim << "D using breakthrough config"
                      << std::endl;
        }
        catch (const std::exception& e) {
            return Result<void>::failure("Failed to initialize VAE for " + data_type +
                                         " (std::exception): " + e.what());
        }
        catch (...) {
            return Result<void>::failure(
                "Failed to initialize VAE for " + data_type +
                " (unknown exception): Non-standard exception caught during VAE initialization");
        }
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.vae_models_count = vae_models_.size();
        stats_.last_optimization = std::chrono::steady_clock::now();
    }

    std::cout << "AI-Native Database initialized successfully with " << vae_models_.size()
              << " VAE models" << std::endl;

    return Result<void>::success();
}

Result<void> AINativeDatabase::initialize_lmdb()
{
    // Create directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(config_.db_path, ec);
    if (ec) {
        return Result<void>::failure("Failed to create database directory: " + ec.message());
    }

    // Create environment
    int rc = mdb_env_create(&lmdb_->env);
    if (rc != 0) {
        return Result<void>::failure("Failed to create LMDB environment: " + lmdb_error_string(rc));
    }

    // Set map size
    rc = mdb_env_set_mapsize(lmdb_->env, config_.max_db_size);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return Result<void>::failure("Failed to set LMDB map size: " + lmdb_error_string(rc));
    }

    // Open environment
    rc = mdb_env_open(lmdb_->env, config_.db_path.c_str(), 0, 0664);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return Result<void>::failure("Failed to open LMDB environment: " + lmdb_error_string(rc));
    }

    // Open database
    MDB_txn* txn;
    rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return Result<void>::failure("Failed to begin transaction: " + lmdb_error_string(rc));
    }

    rc = mdb_dbi_open(txn, nullptr, 0, &lmdb_->dbi);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return Result<void>::failure("Failed to open database: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return Result<void>::failure("Failed to commit transaction: " + lmdb_error_string(rc));
    }

    std::cout << "LMDB initialized successfully at: " << config_.db_path.string() << std::endl;
    return Result<void>::success();
}

bool AINativeDatabase::contains(const Key& key) const noexcept
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return false;
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) {
        return false;
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
    MDB_val mdb_value;

    rc = mdb_get(txn, lmdb_->dbi, &mdb_key, &mdb_value);
    mdb_txn_abort(txn);

    return rc == 0;
}

Result<void> AINativeDatabase::store_raw(const Key& key, const std::vector<uint8_t>& data)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return Result<void>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        return Result<void>::failure("Failed to begin write transaction: " + lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
    MDB_val mdb_value{data.size(), const_cast<uint8_t*>(data.data())};

    rc = mdb_put(txn, lmdb_->dbi, &mdb_key, &mdb_value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return Result<void>::failure("Failed to store data: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        return Result<void>::failure("Failed to commit write transaction: " +
                                     lmdb_error_string(rc));
    }

    return Result<void>::success();
}

Result<std::vector<uint8_t>> AINativeDatabase::retrieve_raw(const Key& key) const
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return Result<std::vector<uint8_t>>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) {
        return Result<std::vector<uint8_t>>::failure("Failed to begin read transaction: " +
                                                     lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
    MDB_val mdb_value;

    rc = mdb_get(txn, lmdb_->dbi, &mdb_key, &mdb_value);
    if (rc == MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return Result<std::vector<uint8_t>>::failure("Key not found: " + key);
    }
    else if (rc != 0) {
        mdb_txn_abort(txn);
        return Result<std::vector<uint8_t>>::failure("Failed to retrieve data: " +
                                                     lmdb_error_string(rc));
    }

    std::vector<uint8_t> result(static_cast<uint8_t*>(mdb_value.mv_data),
                                static_cast<uint8_t*>(mdb_value.mv_data) + mdb_value.mv_size);
    mdb_txn_abort(txn);

    return Result<std::vector<uint8_t>>::success(std::move(result));
}

Result<void> AINativeDatabase::remove(const Key& key)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return Result<void>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        return Result<void>::failure("Failed to begin write transaction: " + lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};

    rc = mdb_del(txn, lmdb_->dbi, &mdb_key, nullptr);
    if (rc == MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return Result<void>::failure("Key not found: " + key);
    }
    else if (rc != 0) {
        mdb_txn_abort(txn);
        return Result<void>::failure("Failed to delete data: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        return Result<void>::failure("Failed to commit delete transaction: " +
                                     lmdb_error_string(rc));
    }

    return Result<void>::success();
}

std::vector<AINativeDatabase::Key> AINativeDatabase::keys() const
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<Key> result;

    if (!lmdb_->env) {
        return result;
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) {
        return result;
    }

    MDB_cursor* cursor;
    rc = mdb_cursor_open(txn, lmdb_->dbi, &cursor);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return result;
    }

    MDB_val key, data;
    while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0) {
        result.emplace_back(static_cast<char*>(key.mv_data), key.mv_size);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);

    return result;
}

template <typename T>
std::vector<uint8_t> AINativeDatabase::serialize_data(const std::vector<T>& data) const
{
    // Simple serialization - just copy the bytes
    // In production, you might want more sophisticated serialization with versioning
    std::vector<uint8_t> result(sizeof(T) * data.size());
    std::memcpy(result.data(), data.data(), result.size());
    return result;
}

template <typename T>
Result<std::vector<T>> AINativeDatabase::deserialize_data(const std::vector<uint8_t>& data) const
{
    if (data.size() % sizeof(T) != 0) {
        return Result<std::vector<T>>::failure("Invalid data size for deserialization");
    }

    size_t element_count = data.size() / sizeof(T);
    std::vector<T> result(element_count);
    std::memcpy(result.data(), data.data(), data.size());

    return Result<std::vector<T>>::success(std::move(result));
}

// VAE-integrated store implementation with real compression
template <typename T>
Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store(const Key& key,
                                                                     const std::vector<T>& data,
                                                                     const std::string& data_type)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    CompressionMetrics metrics;
    metrics.original_bytes = data.size() * sizeof(T);

    try {
        // Get or create VAE for this data type
        auto* vae = get_or_create_vae(data.size(), data_type);
        if (!vae) {
            // Fallback to raw storage if VAE creation fails
            core::Logger::warning("VAE creation failed for data_type '" + data_type +
                                  "', falling back to raw storage");
            auto serialized = serialize_data(data);
            auto store_result = store_raw(key, serialized);
            if (!store_result) {
                return Result<CompressionMetrics>::failure(store_result.error);
            }

            metrics.compressed_bytes = serialized.size();
            metrics.ratio = 1.0;
            metrics.error = 0.0;
            metrics.success = true;
        }
        else {
            // Convert data to float for VAE processing
            std::vector<float> float_data;
            float_data.reserve(data.size());
            for (const auto& value : data) {
                float_data.push_back(static_cast<float>(value));
            }

            // Apply preprocessing (z-score normalization)
            auto preprocessed_data = preprocess_data(float_data);

            // VAE compression: encode -> sample -> store latent
            auto [mean, log_var] = vae->encode(preprocessed_data);
            auto latent = vae->sample(mean, log_var);

            // Create compressed data package with metadata
            CompressedDataPackage package;
            package.latent_data = latent;
            package.original_size = data.size();
            package.data_type = data_type;

            // Calculate preprocessing statistics for reconstruction
            calculate_preprocessing_stats(float_data, package.preprocessing_stats);

            // Serialize compressed package
            auto compressed_serialized = serialize_compressed_package(package);
            auto store_result = store_raw(key, compressed_serialized);
            if (!store_result) {
                return Result<CompressionMetrics>::failure(store_result.error);
            }

            // Calculate compression metrics
            metrics.compressed_bytes = compressed_serialized.size();
            metrics.ratio = static_cast<double>(metrics.original_bytes) / metrics.compressed_bytes;

            // Calculate reconstruction error for quality assessment
            auto reconstructed = vae->decode(latent);
            auto denormalized = denormalize_data(reconstructed, package.preprocessing_stats);
            metrics.error = calculate_reconstruction_error(preprocessed_data, reconstructed);
            metrics.success = true;

            core::Logger::info("VAE compression successful: " + std::to_string(data.size()) +
                               "D -> " + std::to_string(latent.size()) +
                               "D, ratio: " + std::to_string(metrics.ratio) +
                               ":1, error: " + std::to_string(metrics.error));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        metrics.encode_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        metrics.decode_time = std::chrono::milliseconds(0);

        update_statistics(metrics);
        return Result<CompressionMetrics>::success(metrics);
    }
    catch (const std::exception& e) {
        return Result<CompressionMetrics>::failure("VAE compression failed: " +
                                                   std::string(e.what()));
    }
    catch (...) {
        return Result<CompressionMetrics>::failure("VAE compression failed: Unknown exception");
    }
}

template <typename T>
Result<std::pair<std::vector<T>, AINativeDatabase::CompressionMetrics>> AINativeDatabase::retrieve(
    const Key& key)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    CompressionMetrics metrics;

    try {
        // Retrieve raw data
        auto retrieve_result = retrieve_raw(key);
        if (!retrieve_result) {
            return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
                retrieve_result.error);
        }

        metrics.compressed_bytes = retrieve_result->size();

        // Try to deserialize as compressed package first
        auto package_result = deserialize_compressed_package(*retrieve_result);
        if (package_result) {
            // VAE-compressed data found
            auto& package = *package_result;

            // Get VAE for decompression
            auto* vae = get_or_create_vae(package.original_size, package.data_type);
            if (!vae) {
                return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
                    "Failed to get VAE for decompression of data_type: " + package.data_type);
            }

            // VAE decompression: decode latent -> denormalize
            auto reconstructed_float = vae->decode(package.latent_data);
            auto denormalized = denormalize_data(reconstructed_float, package.preprocessing_stats);

            // Convert back to original type T
            std::vector<T> reconstructed_data;
            reconstructed_data.reserve(denormalized.size());
            for (const auto& value : denormalized) {
                reconstructed_data.push_back(static_cast<T>(value));
            }

            // Calculate metrics
            metrics.original_bytes = package.original_size * sizeof(T);
            metrics.ratio = static_cast<double>(metrics.original_bytes) / metrics.compressed_bytes;
            metrics.error =
                calculate_reconstruction_error(preprocess_data(denormalized), reconstructed_float);
            metrics.success = true;

            auto end_time = std::chrono::high_resolution_clock::now();
            metrics.decode_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            metrics.encode_time = std::chrono::milliseconds(0);

            core::Logger::info(
                "VAE decompression successful: " + std::to_string(package.latent_data.size()) +
                "D -> " + std::to_string(reconstructed_data.size()) + "D, ratio: " +
                std::to_string(metrics.ratio) + ":1, error: " + std::to_string(metrics.error));

            return Result<std::pair<std::vector<T>, CompressionMetrics>>::success(
                std::make_pair(std::move(reconstructed_data), metrics));
        }
        else {
            // Fallback: try raw data deserialization
            auto deserialize_result = deserialize_data<T>(*retrieve_result);
            if (!deserialize_result) {
                return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
                    "Failed to deserialize both compressed and raw data formats");
            }

            // Raw data metrics
            metrics.original_bytes = retrieve_result->size();
            metrics.ratio = 1.0;
            metrics.error = 0.0;
            metrics.success = true;

            auto end_time = std::chrono::high_resolution_clock::now();
            metrics.decode_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            metrics.encode_time = std::chrono::milliseconds(0);

            return Result<std::pair<std::vector<T>, CompressionMetrics>>::success(
                std::make_pair(std::move(*deserialize_result), metrics));
        }
    }
    catch (const std::exception& e) {
        return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            "VAE decompression failed: " + std::string(e.what()));
    }
    catch (...) {
        return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            "VAE decompression failed: Unknown exception");
    }
}

// Async method implementations
template <typename T>
std::future<Result<AINativeDatabase::CompressionMetrics>> AINativeDatabase::store_async(
    const Key& key, const std::vector<T>& data, const std::string& data_type)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");

    return std::async(std::launch::async,
                      [this, key, data, data_type]() -> Result<CompressionMetrics> {
                          return store(key, data, data_type);
                      });
}

template <typename T>
std::future<Result<std::pair<std::vector<T>, AINativeDatabase::CompressionMetrics>>>
AINativeDatabase::retrieve_async(const Key& key)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");

    return std::async(std::launch::async,
                      [this, key]() -> Result<std::pair<std::vector<T>, CompressionMetrics>> {
                          return retrieve<T>(key);
                      });
}

void AINativeDatabase::update_statistics(const CompressionMetrics& metrics)
{
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_entries++;
    stats_.total_original_bytes += metrics.original_bytes;
    stats_.total_compressed_bytes += metrics.compressed_bytes;

    // Update running averages
    double n = static_cast<double>(stats_.total_entries);
    stats_.average_compression_ratio =
        ((n - 1) * stats_.average_compression_ratio + metrics.ratio) / n;
    stats_.average_reconstruction_error =
        ((n - 1) * stats_.average_reconstruction_error + metrics.error) / n;
}

AINativeDatabase::Statistics AINativeDatabase::get_statistics() const
{
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AINativeDatabase::start_background_optimization()
{
    if (config_.enable_background_optimization) {
        // Use atomic compare_exchange to prevent race condition
        bool expected = false;
        if (optimization_running_.compare_exchange_strong(expected, true)) {
            // Successfully changed from false to true - we're the only thread that succeeded
            optimization_thread_ = std::make_unique<std::thread>(
                &AINativeDatabase::background_optimization_loop, this);
            std::cout << "Background optimization started" << std::endl;
        }
        else {
            // Another thread already started optimization
            std::cout << "Background optimization already running" << std::endl;
        }
    }
}

void AINativeDatabase::stop_background_optimization()
{
    // Ensure only one thread can stop the optimization at a time
    std::lock_guard<std::mutex> lock(optimization_mutex_);

    // Atomically set optimization_running_ to false only if it was true
    bool expected = true;
    if (optimization_running_.compare_exchange_strong(expected, false)) {
        // Successfully changed from true to false - we're the only thread that succeeded
        if (optimization_thread_ && optimization_thread_->joinable()) {
            optimization_thread_->join();
            optimization_thread_.reset();
            std::cout << "Background optimization stopped" << std::endl;
        }
    }
    else {
        // Another thread already stopped optimization or it wasn't running
        std::cout << "Background optimization already stopped or not running" << std::endl;
    }
}

void AINativeDatabase::background_optimization_loop()
{
    while (optimization_running_) {
        std::this_thread::sleep_for(config_.optimization_interval);

        if (!optimization_running_) break;

        // Perform optimization tasks
        auto result = optimize_now();
        if (!result) {
            std::cout << "Background optimization failed: " << result.error << std::endl;
        }
    }
}

Result<void> AINativeDatabase::optimize_now()
{
    std::cout << "Performing database optimization..." << std::endl;

    // Update optimization timestamp
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.last_optimization = std::chrono::steady_clock::now();
    }

    // TODO: Implement optimization strategies:
    // - Retrain VAE models with recent data
    // - Compress frequently accessed data with better models
    // - Clean up temporary data
    // - Defragment LMDB database

    std::cout << "Database optimization completed" << std::endl;
    return Result<void>::success();
}

research::VariationalAutoencoder<float>* AINativeDatabase::get_or_create_vae(
    size_t input_dim, const std::string& data_type)
{
    std::lock_guard<std::mutex> lock(vae_mutex_);

    auto it = vae_models_.find(data_type);
    if (it != vae_models_.end() && it->second) {
        return it->second.get();
    }

    // Use BREAKTHROUGH optimal configurations discovered through Monte Carlo tuning
    research::VAEConfig vae_config;
    std::vector<size_t> hidden_dims;

    // Data-type-specific configuration using proven breakthrough discoveries
    if (data_type == "telemetry" || data_type == "default") {
        // Use BREAKTHROUGH compression config: 4:1 ratio with ~0.96 reconstruction error
        vae_config = research::OptimalConfigs::getCompressionConfig();
        hidden_dims = research::OptimalConfigs::getCompressionArchitecture();

        core::Logger::info("DEBUG: Original compression config latent_dim=" +
                           std::to_string(vae_config.latent_dim));
        core::Logger::info("DEBUG: Input dimension=" + std::to_string(input_dim));

        // Scale latent dimension proportionally for non-12D telemetry
        if (input_dim != 12) {
            vae_config.latent_dim = std::max(size_t(2), input_dim / 4);  // Maintain 4:1 ratio
            core::Logger::info("DEBUG: Scaled latent_dim=" + std::to_string(vae_config.latent_dim));
        }
        else {
            core::Logger::info("DEBUG: Using original latent_dim=" +
                               std::to_string(vae_config.latent_dim) + " for 12D input");
        }

        core::Logger::info(
            "Using BREAKTHROUGH telemetry compression config: " + std::to_string(input_dim) + "D→" +
            std::to_string(vae_config.latent_dim) +
            "D (4:1 ratio, β=" + std::to_string(vae_config.beta) + ")");
    }
    else if (data_type == "anomaly_detection" || data_type == "monitoring" ||
             data_type == "anomaly") {
        // Use VALIDATED anomaly detection config: 2-3x separation with F1=0.69
        vae_config = research::OptimalConfigs::getAnomalyDetectionConfig();
        hidden_dims = research::OptimalConfigs::getAnomalyDetectionArchitecture();

        // Scale latent dimension proportionally for non-12D data
        if (input_dim != 12) {
            vae_config.latent_dim =
                std::min(input_dim / 2, size_t(16));  // Preserve detection capability
        }

        core::Logger::info(
            "Using VALIDATED anomaly detection config: " + std::to_string(input_dim) + "D→" +
            std::to_string(vae_config.latent_dim) + "D (β=" + std::to_string(vae_config.beta) +
            ", 2-3x separation)");
    }
    else if (data_type == "sensors" || data_type.find("sensor") != std::string::npos) {
        // Use high-quality compression for sensor data (better reconstruction)
        vae_config = research::OptimalConfigs::ImprovedConfigs::getHighQualityCompressionConfig();
        hidden_dims =
            research::OptimalConfigs::ImprovedConfigs::getHighQualityCompressionArchitecture();

        // Scale for different input dimensions
        if (input_dim != 12) {
            vae_config.latent_dim = std::max(size_t(3), input_dim / 3);  // ~3:1 ratio for quality
        }

        core::Logger::info(
            "Using HIGH-QUALITY sensor compression config: " + std::to_string(input_dim) + "D→" +
            std::to_string(vae_config.latent_dim) + "D (β=" + std::to_string(vae_config.beta) +
            ", quality-focused)");
    }
    else {
        // Use balanced configuration for unknown data types
        vae_config = research::OptimalConfigs::getBalancedConfig();
        hidden_dims = research::OptimalConfigs::getBalancedArchitecture();

        // Adaptive scaling for different input dimensions
        if (input_dim != 12) {
            vae_config.latent_dim = std::max(size_t(2), std::min(input_dim / 3, size_t(16)));

            // Adjust architecture complexity based on input size
            if (input_dim >= 64) {
                hidden_dims = {128, 64, 32};  // Complex architecture for large inputs
            }
            else if (input_dim >= 32) {
                hidden_dims = {64, 32};  // Moderate architecture
            }
            else {
                hidden_dims = {32};  // Simple architecture for small inputs
            }
        }

        core::Logger::info("Using BALANCED config for '" + data_type +
                           "': " + std::to_string(input_dim) + "D→" +
                           std::to_string(vae_config.latent_dim) +
                           "D (β=" + std::to_string(vae_config.beta) + ")");
    }

    try {
        vae_models_[data_type] = std::make_unique<research::VariationalAutoencoder<float>>(
            input_dim, vae_config.latent_dim, hidden_dims, neural::ProtectionLevel::NONE,
            vae_config);

        core::Logger::info("✅ Created OPTIMAL VAE for '" + data_type +
                           "' using breakthrough config");

        return vae_models_[data_type].get();
    }
    catch (const std::exception& e) {
        core::Logger::error("Failed to create VAE (std::exception): " + std::string(e.what()));
        return nullptr;
    }
    catch (...) {
        core::Logger::error(
            "Failed to create VAE (unknown exception): Non-standard exception caught during VAE "
            "creation for data type '" +
            data_type + "'");
        return nullptr;
    }
}

template <typename T>
Result<void> AINativeDatabase::train_vae(const std::vector<std::vector<T>>& training_data,
                                         const std::string& data_type)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");

    if (training_data.empty()) {
        return Result<void>::failure("Training data cannot be empty");
    }

    size_t input_dim = training_data[0].size();
    auto* vae = get_or_create_vae(input_dim, data_type);
    if (!vae) {
        return Result<void>::failure("Failed to create VAE model");
    }

    // Get the training parameters that match the VAE configuration
    research::VAEConfig training_config;

    // Use the same data-type-specific configurations as get_or_create_vae
    if (data_type == "telemetry" || data_type == "default") {
        // Compression-optimized training parameters
        training_config.epochs = 50;             // Optimal for compression
        training_config.batch_size = 32;         // Efficient batch size
        training_config.learning_rate = 0.001f;  // Stable learning rate
    }
    else if (data_type == "anomaly_detection" || data_type == "monitoring") {
        // Anomaly detection-optimized training parameters
        training_config.epochs = 100;             // More epochs for pattern learning
        training_config.batch_size = 64;          // Larger batches for stability
        training_config.learning_rate = 0.0005f;  // Lower learning rate for precision
    }
    else {
        // Adaptive training parameters for unknown data types
        training_config.epochs = 75;             // Balanced training duration
        training_config.batch_size = 32;         // Standard batch size
        training_config.learning_rate = 0.001f;  // Standard learning rate
    }

    try {
        // Convert training data to float if necessary
        std::vector<std::vector<float>> float_data;
        for (const auto& sample : training_data) {
            std::vector<float> float_sample;
            for (const auto& value : sample) {
                float_sample.push_back(static_cast<float>(value));
            }
            float_data.push_back(float_sample);
        }

        // Apply CONSISTENT preprocessing to training data (same as store/retrieve)
        std::vector<std::vector<float>> preprocessed_training_data;
        preprocessed_training_data.reserve(float_data.size());

        core::Logger::info("Applying z-score preprocessing to training data for consistency...");
        for (const auto& sample : float_data) {
            auto preprocessed_sample = preprocess_data(sample);
            preprocessed_training_data.push_back(preprocessed_sample);
        }

        // Train the VAE using PREPROCESSED data (consistent with store/retrieve)
        vae->train(preprocessed_training_data, training_config.epochs, training_config.batch_size,
                   training_config.learning_rate);

        core::Logger::info("VAE training completed for data type '" + data_type + "' with " +
                           std::to_string(training_config.epochs) + " epochs, " +
                           "batch_size=" + std::to_string(training_config.batch_size) +
                           ", lr=" + std::to_string(training_config.learning_rate));
        return Result<void>::success();
    }
    catch (const std::exception& e) {
        return Result<void>::failure("VAE training failed (std::exception): " +
                                     std::string(e.what()));
    }
    catch (...) {
        return Result<void>::failure(
            "VAE training failed (unknown exception): Non-standard exception caught during "
            "training for data type '" +
            data_type + "'");
    }
}

// VAE integration helper methods implementation
std::vector<float> AINativeDatabase::preprocess_data(const std::vector<float>& data) const
{
    if (data.empty()) return data;

    // Calculate mean
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();

    // Calculate standard deviation
    float variance = 0.0f;
    for (float value : data) {
        variance += (value - mean) * (value - mean);
    }
    variance /= data.size();
    float std_dev = std::sqrt(variance + 1e-8f);  // Add epsilon for numerical stability

    // Z-score normalization: (x - μ) / σ
    std::vector<float> normalized;
    normalized.reserve(data.size());
    for (float value : data) {
        normalized.push_back((value - mean) / std_dev);
    }

    return normalized;
}

std::vector<float> AINativeDatabase::denormalize_data(const std::vector<float>& data,
                                                      const PreprocessingStats& stats) const
{
    if (data.empty() || stats.means.empty() || stats.stds.empty()) return data;

    std::vector<float> denormalized;
    denormalized.reserve(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        // Reverse z-score: x = (normalized * σ) + μ
        size_t stat_idx = i < stats.means.size() ? i : 0;  // Handle size mismatch gracefully
        float denorm_value = (data[i] * stats.stds[stat_idx]) + stats.means[stat_idx];
        denormalized.push_back(denorm_value);
    }

    return denormalized;
}

void AINativeDatabase::calculate_preprocessing_stats(const std::vector<float>& data,
                                                     PreprocessingStats& stats) const
{
    if (data.empty()) return;

    // For simplicity, treat as single-channel data
    // In practice, you might want per-channel statistics for multi-dimensional data
    stats.means.clear();
    stats.stds.clear();

    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    stats.means.push_back(mean);

    float variance = 0.0f;
    for (float value : data) {
        variance += (value - mean) * (value - mean);
    }
    variance /= data.size();
    float std_dev = std::sqrt(variance + 1e-8f);
    stats.stds.push_back(std_dev);
}

float AINativeDatabase::calculate_reconstruction_error(
    const std::vector<float>& original, const std::vector<float>& reconstructed) const
{
    if (original.size() != reconstructed.size()) {
        return std::numeric_limits<float>::max();  // Invalid comparison
    }

    float mse = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        float diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }

    return mse / original.size();
}

std::vector<uint8_t> AINativeDatabase::serialize_compressed_package(
    const CompressedDataPackage& package) const
{
    // Simple binary serialization format:
    // [magic_bytes:4][data_type_len:4][data_type:variable][original_size:8]
    // [latent_size:4][latent_data:variable][stats_means_size:4][means:variable]
    // [stats_stds_size:4][stds:variable]

    std::vector<uint8_t> result;

    // Magic bytes to identify VAE-compressed data
    const uint32_t magic = 0x56414531;  // "VAE1" in hex
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&magic),
                  reinterpret_cast<const uint8_t*>(&magic) + sizeof(magic));

    // Data type
    uint32_t data_type_len = static_cast<uint32_t>(package.data_type.size());
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&data_type_len),
                  reinterpret_cast<const uint8_t*>(&data_type_len) + sizeof(data_type_len));
    result.insert(result.end(), package.data_type.begin(), package.data_type.end());

    // Original size
    uint64_t original_size = static_cast<uint64_t>(package.original_size);
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&original_size),
                  reinterpret_cast<const uint8_t*>(&original_size) + sizeof(original_size));

    // Latent data
    uint32_t latent_size = static_cast<uint32_t>(package.latent_data.size());
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&latent_size),
                  reinterpret_cast<const uint8_t*>(&latent_size) + sizeof(latent_size));
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(package.latent_data.data()),
                  reinterpret_cast<const uint8_t*>(package.latent_data.data()) +
                      package.latent_data.size() * sizeof(float));

    // Preprocessing stats - means
    uint32_t means_size = static_cast<uint32_t>(package.preprocessing_stats.means.size());
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&means_size),
                  reinterpret_cast<const uint8_t*>(&means_size) + sizeof(means_size));
    if (means_size > 0) {
        result.insert(result.end(),
                      reinterpret_cast<const uint8_t*>(package.preprocessing_stats.means.data()),
                      reinterpret_cast<const uint8_t*>(package.preprocessing_stats.means.data()) +
                          means_size * sizeof(float));
    }

    // Preprocessing stats - stds
    uint32_t stds_size = static_cast<uint32_t>(package.preprocessing_stats.stds.size());
    result.insert(result.end(), reinterpret_cast<const uint8_t*>(&stds_size),
                  reinterpret_cast<const uint8_t*>(&stds_size) + sizeof(stds_size));
    if (stds_size > 0) {
        result.insert(result.end(),
                      reinterpret_cast<const uint8_t*>(package.preprocessing_stats.stds.data()),
                      reinterpret_cast<const uint8_t*>(package.preprocessing_stats.stds.data()) +
                          stds_size * sizeof(float));
    }

    return result;
}

Result<AINativeDatabase::CompressedDataPackage> AINativeDatabase::deserialize_compressed_package(
    const std::vector<uint8_t>& data) const
{
    if (data.size() < sizeof(uint32_t)) {
        return Result<CompressedDataPackage>::failure("Data too small to contain magic bytes");
    }

    size_t offset = 0;

    // Check magic bytes
    uint32_t magic;
    std::memcpy(&magic, data.data() + offset, sizeof(magic));
    offset += sizeof(magic);

    if (magic != 0x56414531) {
        return Result<CompressedDataPackage>::failure(
            "Invalid magic bytes - not VAE compressed data");
    }

    CompressedDataPackage package;

    try {
        // Data type
        uint32_t data_type_len;
        std::memcpy(&data_type_len, data.data() + offset, sizeof(data_type_len));
        offset += sizeof(data_type_len);

        package.data_type.resize(data_type_len);
        std::memcpy(&package.data_type[0], data.data() + offset, data_type_len);
        offset += data_type_len;

        // Original size
        uint64_t original_size;
        std::memcpy(&original_size, data.data() + offset, sizeof(original_size));
        package.original_size = static_cast<size_t>(original_size);
        offset += sizeof(original_size);

        // Latent data
        uint32_t latent_size;
        std::memcpy(&latent_size, data.data() + offset, sizeof(latent_size));
        offset += sizeof(latent_size);

        package.latent_data.resize(latent_size);
        std::memcpy(package.latent_data.data(), data.data() + offset, latent_size * sizeof(float));
        offset += latent_size * sizeof(float);

        // Preprocessing stats - means
        uint32_t means_size;
        std::memcpy(&means_size, data.data() + offset, sizeof(means_size));
        offset += sizeof(means_size);

        if (means_size > 0) {
            package.preprocessing_stats.means.resize(means_size);
            std::memcpy(package.preprocessing_stats.means.data(), data.data() + offset,
                        means_size * sizeof(float));
            offset += means_size * sizeof(float);
        }

        // Preprocessing stats - stds
        uint32_t stds_size;
        std::memcpy(&stds_size, data.data() + offset, sizeof(stds_size));
        offset += sizeof(stds_size);

        if (stds_size > 0) {
            package.preprocessing_stats.stds.resize(stds_size);
            std::memcpy(package.preprocessing_stats.stds.data(), data.data() + offset,
                        stds_size * sizeof(float));
            offset += stds_size * sizeof(float);
        }

        return Result<CompressedDataPackage>::success(std::move(package));
    }
    catch (const std::exception& e) {
        return Result<CompressedDataPackage>::failure("Deserialization failed: " +
                                                      std::string(e.what()));
    }
}

std::string AINativeDatabase::lmdb_error_string(int error_code) const
{
    return std::string(mdb_strerror(error_code));
}

// Template instantiations for common types
template Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store<float>(
    const Key&, const std::vector<float>&, const std::string&);
template Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store<double>(
    const Key&, const std::vector<double>&, const std::string&);
template Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store<int>(
    const Key&, const std::vector<int>&, const std::string&);
template Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store<int64_t>(
    const Key&, const std::vector<int64_t>&, const std::string&);

template Result<std::pair<std::vector<float>, AINativeDatabase::CompressionMetrics>>
AINativeDatabase::retrieve<float>(const Key&);
template Result<std::pair<std::vector<double>, AINativeDatabase::CompressionMetrics>>
AINativeDatabase::retrieve<double>(const Key&);
template Result<std::pair<std::vector<int>, AINativeDatabase::CompressionMetrics>>
AINativeDatabase::retrieve<int>(const Key&);
template Result<std::pair<std::vector<int64_t>, AINativeDatabase::CompressionMetrics>>
AINativeDatabase::retrieve<int64_t>(const Key&);

// Async template instantiations
template std::future<Result<AINativeDatabase::CompressionMetrics>>
AINativeDatabase::store_async<float>(const Key&, const std::vector<float>&, const std::string&);
template std::future<Result<AINativeDatabase::CompressionMetrics>>
AINativeDatabase::store_async<double>(const Key&, const std::vector<double>&, const std::string&);
template std::future<Result<AINativeDatabase::CompressionMetrics>>
AINativeDatabase::store_async<int>(const Key&, const std::vector<int>&, const std::string&);

template std::future<Result<std::pair<std::vector<float>, AINativeDatabase::CompressionMetrics>>>
AINativeDatabase::retrieve_async<float>(const Key&);
template std::future<Result<std::pair<std::vector<double>, AINativeDatabase::CompressionMetrics>>>
AINativeDatabase::retrieve_async<double>(const Key&);
template std::future<Result<std::pair<std::vector<int>, AINativeDatabase::CompressionMetrics>>>
AINativeDatabase::retrieve_async<int>(const Key&);

// Train VAE template instantiations
template Result<void> AINativeDatabase::train_vae<float>(const std::vector<std::vector<float>>&,
                                                         const std::string&);
template Result<void> AINativeDatabase::train_vae<double>(const std::vector<std::vector<double>>&,
                                                          const std::string&);
template Result<void> AINativeDatabase::train_vae<int>(const std::vector<std::vector<int>>&,
                                                       const std::string&);

}  // namespace rad_ml::storage
