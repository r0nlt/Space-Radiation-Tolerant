/**
 * @file ai_native_database.cpp
 * @brief Implementation of AI-Native Database for Datacenter Applications
 */

#include "rad_ml/storage/ai_native_database.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "rad_ml/core/logger.hpp"

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

    // Initialize VAE models for each data type
    std::lock_guard<std::mutex> vae_lock(vae_mutex_);
    for (const auto& [data_type, dimension] : data_dimensions) {
        try {
            size_t latent_dim = std::min(config_.default_latent_dim, dimension / 2);

            // Create a simple VAE placeholder for now
            // In a real implementation, this would use the actual VAE from your framework
            vae_models_[data_type] = nullptr;  // Placeholder for now

            std::cout << "Initialized VAE for data type '" << data_type << "' with input dimension "
                      << dimension << " and latent dimension " << latent_dim << std::endl;
        }
        catch (const std::exception& e) {
            return Result<void>::failure("Failed to initialize VAE for " + data_type + ": " +
                                         e.what());
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

// Simplified store/retrieve implementations without VAE compression for now
template <typename T>
Result<AINativeDatabase::CompressionMetrics> AINativeDatabase::store(
    const Key& key, const std::vector<T>& data, const std::string& /*data_type*/)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    // Serialize data
    auto serialized = serialize_data(data);

    // Store raw data (without compression for now)
    auto store_result = store_raw(key, serialized);
    if (!store_result) {
        return Result<CompressionMetrics>::failure(store_result.error);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    CompressionMetrics metrics;
    metrics.original_bytes = serialized.size();
    metrics.compressed_bytes = serialized.size();  // No compression for now
    metrics.ratio = 1.0;                           // No compression
    metrics.error = 0.0;                           // No loss
    metrics.encode_time = encode_time;
    metrics.decode_time = std::chrono::milliseconds(0);
    metrics.success = true;

    update_statistics(metrics);

    return Result<CompressionMetrics>::success(metrics);
}

template <typename T>
Result<std::pair<std::vector<T>, AINativeDatabase::CompressionMetrics>> AINativeDatabase::retrieve(
    const Key& key)
{
    static_assert(is_storable_data_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    // Retrieve raw data
    auto retrieve_result = retrieve_raw(key);
    if (!retrieve_result) {
        return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            retrieve_result.error);
    }

    // Deserialize data
    auto deserialize_result = deserialize_data<T>(*retrieve_result);
    if (!deserialize_result) {
        return Result<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            deserialize_result.error);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    CompressionMetrics metrics;
    metrics.original_bytes = retrieve_result->size();
    metrics.compressed_bytes = retrieve_result->size();
    metrics.ratio = 1.0;
    metrics.error = 0.0;
    metrics.encode_time = std::chrono::milliseconds(0);
    metrics.decode_time = decode_time;
    metrics.success = true;

    return Result<std::pair<std::vector<T>, CompressionMetrics>>::success(
        std::make_pair(std::move(*deserialize_result), metrics));
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

}  // namespace rad_ml::storage
