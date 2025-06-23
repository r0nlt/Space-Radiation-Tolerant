/**
 * @file simple_ai_native_database.cpp
 * @brief Simplified implementation of AI-Native Database
 */

#include <cstring>
#include <iostream>

#include "rad_ml/storage/ai_native_database_simple.hpp"

namespace rad_ml::storage {

// LMDBEnv implementation
SimpleAINativeDatabase::LMDBEnv::~LMDBEnv()
{
    if (env) {
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        std::cout << "LMDB environment closed cleanly" << std::endl;
    }
}

SimpleAINativeDatabase::LMDBEnv::LMDBEnv(LMDBEnv&& other) noexcept : env(other.env), dbi(other.dbi)
{
    other.env = nullptr;
    other.dbi = 0;
}

SimpleAINativeDatabase::LMDBEnv& SimpleAINativeDatabase::LMDBEnv::operator=(
    LMDBEnv&& other) noexcept
{
    if (this != &other) {
        if (env) {
            mdb_dbi_close(env, dbi);
            mdb_env_close(env);
        }
        env = other.env;
        dbi = other.dbi;
        other.env = nullptr;
        other.dbi = 0;
    }
    return *this;
}

// SimpleAINativeDatabase implementation
SimpleAINativeDatabase::SimpleAINativeDatabase(Config config)
    : config_(std::move(config)), lmdb_(std::make_unique<LMDBEnv>())
{
}

SimpleAINativeDatabase::SimpleAINativeDatabase() : SimpleAINativeDatabase(Config{})
{
    std::cout << "Simple AI-Native Database created with path: " << config_.db_path.string()
              << std::endl;
}

SimpleAINativeDatabase::~SimpleAINativeDatabase()
{
    std::cout << "Simple AI-Native Database destroyed" << std::endl;
}

SimpleAINativeDatabase::SimpleAINativeDatabase(SimpleAINativeDatabase&& other) noexcept
    : config_(std::move(other.config_)), lmdb_(std::move(other.lmdb_)), stats_(other.stats_)
{
}

SimpleAINativeDatabase& SimpleAINativeDatabase::operator=(SimpleAINativeDatabase&& other) noexcept
{
    if (this != &other) {
        config_ = std::move(other.config_);
        lmdb_ = std::move(other.lmdb_);
        stats_ = other.stats_;
    }
    return *this;
}

SimpleResult<void> SimpleAINativeDatabase::initialize()
{
    std::cout << "Initializing Simple AI-Native Database..." << std::endl;
    return initialize_lmdb();
}

SimpleResult<void> SimpleAINativeDatabase::initialize_lmdb()
{
    std::error_code ec;
    std::filesystem::create_directories(config_.db_path, ec);
    if (ec) {
        return SimpleResult<void>::failure("Failed to create database directory: " + ec.message());
    }

    int rc = mdb_env_create(&lmdb_->env);
    if (rc != 0) {
        return SimpleResult<void>::failure("Failed to create LMDB environment: " +
                                           lmdb_error_string(rc));
    }

    rc = mdb_env_set_mapsize(lmdb_->env, config_.max_db_size);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return SimpleResult<void>::failure("Failed to set LMDB map size: " + lmdb_error_string(rc));
    }

    rc = mdb_env_open(lmdb_->env, config_.db_path.c_str(), 0, 0664);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return SimpleResult<void>::failure("Failed to open LMDB environment: " +
                                           lmdb_error_string(rc));
    }

    MDB_txn* txn;
    rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return SimpleResult<void>::failure("Failed to begin transaction: " + lmdb_error_string(rc));
    }

    rc = mdb_dbi_open(txn, nullptr, 0, &lmdb_->dbi);
    if (rc != 0) {
        mdb_txn_abort(txn);
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return SimpleResult<void>::failure("Failed to open database: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        mdb_env_close(lmdb_->env);
        lmdb_->env = nullptr;
        return SimpleResult<void>::failure("Failed to commit transaction: " +
                                           lmdb_error_string(rc));
    }

    std::cout << "LMDB initialized successfully at: " << config_.db_path.string() << std::endl;
    return SimpleResult<void>::success();
}

bool SimpleAINativeDatabase::contains(const Key& key) const noexcept
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

SimpleResult<void> SimpleAINativeDatabase::store_raw(const Key& key,
                                                     const std::vector<uint8_t>& data)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return SimpleResult<void>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        return SimpleResult<void>::failure("Failed to begin write transaction: " +
                                           lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
    MDB_val mdb_value{data.size(), const_cast<uint8_t*>(data.data())};

    rc = mdb_put(txn, lmdb_->dbi, &mdb_key, &mdb_value, 0);
    if (rc != 0) {
        mdb_txn_abort(txn);
        return SimpleResult<void>::failure("Failed to store data: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        return SimpleResult<void>::failure("Failed to commit write transaction: " +
                                           lmdb_error_string(rc));
    }

    return SimpleResult<void>::success();
}

SimpleResult<std::vector<uint8_t>> SimpleAINativeDatabase::retrieve_raw(const Key& key) const
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return SimpleResult<std::vector<uint8_t>>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) {
        return SimpleResult<std::vector<uint8_t>>::failure("Failed to begin read transaction: " +
                                                           lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
    MDB_val mdb_value;

    rc = mdb_get(txn, lmdb_->dbi, &mdb_key, &mdb_value);
    if (rc == MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return SimpleResult<std::vector<uint8_t>>::failure("Key not found: " + key);
    }
    else if (rc != 0) {
        mdb_txn_abort(txn);
        return SimpleResult<std::vector<uint8_t>>::failure("Failed to retrieve data: " +
                                                           lmdb_error_string(rc));
    }

    std::vector<uint8_t> result(static_cast<uint8_t*>(mdb_value.mv_data),
                                static_cast<uint8_t*>(mdb_value.mv_data) + mdb_value.mv_size);
    mdb_txn_abort(txn);

    return SimpleResult<std::vector<uint8_t>>::success(std::move(result));
}

SimpleResult<void> SimpleAINativeDatabase::remove(const Key& key)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!lmdb_->env) {
        return SimpleResult<void>::failure("Database not initialized");
    }

    MDB_txn* txn;
    int rc = mdb_txn_begin(lmdb_->env, nullptr, 0, &txn);
    if (rc != 0) {
        return SimpleResult<void>::failure("Failed to begin write transaction: " +
                                           lmdb_error_string(rc));
    }

    MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};

    rc = mdb_del(txn, lmdb_->dbi, &mdb_key, nullptr);
    if (rc == MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return SimpleResult<void>::failure("Key not found: " + key);
    }
    else if (rc != 0) {
        mdb_txn_abort(txn);
        return SimpleResult<void>::failure("Failed to delete data: " + lmdb_error_string(rc));
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) {
        return SimpleResult<void>::failure("Failed to commit delete transaction: " +
                                           lmdb_error_string(rc));
    }

    return SimpleResult<void>::success();
}

std::vector<SimpleAINativeDatabase::Key> SimpleAINativeDatabase::keys() const
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
std::vector<uint8_t> SimpleAINativeDatabase::serialize_data(const std::vector<T>& data) const
{
    std::vector<uint8_t> result(sizeof(T) * data.size());
    std::memcpy(result.data(), data.data(), result.size());
    return result;
}

template <typename T>
SimpleResult<std::vector<T>> SimpleAINativeDatabase::deserialize_data(
    const std::vector<uint8_t>& data) const
{
    if (data.size() % sizeof(T) != 0) {
        return SimpleResult<std::vector<T>>::failure("Invalid data size for deserialization");
    }

    size_t element_count = data.size() / sizeof(T);
    std::vector<T> result(element_count);
    std::memcpy(result.data(), data.data(), data.size());

    return SimpleResult<std::vector<T>>::success(std::move(result));
}

template <typename T>
SimpleResult<SimpleAINativeDatabase::CompressionMetrics> SimpleAINativeDatabase::store(
    const Key& key, const std::vector<T>& data)
{
    static_assert(is_storable_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto serialized = serialize_data(data);
    auto store_result = store_raw(key, serialized);
    if (!store_result) {
        return SimpleResult<CompressionMetrics>::failure(store_result.error);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    CompressionMetrics metrics;
    metrics.original_bytes = serialized.size();
    metrics.compressed_bytes = serialized.size();
    metrics.ratio = 1.0;
    metrics.error = 0.0;
    metrics.encode_time = encode_time;
    metrics.success = true;

    update_statistics(metrics);

    return SimpleResult<CompressionMetrics>::success(metrics);
}

template <typename T>
SimpleResult<std::pair<std::vector<T>, SimpleAINativeDatabase::CompressionMetrics>>
SimpleAINativeDatabase::retrieve(const Key& key)
{
    static_assert(is_storable_v<T>, "Type must be arithmetic and trivially copyable");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto retrieve_result = retrieve_raw(key);
    if (!retrieve_result) {
        return SimpleResult<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            retrieve_result.error);
    }

    auto deserialize_result = deserialize_data<T>(*retrieve_result);
    if (!deserialize_result) {
        return SimpleResult<std::pair<std::vector<T>, CompressionMetrics>>::failure(
            deserialize_result.error);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    CompressionMetrics metrics;
    metrics.original_bytes = (*retrieve_result).size();
    metrics.compressed_bytes = (*retrieve_result).size();
    metrics.ratio = 1.0;
    metrics.error = 0.0;
    metrics.decode_time = decode_time;
    metrics.success = true;

    return SimpleResult<std::pair<std::vector<T>, CompressionMetrics>>::success(
        std::make_pair(std::move(*deserialize_result), metrics));
}

void SimpleAINativeDatabase::update_statistics(const CompressionMetrics& metrics)
{
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_entries++;
    stats_.total_original_bytes += metrics.original_bytes;
    stats_.total_compressed_bytes += metrics.compressed_bytes;

    double n = static_cast<double>(stats_.total_entries);
    stats_.average_compression_ratio =
        ((n - 1) * stats_.average_compression_ratio + metrics.ratio) / n;
    stats_.average_reconstruction_error =
        ((n - 1) * stats_.average_reconstruction_error + metrics.error) / n;
}

SimpleAINativeDatabase::Statistics SimpleAINativeDatabase::get_statistics() const
{
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::string SimpleAINativeDatabase::lmdb_error_string(int error_code) const
{
    return std::string(mdb_strerror(error_code));
}

// Template instantiations
template SimpleResult<SimpleAINativeDatabase::CompressionMetrics>
SimpleAINativeDatabase::store<float>(const Key&, const std::vector<float>&);

template SimpleResult<SimpleAINativeDatabase::CompressionMetrics>
SimpleAINativeDatabase::store<double>(const Key&, const std::vector<double>&);

template SimpleResult<SimpleAINativeDatabase::CompressionMetrics>
SimpleAINativeDatabase::store<int>(const Key&, const std::vector<int>&);

template SimpleResult<std::pair<std::vector<float>, SimpleAINativeDatabase::CompressionMetrics>>
SimpleAINativeDatabase::retrieve<float>(const Key&);

template SimpleResult<std::pair<std::vector<double>, SimpleAINativeDatabase::CompressionMetrics>>
SimpleAINativeDatabase::retrieve<double>(const Key&);

template SimpleResult<std::pair<std::vector<int>, SimpleAINativeDatabase::CompressionMetrics>>
SimpleAINativeDatabase::retrieve<int>(const Key&);

}  // namespace rad_ml::storage
