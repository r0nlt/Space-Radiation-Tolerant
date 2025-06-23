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

namespace rad_ml::storage {

/**
 * @brief Simple result type for operations that can fail
 */
template <typename T>
struct SimpleResult {
    std::optional<T> value;
    std::string error;

    explicit operator bool() const noexcept { return value.has_value(); }
    T& operator*() { return *value; }
    const T& operator*() const { return *value; }

    static SimpleResult success(T val) { return SimpleResult{std::move(val), ""}; }
    static SimpleResult failure(std::string err)
    {
        return SimpleResult{std::nullopt, std::move(err)};
    }
};

// Specialization for void
template <>
struct SimpleResult<void> {
    bool is_success;
    std::string error;

    explicit operator bool() const noexcept { return is_success; }

    static SimpleResult success() { return SimpleResult{true, ""}; }
    static SimpleResult failure(std::string err) { return SimpleResult{false, std::move(err)}; }
};

/**
 * @brief Simplified AI-Native Database for testing
 */
class SimpleAINativeDatabase {
   public:
    using Key = std::string;
    using CompressionRatio = double;
    using ReconstructionError = double;

    struct Config {
        std::filesystem::path db_path{"./simple_ai_db"};
        size_t max_db_size = 100 * 1024 * 1024;  // 100MB
    };

    struct CompressionMetrics {
        CompressionRatio ratio = 1.0;
        ReconstructionError error = 0.0;
        std::chrono::milliseconds encode_time{0};
        std::chrono::milliseconds decode_time{0};
        size_t original_bytes = 0;
        size_t compressed_bytes = 0;
        bool success = false;
    };

    struct Statistics {
        size_t total_entries = 0;
        size_t total_original_bytes = 0;
        size_t total_compressed_bytes = 0;
        CompressionRatio average_compression_ratio = 0.0;
        ReconstructionError average_reconstruction_error = 0.0;
    };

    // Type trait for storable data
    template <typename T>
    static constexpr bool is_storable_v =
        std::is_arithmetic_v<T> && std::is_trivially_copyable_v<T>;

    explicit SimpleAINativeDatabase(Config config);
    SimpleAINativeDatabase();  // Default constructor
    ~SimpleAINativeDatabase();

    // Delete copy operations
    SimpleAINativeDatabase(const SimpleAINativeDatabase&) = delete;
    SimpleAINativeDatabase& operator=(const SimpleAINativeDatabase&) = delete;

    // Enable move operations
    SimpleAINativeDatabase(SimpleAINativeDatabase&&) noexcept;
    SimpleAINativeDatabase& operator=(SimpleAINativeDatabase&&) noexcept;

    SimpleResult<void> initialize();

    template <typename T>
    SimpleResult<CompressionMetrics> store(const Key& key, const std::vector<T>& data);

    template <typename T>
    SimpleResult<std::pair<std::vector<T>, CompressionMetrics>> retrieve(const Key& key);

    bool contains(const Key& key) const noexcept;
    SimpleResult<void> remove(const Key& key);
    std::vector<Key> keys() const;
    Statistics get_statistics() const;

   private:
    struct LMDBEnv {
        MDB_env* env = nullptr;
        MDB_dbi dbi = 0;

        LMDBEnv() = default;
        ~LMDBEnv();
        LMDBEnv(const LMDBEnv&) = delete;
        LMDBEnv& operator=(const LMDBEnv&) = delete;
        LMDBEnv(LMDBEnv&&) noexcept;
        LMDBEnv& operator=(LMDBEnv&&) noexcept;
    };

    Config config_;
    std::unique_ptr<LMDBEnv> lmdb_;
    mutable std::mutex data_mutex_;
    mutable std::mutex stats_mutex_;
    mutable Statistics stats_;

    SimpleResult<void> initialize_lmdb();
    SimpleResult<void> store_raw(const Key& key, const std::vector<uint8_t>& data);
    SimpleResult<std::vector<uint8_t>> retrieve_raw(const Key& key) const;

    template <typename T>
    std::vector<uint8_t> serialize_data(const std::vector<T>& data) const;

    template <typename T>
    SimpleResult<std::vector<T>> deserialize_data(const std::vector<uint8_t>& data) const;

    void update_statistics(const CompressionMetrics& metrics);
    std::string lmdb_error_string(int error_code) const;
};

}  // namespace rad_ml::storage
