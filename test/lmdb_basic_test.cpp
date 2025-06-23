/**
 * @file lmdb_basic_test.cpp
 * @brief Basic test to verify LMDB installation and modern C++ wrapper
 */

#include <lmdb.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief Modern C++ RAII wrapper for LMDB
 */
class LMDBWrapper {
   public:
    /**
     * @brief Constructor - opens or creates database
     * @param db_path Path to database directory
     * @param max_size Maximum database size in bytes
     */
    explicit LMDBWrapper(const std::string& db_path,
                         size_t max_size = 1024 * 1024 * 1024)  // 1GB default
        : db_path_(db_path)
    {
        // Create directory if it doesn't exist
        if (!fs::exists(db_path_)) {
            fs::create_directories(db_path_);
        }

        // Create environment
        int rc = mdb_env_create(&env_);
        if (rc != 0) {
            throw std::runtime_error("Failed to create LMDB environment: " +
                                     std::string(mdb_strerror(rc)));
        }

        // Set map size
        rc = mdb_env_set_mapsize(env_, max_size);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to set LMDB map size: " +
                                     std::string(mdb_strerror(rc)));
        }

        // Open environment
        rc = mdb_env_open(env_, db_path_.c_str(), 0, 0664);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to open LMDB environment: " +
                                     std::string(mdb_strerror(rc)));
        }

        // Open database
        MDB_txn* txn;
        rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to begin transaction: " +
                                     std::string(mdb_strerror(rc)));
        }

        rc = mdb_dbi_open(txn, nullptr, 0, &dbi_);
        if (rc != 0) {
            mdb_txn_abort(txn);
            mdb_env_close(env_);
            throw std::runtime_error("Failed to open database: " + std::string(mdb_strerror(rc)));
        }

        rc = mdb_txn_commit(txn);
        if (rc != 0) {
            mdb_env_close(env_);
            throw std::runtime_error("Failed to commit transaction: " +
                                     std::string(mdb_strerror(rc)));
        }

        std::cout << "✓ LMDB database opened successfully at: " << db_path_ << std::endl;
    }

    /**
     * @brief Destructor - properly closes database
     */
    ~LMDBWrapper()
    {
        if (env_) {
            mdb_dbi_close(env_, dbi_);
            mdb_env_close(env_);
            std::cout << "✓ LMDB database closed cleanly" << std::endl;
        }
    }

    // Delete copy constructor and assignment operator (RAII)
    LMDBWrapper(const LMDBWrapper&) = delete;
    LMDBWrapper& operator=(const LMDBWrapper&) = delete;

    // Move constructor and assignment operator
    LMDBWrapper(LMDBWrapper&& other) noexcept
        : env_(other.env_), dbi_(other.dbi_), db_path_(std::move(other.db_path_))
    {
        other.env_ = nullptr;
        other.dbi_ = 0;
    }

    LMDBWrapper& operator=(LMDBWrapper&& other) noexcept
    {
        if (this != &other) {
            // Clean up current resources
            if (env_) {
                mdb_dbi_close(env_, dbi_);
                mdb_env_close(env_);
            }

            // Move resources
            env_ = other.env_;
            dbi_ = other.dbi_;
            db_path_ = std::move(other.db_path_);

            // Clear other object
            other.env_ = nullptr;
            other.dbi_ = 0;
        }
        return *this;
    }

    /**
     * @brief Store a key-value pair
     * @param key The key
     * @param value The value
     * @return true if successful
     */
    bool put(const std::string& key, const std::string& value)
    {
        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            std::cout << "Failed to begin write transaction: " << mdb_strerror(rc) << std::endl;
            return false;
        }

        MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
        MDB_val mdb_value{value.size(), const_cast<char*>(value.data())};

        rc = mdb_put(txn, dbi_, &mdb_key, &mdb_value, 0);
        if (rc != 0) {
            std::cout << "Failed to put data: " << mdb_strerror(rc) << std::endl;
            mdb_txn_abort(txn);
            return false;
        }

        rc = mdb_txn_commit(txn);
        if (rc != 0) {
            std::cout << "Failed to commit write transaction: " << mdb_strerror(rc) << std::endl;
            return false;
        }

        return true;
    }

    /**
     * @brief Retrieve a value by key
     * @param key The key
     * @return The value, or empty string if not found
     */
    std::string get(const std::string& key)
    {
        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            std::cout << "Failed to begin read transaction: " << mdb_strerror(rc) << std::endl;
            return "";
        }

        MDB_val mdb_key{key.size(), const_cast<char*>(key.data())};
        MDB_val mdb_value;

        rc = mdb_get(txn, dbi_, &mdb_key, &mdb_value);
        if (rc == MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            return "";
        }
        else if (rc != 0) {
            std::cout << "Failed to get data: " << mdb_strerror(rc) << std::endl;
            mdb_txn_abort(txn);
            return "";
        }

        std::string result(static_cast<char*>(mdb_value.mv_data), mdb_value.mv_size);
        mdb_txn_abort(txn);

        return result;
    }

    /**
     * @brief Get database statistics
     */
    void printStats()
    {
        MDB_stat stat;
        MDB_txn* txn;

        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc == 0) {
            rc = mdb_stat(txn, dbi_, &stat);
            if (rc == 0) {
                std::cout << "Database Statistics:" << std::endl;
                std::cout << "  Entries: " << stat.ms_entries << std::endl;
                std::cout << "  Page size: " << stat.ms_psize << " bytes" << std::endl;
                std::cout << "  Tree depth: " << stat.ms_depth << std::endl;
            }
            mdb_txn_abort(txn);
        }
    }

   private:
    MDB_env* env_ = nullptr;
    MDB_dbi dbi_ = 0;
    std::string db_path_;
};

/**
 * @brief Test basic LMDB operations
 */
void testBasicOperations()
{
    std::cout << "\n=== Testing Basic LMDB Operations ===" << std::endl;

    try {
        // Create database in temporary directory
        const std::string test_db_path = "./test_lmdb_db";

        // Clean up any existing test database
        if (fs::exists(test_db_path)) {
            fs::remove_all(test_db_path);
        }

        {
            LMDBWrapper db(test_db_path);

            // Test storing data
            std::cout << "Testing data storage..." << std::endl;
            bool success = db.put("test_key_1", "Hello, LMDB!");
            if (success) {
                std::cout << "✓ Successfully stored key-value pair" << std::endl;
            }
            else {
                std::cout << "✗ Failed to store key-value pair" << std::endl;
                return;
            }

            // Test retrieving data
            std::cout << "Testing data retrieval..." << std::endl;
            std::string value = db.get("test_key_1");
            if (value == "Hello, LMDB!") {
                std::cout << "✓ Successfully retrieved value: " << value << std::endl;
            }
            else {
                std::cout << "✗ Failed to retrieve correct value. Got: " << value << std::endl;
                return;
            }

            // Test multiple entries
            std::cout << "Testing multiple entries..." << std::endl;
            std::vector<std::pair<std::string, std::string>> test_data = {
                {"sensor_1", "temperature: 23.5°C"},
                {"sensor_2", "pressure: 1013.25 hPa"},
                {"sensor_3", "humidity: 45%"},
                {"config", "datacenter_mode: enabled"}};

            for (const auto& [key, val] : test_data) {
                if (!db.put(key, val)) {
                    std::cout << "✗ Failed to store " << key << std::endl;
                    return;
                }
            }

            // Verify all entries
            for (const auto& [key, expected_val] : test_data) {
                std::string retrieved_val = db.get(key);
                if (retrieved_val == expected_val) {
                    std::cout << "✓ " << key << " -> " << retrieved_val << std::endl;
                }
                else {
                    std::cout << "✗ " << key << " mismatch. Expected: " << expected_val
                              << ", Got: " << retrieved_val << std::endl;
                    return;
                }
            }

            // Print database statistics
            db.printStats();
        }

        // Test persistence (database should survive restart)
        std::cout << "\nTesting persistence..." << std::endl;
        {
            LMDBWrapper db2(test_db_path);
            std::string persisted_value = db2.get("test_key_1");
            if (persisted_value == "Hello, LMDB!") {
                std::cout << "✓ Data persisted correctly across database restart" << std::endl;
            }
            else {
                std::cout << "✗ Data not persisted. Got: " << persisted_value << std::endl;
                return;
            }
        }

        std::cout << "\n✓ All LMDB basic tests passed!" << std::endl;

        // Clean up test database
        if (fs::exists(test_db_path)) {
            fs::remove_all(test_db_path);
            std::cout << "✓ Test database cleaned up" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
    }
}

int main()
{
    std::cout << "LMDB Basic Test - Modern C++ Implementation" << std::endl;
    std::cout << "===========================================" << std::endl;

    // Check LMDB version
    std::cout << "LMDB Version: " << MDB_VERSION_STRING << std::endl;

    testBasicOperations();

    return 0;
}
