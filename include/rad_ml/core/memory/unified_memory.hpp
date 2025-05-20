#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "../../error/error_handling.hpp"

namespace rad_ml {
namespace memory {

/**
 * @brief Memory allocation tracking information
 */
struct MemoryAllocationInfo {
    void* ptr;                                              ///< Memory address
    size_t size;                                            ///< Size in bytes
    std::chrono::steady_clock::time_point allocation_time;  ///< When allocation occurred
    std::string location;                                   ///< Source location of allocation
    std::string type_name;                                  ///< Type of allocated object if known
    bool is_array;                                          ///< Whether this is an array allocation
    std::atomic<bool> is_protected;                         ///< Whether this memory is protected

    MemoryAllocationInfo(void* ptr, size_t size, std::string location, std::string type_name = "",
                         bool is_array = false)
        : ptr(ptr),
          size(size),
          allocation_time(std::chrono::steady_clock::now()),
          location(std::move(location)),
          type_name(std::move(type_name)),
          is_array(is_array),
          is_protected(false)
    {
    }
};

/**
 * @brief Memory allocation statistics
 */
struct MemoryStats {
    size_t current_allocations = 0;    ///< Number of current allocations
    size_t peak_allocations = 0;       ///< Peak number of allocations
    size_t total_allocations = 0;      ///< Total number of allocations
    size_t total_deallocations = 0;    ///< Total number of deallocations
    size_t current_bytes = 0;          ///< Current allocated bytes
    size_t peak_bytes = 0;             ///< Peak allocated bytes
    size_t protected_allocations = 0;  ///< Number of protected allocations
    size_t protected_bytes = 0;        ///< Number of protected bytes

    // Memory error tracking
    size_t detected_corruption = 0;  ///< Number of detected memory corruptions
    size_t repaired_corruption = 0;  ///< Number of repaired memory corruptions
    size_t leaked_allocations = 0;   ///< Number of leaked allocations
};

/**
 * @brief Memory protection level
 */
enum class MemoryProtectionLevel {
    NONE,    ///< No protection
    CANARY,  ///< Canary values to detect overflow/underflow
    CRC,     ///< Checksum to detect corruption
    ECC,     ///< Error correcting code
    TMR      ///< Triple modular redundancy
};

/**
 * @brief Memory allocation flags
 */
enum class MemoryFlags {
    DEFAULT = 0,           ///< Default allocation
    ZERO_INITIALIZED = 1,  ///< Zero-initialize memory
    ALIGNED = 2,           ///< Aligned allocation
    FAULT_TOLERANT = 4,    ///< Fault-tolerant allocation
    NO_THROW = 8           ///< Don't throw exceptions
};

inline MemoryFlags operator|(MemoryFlags a, MemoryFlags b)
{
    return static_cast<MemoryFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(MemoryFlags a, MemoryFlags b)
{
    return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

/**
 * @brief Unified memory manager for radiation-tolerant allocations
 *
 * This class provides:
 * - Memory allocation tracking
 * - Error detection and correction
 * - Memory statistics
 * - Memory protection options
 */
class UnifiedMemoryManager {
   public:
    /**
     * @brief Get the singleton instance
     *
     * @return Reference to the singleton instance
     */
    static UnifiedMemoryManager& getInstance()
    {
        static UnifiedMemoryManager instance;
        return instance;
    }

    /**
     * @brief Allocate memory with protection
     *
     * @param size Size in bytes
     * @param flags Allocation flags
     * @param protection_level Protection level
     * @param location Source location (for debugging)
     * @param alignment Alignment for aligned allocations (must be power of 2)
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size, MemoryFlags flags = MemoryFlags::DEFAULT,
                   MemoryProtectionLevel protection_level = MemoryProtectionLevel::NONE,
                   const std::string& location = "unknown", size_t alignment = 64)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Adjust size for protection if needed
        size_t adjusted_size = size;
        if (protection_level != MemoryProtectionLevel::NONE) {
            adjusted_size = calculateProtectedSize(size, protection_level);
        }

        // Perform allocation
        void* ptr = nullptr;
        try {
            if (flags & MemoryFlags::ALIGNED) {
                // Round up adjusted_size to multiple of alignment
                // std::aligned_alloc requires size to be a multiple of alignment
                size_t aligned_size = ((adjusted_size + alignment - 1) / alignment) * alignment;
                ptr = std::aligned_alloc(alignment, aligned_size);
            }
            else {
                ptr = std::malloc(adjusted_size);
            }

            if (!ptr) {
                if (flags & MemoryFlags::NO_THROW) {
                    return nullptr;
                }
                throw std::bad_alloc();
            }

            // Zero initialize if requested
            if (flags & MemoryFlags::ZERO_INITIALIZED) {
                std::memset(ptr, 0, adjusted_size);
            }

            // Setup protection if needed
            if (protection_level != MemoryProtectionLevel::NONE) {
                setupMemoryProtection(ptr, size, protection_level);
            }

            // Track allocation
            trackAllocation(ptr, size, location);

            return ptr;
        }
        catch (const std::exception& e) {
            if (ptr) {
                std::free(ptr);
            }

            if (flags & MemoryFlags::NO_THROW) {
                return nullptr;
            }
            throw;
        }
    }

    /**
     * @brief Allocate memory for an object
     *
     * @tparam T Object type
     * @param flags Allocation flags
     * @param protection_level Protection level
     * @param location Source location
     * @return Pointer to allocated memory
     */
    template <typename T>
    T* allocateObject(MemoryFlags flags = MemoryFlags::DEFAULT,
                      MemoryProtectionLevel protection_level = MemoryProtectionLevel::NONE,
                      const std::string& location = "unknown")
    {
        void* ptr = allocate(sizeof(T), flags, protection_level, location);

        if (!ptr) {
            return nullptr;
        }

        // Update type info in allocation tracking
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                it->second.type_name = typeid(T).name();
            }
        }

        return static_cast<T*>(ptr);
    }

    /**
     * @brief Allocate an array
     *
     * @tparam T Element type
     * @param count Number of elements
     * @param flags Allocation flags
     * @param protection_level Protection level
     * @param location Source location
     * @return Pointer to allocated array
     */
    template <typename T>
    T* allocateArray(size_t count, MemoryFlags flags = MemoryFlags::DEFAULT,
                     MemoryProtectionLevel protection_level = MemoryProtectionLevel::NONE,
                     const std::string& location = "unknown")
    {
        void* ptr = allocate(sizeof(T) * count, flags, protection_level, location);

        if (!ptr) {
            return nullptr;
        }

        // Update type info in allocation tracking
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                it->second.type_name = typeid(T).name();
                it->second.is_array = true;
            }
        }

        return static_cast<T*>(ptr);
    }

    /**
     * @brief Deallocate memory
     *
     * @param ptr Pointer to memory
     * @return True if deallocation was successful
     */
    bool deallocate(void* ptr)
    {
        if (!ptr) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            // Double free or invalid pointer
            error::ErrorHandler::logError(error::ErrorInfo(
                error::ErrorCode::MEMORY_ACCESS_VIOLATION, error::ErrorCategory::MEMORY,
                error::ErrorSeverity::ERROR, "Attempted to free unallocated memory",
                std::source_location::current(),
                "Address: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))));
            return false;
        }

        // Check for corruption before freeing
        if (it->second.is_protected.load()) {
            if (!verifyMemoryIntegrity(ptr)) {
                stats_.detected_corruption++;

                error::ErrorHandler::logError(error::ErrorInfo(
                    error::ErrorCode::MEMORY_CORRUPTION_DETECTED, error::ErrorCategory::MEMORY,
                    error::ErrorSeverity::ERROR, "Memory corruption detected during deallocation",
                    std::source_location::current(),
                    "Address: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))));

                // Attempt to repair if possible
                if (tryRepairMemory(ptr)) {
                    stats_.repaired_corruption++;
                }
            }
        }

        // Update stats
        stats_.total_deallocations++;
        stats_.current_allocations--;
        stats_.current_bytes -= it->second.size;

        if (it->second.is_protected.load()) {
            stats_.protected_allocations--;
            stats_.protected_bytes -= it->second.size;
        }

        // Remove from tracking
        allocations_.erase(it);

        // Free the memory
        std::free(ptr);

        return true;
    }

    /**
     * @brief Get allocation information for a pointer
     *
     * @param ptr Pointer to check
     * @return Optional with allocation info, or empty if not found
     */
    std::optional<MemoryAllocationInfo> getAllocationInfo(void* ptr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            return it->second;
        }

        return std::nullopt;
    }

    /**
     * @brief Check if a pointer is currently allocated
     *
     * @param ptr Pointer to check
     * @return True if pointer is currently allocated by this manager
     */
    bool isAllocated(void* ptr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocations_.find(ptr) != allocations_.end();
    }

    /**
     * @brief Get memory statistics
     *
     * @return Current memory statistics
     */
    MemoryStats getStats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    /**
     * @brief Reset memory statistics
     */
    void resetStats()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = MemoryStats();
        stats_.current_allocations = allocations_.size();

        // Recalculate current bytes and protected allocations
        stats_.current_bytes = 0;
        stats_.protected_allocations = 0;
        stats_.protected_bytes = 0;

        for (const auto& pair : allocations_) {
            stats_.current_bytes += pair.second.size;

            if (pair.second.is_protected.load()) {
                stats_.protected_allocations++;
                stats_.protected_bytes += pair.second.size;
            }
        }
    }

    /**
     * @brief Check for memory leaks
     *
     * @param report_to_log Whether to report leaks to the error log
     * @return Number of detected leaks
     */
    size_t checkForLeaks(bool report_to_log = true)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        stats_.leaked_allocations = allocations_.size();

        if (report_to_log && !allocations_.empty()) {
            std::string details = "Leaked allocations:\n";

            size_t count = 0;
            for (const auto& pair : allocations_) {
                if (count++ > 10) {
                    details += "... and " + std::to_string(allocations_.size() - 10) + " more\n";
                    break;
                }

                const auto& info = pair.second;
                details += "  - " + std::to_string(reinterpret_cast<uintptr_t>(info.ptr)) + " (" +
                           std::to_string(info.size) + " bytes)";

                if (!info.type_name.empty()) {
                    details += " type: " + info.type_name;
                }

                details += " allocated at: " + info.location + "\n";
            }

            error::ErrorHandler::logError(error::ErrorInfo(
                error::ErrorCode::MEMORY_CORRUPTION_DETECTED, error::ErrorCategory::MEMORY,
                error::ErrorSeverity::WARNING,
                "Memory leaks detected: " + std::to_string(allocations_.size()) + " allocations",
                std::source_location::current(), details));
        }

        return allocations_.size();
    }

    /**
     * @brief Protect memory region
     *
     * @param ptr Pointer to memory
     * @param level Protection level
     * @return True if protection was successful
     */
    bool protectMemory(void* ptr, MemoryProtectionLevel level)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        // If already protected, remove old protection first
        if (it->second.is_protected.load()) {
            removeMemoryProtection(ptr);
        }

        // Setup protection
        if (setupMemoryProtection(ptr, it->second.size, level)) {
            it->second.is_protected.store(true);
            stats_.protected_allocations++;
            stats_.protected_bytes += it->second.size;
            return true;
        }

        return false;
    }

    /**
     * @brief Unprotect memory region
     *
     * @param ptr Pointer to memory
     * @return True if unprotection was successful
     */
    bool unprotectMemory(void* ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        if (!it->second.is_protected.load()) {
            return true;  // Already unprotected
        }

        if (removeMemoryProtection(ptr)) {
            it->second.is_protected.store(false);
            stats_.protected_allocations--;
            stats_.protected_bytes -= it->second.size;
            return true;
        }

        return false;
    }

    /**
     * @brief Verify memory integrity
     *
     * @param ptr Pointer to memory
     * @return True if memory is intact, false if corrupted
     */
    bool verifyMemoryIntegrity(void* ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        if (!it->second.is_protected.load()) {
            return true;  // Can't verify unprotected memory
        }

        // Implement verification logic based on protection type
        auto protection_level = getDefaultProtectionLevel();
        uint8_t* mem_ptr = static_cast<uint8_t*>(ptr);
        size_t size = it->second.size;
        bool is_valid = true;

        switch (protection_level) {
            case MemoryProtectionLevel::CANARY: {
                // Check canary values at beginning and end
                uint64_t* start_canary = reinterpret_cast<uint64_t*>(mem_ptr - 8);
                uint64_t* end_canary = reinterpret_cast<uint64_t*>(mem_ptr + size);

                // Canary value is typically a known pattern
                constexpr uint64_t CANARY_VALUE = 0xFEEDFACEDEADBEEF;

                if (*start_canary != CANARY_VALUE || *end_canary != CANARY_VALUE) {
                    is_valid = false;
                }
                break;
            }

            case MemoryProtectionLevel::CRC: {
                // Calculate CRC of memory region and compare with stored CRC
                uint32_t* stored_crc = reinterpret_cast<uint32_t*>(mem_ptr + size);
                uint32_t calculated_crc = calculateCRC32(mem_ptr, size);

                if (*stored_crc != calculated_crc) {
                    is_valid = false;
                }
                break;
            }

            case MemoryProtectionLevel::ECC: {
                // Simplified ECC check
                // In a real implementation, this would use proper ECC algorithms
                // such as Hamming code or Reed-Solomon
                uint8_t* ecc_data = mem_ptr + size;
                is_valid = verifyECC(mem_ptr, size, ecc_data);
                break;
            }

            case MemoryProtectionLevel::TMR: {
                // Triple Modular Redundancy - compare three copies
                size_t chunk_size = size;
                uint8_t* copy1 = mem_ptr;
                uint8_t* copy2 = mem_ptr + chunk_size;
                uint8_t* copy3 = mem_ptr + 2 * chunk_size;

                // Compare byte by byte
                for (size_t i = 0; i < chunk_size; ++i) {
                    // Majority voting
                    int correct_value = majorityVote(copy1[i], copy2[i], copy3[i]);

                    // If any copy is corrupt, the memory is corrupt
                    if (copy1[i] != correct_value || copy2[i] != correct_value ||
                        copy3[i] != correct_value) {
                        is_valid = false;
                        break;
                    }
                }
                break;
            }

            default:
                // No additional checks for unknown protection types
                break;
        }

        return is_valid;
    }

    /**
     * @brief Set default protection level for new allocations
     *
     * @param level Default protection level
     */
    void setDefaultProtectionLevel(MemoryProtectionLevel level)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        default_protection_level_ = level;
    }

    /**
     * @brief Get default protection level
     *
     * @return Default protection level
     */
    MemoryProtectionLevel getDefaultProtectionLevel() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return default_protection_level_;
    }

    /**
     * @brief Register a callback for memory corruption events
     *
     * @param callback Function to call when corruption is detected
     * @return ID that can be used to unregister the callback
     */
    size_t registerCorruptionCallback(
        std::function<void(void*, size_t, const std::string&)> callback)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        static size_t next_id = 1;
        size_t id = next_id++;

        corruption_callbacks_[id] = std::move(callback);
        return id;
    }

    /**
     * @brief Unregister a corruption callback
     *
     * @param id ID returned by registerCorruptionCallback
     * @return True if callback was found and removed
     */
    bool unregisterCorruptionCallback(size_t id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return corruption_callbacks_.erase(id) > 0;
    }

   private:
    // Private constructor for singleton
    UnifiedMemoryManager() = default;

    // Prevent copying or moving
    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager(UnifiedMemoryManager&&) = delete;
    UnifiedMemoryManager& operator=(UnifiedMemoryManager&&) = delete;

    /**
     * @brief Track a new allocation
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param location Source location
     */
    void trackAllocation(void* ptr, size_t size, const std::string& location)
    {
        allocations_.emplace(ptr, MemoryAllocationInfo(ptr, size, location));

        // Update stats
        stats_.current_allocations++;
        stats_.total_allocations++;
        stats_.current_bytes += size;

        if (stats_.current_allocations > stats_.peak_allocations) {
            stats_.peak_allocations = stats_.current_allocations;
        }

        if (stats_.current_bytes > stats_.peak_bytes) {
            stats_.peak_bytes = stats_.current_bytes;
        }
    }

    /**
     * @brief Calculate size needed for protected allocation
     *
     * @param original_size Original size
     * @param level Protection level
     * @return Adjusted size
     */
    size_t calculateProtectedSize(size_t original_size, MemoryProtectionLevel level)
    {
        switch (level) {
            case MemoryProtectionLevel::NONE:
                return original_size;

            case MemoryProtectionLevel::CANARY:
                return original_size + 16;  // 8 bytes at beginning and end

            case MemoryProtectionLevel::CRC:
                return original_size + 8;  // 4 bytes for CRC, 4 for alignment

            case MemoryProtectionLevel::ECC:
                // ECC typically adds 12.5% overhead
                return original_size + (original_size / 8) + 8;

            case MemoryProtectionLevel::TMR:
                // TMR requires 3x the memory + metadata
                return original_size * 3 + 16;

            default:
                return original_size;
        }
    }

    /**
     * @brief Setup memory protection for an allocation
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param level Protection level
     * @return True if protection was set up successfully
     */
    bool setupMemoryProtection(void* ptr, size_t size, MemoryProtectionLevel level)
    {
        if (!ptr) {
            return false;
        }

        uint8_t* mem_ptr = static_cast<uint8_t*>(ptr);
        auto it = allocations_.find(ptr);

        if (it == allocations_.end()) {
            return false;
        }

        // Mark as protected
        it->second.is_protected.store(true);
        stats_.protected_allocations++;
        stats_.protected_bytes += size;

        // Apply protection based on the level
        switch (level) {
            case MemoryProtectionLevel::CANARY: {
                // Set canary values at the beginning and end
                uint64_t* start_canary = reinterpret_cast<uint64_t*>(mem_ptr - 8);
                uint64_t* end_canary = reinterpret_cast<uint64_t*>(mem_ptr + size);

                constexpr uint64_t CANARY_VALUE = 0xFEEDFACEDEADBEEF;
                *start_canary = CANARY_VALUE;
                *end_canary = CANARY_VALUE;
                break;
            }

            case MemoryProtectionLevel::CRC: {
                // Calculate and store CRC
                uint32_t* crc_ptr = reinterpret_cast<uint32_t*>(mem_ptr + size);
                *crc_ptr = calculateCRC32(mem_ptr, size);
                break;
            }

            case MemoryProtectionLevel::ECC: {
                // Calculate and store ECC data
                uint8_t* ecc_data = mem_ptr + size;
                size_t ecc_size = (size + 7) / 8;  // 1 bit per byte

                // Clear ECC data area
                std::memset(ecc_data, 0, ecc_size);

                // Calculate and store parity bits
                for (size_t i = 0; i < size; ++i) {
                    uint8_t byte = mem_ptr[i];
                    uint8_t parity = 0;

                    // Calculate parity
                    for (int bit = 0; bit < 8; ++bit) {
                        parity ^= ((byte >> bit) & 1);
                    }

                    // Store parity bit
                    if (parity) {
                        ecc_data[i / 8] |= (1 << (i % 8));
                    }
                }
                break;
            }

            case MemoryProtectionLevel::TMR: {
                // Triplicate the data
                size_t chunk_size = size;

                // Make two additional copies
                std::memcpy(mem_ptr + chunk_size, mem_ptr, chunk_size);
                std::memcpy(mem_ptr + 2 * chunk_size, mem_ptr, chunk_size);
                break;
            }

            case MemoryProtectionLevel::NONE:
            default:
                // No protection
                break;
        }

        return true;
    }

    /**
     * @brief Remove memory protection from an allocation
     *
     * @param ptr Memory pointer
     * @return True if protection was removed successfully
     */
    bool removeMemoryProtection(void* ptr)
    {
        if (!ptr) {
            return false;
        }

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        // If already unprotected, nothing to do
        if (!it->second.is_protected.load()) {
            return true;
        }

        // Update protection status
        it->second.is_protected.store(false);

        // Update statistics
        stats_.protected_allocations--;
        stats_.protected_bytes -= it->second.size;

        // No need to clear the protection data as it will be overwritten
        // by future allocations, and the space is already accounted for

        return true;
    }

    /**
     * @brief Try to repair corrupted memory
     *
     * @param ptr Memory pointer
     * @return True if repair was successful
     */
    bool tryRepairMemory(void* ptr)
    {
        // Implementation of memory repair logic based on protection level
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        // Notify all callbacks
        for (const auto& callback_pair : corruption_callbacks_) {
            try {
                callback_pair.second(ptr, it->second.size, it->second.type_name);
            }
            catch (...) {
                // Ignore callback errors
            }
        }

        if (!it->second.is_protected.load()) {
            // Cannot repair unprotected memory
            return false;
        }

        uint8_t* mem_ptr = static_cast<uint8_t*>(ptr);
        size_t size = it->second.size;
        bool repaired = false;
        auto protection_level = getDefaultProtectionLevel();

        switch (protection_level) {
            case MemoryProtectionLevel::CANARY: {
                // For canary, we can't actually repair the data,
                // we can only detect the corruption
                return false;
            }

            case MemoryProtectionLevel::CRC: {
                // For CRC, we can only detect, not repair
                return false;
            }

            case MemoryProtectionLevel::ECC: {
                // For ECC, we can repair single-bit errors
                uint8_t* ecc_data = mem_ptr + size;

                // Check each byte's parity and try to repair
                for (size_t i = 0; i < size; ++i) {
                    uint8_t parity = 0;
                    uint8_t byte = mem_ptr[i];

                    // Calculate parity
                    for (int bit = 0; bit < 8; ++bit) {
                        parity ^= ((byte >> bit) & 1);
                    }

                    // Compare with stored parity
                    uint8_t stored_parity = (ecc_data[i / 8] >> (i % 8)) & 1;

                    if (parity != stored_parity) {
                        // Try to find the flipped bit
                        for (int bit = 0; bit < 8; ++bit) {
                            // Flip this bit and see if parity matches
                            uint8_t test_byte = byte ^ (1 << bit);
                            uint8_t test_parity = 0;

                            for (int j = 0; j < 8; ++j) {
                                test_parity ^= ((test_byte >> j) & 1);
                            }

                            if (test_parity == stored_parity) {
                                // Found the likely bit flip, repair it
                                mem_ptr[i] = test_byte;
                                repaired = true;
                                break;
                            }
                        }
                    }
                }
                return repaired;
            }

            case MemoryProtectionLevel::TMR: {
                // TMR can repair using majority voting
                size_t chunk_size = size;
                uint8_t* copy1 = mem_ptr;
                uint8_t* copy2 = mem_ptr + chunk_size;
                uint8_t* copy3 = mem_ptr + 2 * chunk_size;

                // Repair using majority voting
                for (size_t i = 0; i < chunk_size; ++i) {
                    // Find correct value using majority voting
                    uint8_t correct_value = majorityVote(copy1[i], copy2[i], copy3[i]);

                    // Repair any copies that differ from majority
                    if (copy1[i] != correct_value) {
                        copy1[i] = correct_value;
                        repaired = true;
                    }

                    if (copy2[i] != correct_value) {
                        copy2[i] = correct_value;
                        repaired = true;
                    }

                    if (copy3[i] != correct_value) {
                        copy3[i] = correct_value;
                        repaired = true;
                    }
                }
                return repaired;
            }

            default:
                return false;
        }
    }

    /**
     * @brief Calculate CRC32 checksum for a memory region
     *
     * @param data Pointer to data
     * @param size Size of data in bytes
     * @return CRC32 checksum
     */
    uint32_t calculateCRC32(const uint8_t* data, size_t size) const
    {
        // Basic CRC-32 implementation
        constexpr uint32_t CRC32_POLYNOMIAL = 0xEDB88320;
        uint32_t crc = 0xFFFFFFFF;

        for (size_t i = 0; i < size; ++i) {
            crc ^= data[i];
            for (int j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ ((crc & 1) ? CRC32_POLYNOMIAL : 0);
            }
        }

        return ~crc;
    }

    /**
     * @brief Verify Error Correction Code (ECC) for a memory region
     *
     * @param data Pointer to data
     * @param size Size of data in bytes
     * @param ecc_data Pointer to ECC data
     * @return True if data is valid or corrected
     */
    bool verifyECC(uint8_t* data, size_t size, const uint8_t* ecc_data) const
    {
        // Simplified ECC implementation
        // Just verify parity for demonstration

        bool is_valid = true;

        // Check each byte's parity
        for (size_t i = 0; i < size; ++i) {
            uint8_t parity = 0;
            uint8_t byte = data[i];

            // Calculate parity
            for (int bit = 0; bit < 8; ++bit) {
                parity ^= ((byte >> bit) & 1);
            }

            // Compare with stored parity (1 bit per byte)
            uint8_t stored_parity = (ecc_data[i / 8] >> (i % 8)) & 1;

            if (parity != stored_parity) {
                is_valid = false;
                // For real ECC, this would attempt to correct the error
            }
        }

        return is_valid;
    }

    /**
     * @brief Perform majority vote on three values
     *
     * @param a First value
     * @param b Second value
     * @param c Third value
     * @return Majority value
     */
    template <typename T>
    T majorityVote(T a, T b, T c) const
    {
        // Return majority value
        if (a == b || a == c) {
            return a;
        }
        else if (b == c) {
            return b;
        }
        else {
            // All values are different
            // In this case, we can't determine the correct value
            // For simplicity, return the first value
            return a;
        }
    }

    // Member variables
    mutable std::mutex mutex_;
    std::unordered_map<void*, MemoryAllocationInfo> allocations_;
    MemoryStats stats_;
    MemoryProtectionLevel default_protection_level_ = MemoryProtectionLevel::NONE;
    std::unordered_map<size_t, std::function<void(void*, size_t, const std::string&)>>
        corruption_callbacks_;
};

/**
 * @brief Smart pointer with radiation tolerance
 *
 * This is a wrapper around std::unique_ptr that uses the UnifiedMemoryManager
 * for allocation and deallocation.
 *
 * @tparam T Object type
 */
template <typename T>
class RadiationTolerantPtr {
   public:
    /**
     * @brief Default constructor - creates a null pointer
     */
    RadiationTolerantPtr() noexcept : ptr_(nullptr) {}

    /**
     * @brief Constructor from raw pointer
     *
     * Takes ownership of the pointer.
     *
     * @param ptr Raw pointer
     */
    explicit RadiationTolerantPtr(T* ptr) noexcept : ptr_(ptr) {}

    /**
     * @brief Move constructor
     *
     * @param other Pointer to move from
     */
    RadiationTolerantPtr(RadiationTolerantPtr&& other) noexcept : ptr_(other.release()) {}

    /**
     * @brief Move assignment
     *
     * @param other Pointer to move from
     * @return Reference to this
     */
    RadiationTolerantPtr& operator=(RadiationTolerantPtr&& other) noexcept
    {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    /**
     * @brief Destructor
     */
    ~RadiationTolerantPtr() { reset(); }

    /**
     * @brief Access the managed object
     *
     * @return Pointer to the managed object
     */
    T* get() const noexcept { return ptr_; }

    /**
     * @brief Dereference operator
     *
     * @return Reference to the managed object
     */
    T& operator*() const { return *ptr_; }

    /**
     * @brief Member access operator
     *
     * @return Pointer to the managed object
     */
    T* operator->() const noexcept { return ptr_; }

    /**
     * @brief Boolean conversion operator
     *
     * @return True if pointer is not null
     */
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /**
     * @brief Release ownership of the pointer
     *
     * @return Raw pointer
     */
    T* release() noexcept
    {
        T* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    /**
     * @brief Reset the pointer
     *
     * @param ptr New pointer to manage (default: nullptr)
     */
    void reset(T* ptr = nullptr) noexcept
    {
        if (ptr_ != nullptr) {
            UnifiedMemoryManager::getInstance().deallocate(ptr_);
        }
        ptr_ = ptr;
    }

    /**
     * @brief Check if memory is protected
     *
     * @return True if memory is protected
     */
    bool isProtected() const
    {
        if (!ptr_) {
            return false;
        }

        auto info = UnifiedMemoryManager::getInstance().getAllocationInfo(ptr_);
        return info && info->is_protected.load();
    }

    /**
     * @brief Protect the memory
     *
     * @param level Protection level
     * @return True if protection was successful
     */
    bool protect(MemoryProtectionLevel level)
    {
        if (!ptr_) {
            return false;
        }

        return UnifiedMemoryManager::getInstance().protectMemory(ptr_, level);
    }

    /**
     * @brief Unprotect the memory
     *
     * @return True if unprotection was successful
     */
    bool unprotect()
    {
        if (!ptr_) {
            return false;
        }

        return UnifiedMemoryManager::getInstance().unprotectMemory(ptr_);
    }

    /**
     * @brief Verify memory integrity
     *
     * @return True if memory is intact
     */
    bool verifyIntegrity()
    {
        if (!ptr_) {
            return false;
        }

        return UnifiedMemoryManager::getInstance().verifyMemoryIntegrity(ptr_);
    }

    /**
     * @brief Factory method to create a RadiationTolerantPtr
     *
     * @tparam U Object type
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     * @return RadiationTolerantPtr managing the new object
     */
    template <typename U = T, typename... Args>
    static RadiationTolerantPtr<U> make(Args&&... args)
    {
        auto* ptr = UnifiedMemoryManager::getInstance().allocateObject<U>(
            MemoryFlags::DEFAULT, UnifiedMemoryManager::getInstance().getDefaultProtectionLevel(),
            "RadiationTolerantPtr::make at " + std::to_string(__LINE__));

        if (!ptr) {
            throw std::bad_alloc();
        }

        try {
            new (ptr) U(std::forward<Args>(args)...);
        }
        catch (...) {
            UnifiedMemoryManager::getInstance().deallocate(ptr);
            throw;
        }

        return RadiationTolerantPtr<U>(ptr);
    }

    /**
     * @brief Factory method to create a protected RadiationTolerantPtr
     *
     * @tparam U Object type
     * @tparam Args Constructor argument types
     * @param protection_level Protection level
     * @param args Constructor arguments
     * @return RadiationTolerantPtr managing the new object
     */
    template <typename U = T, typename... Args>
    static RadiationTolerantPtr<U> makeProtected(MemoryProtectionLevel protection_level,
                                                 Args&&... args)
    {
        auto* ptr = UnifiedMemoryManager::getInstance().allocateObject<U>(
            MemoryFlags::DEFAULT, protection_level,
            "RadiationTolerantPtr::makeProtected at " + std::to_string(__LINE__));

        if (!ptr) {
            throw std::bad_alloc();
        }

        try {
            new (ptr) U(std::forward<Args>(args)...);
        }
        catch (...) {
            UnifiedMemoryManager::getInstance().deallocate(ptr);
            throw;
        }

        return RadiationTolerantPtr<U>(ptr);
    }

   private:
    T* ptr_;

    // Disable copy operations
    RadiationTolerantPtr(const RadiationTolerantPtr&) = delete;
    RadiationTolerantPtr& operator=(const RadiationTolerantPtr&) = delete;
};

// Convenience function for creating RadiationTolerantPtr objects
template <typename T, typename... Args>
RadiationTolerantPtr<T> makeRadTolerant(Args&&... args)
{
    return RadiationTolerantPtr<T>::make(std::forward<Args>(args)...);
}

// Convenience function for creating protected RadiationTolerantPtr objects
template <typename T, typename... Args>
RadiationTolerantPtr<T> makeRadTolerantProtected(MemoryProtectionLevel protection_level,
                                                 Args&&... args)
{
    return RadiationTolerantPtr<T>::makeProtected(protection_level, std::forward<Args>(args)...);
}

}  // namespace memory
}  // namespace rad_ml
