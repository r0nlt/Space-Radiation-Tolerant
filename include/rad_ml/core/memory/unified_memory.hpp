#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
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

// Canary value for memory protection (read/written via std::memcpy because the
// trailing canary is not guaranteed to be aligned for uint64_t access)
using CanaryType = std::uint64_t;
constexpr CanaryType CANARY_VALUE = 0xDEADBEEFDEADBEEFULL;

/**
 * @brief Memory protection level
 */
enum class MemoryProtectionLevel {
    NONE,       ///< No protection
    MINIMAL,    ///< Minimal protection
    MODERATE,   ///< Moderate protection
    HIGH,       ///< High protection
    VERY_HIGH,  ///< Very high protection
    ADAPTIVE,   ///< Adaptive protection
    CANARY,     ///< Canary values to detect overflow/underflow
    CRC,        ///< Checksum to detect corruption
    ECC,        ///< Error correcting code
    TMR         ///< Triple modular redundancy
};

/**
 * @brief Memory allocation tracking information
 */
struct MemoryAllocationInfo {
    void* ptr;                                              ///< Memory address (user-facing)
    void* original_ptr;                                     ///< Original allocation address
    size_t size;                                            ///< Size in bytes
    std::chrono::steady_clock::time_point allocation_time;  ///< When allocation occurred
    std::string location;                                   ///< Source location of allocation
    std::string type_name;                                  ///< Type of allocated object if known
    bool is_array;                                          ///< Whether this is an array allocation
    std::atomic<bool> is_protected;                         ///< Whether this memory is protected
    MemoryProtectionLevel protection_level;                 ///< Protection level used

    MemoryAllocationInfo(void* ptr, size_t size, std::string location, std::string type_name = "",
                         bool is_array = false)
        : ptr(ptr),
          original_ptr(ptr),  // By default, they're the same
          size(size),
          allocation_time(std::chrono::steady_clock::now()),
          location(std::move(location)),
          type_name(std::move(type_name)),
          is_array(is_array),
          is_protected(false),
          protection_level(MemoryProtectionLevel::NONE)
    {
    }

    /**
     * @brief Copy constructor
     *
     * @param other MemoryAllocationInfo to copy
     */
    MemoryAllocationInfo(const MemoryAllocationInfo& other)
        : ptr(other.ptr),
          original_ptr(other.original_ptr),
          size(other.size),
          allocation_time(other.allocation_time),
          location(other.location),
          type_name(other.type_name),
          is_array(other.is_array),
          is_protected(other.is_protected.load()),
          protection_level(other.protection_level)
    {
    }

    /**
     * @brief Copy assignment operator
     *
     * @param other MemoryAllocationInfo to copy
     * @return Reference to this object
     */
    MemoryAllocationInfo& operator=(const MemoryAllocationInfo& other)
    {
        if (this != &other) {
            ptr = other.ptr;
            original_ptr = other.original_ptr;
            size = other.size;
            allocation_time = other.allocation_time;
            location = other.location;
            type_name = other.type_name;
            is_array = other.is_array;
            is_protected.store(other.is_protected.load());
            protection_level = other.protection_level;
        }
        return *this;
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

        // Extra space required before/after the user region for this protection level
        const size_t header = protectionHeaderSize(protection_level, flags, alignment);
        const size_t trailer = protectionTrailerSize(protection_level, size);
        const size_t adjusted_size = header + size + trailer;

        // Perform allocation
        void* allocated_ptr = nullptr;
        void* return_ptr = nullptr;
        try {
            if (flags & MemoryFlags::ALIGNED) {
                // Round up adjusted_size to multiple of alignment
                // std::aligned_alloc requires size to be a multiple of alignment
                size_t aligned_size = ((adjusted_size + alignment - 1) / alignment) * alignment;
                allocated_ptr = std::aligned_alloc(alignment, aligned_size);
            }
            else {
                allocated_ptr = std::malloc(adjusted_size);
            }

            if (!allocated_ptr) {
                if (flags & MemoryFlags::NO_THROW) {
                    return nullptr;
                }
                throw std::bad_alloc();
            }

            // Zero initialize if requested
            if (flags & MemoryFlags::ZERO_INITIALIZED) {
                std::memset(allocated_ptr, 0, adjusted_size);
            }

            // The user-facing pointer is offset past any protection header
            return_ptr = static_cast<uint8_t*>(allocated_ptr) + header;

            // Track by the user-facing pointer, since that is what callers hand
            // back to deallocate()/verifyMemoryIntegrity()
            MemoryAllocationInfo& info = trackAllocation(return_ptr, allocated_ptr, size, location);
            info.protection_level = protection_level;

            if (isConcreteProtection(protection_level)) {
                setupProtectionLocked(info);
                info.is_protected.store(true);
                stats_.protected_allocations++;
                stats_.protected_bytes += size;
            }

            return return_ptr;
        }
        catch (const std::exception& e) {
            if (allocated_ptr) {
                if (return_ptr) {
                    allocations_.erase(return_ptr);
                }
                std::free(allocated_ptr);
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
                error::SourceLocation(__FILE__, __LINE__, __func__),
                "Address: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))));
            return false;
        }

        // Check for corruption before freeing
        if (it->second.is_protected.load()) {
            if (!verifyIntegrityLocked(it->second)) {
                error::ErrorHandler::logError(error::ErrorInfo(
                    error::ErrorCode::MEMORY_CORRUPTION_DETECTED, error::ErrorCategory::MEMORY,
                    error::ErrorSeverity::ERROR, "Memory corruption detected during deallocation",
                    error::SourceLocation(__FILE__, __LINE__, __func__),
                    "Address: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))));

                // Attempt to repair if possible
                if (tryRepairMemoryLocked(it->second)) {
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

        // Free the underlying allocation (may differ from the user-facing
        // pointer when a protection header precedes the user region)
        void* original_ptr = it->second.original_ptr;

        // Remove from tracking
        allocations_.erase(it);

        std::free(original_ptr);

        return true;
    }

    /**
     * @brief Get allocation information for a pointer
     *
     * @param ptr Pointer to check
     * @return Pointer to allocation info if found, nullptr otherwise
     */
    const MemoryAllocationInfo* getAllocationInfo(void* ptr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            return &(it->second);
        }
        return nullptr;
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
                error::SourceLocation(__FILE__, __LINE__, __func__), details));
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

        MemoryAllocationInfo& info = it->second;

        // Protection metadata space (canaries, CRC, parity bits, TMR copies) is
        // reserved at allocation time. Applying a different concrete protection
        // level after the fact would write past the allocation, so it is
        // rejected instead of corrupting the heap.
        if (isConcreteProtection(level) && level != info.protection_level) {
            error::ErrorHandler::logError(error::ErrorInfo(
                error::ErrorCode::MEMORY_ACCESS_VIOLATION, error::ErrorCategory::MEMORY,
                error::ErrorSeverity::ERROR,
                "Cannot change protection level after allocation; allocate with the desired level",
                error::SourceLocation(__FILE__, __LINE__, __func__),
                "Address: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))));
            return false;
        }

        if (!isConcreteProtection(level)) {
            return false;
        }

        // Re-arm protection (refresh canaries/CRC/parity/TMR copies from the
        // current contents of the user region)
        setupProtectionLocked(info);

        if (!info.is_protected.load()) {
            info.is_protected.store(true);
            stats_.protected_allocations++;
            stats_.protected_bytes += info.size;
        }
        return true;
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

        // Protection metadata is left in place; it is simply no longer checked.
        it->second.is_protected.store(false);
        stats_.protected_allocations--;
        stats_.protected_bytes -= it->second.size;
        return true;
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
            // Not found - not our memory
            return false;
        }

        return verifyIntegrityLocked(it->second);
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
     * @brief Track a new allocation (caller must hold mutex_)
     *
     * @param user_ptr User-facing pointer (map key)
     * @param original_ptr Pointer returned by malloc/aligned_alloc
     * @param size User-visible size in bytes
     * @param location Source location
     * @return Reference to the new tracking entry
     */
    MemoryAllocationInfo& trackAllocation(void* user_ptr, void* original_ptr, size_t size,
                                          const std::string& location)
    {
        auto result = allocations_.emplace(user_ptr, MemoryAllocationInfo(user_ptr, size, location));
        MemoryAllocationInfo& info = result.first->second;
        info.original_ptr = original_ptr;

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

        return info;
    }

    /**
     * @brief Whether a protection level has a concrete in-memory scheme
     *
     * The abstract levels (MINIMAL..ADAPTIVE) are policy hints with no
     * metadata layout of their own.
     */
    static bool isConcreteProtection(MemoryProtectionLevel level)
    {
        switch (level) {
            case MemoryProtectionLevel::CANARY:
            case MemoryProtectionLevel::CRC:
            case MemoryProtectionLevel::ECC:
            case MemoryProtectionLevel::TMR:
                return true;
            default:
                return false;
        }
    }

    /**
     * @brief Bytes reserved before the user region for this protection level
     *
     * Only CANARY uses a header (the leading canary). The header is padded so
     * the user pointer keeps the alignment guarantee of the allocator.
     */
    static size_t protectionHeaderSize(MemoryProtectionLevel level, MemoryFlags flags,
                                       size_t alignment)
    {
        if (level != MemoryProtectionLevel::CANARY) {
            return 0;
        }

        const size_t min_header =
            (flags & MemoryFlags::ALIGNED) ? alignment : alignof(std::max_align_t);
        return ((sizeof(CanaryType) + min_header - 1) / min_header) * min_header;
    }

    /**
     * @brief Bytes reserved after the user region for this protection level
     */
    static size_t protectionTrailerSize(MemoryProtectionLevel level, size_t size)
    {
        switch (level) {
            case MemoryProtectionLevel::CANARY:
                return sizeof(CanaryType);
            case MemoryProtectionLevel::CRC:
                return sizeof(uint32_t);
            case MemoryProtectionLevel::ECC:
                return (size + 7) / 8;  // 1 parity bit per byte
            case MemoryProtectionLevel::TMR:
                return 2 * size;  // Two extra copies
            default:
                return 0;
        }
    }

    /**
     * @brief Write protection metadata from the current user-region contents
     * (caller must hold mutex_)
     *
     * Layout, relative to the user pointer P with user size S:
     * - CANARY: canary at [P - 8, P) and [P + S, P + S + 8)
     * - CRC:    stored CRC32 at [P + S, P + S + 4)
     * - ECC:    parity bits at [P + S, P + S + ceil(S/8))
     * - TMR:    copies at [P + S, P + 2S) and [P + 2S, P + 3S)
     */
    void setupProtectionLocked(MemoryAllocationInfo& info)
    {
        uint8_t* user = static_cast<uint8_t*>(info.ptr);
        const size_t size = info.size;

        switch (info.protection_level) {
            case MemoryProtectionLevel::CANARY: {
                std::memcpy(user - sizeof(CanaryType), &CANARY_VALUE, sizeof(CanaryType));
                std::memcpy(user + size, &CANARY_VALUE, sizeof(CanaryType));
                break;
            }

            case MemoryProtectionLevel::CRC: {
                const uint32_t crc = calculateCRC32(user, size);
                std::memcpy(user + size, &crc, sizeof(crc));
                break;
            }

            case MemoryProtectionLevel::ECC: {
                uint8_t* ecc_data = user + size;
                const size_t ecc_size = (size + 7) / 8;

                std::memset(ecc_data, 0, ecc_size);

                for (size_t i = 0; i < size; ++i) {
                    if (byteParity(user[i])) {
                        ecc_data[i / 8] |= (1 << (i % 8));
                    }
                }
                break;
            }

            case MemoryProtectionLevel::TMR: {
                std::memcpy(user + size, user, size);
                std::memcpy(user + 2 * size, user, size);
                break;
            }

            default:
                break;
        }
    }

    /**
     * @brief Verify integrity of one allocation (caller must hold mutex_)
     *
     * Uses the protection level the allocation was created with. On failure,
     * updates corruption stats and fires corruption callbacks. Callbacks are
     * invoked while the lock is held and must not call back into the manager.
     *
     * @return True if memory is intact
     */
    bool verifyIntegrityLocked(MemoryAllocationInfo& info)
    {
        uint8_t* user = static_cast<uint8_t*>(info.ptr);
        const size_t size = info.size;

        bool intact = true;
        std::string failure_reason;

        switch (info.protection_level) {
            case MemoryProtectionLevel::CANARY: {
                CanaryType start_canary = 0;
                CanaryType end_canary = 0;
                std::memcpy(&start_canary, user - sizeof(CanaryType), sizeof(CanaryType));
                std::memcpy(&end_canary, user + size, sizeof(CanaryType));

                if (start_canary != CANARY_VALUE || end_canary != CANARY_VALUE) {
                    intact = false;
                    failure_reason = "canary value modified";
                }
                break;
            }

            case MemoryProtectionLevel::CRC: {
                uint32_t stored_crc = 0;
                std::memcpy(&stored_crc, user + size, sizeof(stored_crc));

                if (calculateCRC32(user, size) != stored_crc) {
                    intact = false;
                    failure_reason = "CRC mismatch";
                }
                break;
            }

            case MemoryProtectionLevel::ECC: {
                if (!verifyECC(user, size, user + size)) {
                    intact = false;
                    failure_reason = "parity mismatch";
                }
                break;
            }

            case MemoryProtectionLevel::TMR: {
                const uint8_t* copy1 = user;
                const uint8_t* copy2 = user + size;
                const uint8_t* copy3 = user + 2 * size;

                for (size_t i = 0; i < size; ++i) {
                    if (copy1[i] != copy2[i] || copy1[i] != copy3[i]) {
                        intact = false;
                        failure_reason = "TMR copies disagree";
                        break;
                    }
                }
                break;
            }

            default:
                // No concrete protection - nothing to check
                return true;
        }

        if (!intact) {
            stats_.detected_corruption++;

            for (const auto& [id, callback] : corruption_callbacks_) {
                try {
                    callback(info.ptr, size, "Memory corruption detected: " + failure_reason);
                }
                catch (...) {
                    // Ignore callback errors
                }
            }
        }

        return intact;
    }

    /**
     * @brief Try to repair corrupted memory (caller must hold mutex_)
     *
     * @return True if a repair was performed
     */
    bool tryRepairMemoryLocked(MemoryAllocationInfo& info)
    {
        if (!info.is_protected.load()) {
            // Cannot repair unprotected memory
            return false;
        }

        uint8_t* user = static_cast<uint8_t*>(info.ptr);
        const size_t size = info.size;
        bool repaired = false;

        switch (info.protection_level) {
            case MemoryProtectionLevel::CANARY:
            case MemoryProtectionLevel::CRC:
                // Detection-only schemes - cannot repair
                return false;

            case MemoryProtectionLevel::ECC: {
                // Parity can detect single-bit errors per byte but cannot
                // locate the flipped bit; restoring parity by guessing a bit
                // would corrupt data further, so only report.
                uint8_t* ecc_data = user + size;

                for (size_t i = 0; i < size; ++i) {
                    const uint8_t stored_parity = (ecc_data[i / 8] >> (i % 8)) & 1;
                    if (byteParity(user[i]) != stored_parity) {
                        return false;  // Corruption present but not repairable
                    }
                }
                return false;
            }

            case MemoryProtectionLevel::TMR: {
                // TMR can repair using majority voting
                uint8_t* copy1 = user;
                uint8_t* copy2 = user + size;
                uint8_t* copy3 = user + 2 * size;

                for (size_t i = 0; i < size; ++i) {
                    const uint8_t correct_value = majorityVote(copy1[i], copy2[i], copy3[i]);

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
     * @brief Compute the parity of a byte (1 if odd number of set bits)
     */
    static uint8_t byteParity(uint8_t byte)
    {
        uint8_t parity = 0;
        for (int bit = 0; bit < 8; ++bit) {
            parity ^= ((byte >> bit) & 1);
        }
        return parity;
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
            ptr_->~T();  // Objects are placement-constructed in make()/makeProtected()
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

        // Re-arm protection so CRC/TMR/ECC metadata reflects the constructed
        // object rather than the uninitialized allocation
        auto level = UnifiedMemoryManager::getInstance().getDefaultProtectionLevel();
        UnifiedMemoryManager::getInstance().protectMemory(ptr, level);

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

        // Re-arm protection so CRC/TMR/ECC metadata reflects the constructed
        // object rather than the uninitialized allocation
        UnifiedMemoryManager::getInstance().protectMemory(ptr, protection_level);

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
