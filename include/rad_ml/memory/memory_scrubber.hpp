#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "rad_ml/core/crc32.hpp"

namespace rad_ml {
namespace memory {

/**
 * @brief Memory Scrubber for error detection and correction
 *
 * The single memory scrubber for the framework (it unifies what were
 * previously two separate MemoryScrubber classes). It periodically scans
 * registered memory regions and supports two kinds of region:
 *
 *  - CRC-verified regions (registerMemoryRegion(ptr, size)): a CRC-32 is
 *    stored per 64-byte block at registration time; scrubbing recomputes and
 *    compares each block, counting mismatches as detected errors. This is
 *    detection-only -- after a scrub cycle the checksums are re-armed to the
 *    current memory contents.
 *
 *  - Callback-repaired regions (registerMemoryRegion<T>(ptr, size, fn)): the
 *    caller provides the repair routine (e.g. invoking TMR repair() across an
 *    array of protected values); scrubbing simply invokes it.
 *
 * Scrubbing can be driven manually (scrubMemory()/scrubOnce()) or by the
 * background thread (startBackgroundThread()/start()). The constructor never
 * starts the thread; it only stores the interval.
 *
 * Thread-safe. Note that region callbacks run while the scrubber's lock is
 * held, so they must not call back into this scrubber.
 */
class MemoryScrubber {
   public:
    /**
     * @brief Constructor
     *
     * @param scrub_interval_ms Interval for background scrubbing in
     *        milliseconds (used when the background thread is started)
     */
    explicit MemoryScrubber(unsigned long scrub_interval_ms = 0)
        : scrub_interval_ms_(scrub_interval_ms), running_(false), terminate_requested_(false)
    {
    }

    /**
     * @brief Destructor - ensures scrubbing is stopped
     */
    ~MemoryScrubber() { stopBackgroundThread(); }

    MemoryScrubber(const MemoryScrubber&) = delete;
    MemoryScrubber& operator=(const MemoryScrubber&) = delete;
    MemoryScrubber(MemoryScrubber&&) = delete;
    MemoryScrubber& operator=(MemoryScrubber&&) = delete;

    /**
     * @brief Register a CRC-verified memory region
     *
     * A CRC-32 per 64-byte block is stored at registration time; scrubbing
     * detects any block whose contents changed since registration (or since
     * the previous scrub cycle, which re-arms the checksums).
     *
     * @param memory_ptr Pointer to the memory region
     * @param memory_size Size of the memory region in bytes
     * @return Handle for unregisterMemoryRegion
     */
    size_t registerMemoryRegion(void* memory_ptr, size_t memory_size)
    {
        MemoryRegion region;
        region.memory_ptr = memory_ptr;
        region.memory_size = memory_size;
        computeBlockCrcs(region);

        std::lock_guard<std::mutex> lock(mutex_);
        region.id = next_region_id_++;
        memory_regions_.push_back(std::move(region));
        return memory_regions_.back().id;
    }

    /**
     * @brief Register a callback-repaired memory region
     *
     * The scrub function is invoked on every scrub cycle and is responsible
     * for verification and correction (e.g. calling repair() on each TMR
     * element in the region).
     *
     * @param memory_ptr Pointer to the memory region
     * @param size_bytes Size of the memory region in bytes
     * @param scrub_function Function performing verification and correction
     * @return Handle for unregisterMemoryRegion
     */
    template <typename T>
    size_t registerMemoryRegion(T* memory_ptr, size_t size_bytes,
                                std::function<void(T*, size_t)> scrub_function)
    {
        MemoryRegion region;
        region.memory_ptr = memory_ptr;
        region.memory_size = size_bytes;
        region.scrub_callback = [memory_ptr, size_bytes, scrub_function]() {
            scrub_function(memory_ptr, size_bytes);
        };

        std::lock_guard<std::mutex> lock(mutex_);
        region.id = next_region_id_++;
        memory_regions_.push_back(std::move(region));
        return memory_regions_.back().id;
    }

    /**
     * @brief Unregister a memory region
     *
     * @param handle The handle returned from registerMemoryRegion
     * @return True if region was found and unregistered, false otherwise
     */
    bool unregisterMemoryRegion(size_t handle)
    {
        if (handle == 0) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(memory_regions_.begin(), memory_regions_.end(),
                               [handle](const MemoryRegion& region) { return region.id == handle; });

        if (it != memory_regions_.end()) {
            memory_regions_.erase(it);
            return true;
        }

        return false;
    }

    /**
     * @brief Scrub all registered memory regions once
     *
     * CRC-verified regions are checked block by block (and their checksums
     * re-armed); callback regions have their scrub function invoked.
     *
     * @return Number of CRC errors detected
     */
    size_t scrubMemory()
    {
        size_t errors_detected = 0;

        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& region : memory_regions_) {
            if (region.scrub_callback) {
                region.scrub_callback();
            }
            else {
                errors_detected += scrubCrcRegion(region);
                // Re-arm checksums to the (possibly externally rewritten)
                // current contents for the next cycle
                computeBlockCrcs(region);
            }
        }

        stats_.scrub_cycles++;
        return errors_detected;
    }

    /**
     * @brief Perform one scrubbing cycle (alias for scrubMemory)
     *
     * @return Number of CRC errors detected
     */
    size_t scrubOnce() { return scrubMemory(); }

    /**
     * @brief Start background scrubbing thread
     *
     * @param interval_ms New interval in milliseconds (0 means use the
     *        interval passed to the constructor)
     * @return True if thread was started successfully
     */
    bool startBackgroundThread(unsigned long interval_ms = 0)
    {
        std::lock_guard<std::mutex> lock(thread_mutex_);

        if (running_.load()) {
            return true;
        }

        if (interval_ms > 0) {
            scrub_interval_ms_ = interval_ms;
        }

        if (scrub_interval_ms_ == 0) {
            return false;
        }

        terminate_requested_.store(false);

        try {
            scrub_thread_ = std::thread(&MemoryScrubber::scrubThreadFunction, this);
            running_.store(true);
            return true;
        }
        catch (const std::exception&) {
            running_.store(false);
            return false;
        }
    }

    /**
     * @brief Stop background scrubbing thread
     */
    void stopBackgroundThread()
    {
        std::lock_guard<std::mutex> lock(thread_mutex_);

        if (!running_.load()) {
            return;
        }

        terminate_requested_.store(true);

        if (scrub_thread_.joinable()) {
            scrub_thread_.join();
        }

        running_.store(false);
    }

    /// Alias for startBackgroundThread() using the constructor interval
    void start() { startBackgroundThread(); }

    /// Alias for stopBackgroundThread()
    void stop() { stopBackgroundThread(); }

    /**
     * @brief Check if background thread is running
     *
     * @return True if running
     */
    bool isRunning() const { return running_.load(); }

    /**
     * @brief Get the number of registered memory regions
     *
     * @return Number of registered memory regions
     */
    size_t getRegionCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return memory_regions_.size();
    }

    /**
     * @brief Get total memory size being scrubbed
     *
     * @return Total size in bytes
     */
    size_t getTotalMemorySize() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t total_size = 0;
        for (const auto& region : memory_regions_) {
            total_size += region.memory_size;
        }
        return total_size;
    }

    /**
     * @brief Statistics structure
     */
    struct Statistics {
        size_t scrub_cycles = 0;
        size_t errors_detected = 0;      ///< CRC block mismatches (detection-only)
        size_t last_error_time_ms = 0;   ///< Time since epoch of last error

        double error_rate = 0.0;         ///< Errors per megabyte per hour

        void updateErrorRate(size_t total_memory_bytes, unsigned long interval_ms)
        {
            if (scrub_cycles == 0 || total_memory_bytes == 0 || interval_ms == 0) {
                error_rate = 0.0;
                return;
            }

            const double errors_per_mb =
                static_cast<double>(errors_detected) /
                (static_cast<double>(total_memory_bytes) / 1024.0 / 1024.0);

            constexpr double ms_per_hour = 3600.0 * 1000.0;
            error_rate = errors_per_mb / (static_cast<double>(scrub_cycles) *
                                          static_cast<double>(interval_ms) / ms_per_hour);
        }
    };

    /**
     * @brief Get statistics
     *
     * @param update_rates Whether to update rate calculations before returning
     * @return Current statistics
     */
    Statistics getStatistics(bool update_rates = true) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (update_rates) {
            size_t total_size = 0;
            for (const auto& region : memory_regions_) {
                total_size += region.memory_size;
            }
            stats_.updateErrorRate(total_size, scrub_interval_ms_);
        }

        return stats_;
    }

    /**
     * @brief Reset statistics
     */
    void resetStatistics()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = Statistics();
    }

   private:
    static constexpr size_t kBlockSize = 64;

    struct MemoryRegion {
        size_t id = 0;
        void* memory_ptr = nullptr;
        size_t memory_size = 0;
        std::vector<uint32_t> block_crcs;      ///< One CRC per 64-byte block (CRC regions)
        std::function<void()> scrub_callback;  ///< Set for callback regions
    };

    mutable std::mutex mutex_;                  // Protects memory_regions_ and stats_
    mutable std::mutex thread_mutex_;           // Protects thread-related members
    std::vector<MemoryRegion> memory_regions_;  // Protected by mutex_
    unsigned long scrub_interval_ms_;           // Immutable after thread start
    std::atomic<bool> running_;                 // Thread running state
    std::atomic<bool> terminate_requested_;     // Signal to terminate thread
    std::thread scrub_thread_;                  // Background thread
    mutable Statistics stats_;                  // Protected by mutex_
    size_t next_region_id_ = 1;                 // Next available region ID

    /**
     * @brief Compute and store one CRC per 64-byte block of the region
     */
    static void computeBlockCrcs(MemoryRegion& region)
    {
        region.block_crcs.clear();
        const uint8_t* data = static_cast<const uint8_t*>(region.memory_ptr);
        for (size_t offset = 0; offset < region.memory_size; offset += kBlockSize) {
            const size_t block_size = std::min<size_t>(kBlockSize, region.memory_size - offset);
            region.block_crcs.push_back(core::Crc32::compute(data + offset, block_size));
        }
    }

    /**
     * @brief Verify a CRC region block by block
     *
     * @return Number of blocks whose CRC no longer matches
     */
    size_t scrubCrcRegion(MemoryRegion& region)
    {
        size_t errors_detected = 0;
        const uint8_t* data = static_cast<const uint8_t*>(region.memory_ptr);

        for (size_t block = 0; block < region.block_crcs.size(); ++block) {
            const size_t offset = block * kBlockSize;
            const size_t block_size = std::min<size_t>(kBlockSize, region.memory_size - offset);
            const uint32_t current_crc = core::Crc32::compute(data + offset, block_size);

            if (current_crc != region.block_crcs[block]) {
                errors_detected++;
                stats_.errors_detected++;
                stats_.last_error_time_ms = static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count());
            }
        }

        return errors_detected;
    }

    /**
     * @brief Background thread function
     */
    void scrubThreadFunction()
    {
        while (!terminate_requested_.load()) {
            // Sleep first (in 10ms slices so termination stays responsive)
            for (unsigned long i = 0; i < scrub_interval_ms_; i += 10) {
                if (terminate_requested_.load()) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            scrubMemory();
        }
    }
};

}  // namespace memory
}  // namespace rad_ml
