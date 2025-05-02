#pragma once

#include <chrono>
#include <functional>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

namespace rad_ml {
namespace core {
namespace memory {

/**
 * @brief Memory scrubber for radiation environments
 * 
 * Periodically scans memory regions to detect and correct bit flips caused
 * by radiation. This is essential in space environments where Single Event
 * Upsets (SEUs) can corrupt memory even when it's not being accessed.
 */
class MemoryScrubber {
public:
    /**
     * @brief Constructor
     * @param scrub_interval_ms How often to perform scrubbing in milliseconds
     */
    explicit MemoryScrubber(unsigned long scrub_interval_ms = 1000)
        : scrub_interval_ms_(scrub_interval_ms)
        , running_(false) {}
    
    /**
     * @brief Destructor - ensures scrubbing is stopped
     */
    ~MemoryScrubber() {
        stop();
    }
    
    /**
     * @brief Register a memory region to be scrubbed
     * 
     * @param memory_ptr Pointer to the memory region
     * @param size_bytes Size of the memory region in bytes
     * @param scrub_function Function to perform the scrubbing (verification and correction)
     * @return A handle that can be used to unregister the region
     */
    template <typename T>
    size_t registerMemoryRegion(
        T* memory_ptr,
        size_t size_bytes,
        std::function<void(T*, size_t)> scrub_function
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t handle = next_handle_++;
        
        regions_.push_back({
            handle,
            memory_ptr,
            size_bytes,
            [memory_ptr, size_bytes, scrub_function]() {
                scrub_function(memory_ptr, size_bytes);
            }
        });
        
        return handle;
    }
    
    /**
     * @brief Unregister a memory region
     * 
     * @param handle The handle returned from registerMemoryRegion
     * @return true if the region was found and unregistered, false otherwise
     */
    bool unregisterMemoryRegion(size_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto it = regions_.begin(); it != regions_.end(); ++it) {
            if (it->handle == handle) {
                regions_.erase(it);
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief Start the scrubbing thread
     */
    void start() {
        if (running_) {
            return;
        }
        
        running_ = true;
        scrub_thread_ = std::thread(&MemoryScrubber::scrubThreadFunction, this);
    }
    
    /**
     * @brief Stop the scrubbing thread
     */
    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        if (scrub_thread_.joinable()) {
            scrub_thread_.join();
        }
    }
    
    /**
     * @brief Perform one scrubbing cycle manually
     */
    void scrubOnce() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (const auto& region : regions_) {
            region.scrub_function();
        }
    }
    
private:
    struct MemoryRegion {
        size_t handle;
        void* memory_ptr;
        size_t size_bytes;
        std::function<void()> scrub_function;
    };
    
    std::vector<MemoryRegion> regions_;
    unsigned long scrub_interval_ms_;
    std::atomic<bool> running_;
    std::thread scrub_thread_;
    std::mutex mutex_;
    size_t next_handle_ = 0;
    
    void scrubThreadFunction() {
        while (running_) {
            scrubOnce();
            std::this_thread::sleep_for(std::chrono::milliseconds(scrub_interval_ms_));
        }
    }
};

} // namespace memory
} // namespace core
} // namespace rad_ml 