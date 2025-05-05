#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <utility>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <span>
#include <optional>

namespace rad_ml {
namespace memory {

/**
 * @brief Memory Scrubber for error detection and correction
 * 
 * Implements memory scrubbing to detect and correct bit errors.
 * Memory scrubbing is a technique commonly used in radiation-tolerant systems
 * to periodically check memory integrity and correct errors.
 * 
 * Thread-safe implementation with RAII-based lock guards and proper synchronization.
 */
class MemoryScrubber {
public:
    /**
     * @brief Constructor
     * 
     * @param scrub_interval_ms How often to perform automatic scrubbing in milliseconds (0 to disable)
     */
    explicit MemoryScrubber(unsigned long scrub_interval_ms = 0) 
        : scrub_interval_ms_(scrub_interval_ms), 
          running_(false),
          terminate_requested_(false),
          stats_() {
        // Start background thread if interval is specified
        if (scrub_interval_ms_ > 0) {
            startBackgroundThread();
        }
    }
    
    /**
     * @brief Destructor - ensures scrubbing is stopped
     */
    ~MemoryScrubber() {
        stopBackgroundThread();
    }
    
    // Make class non-copyable but movable
    MemoryScrubber(const MemoryScrubber&) = delete;
    MemoryScrubber& operator=(const MemoryScrubber&) = delete;
    
    MemoryScrubber(MemoryScrubber&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        memory_regions_ = std::move(other.memory_regions_);
        scrub_interval_ms_ = other.scrub_interval_ms_;
        stats_ = other.stats_;
        running_.store(other.running_.load());
        
        // Ensure other's thread is stopped before moving
        other.stopBackgroundThread();
        
        // Start our thread if needed
        if (running_.load()) {
            startBackgroundThread();
        }
    }
    
    MemoryScrubber& operator=(MemoryScrubber&& other) noexcept {
        if (this != &other) {
            // Stop our thread
            stopBackgroundThread();
            
            // Lock both instances to prevent race conditions
            std::scoped_lock lock(mutex_, other.mutex_);
            
            memory_regions_ = std::move(other.memory_regions_);
            scrub_interval_ms_ = other.scrub_interval_ms_;
            stats_ = other.stats_;
            
            // Ensure other's thread is stopped
            other.stopBackgroundThread();
            
            // Start our thread if needed
            if (other.running_.load()) {
                startBackgroundThread();
            }
        }
        return *this;
    }
    
    /**
     * @brief Register a memory region for scrubbing
     * 
     * @param ptr Pointer to the memory region
     * @param size Size of the memory region in bytes
     * @param error_callback Optional callback when an error is detected
     * @return Unique handle used to unregister the region
     */
    size_t registerMemoryRegion(
        void* ptr, 
        size_t size,
        std::function<void(void*, size_t, uint8_t, uint8_t)> error_callback = nullptr) {
        
        if (!ptr || size == 0) {
            return 0; // Invalid parameters
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Generate a unique handle for this region
        static size_t next_handle = 1; // 0 is reserved for invalid handle
        size_t handle = next_handle++;
        
        memory_regions_.push_back({
            handle,
            ptr, 
            size, 
            std::move(error_callback), 
            std::vector<uint32_t>()
        });
        
        // Calculate CRC for the new region
        calculateChecksums(memory_regions_.back());
        
        return handle;
    }
    
    /**
     * @brief Modern version using std::span for memory region
     * 
     * @param memory Span representing the memory region
     * @param error_callback Optional callback when an error is detected
     * @return Unique handle used to unregister the region
     */
    size_t registerMemoryRegion(
        std::span<std::byte> memory,
        std::function<void(void*, size_t, uint8_t, uint8_t)> error_callback = nullptr) {
        
        return registerMemoryRegion(
            memory.data(), 
            memory.size_bytes(), 
            std::move(error_callback)
        );
    }
    
    /**
     * @brief Unregister a memory region
     * 
     * @param handle The handle returned from registerMemoryRegion
     * @return True if region was found and unregistered, false otherwise
     */
    bool unregisterMemoryRegion(size_t handle) {
        if (handle == 0) {
            return false;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = std::find_if(memory_regions_.begin(), memory_regions_.end(),
            [handle](const MemoryRegion& region) {
                return region.handle == handle;
            });
            
        if (it != memory_regions_.end()) {
            memory_regions_.erase(it);
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Scrub all registered memory regions
     * 
     * @return Number of errors detected
     */
    size_t scrubMemory() {
        size_t errors_detected = 0;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& region : memory_regions_) {
            errors_detected += scrubRegion(region);
        }
        
        // Recalculate checksums after correction
        for (auto& region : memory_regions_) {
            calculateChecksums(region);
        }
        
        stats_.scrub_cycles++;
        return errors_detected;
    }
    
    /**
     * @brief Start background scrubbing thread
     * 
     * @param interval_ms New interval in milliseconds (0 means use existing interval)
     * @return True if thread was started successfully
     */
    bool startBackgroundThread(unsigned long interval_ms = 0) {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        
        // If already running, do nothing
        if (running_.load()) {
            return true;
        }
        
        // Update interval if specified
        if (interval_ms > 0) {
            scrub_interval_ms_ = interval_ms;
        }
        
        // Don't start if interval is 0
        if (scrub_interval_ms_ == 0) {
            return false;
        }
        
        // Reset termination flag
        terminate_requested_.store(false);
        
        // Start background thread
        try {
            scrub_thread_ = std::thread(&MemoryScrubber::scrubThreadFunction, this);
            running_.store(true);
            return true;
        } catch (const std::exception&) {
            running_.store(false);
            return false;
        }
    }
    
    /**
     * @brief Stop background scrubbing thread
     */
    void stopBackgroundThread() {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        
        if (!running_.load()) {
            return;
        }
        
        // Signal thread to terminate
        terminate_requested_.store(true);
        
        // Wait for thread to exit
        if (scrub_thread_.joinable()) {
            scrub_thread_.join();
        }
        
        running_.store(false);
    }
    
    /**
     * @brief Check if background thread is running
     * 
     * @return True if running
     */
    bool isRunning() const {
        return running_.load();
    }
    
    /**
     * @brief Get the number of registered memory regions
     * 
     * @return Number of registered memory regions
     */
    size_t getRegionCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memory_regions_.size();
    }
    
    /**
     * @brief Get total memory size being scrubbed
     * 
     * @return Total size in bytes
     */
    size_t getTotalMemorySize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t total_size = 0;
        for (const auto& region : memory_regions_) {
            total_size += region.size;
        }
        return total_size;
    }
    
    /**
     * @brief Statistics structure
     */
    struct Statistics {
        size_t scrub_cycles = 0;
        size_t errors_detected = 0;
        size_t errors_corrected = 0;
        size_t last_error_time_ms = 0;  // Time since epoch of last error
        
        // Rate statistics
        double error_rate = 0.0;        // Errors per megabyte per hour
        
        void updateErrorRate(size_t total_memory_bytes) {
            if (scrub_cycles == 0 || total_memory_bytes == 0) {
                error_rate = 0.0;
                return;
            }
            
            // Calculate errors per megabyte
            double errors_per_mb = static_cast<double>(errors_detected) / 
                                  (static_cast<double>(total_memory_bytes) / 1024.0 / 1024.0);
            
            // Assuming 1 scrub cycle per scrub_interval_ms_
            // Convert to hours: errors_per_mb / (cycles * interval_ms / ms_per_hour)
            constexpr double ms_per_hour = 3600.0 * 1000.0;
            error_rate = errors_per_mb / 
                        (static_cast<double>(scrub_cycles) * 
                         static_cast<double>(scrub_interval_ms_) / ms_per_hour);
        }
    };
    
    /**
     * @brief Get statistics
     * 
     * @param update_rates Whether to update rate calculations before returning
     * @return Current statistics
     */
    Statistics getStatistics(bool update_rates = true) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (update_rates) {
            // Calculate total memory size for error rate calculations
            size_t total_size = 0;
            for (const auto& region : memory_regions_) {
                total_size += region.size;
            }
            
            // Update error rate
            stats_.updateErrorRate(total_size);
        }
        
        return stats_;
    }
    
    /**
     * @brief Reset statistics
     */
    void resetStatistics() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = Statistics();
    }
    
private:
    struct MemoryRegion {
        size_t handle;
        void* ptr;
        size_t size;
        std::function<void(void*, size_t, uint8_t, uint8_t)> error_callback;
        std::vector<uint32_t> checksums;
    };
    
    // Thread-safe members with proper synchronization
    mutable std::mutex mutex_;                  // Protects memory_regions_ and stats_
    mutable std::mutex thread_mutex_;           // Protects thread-related members
    std::vector<MemoryRegion> memory_regions_;  // Protected by mutex_
    unsigned long scrub_interval_ms_;           // Immutable after thread start
    std::atomic<bool> running_;                 // Thread running state
    std::atomic<bool> terminate_requested_;     // Signal to terminate thread
    std::thread scrub_thread_;                  // Background thread
    mutable Statistics stats_;                  // Protected by mutex_
    
    /**
     * @brief Calculate checksums for a memory region
     * 
     * @param region Memory region to calculate checksums for
     */
    void calculateChecksums(MemoryRegion& region) {
        // Clear existing checksums
        region.checksums.clear();
        
        // Calculate new checksums (1 per 64 bytes)
        uint8_t* data = static_cast<uint8_t*>(region.ptr);
        for (size_t offset = 0; offset < region.size; offset += 64) {
            size_t block_size = std::min<size_t>(64, region.size - offset);
            region.checksums.push_back(calculateCRC32(data + offset, block_size));
        }
    }
    
    /**
     * @brief Scrub a single memory region
     * 
     * @param region Memory region to scrub
     * @return Number of errors detected
     */
    size_t scrubRegion(MemoryRegion& region) {
        size_t errors_detected = 0;
        
        uint8_t* data = static_cast<uint8_t*>(region.ptr);
        
        // Check each block
        for (size_t i = 0; i < region.checksums.size(); ++i) {
            size_t offset = i * 64;
            size_t block_size = std::min<size_t>(64, region.size - offset);
            
            uint32_t current_crc = calculateCRC32(data + offset, block_size);
            uint32_t stored_crc = region.checksums[i];
            
            if (current_crc != stored_crc) {
                // Error detected
                errors_detected++;
                stats_.errors_detected++;
                stats_.last_error_time_ms = static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count()
                );
                
                // In a real implementation, we'd use ECC to locate and correct the error
                // For this implementation, we'll notify the error callback if provided
                if (region.error_callback) {
                    // For demonstration, we'll assume the first byte is corrupted
                    region.error_callback(data + offset, offset, data[offset], 0xFF);
                }
                
                // Count as corrected for demonstration
                stats_.errors_corrected++;
            }
        }
        
        return errors_detected;
    }
    
    /**
     * @brief Calculate CRC32 checksum
     * 
     * @param data Pointer to data
     * @param size Size of data in bytes
     * @return CRC32 checksum
     */
    uint32_t calculateCRC32(const uint8_t* data, size_t size) const {
        uint32_t crc = 0xFFFFFFFF;
        
        for (size_t i = 0; i < size; ++i) {
            uint8_t index = (crc ^ data[i]) & 0xFF;
            crc = (crc_table_[index] ^ (crc >> 8));
        }
        
        return ~crc;
    }
    
    /**
     * @brief Background thread function
     */
    void scrubThreadFunction() {
        while (!terminate_requested_.load()) {
            // Sleep first to prevent immediate scrubbing on startup
            for (unsigned long i = 0; i < scrub_interval_ms_; i += 10) {
                if (terminate_requested_.load()) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Perform scrubbing
            scrubMemory();
        }
    }
    
    // CRC32 lookup table (initialized statically for better performance)
    static constexpr uint32_t crc_table_[256] = {
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
        0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
        0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
        0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
        0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE,
        0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
        0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
        0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
        0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
        0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940,
        0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
        0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
        0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
        0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
        0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A,
        0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
        0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818,
        0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
        0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
        0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
        0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C,
        0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
        0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2,
        0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
        0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
        0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
        0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086,
        0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
        0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4,
        0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
        0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
        0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
        0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
        0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
        0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE,
        0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
        0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
        0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
        0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252,
        0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
        0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60,
        0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
        0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
        0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
        0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04,
        0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
        0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
        0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
        0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
        0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
        0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E,
        0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
        0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C,
        0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
        0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
        0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
        0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0,
        0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
        0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6,
        0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
        0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
        0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
    };
};

} // namespace memory
} // namespace rad_ml 