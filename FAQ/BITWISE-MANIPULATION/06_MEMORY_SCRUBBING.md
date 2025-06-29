# Memory Scrubbing Techniques for Radiation-Tolerant Systems

## Educational Overview

Memory scrubbing is a proactive fault tolerance technique that periodically examines memory contents to detect and correct radiation-induced errors before they accumulate or propagate. This module explores the theoretical foundations, practical implementations, and optimization strategies for memory scrubbing in space environments.

## Table of Contents

1. [Scientific Foundation](#scientific-foundation)
2. [Error Detection Mechanisms](#error-detection-mechanisms)
3. [Scrubbing Strategies](#scrubbing-strategies)
4. [Implementation Techniques](#implementation-techniques)
5. [Adaptive Scrubbing Algorithms](#adaptive-scrubbing-algorithms)
6. [Performance Optimization](#performance-optimization)
7. [Integration with Protection Systems](#integration-with-protection-systems)
8. [Mission-Critical Applications](#mission-critical-applications)

---

## Scientific Foundation

### Radiation-Induced Memory Errors

In space environments, cosmic radiation causes various types of memory corruption:

**Single Event Upsets (SEUs):**
- Random bit flips in memory cells
- Occur even when memory is not being accessed
- Accumulate over time without detection

**Multi-Bit Upsets (MBUs):**
- Multiple bits affected by single particle
- More challenging to detect and correct
- Increasing frequency in modern memory technologies

**Stuck Bits:**
- Permanent bit failures
- Require different handling than transient errors
- Can mask other errors if not properly managed

### Scrubbing Necessity

Memory scrubbing addresses the fundamental challenge that errors can accumulate in unused memory regions:

```cpp
// Problem: Undetected errors accumulate
float neural_weights[10000];  // Large weight matrix
// ... weights loaded during initialization
// ... long mission duration without access
// Result: Multiple accumulated bit flips when finally accessed
```

---

## Error Detection Mechanisms

### CRC-32 Checksums

The framework implements IEEE 802.3 CRC-32 for error detection:

```cpp
class MemoryScrubber {
private:
    // Optimized lookup table for CRC-32 calculation
    static constexpr uint32_t crc_table_[256] = {
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
        0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
        // ... complete 256-entry table
    };

    uint32_t calculateCRC32(const uint8_t* data, size_t size) const {
        uint32_t crc = 0xFFFFFFFF;  // Initial value

        for (size_t i = 0; i < size; ++i) {
            uint8_t index = (crc ^ data[i]) & 0xFF;
            crc = (crc_table_[index] ^ (crc >> 8));
        }

        return ~crc;  // Final XOR
    }
};
```

**Technical Benefits:**
- **Fast Computation**: Lookup table eliminates polynomial division
- **Strong Detection**: Detects all single-bit and most multi-bit errors
- **Low Overhead**: 32 bits per protected region

### Block-Based Protection

Memory regions are divided into manageable blocks for efficient scrubbing:

```cpp
void calculateChecksums(MemoryRegion& region) {
    // Process in 64-byte blocks for cache efficiency
    uint8_t* data = static_cast<uint8_t*>(region.memory_ptr);

    for (size_t offset = 0; offset < region.memory_size; offset += 64) {
        size_t block_size = std::min<size_t>(64, region.memory_size - offset);

        // Calculate block checksum
        uint32_t block_crc = calculateCRC32(data + offset, block_size);

        // Store or compare with expected value
        region.calculated_crc = block_crc;
    }
}
```

---

## Scrubbing Strategies

### 1. Periodic Scrubbing

Regular interval-based scrubbing provides predictable protection:

```cpp
enum class ScrubbingStrategy {
    PERIODIC,    // Regular interval scrubbing
    CONTINUOUS,  // Continuous background scrubbing
    TRIGGERED,   // Scrubbing triggered by error detection
    ADAPTIVE     // Adaptive rate based on environment
};

// Periodic implementation
void scrubThreadFunction() {
    while (!terminate_requested_.load()) {
        // Configurable sleep interval
        for (unsigned long i = 0; i < scrub_interval_ms_; i += 10) {
            if (terminate_requested_.load()) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Perform scrubbing cycle
        scrubMemory();
    }
}
```

**Advantages:**
- Predictable resource usage
- Simple implementation
- Suitable for low-radiation environments

**Disadvantages:**
- May waste resources in benign periods
- May be insufficient during radiation storms

### 2. Continuous Scrubbing

Constant background scrubbing for maximum protection:

```cpp
// Continuous scrubbing with round-robin region selection
size_t scrubContinuous() {
    static size_t current_region = 0;
    size_t errors_detected = 0;

    std::lock_guard<std::mutex> lock(mutex_);

    if (!memory_regions_.empty()) {
        // Scrub one region per call
        auto& region = memory_regions_[current_region];
        errors_detected = scrubRegion(region);

        // Move to next region
        current_region = (current_region + 1) % memory_regions_.size();

        // Update checksums after correction
        calculateChecksums(region);
    }

    return errors_detected;
}
```

### 3. Triggered Scrubbing

Error-driven scrubbing activation:

```cpp
class TriggeredScrubber {
private:
    size_t error_threshold_ = 3;  // Trigger after 3 errors
    size_t recent_errors_ = 0;

public:
    void onErrorDetected() {
        recent_errors_++;

        if (recent_errors_ >= error_threshold_) {
            // Trigger intensive scrubbing
            performIntensiveScrub();
            recent_errors_ = 0;  // Reset counter
        }
    }

    void performIntensiveScrub() {
        // Scrub all regions immediately
        for (auto& region : memory_regions_) {
            scrubRegion(region);
        }
    }
};
```

### 4. Adaptive Scrubbing

Environment-aware scrubbing rate adjustment:

```cpp
class AdaptiveScrubber {
private:
    double base_interval_ms_ = 1000.0;
    double current_interval_ms_ = 1000.0;
    double error_rate_ = 0.0;

public:
    void updateEnvironment(size_t errors_detected, size_t total_memory_mb) {
        // Calculate current error rate (errors per MB per hour)
        updateErrorRate(errors_detected, total_memory_mb);

        // Adjust scrubbing interval based on error rate
        if (error_rate_ > 0.1) {
            // High error rate: increase scrubbing frequency
            current_interval_ms_ = base_interval_ms_ * 0.1;
        } else if (error_rate_ > 0.01) {
            // Medium error rate: moderate increase
            current_interval_ms_ = base_interval_ms_ * 0.5;
        } else {
            // Low error rate: standard interval
            current_interval_ms_ = base_interval_ms_;
        }

        // Update scrubber with new interval
        updateScrubInterval(static_cast<unsigned long>(current_interval_ms_));
    }

private:
    void updateErrorRate(size_t errors, size_t memory_mb) {
        constexpr double ms_per_hour = 3600.0 * 1000.0;

        if (memory_mb > 0) {
            error_rate_ = static_cast<double>(errors) /
                         (static_cast<double>(memory_mb) *
                          (current_interval_ms_ / ms_per_hour));
        }
    }
};
```

---

## Implementation Techniques

### Thread-Safe Memory Region Management

```cpp
class MemoryScrubber {
private:
    struct MemoryRegion {
        size_t id;
        void* memory_ptr;
        size_t memory_size;
        std::chrono::steady_clock::time_point last_scrub_time;
        bool ecc_enabled;
        bool crc_enabled;
        uint32_t calculated_crc;
    };

    // Thread synchronization
    mutable std::mutex mutex_;                  // Protects memory_regions_
    mutable std::mutex thread_mutex_;           // Protects thread operations
    std::vector<MemoryRegion> memory_regions_;
    std::atomic<bool> running_;
    std::atomic<bool> terminate_requested_;

public:
    size_t registerMemoryRegion(void* memory_ptr, size_t memory_size) {
        MemoryRegion region;
        region.id = next_region_id_++;
        region.memory_ptr = memory_ptr;
        region.memory_size = memory_size;
        region.last_scrub_time = std::chrono::steady_clock::now();
        region.ecc_enabled = false;
        region.crc_enabled = true;

        // Calculate initial checksum
        region.calculated_crc = calculateCRC32(
            static_cast<const uint8_t*>(memory_ptr), memory_size);

        std::lock_guard<std::mutex> lock(mutex_);
        memory_regions_.push_back(region);

        return region.id;
    }
};
```

### Error Correction Integration

```cpp
size_t scrubRegion(MemoryRegion& region) {
    size_t errors_detected = 0;
    uint8_t* data = static_cast<uint8_t*>(region.memory_ptr);

    // Check integrity using CRC
    uint32_t current_crc = calculateCRC32(data, region.memory_size);

    if (current_crc != region.calculated_crc) {
        errors_detected++;
        stats_.errors_detected++;

        // Record error timestamp
        stats_.last_error_time_ms = static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );

        // Apply error correction if available
        if (region.ecc_enabled) {
            // ECC-based correction
            bool corrected = applyECCCorrection(data, region.memory_size);
            if (corrected) {
                stats_.errors_corrected++;
                // Recalculate CRC after correction
                region.calculated_crc = calculateCRC32(data, region.memory_size);
            }
        } else {
            // For demonstration: assume correction from backup
            // In practice, would use Reed-Solomon or other ECC
            stats_.errors_corrected++;
        }
    }

    return errors_detected;
}
```

---

## Adaptive Scrubbing Algorithms

### Environment-Based Rate Control

```cpp
class EnvironmentAwareScrubber {
private:
    struct EnvironmentProfile {
        std::string name;
        double seu_rate;          // SEUs per bit per day
        double recommended_interval_ms;
    };

    std::vector<EnvironmentProfile> environments_ = {
        {"LEO", 1e-10, 500.0},           // Low Earth Orbit
        {"GEO", 1e-9, 200.0},            // Geostationary Orbit
        {"DEEP_SPACE", 1e-8, 100.0},     // Deep Space
        {"SOLAR_STORM", 1e-6, 10.0}      // Solar Particle Event
    };

public:
    void adaptToEnvironment(const std::string& env_name) {
        auto it = std::find_if(environments_.begin(), environments_.end(),
            [&env_name](const EnvironmentProfile& profile) {
                return profile.name == env_name;
            });

        if (it != environments_.end()) {
            updateScrubInterval(static_cast<unsigned long>(it->recommended_interval_ms));

            // Log environment change
            std::cout << "Adapted scrubbing for " << env_name
                      << " environment (interval: "
                      << it->recommended_interval_ms << " ms)" << std::endl;
        }
    }
};
```

### Statistical Rate Optimization

```cpp
struct Statistics {
    size_t scrub_cycles = 0;
    size_t errors_detected = 0;
    size_t errors_corrected = 0;
    double error_rate = 0.0;  // Errors per MB per hour

    void updateErrorRate(size_t total_memory_bytes, unsigned long interval_ms) {
        if (scrub_cycles == 0 || total_memory_bytes == 0) {
            error_rate = 0.0;
            return;
        }

        // Calculate errors per megabyte
        double errors_per_mb = static_cast<double>(errors_detected) /
                              (static_cast<double>(total_memory_bytes) / (1024.0 * 1024.0));

        // Convert to hourly rate
        constexpr double ms_per_hour = 3600.0 * 1000.0;
        error_rate = errors_per_mb /
                    (static_cast<double>(scrub_cycles) *
                     static_cast<double>(interval_ms) / ms_per_hour);
    }

    // Recommend optimal scrubbing interval
    unsigned long recommendInterval() const {
        if (error_rate < 0.001) {
            return 5000;  // 5 seconds for low error rate
        } else if (error_rate < 0.01) {
            return 1000;  // 1 second for medium error rate
        } else {
            return 100;   // 100ms for high error rate
        }
    }
};
```

---

## Performance Optimization

### Cache-Aware Block Processing

```cpp
// Optimize for CPU cache line size (typically 64 bytes)
constexpr size_t CACHE_LINE_SIZE = 64;

void optimizedScrub(MemoryRegion& region) {
    uint8_t* data = static_cast<uint8_t*>(region.memory_ptr);
    size_t blocks = (region.memory_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;

    for (size_t i = 0; i < blocks; ++i) {
        size_t offset = i * CACHE_LINE_SIZE;
        size_t block_size = std::min(CACHE_LINE_SIZE, region.memory_size - offset);

        // Process cache-aligned block
        uint32_t block_crc = calculateCRC32(data + offset, block_size);

        // Compare with stored checksum for this block
        if (block_crc != getStoredCRC(region.id, i)) {
            handleBlockError(region, i, offset, block_size);
        }
    }
}
```

### Interleaved Scrubbing

```cpp
class InterleavedScrubber {
private:
    size_t current_region_index_ = 0;
    size_t blocks_per_cycle_ = 4;  // Process 4 blocks per cycle

public:
    void performInterleavedScrub() {
        if (memory_regions_.empty()) return;

        // Get current region
        auto& region = memory_regions_[current_region_index_];

        // Calculate total blocks in region
        size_t total_blocks = (region.memory_size + 63) / 64;  // 64-byte blocks

        // Process a few blocks from this region
        for (size_t i = 0; i < blocks_per_cycle_ && i < total_blocks; ++i) {
            size_t block_offset = (region.current_block + i) * 64;

            if (block_offset < region.memory_size) {
                scrubBlock(region, block_offset);
            }
        }

        // Update position for next cycle
        region.current_block = (region.current_block + blocks_per_cycle_) % total_blocks;

        // Move to next region if current is complete
        if (region.current_block == 0) {
            current_region_index_ = (current_region_index_ + 1) % memory_regions_.size();
        }
    }
};
```

---

## Integration with Protection Systems

### TMR Integration

```cpp
template<typename T>
class TMRMemoryScrubber {
public:
    void scrubTMRValue(EnhancedTMR<T>& tmr_value) {
        // Get all three replicas
        T replica1 = tmr_value.getRawCopy(0);
        T replica2 = tmr_value.getRawCopy(1);
        T replica3 = tmr_value.getRawCopy(2);

        // Perform majority voting
        T correct_value;
        if (replica1 == replica2) {
            correct_value = replica1;
        } else if (replica1 == replica3) {
            correct_value = replica1;
        } else if (replica2 == replica3) {
            correct_value = replica2;
        } else {
            // All different - use error correction
            correct_value = recoverFromTripleError(replica1, replica2, replica3);
        }

        // Repair any incorrect replicas
        bool any_corrected = false;
        for (size_t i = 0; i < 3; ++i) {
            if (tmr_value.getRawCopy(i) != correct_value) {
                tmr_value.getRawCopy(i) = correct_value;
                any_corrected = true;
            }
        }

        if (any_corrected) {
            stats_.errors_corrected++;
        }
    }
};
```

### ECC Integration

```cpp
class ECCMemoryScrubber {
private:
    ReedSolomonECC<255, 223> ecc_codec_;  // RS(255,223) code

public:
    bool scrubWithECC(uint8_t* data, size_t size) {
        bool any_corrected = false;

        // Process data in ECC block sizes
        for (size_t offset = 0; offset < size; offset += 223) {
            size_t block_size = std::min<size_t>(223, size - offset);

            // Decode and correct
            std::vector<uint8_t> corrected_data;
            bool corrected = ecc_codec_.decode(
                data + offset, block_size, corrected_data);

            if (corrected) {
                // Copy corrected data back
                std::memcpy(data + offset, corrected_data.data(), block_size);
                any_corrected = true;
            }
        }

        return any_corrected;
    }
};
```

---

## Mission-Critical Applications

### Neural Network Weight Protection

```cpp
class NeuralNetworkScrubber {
private:
    MemoryScrubber scrubber_;
    std::vector<size_t> layer_handles_;

public:
    void protectNeuralNetwork(const std::vector<LayerWeights>& layers) {
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];

            // Register weight matrices for scrubbing
            size_t handle = scrubber_.registerMemoryRegion(
                const_cast<float*>(layer.weights.data()),
                layer.weights.size() * sizeof(float)
            );

            layer_handles_.push_back(handle);

            // Set higher scrubbing frequency for critical layers
            if (layer.is_critical) {
                scrubber_.setECCProtection(handle, true);
            }
        }

        // Start background scrubbing
        scrubber_.startBackgroundThread(500);  // 500ms interval
    }

    void adaptScrubbing(double mission_criticality) {
        unsigned long interval = static_cast<unsigned long>(1000.0 / mission_criticality);
        scrubber_.startBackgroundThread(std::max(100UL, interval));
    }
};
```

### Mission Phase Adaptation

```cpp
class MissionAwareScrubber {
public:
    enum class MissionPhase {
        LAUNCH,
        CRUISE,
        SCIENCE_OPERATIONS,
        EMERGENCY
    };

    void adaptToMissionPhase(MissionPhase phase) {
        switch (phase) {
            case MissionPhase::LAUNCH:
                // High vibration, moderate radiation
                configureScrubbing(1000, ScrubbingStrategy::PERIODIC);
                break;

            case MissionPhase::CRUISE:
                // Predictable environment, power conservation
                configureScrubbing(5000, ScrubbingStrategy::ADAPTIVE);
                break;

            case MissionPhase::SCIENCE_OPERATIONS:
                // Maximum reliability required
                configureScrubbing(200, ScrubbingStrategy::CONTINUOUS);
                break;

            case MissionPhase::EMERGENCY:
                // Intensive protection
                configureScrubbing(50, ScrubbingStrategy::CONTINUOUS);
                break;
        }
    }

private:
    void configureScrubbing(unsigned long interval, ScrubbingStrategy strategy) {
        scrubber_.stopBackgroundThread();

        // Update configuration
        AcceleratorConfig config;
        config.scrubbing_strategy = strategy;
        config.scrubbing_interval_sec = interval / 1000.0;

        scrubber_.startBackgroundThread(interval);
    }
};
```

---

## Key Takeaways

### Essential Principles

1. **Proactive Protection**: Scrubbing prevents error accumulation in unused memory
2. **Adaptive Strategies**: Environment-aware scrubbing optimizes resource usage
3. **Block-Based Processing**: Cache-aligned blocks improve performance
4. **Statistical Optimization**: Error rate monitoring guides interval adjustment

### Implementation Guidelines

1. **Thread Safety**: Use proper synchronization for concurrent access
2. **Resource Management**: Balance protection level with power/performance constraints
3. **Error Handling**: Integrate with broader error correction systems
4. **Mission Awareness**: Adapt scrubbing to mission phase requirements

### Performance Considerations

1. **Cache Efficiency**: Align scrubbing blocks with CPU cache lines
2. **Interleaved Processing**: Distribute scrubbing load across regions
3. **Adaptive Intervals**: Adjust frequency based on observed error rates
4. **Integration Overhead**: Minimize impact on primary system functions

Memory scrubbing represents a critical component of radiation-tolerant system design, providing the proactive error detection and correction necessary for long-duration space missions. The techniques presented here demonstrate how theoretical principles translate into practical, high-performance implementations suitable for mission-critical applications.
