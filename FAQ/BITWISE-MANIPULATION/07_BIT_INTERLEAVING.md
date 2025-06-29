# Bit Interleaving and Burst Error Protection

## Educational Overview

Bit interleaving is a sophisticated technique that redistributes bits across memory to transform burst errors into correctable single-bit errors. This module explores the theoretical foundations, practical implementations, and optimization strategies for protecting against multi-bit upsets (MBUs) and burst errors in radiation environments.

## Table of Contents

1. [Scientific Foundation](#scientific-foundation)
2. [Multi-Bit Upset Characteristics](#multi-bit-upset-characteristics)
3. [Interleaving Algorithms](#interleaving-algorithms)
4. [Burst Error Transformation](#burst-error-transformation)
5. [Implementation Techniques](#implementation-techniques)
6. [Adaptive Protection Strategies](#adaptive-protection-strategies)
7. [Performance Optimization](#performance-optimization)
8. [Integration with Error Correction](#integration-with-error-correction)

---

## Scientific Foundation

### Radiation-Induced Error Patterns

In space environments, high-energy particles create distinct error patterns that challenge traditional single-bit error correction:

**Single Event Upsets (SEUs):**
- Random, isolated bit flips
- Well-handled by simple parity or Hamming codes
- Probability: ~10⁻¹⁰ errors/bit/day in deep space

**Multi-Bit Upsets (MBUs):**
- 2-8 adjacent bits affected by single particle
- Increasing frequency in modern memory technologies
- Challenge: Adjacent errors appear as uncorrectable burst errors

**Burst Errors:**
- Consecutive bit errors in data stream
- Can overwhelm error correction capacity
- Common in high-radiation environments during solar events

### The Interleaving Solution

Bit interleaving spatially redistributes bits to transform burst errors into correctable patterns:

```cpp
// Problem: Burst error overwhelms correction
Original:  [AAAAAAAA][BBBBBBBB][CCCCCCCC][DDDDDDDD]
Burst:     [XXXXXXXX][BBBBBBBB][CCCCCCCC][DDDDDDDD]  ← 8 consecutive errors

// Solution: Interleaving distributes the damage
Interleaved: [A₀B₀C₀D₀][A₁B₁C₁D₁][A₂B₂C₂D₂][A₃B₃C₃D₃]...
After burst: [XXXXXXXX][A₁B₁C₁D₁][A₂B₂C₂D₂][A₃B₃C₃D₃]...
Result:      [X₀X₁X₂X₃][A₁B₁C₁D₁][A₂B₂C₂D₂][A₃B₃C₃D₃]...  ← Single errors per symbol
```

---

## Multi-Bit Upset Characteristics

### MBU Classification

The framework categorizes multi-bit upsets based on spatial distribution:

```cpp
enum class MultibitUpsetType {
    SINGLE_BIT,     // Single bit flip (SEU)
    ADJACENT_BITS,  // Adjacent bits in the same word
    ROW_UPSET,      // Bits in the same row (memory layout)
    COLUMN_UPSET,   // Bits in the same column (memory layout)
    RANDOM_MULTI    // Random multiple bit flips
};
```

### Adjacent Bit Errors

Most common MBU pattern in modern memories:

```cpp
// Adjacent bits MBU simulation
case MultibitUpsetType::ADJACENT_BITS: {
    if (dist(rng) < error_rate) {
        // Typically 2-3 adjacent bits
        std::uniform_int_distribution<unsigned> bit_dist(0, sizeof(T) * 8 - 3);
        std::uniform_int_distribution<unsigned> len_dist(2, 3);

        unsigned start_bit = bit_dist(rng);
        unsigned num_bits = len_dist(rng);

        // Flip adjacent bits
        for (unsigned i = 0; i < num_bits; ++i) {
            data.bits ^= (1u << (start_bit + i));
        }
    }
    break;
}
```

### Row and Column Upsets

Memory layout-specific error patterns:

```cpp
// Row upset: Multiple bits in same memory row
case MultibitUpsetType::ROW_UPSET: {
    if (dist(rng) < error_rate) {
        // Choose a "row" (byte in this implementation)
        std::uniform_int_distribution<unsigned> byte_dist(0, sizeof(T) - 1);
        unsigned byte_idx = byte_dist(rng);

        // Flip multiple bits in this byte
        unsigned num_flips = 1 + static_cast<unsigned>(error_rate * 4);
        std::uniform_int_distribution<unsigned> bit_dist(0, 7);

        for (unsigned i = 0; i < num_flips; ++i) {
            unsigned bit_pos = bit_dist(rng);
            data.bytes[byte_idx] ^= (1u << bit_pos);
        }
    }
    break;
}

// Column upset: Same bit position in multiple bytes
case MultibitUpsetType::COLUMN_UPSET: {
    if (dist(rng) < error_rate) {
        // Choose a "column" (bit position)
        std::uniform_int_distribution<unsigned> bit_dist(0, 7);
        unsigned bit_pos = bit_dist(rng);

        // Flip this bit in multiple bytes
        unsigned num_bytes = 1 + static_cast<unsigned>(error_rate * (sizeof(T) - 1));

        for (unsigned i = 0; i < num_bytes; ++i) {
            unsigned byte_idx = byte_dist(rng) % sizeof(T);
            data.bytes[byte_idx] ^= (1u << bit_pos);
        }
    }
    break;
}
```

---

## Interleaving Algorithms

### Simple Bit Interleaving

Basic interleaving separates adjacent bits to non-adjacent positions:

```cpp
template<typename T>
T applyBitInterleaving(T value) {
    // Union for safe bit manipulation
    union {
        T value;
        uint32_t bits;
    } original, interleaved;

    original.value = value;
    interleaved.bits = 0;

    // Simple bit interleaving - separate adjacent bits
    for (int i = 0; i < 32; ++i) {
        // Even bits go to first half, odd bits to second half
        if (i % 2 == 0) {
            interleaved.bits |= ((original.bits >> i) & 1) << (i / 2);
        } else {
            interleaved.bits |= ((original.bits >> i) & 1) << (16 + i / 2);
        }
    }

    return interleaved.value;
}

// Deinterleaving function
template<typename T>
T undoBitInterleaving(T interleaved_value) {
    union {
        T value;
        uint32_t bits;
    } original, interleaved;

    interleaved.value = interleaved_value;
    original.bits = 0;

    // Reconstruct original bit pattern
    for (int i = 0; i < 16; ++i) {
        original.bits |= ((interleaved.bits >> i) & 1) << (i * 2);
        original.bits |= ((interleaved.bits >> (i + 16)) & 1) << (i * 2 + 1);
    }

    return original.value;
}
```

**Protection Analysis:**
- **Adjacent 2-bit MBU**: Separated into 2 single-bit errors
- **Adjacent 3-bit MBU**: Becomes 2 single-bit + 1 double-bit error
- **Effectiveness**: Transforms most MBUs into correctable patterns

### Advanced Block Interleaving

For Reed-Solomon and other block codes, interleaving operates on symbol level:

```cpp
template<size_t BlockSize>
class BlockInterleaver {
private:
    static constexpr size_t INTERLEAVE_DEPTH = 8;  // Interleave across 8 symbols

public:
    std::vector<uint8_t> interleave(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> result(data.size());

        // Calculate parameters
        size_t block_count = (data.size() + BlockSize - 1) / BlockSize;
        size_t symbols_per_block = BlockSize;

        // Interleave each bit position across blocks
        for (size_t bit = 0; bit < 8; ++bit) {
            for (size_t block = 0; block < block_count; ++block) {
                for (size_t symbol = 0; symbol < symbols_per_block; ++symbol) {
                    size_t src_idx = block * symbols_per_block + symbol;
                    size_t dst_idx = (bit * block_count + block) * symbols_per_block + symbol;

                    if (src_idx < data.size() && dst_idx < result.size()) {
                        // Extract bit from source
                        bool bit_value = (data[src_idx] >> bit) & 1;

                        // Place in interleaved position
                        if (bit_value) {
                            result[dst_idx / 8] |= (1 << (dst_idx % 8));
                        }
                    }
                }
            }
        }

        return result;
    }

    std::vector<uint8_t> deinterleave(const std::vector<uint8_t>& interleaved_data) {
        std::vector<uint8_t> result(interleaved_data.size());

        // Reverse the interleaving process
        size_t block_count = (interleaved_data.size() + BlockSize - 1) / BlockSize;
        size_t symbols_per_block = BlockSize;

        for (size_t bit = 0; bit < 8; ++bit) {
            for (size_t block = 0; block < block_count; ++block) {
                for (size_t symbol = 0; symbol < symbols_per_block; ++symbol) {
                    size_t src_idx = (bit * block_count + block) * symbols_per_block + symbol;
                    size_t dst_idx = block * symbols_per_block + symbol;

                    if (src_idx < interleaved_data.size() && dst_idx < result.size()) {
                        // Extract bit from interleaved position
                        bool bit_value = (interleaved_data[src_idx / 8] >> (src_idx % 8)) & 1;

                        // Place in original position
                        if (bit_value) {
                            result[dst_idx] |= (1 << bit);
                        }
                    }
                }
            }
        }

        return result;
    }
};
```

### Convolutional Interleaving

Memory-efficient interleaving for continuous data streams:

```cpp
template<size_t Depth>
class ConvolutionalInterleaver {
private:
    std::array<std::queue<uint8_t>, Depth> delay_lines_;
    size_t input_counter_ = 0;

public:
    uint8_t interleave(uint8_t input_symbol) {
        // Select delay line based on input position
        size_t line_index = input_counter_ % Depth;

        // Add to delay line
        delay_lines_[line_index].push(input_symbol);

        // Output symbol with appropriate delay
        uint8_t output = 0;
        if (delay_lines_[line_index].size() > line_index) {
            output = delay_lines_[line_index].front();
            delay_lines_[line_index].pop();
        }

        input_counter_++;
        return output;
    }

    uint8_t deinterleave(uint8_t input_symbol) {
        // Reverse the interleaving process
        size_t line_index = input_counter_ % Depth;

        // Add to delay line with reverse delay
        size_t reverse_delay = Depth - 1 - line_index;
        delay_lines_[reverse_delay].push(input_symbol);

        // Output symbol
        uint8_t output = 0;
        if (delay_lines_[reverse_delay].size() > reverse_delay) {
            output = delay_lines_[reverse_delay].front();
            delay_lines_[reverse_delay].pop();
        }

        input_counter_++;
        return output;
    }
};
```

---

## Burst Error Transformation

### Error Pattern Analysis

Understanding how interleaving transforms error patterns:

```cpp
class ErrorPatternAnalyzer {
public:
    struct ErrorPattern {
        std::vector<size_t> error_positions;
        size_t max_consecutive_errors;
        size_t total_errors;
        double burst_ratio;  // Ratio of burst to total errors
    };

    static ErrorPattern analyzePattern(const std::vector<uint8_t>& original,
                                     const std::vector<uint8_t>& corrupted) {
        ErrorPattern pattern;

        // Find all error positions
        for (size_t i = 0; i < std::min(original.size(), corrupted.size()); ++i) {
            if (original[i] != corrupted[i]) {
                // Check each bit in the byte
                uint8_t diff = original[i] ^ corrupted[i];
                for (int bit = 0; bit < 8; ++bit) {
                    if (diff & (1 << bit)) {
                        pattern.error_positions.push_back(i * 8 + bit);
                    }
                }
            }
        }

        pattern.total_errors = pattern.error_positions.size();

        // Calculate maximum consecutive errors
        pattern.max_consecutive_errors = calculateMaxConsecutive(pattern.error_positions);

        // Calculate burst ratio
        if (pattern.total_errors > 0) {
            pattern.burst_ratio = static_cast<double>(pattern.max_consecutive_errors) /
                                 pattern.total_errors;
        }

        return pattern;
    }

private:
    static size_t calculateMaxConsecutive(const std::vector<size_t>& positions) {
        if (positions.empty()) return 0;

        size_t max_consecutive = 1;
        size_t current_consecutive = 1;

        for (size_t i = 1; i < positions.size(); ++i) {
            if (positions[i] == positions[i-1] + 1) {
                current_consecutive++;
            } else {
                max_consecutive = std::max(max_consecutive, current_consecutive);
                current_consecutive = 1;
            }
        }

        return std::max(max_consecutive, current_consecutive);
    }
};
```

### Interleaving Effectiveness Measurement

```cpp
class InterleavingEffectiveness {
public:
    struct EffectivenessMetrics {
        double burst_reduction_ratio;     // How much burst length is reduced
        double correctable_error_ratio;   // Fraction of errors that become correctable
        double protection_efficiency;     // Overall protection improvement
    };

    template<typename InterleaverType>
    static EffectivenessMetrics measureEffectiveness(
        const std::vector<uint8_t>& original_data,
        const std::vector<uint8_t>& burst_corrupted_data,
        InterleaverType& interleaver) {

        // Apply interleaving to original data
        auto interleaved_original = interleaver.interleave(original_data);

        // Apply same burst error pattern to interleaved data
        auto interleaved_corrupted = simulateBurstOnInterleaved(
            interleaved_original, burst_corrupted_data, original_data);

        // Analyze error patterns
        auto original_pattern = ErrorPatternAnalyzer::analyzePattern(
            original_data, burst_corrupted_data);
        auto interleaved_pattern = ErrorPatternAnalyzer::analyzePattern(
            interleaved_original, interleaved_corrupted);

        EffectivenessMetrics metrics;

        // Calculate burst reduction
        metrics.burst_reduction_ratio =
            static_cast<double>(original_pattern.max_consecutive_errors) /
            std::max(1UL, interleaved_pattern.max_consecutive_errors);

        // Estimate correctable error ratio (assumes Reed-Solomon with t=4)
        constexpr size_t correction_capability = 4;
        metrics.correctable_error_ratio =
            (interleaved_pattern.max_consecutive_errors <= correction_capability) ? 1.0 :
            static_cast<double>(correction_capability) / interleaved_pattern.max_consecutive_errors;

        // Overall protection efficiency
        metrics.protection_efficiency =
            metrics.burst_reduction_ratio * metrics.correctable_error_ratio;

        return metrics;
    }

private:
    static std::vector<uint8_t> simulateBurstOnInterleaved(
        const std::vector<uint8_t>& interleaved_original,
        const std::vector<uint8_t>& burst_corrupted,
        const std::vector<uint8_t>& original) {

        // Find burst error positions in original data
        std::vector<size_t> error_positions;
        for (size_t i = 0; i < std::min(original.size(), burst_corrupted.size()); ++i) {
            if (original[i] != burst_corrupted[i]) {
                error_positions.push_back(i);
            }
        }

        // Apply equivalent errors to interleaved data
        auto result = interleaved_original;
        for (size_t pos : error_positions) {
            if (pos < result.size()) {
                // Apply same error pattern
                result[pos] = burst_corrupted[pos];
            }
        }

        return result;
    }
};
```

---

## Implementation Techniques

### Memory-Efficient Interleaving

For resource-constrained systems, in-place interleaving minimizes memory overhead:

```cpp
template<typename T>
class InPlaceInterleaver {
public:
    static void interleaveBits(T& value) {
        // Use bit manipulation for in-place interleaving
        union {
            T typed_value;
            uint32_t bits;
        } data;

        data.typed_value = value;
        uint32_t original = data.bits;
        data.bits = 0;

        // Separate even and odd bits
        for (int i = 0; i < 32; i += 2) {
            // Even bits go to lower half
            data.bits |= ((original >> i) & 1) << (i / 2);
            // Odd bits go to upper half
            if (i + 1 < 32) {
                data.bits |= ((original >> (i + 1)) & 1) << (16 + i / 2);
            }
        }

        value = data.typed_value;
    }

    static void deinterleaveBits(T& value) {
        union {
            T typed_value;
            uint32_t bits;
        } data;

        data.typed_value = value;
        uint32_t interleaved = data.bits;
        data.bits = 0;

        // Reconstruct original pattern
        for (int i = 0; i < 16; ++i) {
            // Restore even bits
            data.bits |= ((interleaved >> i) & 1) << (i * 2);
            // Restore odd bits
            data.bits |= ((interleaved >> (i + 16)) & 1) << (i * 2 + 1);
        }

        value = data.typed_value;
    }
};
```

### Hardware-Accelerated Interleaving

Leveraging SIMD instructions for performance:

```cpp
#ifdef __AVX2__
#include <immintrin.h>

class SIMDInterleaver {
public:
    static void interleaveBlock256(const uint8_t* input, uint8_t* output) {
        // Load 32 bytes (256 bits) at once
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));

        // Separate even and odd bits using bit manipulation
        __m256i even_mask = _mm256_set1_epi8(0x55);  // 01010101
        __m256i odd_mask = _mm256_set1_epi8(0xAA);   // 10101010

        __m256i even_bits = _mm256_and_si256(data, even_mask);
        __m256i odd_bits = _mm256_and_si256(data, odd_mask);

        // Shift odd bits to even positions
        odd_bits = _mm256_srli_epi16(odd_bits, 1);

        // Pack even bits to lower half
        __m256i packed_even = _mm256_packus_epi16(even_bits, _mm256_setzero_si256());

        // Pack odd bits to upper half
        __m256i packed_odd = _mm256_packus_epi16(_mm256_setzero_si256(), odd_bits);

        // Combine and store
        __m256i result = _mm256_or_si256(packed_even, packed_odd);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), result);
    }
};
#endif
```

### Cache-Optimized Interleaving

Optimizing memory access patterns for performance:

```cpp
class CacheOptimizedInterleaver {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t BLOCK_SIZE = CACHE_LINE_SIZE * 8;  // 512 bytes

public:
    static void interleaveData(const std::vector<uint8_t>& input,
                              std::vector<uint8_t>& output) {
        output.resize(input.size());

        // Process data in cache-friendly blocks
        for (size_t block_start = 0; block_start < input.size(); block_start += BLOCK_SIZE) {
            size_t block_end = std::min(block_start + BLOCK_SIZE, input.size());

            // Interleave within this block
            interleaveBlock(input.data() + block_start,
                          output.data() + block_start,
                          block_end - block_start);
        }
    }

private:
    static void interleaveBlock(const uint8_t* input, uint8_t* output, size_t size) {
        // Process cache lines
        for (size_t i = 0; i < size; i += CACHE_LINE_SIZE) {
            size_t line_size = std::min(CACHE_LINE_SIZE, size - i);

            // Interleave bits within cache line
            for (size_t j = 0; j < line_size; ++j) {
                uint8_t byte = input[i + j];
                uint8_t interleaved = 0;

                // Separate even and odd bits
                for (int bit = 0; bit < 8; bit += 2) {
                    interleaved |= ((byte >> bit) & 1) << (bit / 2);
                    if (bit + 1 < 8) {
                        interleaved |= ((byte >> (bit + 1)) & 1) << (4 + bit / 2);
                    }
                }

                output[i + j] = interleaved;
            }
        }
    }
};
```

---

## Adaptive Protection Strategies

### Environment-Aware Interleaving

Adapting interleaving depth based on radiation environment:

```cpp
class AdaptiveInterleaver {
public:
    enum class RadiationEnvironment {
        LOW_EARTH_ORBIT,    // Minimal interleaving needed
        GEOSTATIONARY,      // Moderate interleaving
        DEEP_SPACE,         // Standard interleaving
        SOLAR_STORM,        // Maximum interleaving
        JUPITER_MISSION     // Extreme interleaving
    };

private:
    struct EnvironmentConfig {
        size_t interleave_depth;
        size_t block_size;
        bool enable_convolutional;
        double mbu_probability;
    };

    std::map<RadiationEnvironment, EnvironmentConfig> configs_ = {
        {RadiationEnvironment::LOW_EARTH_ORBIT, {2, 64, false, 1e-12}},
        {RadiationEnvironment::GEOSTATIONARY, {4, 128, false, 1e-11}},
        {RadiationEnvironment::DEEP_SPACE, {8, 256, true, 1e-10}},
        {RadiationEnvironment::SOLAR_STORM, {16, 512, true, 1e-8}},
        {RadiationEnvironment::JUPITER_MISSION, {32, 1024, true, 1e-7}}
    };

    RadiationEnvironment current_environment_ = RadiationEnvironment::DEEP_SPACE;

public:
    void adaptToEnvironment(RadiationEnvironment env) {
        current_environment_ = env;

        // Log adaptation
        auto& config = configs_[env];
        std::cout << "Adapted interleaving for environment: "
                  << "depth=" << config.interleave_depth
                  << ", block_size=" << config.block_size
                  << ", convolutional=" << (config.enable_convolutional ? "yes" : "no")
                  << std::endl;
    }

    std::vector<uint8_t> interleave(const std::vector<uint8_t>& data) {
        auto& config = configs_[current_environment_];

        if (config.enable_convolutional) {
            return convolutionalInterleave(data, config.interleave_depth);
        } else {
            return blockInterleave(data, config.block_size, config.interleave_depth);
        }
    }

private:
    std::vector<uint8_t> blockInterleave(const std::vector<uint8_t>& data,
                                        size_t block_size,
                                        size_t depth) {
        // Implementation depends on block size and depth
        BlockInterleaver<256> interleaver;  // Use template specialization
        return interleaver.interleave(data);
    }

    std::vector<uint8_t> convolutionalInterleave(const std::vector<uint8_t>& data,
                                                size_t depth) {
        std::vector<uint8_t> result;
        result.reserve(data.size());

        ConvolutionalInterleaver<16> interleaver;  // Use appropriate depth

        for (uint8_t byte : data) {
            result.push_back(interleaver.interleave(byte));
        }

        return result;
    }
};
```

### Real-Time MBU Detection and Response

```cpp
class MBUDetector {
private:
    struct MBUStatistics {
        size_t single_bit_errors = 0;
        size_t adjacent_bit_errors = 0;
        size_t burst_errors = 0;
        double mbu_rate = 0.0;
    };

    MBUStatistics stats_;
    std::chrono::steady_clock::time_point last_update_;

public:
    void reportError(const std::vector<size_t>& error_positions) {
        // Analyze error pattern
        if (error_positions.size() == 1) {
            stats_.single_bit_errors++;
        } else if (areAdjacent(error_positions)) {
            stats_.adjacent_bit_errors++;
        } else {
            stats_.burst_errors++;
        }

        // Update MBU rate
        updateMBURate();

        // Trigger adaptation if needed
        if (stats_.mbu_rate > 0.1) {  // 10% MBU rate threshold
            recommendInterleavingIncrease();
        }
    }

    double getMBURate() const {
        return stats_.mbu_rate;
    }

private:
    bool areAdjacent(const std::vector<size_t>& positions) {
        if (positions.size() < 2) return false;

        auto sorted_positions = positions;
        std::sort(sorted_positions.begin(), sorted_positions.end());

        for (size_t i = 1; i < sorted_positions.size(); ++i) {
            if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                return false;
            }
        }

        return true;
    }

    void updateMBURate() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_update_).count();

        if (duration > 0) {
            size_t total_multi_bit = stats_.adjacent_bit_errors + stats_.burst_errors;
            size_t total_errors = stats_.single_bit_errors + total_multi_bit;

            if (total_errors > 0) {
                stats_.mbu_rate = static_cast<double>(total_multi_bit) / total_errors;
            }
        }

        last_update_ = now;
    }

    void recommendInterleavingIncrease() {
        std::cout << "WARNING: High MBU rate detected ("
                  << (stats_.mbu_rate * 100) << "%). "
                  << "Recommend increasing interleaving depth." << std::endl;
    }
};
```

---

## Performance Optimization

### Lookup Table Acceleration

Pre-computed interleaving patterns for common data sizes:

```cpp
template<size_t DataWidth>
class LookupTableInterleaver {
private:
    static constexpr size_t TABLE_SIZE = 1 << DataWidth;
    std::array<uint32_t, TABLE_SIZE> interleave_table_;
    std::array<uint32_t, TABLE_SIZE> deinterleave_table_;

public:
    LookupTableInterleaver() {
        // Pre-compute all possible interleaving patterns
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            interleave_table_[i] = computeInterleaved(static_cast<uint32_t>(i));
            deinterleave_table_[interleave_table_[i]] = static_cast<uint32_t>(i);
        }
    }

    uint32_t interleave(uint32_t value) const {
        if constexpr (DataWidth <= 16) {
            return interleave_table_[value & ((1 << DataWidth) - 1)];
        } else {
            // Fall back to computation for larger values
            return computeInterleaved(value);
        }
    }

    uint32_t deinterleave(uint32_t value) const {
        if constexpr (DataWidth <= 16) {
            return deinterleave_table_[value & ((1 << DataWidth) - 1)];
        } else {
            return computeDeinterleaved(value);
        }
    }

private:
    uint32_t computeInterleaved(uint32_t value) const {
        uint32_t result = 0;

        // Separate even and odd bits
        for (int i = 0; i < DataWidth; i += 2) {
            result |= ((value >> i) & 1) << (i / 2);
            if (i + 1 < DataWidth) {
                result |= ((value >> (i + 1)) & 1) << (DataWidth / 2 + i / 2);
            }
        }

        return result;
    }

    uint32_t computeDeinterleaved(uint32_t value) const {
        uint32_t result = 0;

        // Reconstruct original pattern
        for (int i = 0; i < DataWidth / 2; ++i) {
            result |= ((value >> i) & 1) << (i * 2);
            result |= ((value >> (DataWidth / 2 + i)) & 1) << (i * 2 + 1);
        }

        return result;
    }
};
```

### Parallel Processing

Multi-threaded interleaving for large datasets:

```cpp
class ParallelInterleaver {
private:
    size_t num_threads_;

public:
    explicit ParallelInterleaver(size_t num_threads = std::thread::hardware_concurrency())
        : num_threads_(num_threads) {}

    std::vector<uint8_t> interleaveParallel(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> result(data.size());

        // Calculate work distribution
        size_t chunk_size = data.size() / num_threads_;
        std::vector<std::thread> threads;

        // Launch worker threads
        for (size_t t = 0; t < num_threads_; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads_ - 1) ? data.size() : start + chunk_size;

            threads.emplace_back([&, start, end]() {
                interleaveChunk(data.data() + start,
                              result.data() + start,
                              end - start);
            });
        }

        // Wait for completion
        for (auto& thread : threads) {
            thread.join();
        }

        return result;
    }

private:
    void interleaveChunk(const uint8_t* input, uint8_t* output, size_t size) {
        InPlaceInterleaver<uint8_t> interleaver;

        for (size_t i = 0; i < size; ++i) {
            uint8_t value = input[i];
            interleaver.interleaveBits(value);
            output[i] = value;
        }
    }
};
```

---

## Integration with Error Correction

### Reed-Solomon with Interleaving

Combining interleaving with Reed-Solomon codes for maximum protection:

```cpp
template<size_t N, size_t K>
class InterleavedReedSolomon {
private:
    ReedSolomonCodec<N, K> rs_codec_;
    BlockInterleaver<N> interleaver_;

public:
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
        // Step 1: Apply Reed-Solomon encoding
        auto encoded = rs_codec_.encode(data);

        // Step 2: Apply interleaving to encoded data
        auto interleaved = interleaver_.interleave(encoded);

        return interleaved;
    }

    std::optional<std::vector<uint8_t>> decode(const std::vector<uint8_t>& received) {
        // Step 1: Deinterleave received data
        auto deinterleaved = interleaver_.deinterleave(received);

        // Step 2: Apply Reed-Solomon decoding
        return rs_codec_.decode(deinterleaved);
    }

    // Test effectiveness against burst errors
    double testBurstProtection(size_t burst_length) {
        // Generate test data
        std::vector<uint8_t> test_data(K);
        std::iota(test_data.begin(), test_data.end(), 0);

        // Encode with interleaving
        auto encoded = encode(test_data);

        // Apply burst error
        auto corrupted = encoded;
        for (size_t i = 0; i < std::min(burst_length, encoded.size()); ++i) {
            corrupted[i] ^= 0xFF;  // Flip all bits
        }

        // Try to decode
        auto decoded = decode(corrupted);

        // Check if successful
        if (decoded && *decoded == test_data) {
            return 1.0;  // 100% success
        } else {
            // Measure partial recovery
            if (decoded) {
                size_t correct_bytes = 0;
                for (size_t i = 0; i < std::min(decoded->size(), test_data.size()); ++i) {
                    if ((*decoded)[i] == test_data[i]) {
                        correct_bytes++;
                    }
                }
                return static_cast<double>(correct_bytes) / test_data.size();
            }
            return 0.0;  // Complete failure
        }
    }
};
```

### TMR with Bit-Level Interleaving

Enhancing Triple Modular Redundancy with interleaving:

```cpp
template<typename T>
class InterleavedTMR {
private:
    std::array<T, 3> replicas_;
    InPlaceInterleaver<T> interleaver_;

public:
    explicit InterleavedTMR(const T& initial_value) {
        set(initial_value);
    }

    void set(const T& value) {
        // Store three copies with different interleaving patterns
        replicas_[0] = value;                                    // Original
        replicas_[1] = value; interleaver_.interleaveBits(replicas_[1]);  // Interleaved
        replicas_[2] = value;
        // Apply double interleaving to third copy
        interleaver_.interleaveBits(replicas_[2]);
        interleaver_.interleaveBits(replicas_[2]);
    }

    T get() const {
        // Normalize all replicas to original pattern
        T normalized[3];
        normalized[0] = replicas_[0];  // Already in original pattern

        normalized[1] = replicas_[1];
        interleaver_.deinterleaveBits(normalized[1]);

        normalized[2] = replicas_[2];
        interleaver_.deinterleaveBits(normalized[2]);
        interleaver_.deinterleaveBits(normalized[2]);

        // Perform majority voting
        if (normalized[0] == normalized[1]) return normalized[0];
        if (normalized[0] == normalized[2]) return normalized[0];
        if (normalized[1] == normalized[2]) return normalized[1];

        // No majority - return first value (could implement more sophisticated recovery)
        return normalized[0];
    }

    bool verify() const {
        T value = get();

        // Check if all normalized replicas match
        T normalized[3];
        normalized[0] = replicas_[0];

        normalized[1] = replicas_[1];
        interleaver_.deinterleaveBits(normalized[1]);

        normalized[2] = replicas_[2];
        interleaver_.deinterleaveBits(normalized[2]);
        interleaver_.deinterleaveBits(normalized[2]);

        return (normalized[0] == value) &&
               (normalized[1] == value) &&
               (normalized[2] == value);
    }

    void repair() {
        T correct_value = get();
        set(correct_value);
    }
};
```

---

## Key Takeaways

### Essential Principles

1. **Spatial Distribution**: Interleaving transforms burst errors into distributed single-bit errors
2. **Adaptive Depth**: Interleaving complexity should match radiation environment severity
3. **Error Pattern Awareness**: Different MBU types require different interleaving strategies
4. **Performance Balance**: Trade-off between protection level and computational overhead

### Implementation Guidelines

1. **Memory Efficiency**: Use in-place algorithms when memory is constrained
2. **Cache Optimization**: Align interleaving blocks with cache line boundaries
3. **SIMD Acceleration**: Leverage vector instructions for performance
4. **Environment Adaptation**: Dynamically adjust interleaving based on error rates

### Integration Strategies

1. **ECC Synergy**: Combine interleaving with Reed-Solomon or other block codes
2. **TMR Enhancement**: Use different interleaving patterns for TMR replicas
3. **Real-Time Monitoring**: Implement MBU detection for adaptive protection
4. **System-Level Design**: Consider interleaving at multiple levels (bit, byte, block)

### Performance Considerations

1. **Lookup Tables**: Pre-compute patterns for frequently used data widths
2. **Parallel Processing**: Distribute interleaving across multiple threads
3. **Hardware Support**: Utilize dedicated interleaving hardware when available
4. **Memory Bandwidth**: Optimize access patterns to minimize bandwidth requirements

Bit interleaving represents a sophisticated approach to radiation tolerance that transforms the fundamental nature of error patterns. By spatially redistributing bits, it converts challenging burst errors into manageable single-bit errors that can be efficiently corrected by conventional error correction codes. The techniques presented here demonstrate how theoretical concepts translate into practical, high-performance implementations suitable for mission-critical space applications.
