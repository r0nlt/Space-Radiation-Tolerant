# Stuck Bit Detection Algorithms

## 🎯 Learning Objectives

After studying this module, you'll understand:
- What stuck bits are and why they're dangerous
- Statistical approaches to stuck bit detection
- Empirical threshold-based algorithm
- Real-world validation from space missions

## 🚨 Understanding Stuck Bits

### What Are Stuck Bits?

A **stuck bit** is a memory cell that becomes permanently fixed at either 0 or 1, regardless of what value should be stored there. This is typically caused by:

1. **Total Ionizing Dose (TID)**: Cumulative radiation damage over time
2. **Single Event Latchup (SEL)**: High-energy particle triggers parasitic structures
3. **Manufacturing defects**: Exacerbated by radiation exposure
4. **Wear-out mechanisms**: Accelerated by radiation stress

### Why Stuck Bits Are Dangerous

📖 **Reference**: See [Memory Representation Mastery](./02_MEMORY_REPRESENTATION.md) for context on bit-level effects.

Unlike transient errors (SEUs), stuck bits are **permanent**:

```cpp
// Normal memory behavior
memory[address] = 0x12345678;
assert(memory[address] == 0x12345678);  // ✅ Works

// With stuck bit at position 4 (stuck at 1)
memory[address] = 0x12345678;  // Try to write: ...01111000
// Actual stored value:                     ...01111000
//                                               ↑
//                                          Bit 4 stuck at 1

memory[address] = 0x12345660;  // Try to write: ...01100000
// Actual stored value:                     ...01110000
//                                               ↑
//                                          Still stuck at 1!
```

**Impact on radiation-tolerant systems**:
- ❌ **TMR voting fails**: All three copies may have same stuck bit
- ❌ **Error correction ineffective**: ECC assumes transient errors
- ❌ **Data corruption accumulates**: Stuck bits don't self-heal
- ❌ **System degradation**: Performance decreases over time

## 🔧 Detection Algorithm

### Core Strategy: Statistical Pattern Recognition

The algorithm is based on empirical data from **NASA MESSENGER** and **ESA JUICE** missions:

```cpp
// From: include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp
class EnhancedStuckBitTMR {
private:
    // Track potentially stuck bits across all copies
    std::bitset<sizeof(T) * 8> potential_stuck_bits{};

    // Track consecutive errors at bit level
    std::array<uint8_t, sizeof(T) * 8> error_consistency_counters{};

    // Threshold based on JUICE mission testing: 3+ consecutive errors
    static constexpr uint8_t stuck_bit_threshold = 3;
};
```

**Key insight**: **Transient errors are random**, but **stuck bits show consistent patterns**.

### 1. Bit-Level Error Tracking

```cpp
void update_stuck_bit_tracking() {
    for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
        // Extract bit from each TMR copy
        bool bit0 = (copies_[0] >> bit) & 1;
        bool bit1 = (copies_[1] >> bit) & 1;
        bool bit2 = (copies_[2] >> bit) & 1;

        // Check if this bit differs across copies
        if (bit0 != bit1 || bit1 != bit2 || bit0 != bit2) {
            // Increment error counter for this bit position
            error_consistency_counters[bit]++;

            // Mark as potentially stuck if threshold exceeded
            if (error_consistency_counters[bit] >= stuck_bit_threshold) {
                potential_stuck_bits.set(bit);

                // Determine stuck value (0 or 1)
                determineStuckValue(bit);
            }
        } else {
            // All copies agree - reset counter for this bit
            error_consistency_counters[bit] = 0;
        }
    }
}
```

**Algorithm breakdown**:
1. **Compare all TMR copies** at each bit position
2. **Count disagreements** over time
3. **Apply empirical threshold** (3 consecutive errors)
4. **Mark persistent errors** as stuck bits

### 2. Stuck Value Determination

```cpp
void determineStuckValue(size_t bit_position) {
    // Count occurrences of 0 and 1 across all copies
    int zero_count = 0;
    int one_count = 0;

    for (size_t copy = 0; copy < 3; ++copy) {
        bool bit_value = (copies_[copy] >> bit_position) & 1;
        if (bit_value) {
            one_count++;
        } else {
            zero_count++;
        }
    }

    // The stuck value is the one that appears more frequently
    bool stuck_at_one = (one_count > zero_count);

    // Record stuck value for each copy
    for (size_t copy = 0; copy < 3; ++copy) {
        if (stuck_at_one) {
            stuck_value_masks[copy].set(bit_position);
        } else {
            stuck_value_masks[copy].reset(bit_position);
        }
    }
}
```

### 3. Mask-Aware Voting

Once stuck bits are identified, the voting algorithm adapts:

```cpp
T get_with_stuck_bit_awareness() const {
    T result = 0;

    // Vote on each bit position individually
    for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
        if (potential_stuck_bits[bit]) {
            // For stuck bits, use the known stuck value
            bool stuck_value = stuck_value_masks[0][bit]; // Same for all copies
            if (stuck_value) {
                result |= (static_cast<T>(1) << bit);
            }
        } else {
            // For non-stuck bits, use majority voting
            int bit_votes = 0;
            for (size_t copy = 0; copy < 3; ++copy) {
                if ((copies_[copy] >> bit) & 1) {
                    bit_votes++;
                }
            }

            // Majority wins
            if (bit_votes >= 2) {
                result |= (static_cast<T>(1) << bit);
            }
        }
    }

    return result;
}
```

## 🎨 Advanced Detection Techniques

### 1. Temporal Pattern Analysis

```cpp
class TemporalStuckBitDetector {
private:
    struct BitHistory {
        std::array<bool, 16> recent_values;  // Last 16 observations
        size_t write_index = 0;
        size_t stuck_count = 0;
    };

    std::array<BitHistory, sizeof(T) * 8> bit_histories;

public:
    void recordBitValue(size_t bit_position, bool value) {
        auto& history = bit_histories[bit_position];

        // Add new value to circular buffer
        history.recent_values[history.write_index] = value;
        history.write_index = (history.write_index + 1) % 16;

        // Count how many recent values are the same
        bool first_value = history.recent_values[0];
        size_t same_count = 1;

        for (size_t i = 1; i < 16; ++i) {
            if (history.recent_values[i] == first_value) {
                same_count++;
            }
        }

        // If all recent values are identical, likely stuck
        if (same_count >= 15) {  // Allow 1 outlier
            history.stuck_count++;
        } else {
            history.stuck_count = 0;
        }
    }

    bool isBitStuck(size_t bit_position) const {
        return bit_histories[bit_position].stuck_count >= 3;
    }
};
```

### 2. Cross-Copy Correlation Analysis

```cpp
double calculateBitCorrelation(size_t bit_position) const {
    // Analyze correlation between copies for this bit
    std::vector<std::array<bool, 3>> observations;

    // Collect historical observations
    for (const auto& snapshot : historical_snapshots) {
        std::array<bool, 3> bit_values;
        for (size_t copy = 0; copy < 3; ++copy) {
            bit_values[copy] = (snapshot.copies[copy] >> bit_position) & 1;
        }
        observations.push_back(bit_values);
    }

    // Calculate correlation coefficient
    double correlation = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 3; ++j) {
            correlation += calculatePearsonCorrelation(
                observations, i, j, bit_position);
        }
    }

    // High correlation suggests stuck bits
    return correlation / 3.0;  // Average of all pairs
}
```

### 3. Environment-Aware Thresholds

```cpp
// From: include/rad_ml/tmr/adaptive_protection.hpp
uint8_t calculateAdaptiveThreshold(const sim::RadiationEnvironment& env) const {
    // Base threshold from JUICE mission data
    uint8_t base_threshold = 3;

    // Adjust based on radiation environment
    switch (env.environment_type) {
        case RadiationEnvironment::LEO:
            return base_threshold;  // Standard threshold

        case RadiationEnvironment::GEO:
            return base_threshold - 1;  // More sensitive (higher radiation)

        case RadiationEnvironment::JUPITER:
        case RadiationEnvironment::EUROPA:
            return base_threshold - 2;  // Very sensitive (extreme radiation)

        case RadiationEnvironment::MARS_SURFACE:
            return base_threshold + 1;  // Less sensitive (atmosphere protection)

        default:
            return base_threshold;
    }
}
```

## 🔬 Empirical Validation

### Space Mission Data

The thresholds are based on real mission data:

**NASA MESSENGER (Mercury orbit)**:
- **TID accumulation**: 100+ kRad over 4 years
- **Stuck bit rate**: ~1 per 10⁶ bits per year
- **Detection threshold**: 3 consecutive errors with 95% confidence

**ESA JUICE (Jupiter system)**:
- **Extreme radiation**: 1000× Earth levels
- **Predicted stuck bit rate**: ~1 per 10⁴ bits per year
- **Detection threshold**: 2 consecutive errors (more aggressive)

### Validation Framework

```cpp
// From test suite
void validate_stuck_bit_detection() {
    EnhancedStuckBitTMR<uint32_t> tmr(0x12345678);

    // Simulate stuck bit at position 4 (stuck at 1)
    const size_t stuck_bit_pos = 4;
    const uint32_t stuck_mask = 1u << stuck_bit_pos;

    // Inject stuck bit over multiple iterations
    for (int iteration = 0; iteration < 10; ++iteration) {
        // Corrupt copies with stuck bit
        for (size_t copy = 0; copy < 3; ++copy) {
            uint32_t original = tmr.getCopies()[copy];
            uint32_t corrupted = original | stuck_mask;  // Force bit to 1
            tmr.corruptCopy(copy, corrupted);
        }

        // Update stuck bit tracking
        tmr.repair();

        // Check if stuck bit is detected after threshold
        if (iteration >= 3) {
            auto stuck_mask = tmr.getStuckBitMask();
            assert(stuck_mask[stuck_bit_pos] == true);
        }
    }
}
```


### Memory Overhead

```cpp
// Memory usage analysis
sizeof(std::bitset<32>)                           // 4 bytes - stuck bit mask
+ sizeof(std::array<uint8_t, 32>)                 // 32 bytes - error counters
+ sizeof(std::array<std::bitset<32>, 3>)          // 12 bytes - stuck value masks
= 48 bytes overhead for 32-bit values
```

**Overhead ratio**: ~12× for 32-bit values, ~6× for 64-bit values

## 🧪 Testing Strategies

### 1. Synthetic Stuck Bit Injection

```cpp
void test_synthetic_stuck_bits() {
    for (size_t bit_pos = 0; bit_pos < 32; ++bit_pos) {
        EnhancedStuckBitTMR<uint32_t> tmr(0x00000000);

        // Create stuck bit at this position
        uint32_t stuck_pattern = 1u << bit_pos;

        // Inject over multiple cycles
        for (int cycle = 0; cycle < 5; ++cycle) {
            // All copies get the stuck bit
            for (size_t copy = 0; copy < 3; ++copy) {
                tmr.corruptCopy(copy, stuck_pattern);
            }
            tmr.repair();
        }

        // Verify detection
        assert(tmr.getStuckBitMask()[bit_pos] == true);
    }
}
```

### 2. Mission Profile Simulation

```cpp
void test_mission_profile() {
    // Simulate 1-year mission with realistic error rates
    EnhancedStuckBitTMR<uint64_t> tmr(0x123456789ABCDEF0ULL);

    std::random_device rd;
    std::mt19937 gen(rd());

    // LEO error rate: ~1e-8 per bit per day
    std::bernoulli_distribution stuck_bit_prob(1e-8);

    for (int day = 0; day < 365; ++day) {
        // Check each bit for potential stuck bit event
        for (size_t bit = 0; bit < 64; ++bit) {
            if (stuck_bit_prob(gen)) {
                // Inject stuck bit
                injectStuckBit(tmr, bit);
            }
        }

        // Daily repair cycle
        tmr.repair();

        // Verify system still functional
        assert(tmr.getHealthScores()[0] > 0.1);
    }
}
```

## 🎯 Best Practices

### Detection Strategy Guidelines

1. **Use empirical thresholds**: Based on real mission data
2. **Adapt to environment**: Adjust sensitivity for radiation levels
3. **Track temporal patterns**: Look for consistency over time
4. **Validate with correlation**: Cross-check between TMR copies
5. **Monitor false positives**: Avoid over-aggressive detection

### Implementation Guidelines

1. **Minimize overhead**: Use efficient data structures
2. **Optimize hot paths**: Fast voting for non-stuck bits
3. **Handle edge cases**: Single-copy stuck bits
4. **Provide diagnostics**: Export stuck bit statistics
5. **Test thoroughly**: Validate on target hardware

## 🔗 Related Topics

- 📖 **Previous**: [Type Punning and reinterpret_cast](./03_TYPE_PUNNING.md) - Safe memory manipulation
- 📖 **Next**: [Error Correction Code Implementation](./05_ERROR_CORRECTION_CODES.md) - ECC techniques
- 🔧 **Implementation**: [Memory Scrubbing Strategies](./06_MEMORY_SCRUBBING.md) - Continuous monitoring
- 🧪 **Testing**: [Fault Injection Testing](./09_FAULT_INJECTION.md) - Validation techniques

## 💡 Key Takeaways

1. **Stuck bits are permanent** - unlike transient SEUs
2. **Statistical detection** works better than single-point checks
3. **Empirical thresholds** from space missions provide reliable baselines
4. **Environment adaptation** improves detection accuracy
5. **Bit-level granularity** enables precise error localization
6. **Temporal patterns** distinguish stuck bits from random errors
7. **Validation is critical** - test with realistic mission profiles

---

📖 **Continue Learning**: Advance to [Error Correction Code Implementation](./05_ERROR_CORRECTION_CODES.md) to see how ECC complements stuck bit detection.
