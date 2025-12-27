# Memory Representation Mastery

## 🎯 Learning Objectives

After studying this module, you'll understand:
- How different data types are represented in memory
- Techniques for bit-level manipulation of floating-point numbers
- Memory layout considerations for radiation tolerance
- Union-based type punning for safe bit manipulation

## 🧠 Fundamental Concepts

### Memory as a Sequence of Bits

Every piece of data in a computer is ultimately stored as a sequence of bits. Understanding that **radiation affects individual bits** is fundamental - protection must operate at the bit level.

```cpp
// A 32-bit integer in memory
int value = 0x12345678;
// Memory layout (little-endian):
// Address:  [n+3] [n+2] [n+1] [n+0]
// Bytes:    0x12  0x34  0x56  0x78
// Bits:     00010010 00110100 01010110 01111000
```

### Why Bit-Level Matters for Radiation

📖 **Reference**: See [Branchless Programming Fundamentals](./01_BRANCHLESS_PROGRAMMING.md) for context on radiation effects.

**Single Event Upsets (SEUs)** typically affect individual bits:
```
Original: 0x12345678 = 00010010001101000101011001111000
After SEU:0x12345679 = 00010010001101000101011001111001
                                                      ↑
                                                   One bit flipped
```

## 🔧 Space-Radiation-Tolerant Bit Manipulation Toolkit

### 1. Union-Based Type Punning

Use unions to safely reinterpret memory:

```cpp
// From: include/rad_ml/utils/bit_manipulation.hpp
static float flipBit(float value, int bit_position) {
    // Validate bit position bounds
    if (bit_position < 0 || bit_position >= 32) {
        return value; // Invalid bit position
    }

    // Use union to reinterpret float as uint32_t for bit manipulation
    union {
        float f;
        uint32_t i;
    } converter;

    converter.f = value;

    // Flip the specified bit using XOR
    converter.i ^= (1u << bit_position);

    return converter.f;
}
```

**Why unions are safe here**:
- Both `float` and `uint32_t` are 32 bits
- No undefined behavior when switching between active members
- Preserves exact bit pattern

### 2. IEEE 754 Floating-Point Manipulation

```
IEEE 754 Single Precision (32-bit):
┌─┬─────────┬───────────────────────┐
│S│ Exponent│      Mantissa         │
└─┴─────────┴───────────────────────┘
 31 30    23 22                   0

S = Sign bit (1 bit)
Exponent = Biased exponent (8 bits)
Mantissa = Fractional part (23 bits)
```

**Radiation hardening strategy**:
- **Sign bit errors**: Change positive to negative (or vice versa)
- **Exponent errors**: Can cause massive magnitude changes
- **Mantissa errors**: Affect precision but may be tolerable

### 3. Byte-Level Access Patterns

```cpp
// From: include/rad_ml/neural/adaptive_protection.hpp
// Example: Counting bit differences between two values
template <typename U>
size_t count_bit_differences(const U& a, const U& b) const {
    const uint8_t* bytes_a = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* bytes_b = reinterpret_cast<const uint8_t*>(&b);

    size_t differences = 0;

    for (size_t i = 0; i < sizeof(U); ++i) {
        uint8_t diff = bytes_a[i] ^ bytes_b[i];

        // Count bits in the difference
        for (int bit = 0; bit < 8; ++bit) {
            if ((diff >> bit) & 1) {
                differences++;
            }
        }
    }

    return differences;
}
```

## 🎨 Advanced Memory Manipulation Techniques

### 1. Stuck Bit Pattern Detection

```cpp
// From: include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp
void update_stuck_bit_tracking() {
    // Get the current value based on standard health-weighted voting
    T voted_value = get_standard();

    // Analyze each copy for potential stuck bits
    for (size_t copy_idx = 0; copy_idx < copies_.size(); ++copy_idx) {
        // Skip completely healthy copies
        if (copies_[copy_idx] == voted_value) {
            continue;
        }

        // Calculate XOR to identify differing bits
        T diff = copies_[copy_idx] ^ voted_value;

        // Check each differing bit
        for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
            if ((diff >> bit) & 1) {
                // Increment error consistency counter
                if (error_consistency_counters[bit] < 255) {
                    error_consistency_counters[bit]++;
                }

                // Record stuck bit value (0 or 1) for this copy
                bool current_bit_value = (copies_[copy_idx] >> bit) & 1;
                stuck_value_masks[copy_idx][bit] = current_bit_value;

                // Mark bit as potentially stuck if threshold reached
                if (error_consistency_counters[bit] >= stuck_bit_threshold) {
                    potential_stuck_bits.set(bit);
                }
            }
        }
    }
}
```

**Key insights**:
- **Bit-level granularity**: Tracks errors at individual bit positions
- **Statistical approach**: Uses error frequency to identify stuck bits
- **Empirical threshold**: Based on JUICE mission data (3 consecutive errors)

### 2. Memory Interleaving for Burst Protection

```cpp
// From: include/rad_ml/neural/advanced_reed_solomon.hpp
std::vector<uint8_t> interleave(const std::vector<uint8_t>& data) const {
    if (data.empty()) return {};

    // Determine how many blocks we need
    size_t block_count = (data.size() + 7) / 8;
    std::vector<uint8_t> result(data.size());

    // Process each bit position across all bytes
    for (size_t bit = 0; bit < 8; ++bit) {
        for (size_t block = 0; block < block_count; ++block) {
            size_t src_idx = block;
            size_t dst_idx = bit * block_count + block;

            if (src_idx < data.size() && dst_idx < result.size()) {
                // Extract bit from source byte
                bool bit_value = (data[src_idx] >> bit) & 1;

                // Set bit in destination byte
                if (bit_value) {
                    result[dst_idx / 8] |= (1 << (dst_idx % 8));
                }
            }
        }
    }

    return result;
}
```

**Burst error mitigation**:
```
Original:    [AAAAAAAA][BBBBBBBB][CCCCCCCC]
After burst: [XXXXXXXX][BBBBBBBB][CCCCCCCC]  ← 8 consecutive errors

Interleaved: [A₀B₀C₀...][A₁B₁C₁...][A₂B₂C₂...]
After burst: [XXXXXXXX][A₁B₁C₁...][A₂B₂C₂...]  ← Distributed errors
```

### 3. Fixed-Point Arithmetic for Determinism

```cpp
// From: include/rad_ml/math/fixed_point.hpp
template <unsigned IntBits, unsigned FracBits, typename T = std::int32_t>
class FixedPoint {
    static_assert(std::is_integral_v<T>, "Base type must be integral");
    static_assert(IntBits + FracBits <= sizeof(T) * 8, "Total bits must fit in underlying type");

public:
    static constexpr T scale = static_cast<T>(1) << FracBits;

    constexpr FixedPoint operator*(const FixedPoint& other) const noexcept {
        // Use wider type to prevent overflow during multiplication
        using wider_t = typename std::conditional<
            sizeof(T) <= 4,
            std::int64_t,
            std::int64_t  // Would use int128_t if available
        >::type;

        wider_t wide_result = static_cast<wider_t>(value_) * other.value_;

        // Scale back down with bit shift
        FixedPoint result;
        result.value_ = static_cast<T>(wide_result >> FracBits);
        return result;
    }

private:
    T value_;  // The fixed-point value
};

// Common type aliases
using Fixed16_16 = FixedPoint<16, 16, std::int32_t>;  // 16.16 fixed-point
using Fixed8_24 = FixedPoint<8, 24, std::int32_t>;    // 8.24 fixed-point
```

**Advantages over floating-point**:
- **Deterministic**: Same inputs always produce same bit patterns
- **No special cases**: No NaN, infinity, or denormal numbers
- **Simpler hardware**: Integer ALU operations only

## 🔬 Deep Dive: Endianness Considerations

### Little-Endian vs Big-Endian

The code is endianness-aware:

```cpp
// 32-bit value: 0x12345678

// Little-endian (x86):
// Address: [n+0] [n+1] [n+2] [n+3]
// Bytes:   0x78  0x56  0x34  0x12

// Big-endian (some ARM, SPARC):
// Address: [n+0] [n+1] [n+2] [n+3]
// Bytes:   0x12  0x34  0x56  0x78
```

**Radiation implications**:
- **Byte-order matters** when interpreting bit positions
- **Cross-platform compatibility** requires endianness handling
- **Error correction** must account for byte ordering

### Portable Bit Manipulation

```cpp
// From: include/rad_ml/utils/bit_manipulation.hpp
template<typename T>
static bool isBitSet(T value, int bit_position) {
    // Validate bit position bounds
    if (bit_position < 0 || bit_position >= sizeof(T) * 8) {
        return false; // Invalid bit position
    }

    // For floating-point types, convert to integer representation
    if constexpr (std::is_floating_point<T>::value) {
        using UIntType = typename std::conditional<
            sizeof(T) == 8, uint64_t, uint32_t
        >::type;

        union {
            T value;
            UIntType bits;
        } converter;

        converter.value = value;
        return (converter.bits & (static_cast<UIntType>(1) << bit_position)) != 0;
    }
    else {
        // Handle integer types directly
        return (value & (static_cast<T>(1) << bit_position)) != 0;
    }
}
```

## 📊 Memory Layout Optimization

### Cache-Friendly Data Structures

```cpp
// From: include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp
template <typename T>
class EnhancedStuckBitTMR {
protected:
    // Hot data: frequently accessed (TMR copies and health tracking)
    std::array<T, 3> copies_;                     // 3 * sizeof(T) bytes
    mutable std::array<double, 3> health_scores_; // 24 bytes

    // Cold data: stuck bit detection and tracking
    std::bitset<sizeof(T) * 8> potential_stuck_bits{};      // Bits identified as stuck
    std::array<std::bitset<sizeof(T) * 8>, 3> stuck_value_masks{}; // Per-copy stuck values
    std::array<uint8_t, sizeof(T) * 8> error_consistency_counters{}; // Error frequency

    // Threshold based on JUICE mission testing (3+ consecutive errors = stuck)
    static constexpr uint8_t stuck_bit_threshold = 3;
};
```

**Memory layout strategy**:
- **Hot data first**: Frequently accessed data in same cache line
- **Cold data last**: Infrequently accessed data separate
- **Alignment considerations**: Natural alignment for performance

### Radiation-Aware Memory Allocation

🔧 **Implementation**: The radiation-mapped allocator places data in memory regions based on criticality and shielding levels. See [Memory Scrubbing](./06_MEMORY_SCRUBBING.md) for related memory protection techniques.

```cpp
// From: include/rad_ml/memory/radiation_mapped_allocator.hpp
enum class DataCriticality {
    MISSION_CRITICAL,    // Place in most protected memory
    HIGHLY_IMPORTANT,    // Place in well-protected memory
    MODERATELY_IMPORTANT,// Place in moderately protected memory
    LOW_IMPORTANCE       // Place in least protected memory
};
```

## 🧪 Testing Memory Representations

### Bit Pattern Validation

```cpp
void test_bit_manipulation() {
    float original = 3.14159f;

    // Test each bit position
    for (int bit = 0; bit < 32; ++bit) {
        float corrupted = BitManipulation::flipBit(original, bit);

        // Verify exactly one bit differs
        uint32_t orig_bits = *reinterpret_cast<uint32_t*>(&original);
        uint32_t corr_bits = *reinterpret_cast<uint32_t*>(&corrupted);

        uint32_t diff = orig_bits ^ corr_bits;

        // Should have exactly one bit set
        assert(__builtin_popcount(diff) == 1);

        // Should be the correct bit
        assert(diff == (1u << bit));
    }
}
```

### Cross-Platform Consistency

```cpp
void test_endianness_independence() {
    uint32_t value = 0x12345678;

    // Bit manipulation should work regardless of endianness
    for (int bit = 0; bit < 32; ++bit) {
        bool is_set = BitManipulation::isBitSet(value, bit);

        // Verify by manual calculation
        bool expected = (value & (1u << bit)) != 0;
        assert(is_set == expected);
    }
}
```

## 🎯 Best Practices

### Memory Safety Guidelines

1. **Use unions for type punning**: Safer than reinterpret_cast for same-size types
2. **Respect alignment**: Ensure proper alignment for target types
3. **Handle endianness**: Write portable code that works on all platforms
4. **Validate bit positions**: Always check bounds before bit manipulation

### Performance Considerations

1. **Minimize cache misses**: Group related data together
2. **Use SIMD when possible**: Vectorize bit operations
3. **Avoid unnecessary copies**: Work with references when possible
4. **Profile memory access**: Measure actual cache performance

## 🔗 Related Topics

- 📖 **Previous**: [Branchless Programming Fundamentals](./01_BRANCHLESS_PROGRAMMING.md) - Foundation concepts
- 📖 **Next**: [Type Punning and reinterpret_cast](./03_TYPE_PUNNING.md) - Safe memory reinterpretation
- 🔧 **Implementation**: [Stuck Bit Detection Algorithms](./04_STUCK_BIT_DETECTION.md) - Advanced bit tracking
- 🔧 **Memory Protection**: [Memory Scrubbing](./06_MEMORY_SCRUBBING.md) - Memory scrubbing techniques
- 📊 **Error Correction**: [Error Correction Codes](./05_ERROR_CORRECTION_CODES.md) - Reed-Solomon and Hamming codes

## 💡 Key Takeaways

1. **Every data type is just bits** - understanding memory layout is fundamental
2. **Radiation affects individual bits** - protection must work at bit granularity
3. **IEEE 754 has vulnerable fields** - different bit positions have different impacts
4. **Unions enable safe type punning** - better than unsafe casting
5. **Memory layout affects performance** - organize data for cache efficiency
6. **Endianness matters** - write portable bit manipulation code

---

📖 **Continue Learning**: Proceed to [Type Punning and reinterpret_cast](./03_TYPE_PUNNING.md) to learn safe techniques for memory reinterpretation.
