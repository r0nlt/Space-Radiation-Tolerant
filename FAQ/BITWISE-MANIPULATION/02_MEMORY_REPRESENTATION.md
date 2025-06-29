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
    // Use union to reinterpret float as uint32_t
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
const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);

// Examine each byte individually
for (size_t i = 0; i < sizeof(T); ++i) {
    uint8_t byte = bytes[i];

    // Check each bit in the byte
    for (int bit = 0; bit < 8; ++bit) {
        if (byte & (1 << bit)) {
            // Process set bits
        }
    }
}
```

## 🎨 Advanced Memory Manipulation Techniques

### 1. Stuck Bit Pattern Detection

```cpp
// From: include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp
template <typename T>
void update_stuck_bit_tracking() {
    for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
        // Extract bit from each copy
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
    std::vector<uint8_t> result(data.size());

    // Process each bit position across all bytes
    for (size_t bit = 0; bit < 8; ++bit) {
        for (size_t block = 0; block < block_count; ++block) {
            // Extract bit from source
            bool bit_value = (data[src_idx] >> bit) & 1;

            // Place in interleaved position
            if (bit_value) {
                result[dst_idx / 8] |= (1 << (dst_idx % 8));
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
template <unsigned IntBits, unsigned FracBits, typename T>
class FixedPoint {
    static constexpr T scale = static_cast<T>(1) << FracBits;

    constexpr FixedPoint operator*(const FixedPoint& other) const noexcept {
        // Use wider type to prevent overflow
        using wider_t = std::conditional_t<sizeof(T) <= 4, std::int64_t, std::int64_t>;

        wider_t wide_result = static_cast<wider_t>(value_) * other.value_;

        // Scale back down with bit shift
        FixedPoint result;
        result.value_ = static_cast<T>(wide_result >> FracBits);
        return result;
    }
};
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
template<typename T>
static bool isBitSet(T value, int bit_position) {
    if constexpr (std::is_floating_point<T>::value) {
        // Handle floating-point types
        using UIntType = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;

        union {
            T value;
            UIntType bits;
        } converter;

        converter.value = value;
        return (converter.bits & (static_cast<UIntType>(1) << bit_position)) != 0;
    } else {
        // Handle integer types
        return (value & (static_cast<T>(1) << bit_position)) != 0;
    }
}
```

## 📊 Memory Layout Optimization

### Cache-Friendly Data Structures

```cpp
// From: include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp
class EnhancedStuckBitTMR {
private:
    // Hot data: frequently accessed
    std::array<T, 3> copies_;                    // 3 * sizeof(T) bytes
    mutable std::array<double, 3> health_scores_; // 24 bytes

    // Cold data: infrequently accessed
    std::bitset<sizeof(T) * 8> potential_stuck_bits;      // sizeof(T) bytes
    std::array<uint8_t, sizeof(T) * 8> error_consistency_counters; // sizeof(T) * 8 bytes
};
```

**Memory layout strategy**:
- **Hot data first**: Frequently accessed data in same cache line
- **Cold data last**: Infrequently accessed data separate
- **Alignment considerations**: Natural alignment for performance

### NUMA-Aware Allocation

🔧 **Implementation**: See [Radiation-Aware Memory Management](./08_RADIATION_MEMORY_MGMT.md) for details.

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
- 📊 **Performance**: [Cache-Friendly Algorithms](./11_CACHE_OPTIMIZATION.md) - Memory access optimization

## 💡 Key Takeaways

1. **Every data type is just bits** - understanding memory layout is fundamental
2. **Radiation affects individual bits** - protection must work at bit granularity
3. **IEEE 754 has vulnerable fields** - different bit positions have different impacts
4. **Unions enable safe type punning** - better than unsafe casting
5. **Memory layout affects performance** - organize data for cache efficiency
6. **Endianness matters** - write portable bit manipulation code

---

📖 **Continue Learning**: Proceed to [Type Punning and reinterpret_cast](./03_TYPE_PUNNING.md) to learn safe techniques for memory reinterpretation.
