# Type Punning and reinterpret_cast

## 🎯 Learning Objectives

After studying this module, you'll understand:
- Safe techniques for reinterpreting memory as different types
- When to use unions vs reinterpret_cast
- Alignment and safety considerations
- How the developer uses type punning for radiation protection

## 🧠 What is Type Punning?

**Type punning** is the practice of reading a value stored in memory through a different type than it was written with. This is essential for radiation-tolerant systems, you'll need to:

1. **Access raw bits** of floating-point numbers
2. **Reinterpret addresses** as different pointer types
3. **Convert between representations** safely
4. **Examine memory layout** at the byte level

## 🔧 Type Punning Techniques

### 1. Union-Based Type Punning

```cpp
// From: include/rad_ml/utils/bit_manipulation.hpp
static float flipBit(float value, int bit_position) {
    // Safe union-based type punning
    union {
        float f;      // Floating-point view
        uint32_t i;   // Integer view of same bits
    } converter;

    converter.f = value;                    // Write as float
    converter.i ^= (1u << bit_position);    // Read/modify as uint32_t
    return converter.f;                     // Read back as float
}
```

**Why unions are safe**:
- ✅ **Well-defined behavior** when both types are same size
- ✅ **Compiler optimized** - no runtime overhead
- ✅ **Type safety** - compiler enforces size matching
- ✅ **Portable** - works consistently across platforms

### 2. reinterpret_cast for Address Manipulation

```cpp
// From: include/rad_ml/memory/radiation_mapped_allocator.hpp
void* allocate(size_t size, DataCriticality criticality) {
    // Calculate memory address
    size_t address = find_available_space(target_zone, size);

    // Convert address to pointer
    void* ptr = reinterpret_cast<void*>(address);

    return ptr;
}

void deallocate(void* ptr) {
    // Convert pointer back to address for bookkeeping
    size_t address = reinterpret_cast<size_t>(ptr);

    // Find and remove from allocation records
    auto it = std::find_if(allocations_.begin(), allocations_.end(),
                          [address](const AllocationRecord& rec) {
                              return rec.address == address;
                          });
}
```

**When reinterpret_cast is appropriate**:
- ✅ **Pointer ↔ integer conversions** for address arithmetic
- ✅ **void* ↔ typed pointer** conversions
- ✅ **Same-size type conversions** with careful alignment
- ❌ **Never for different-sized types**

### 3. Byte-Level Memory Access

```cpp
// From: include/rad_ml/neural/adaptive_protection.hpp
template<typename T>
void analyzeMemoryPattern(const T& value) {
    // Get byte-level view of any type
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);

    // Examine each byte
    for (size_t i = 0; i < sizeof(T); ++i) {
        uint8_t byte = bytes[i];

        // Check for stuck bit patterns
        if (byte == 0x00 || byte == 0xFF) {
            // Potential stuck bits detected
            handleStuckBytePattern(i, byte);
        }

        // Analyze bit patterns within byte
        analyzeBitPattern(byte, i * 8);
    }
}
```

## 🎨 Advanced Type Punning Patterns

### 1. Template-Based Generic Punning

```cpp
// From: include/rad_ml/testing/fault_injection.hpp
template<typename T>
T injectFault(T value, FaultPattern pattern, int bit_position) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");

    constexpr int total_bits = sizeof(T) * 8;

    // Convert to bitset for manipulation
    std::bitset<sizeof(T) * 8> bits =
        *reinterpret_cast<std::bitset<sizeof(T) * 8>*>(&value);

    // Apply bit manipulations
    for (int bit : bits_to_flip) {
        if (pattern == STUCK_AT_ZERO) {
            bits.reset(bit);
        } else if (pattern == STUCK_AT_ONE) {
            bits.set(bit);
        } else {
            bits.flip(bit);
        }
    }

    // Convert back to original type
    value = *reinterpret_cast<T*>(&bits);
    return value;
}
```

**Key techniques**:
- **Template safety**: `static_assert` ensures valid types
- **Bitset conversion**: Safe bit manipulation through `std::bitset`
- **Round-trip conversion**: Preserve exact bit patterns

### 2. Serialization-Safe Type Punning

```cpp
// From: include/rad_ml/storage/ai_native_database.cpp
std::vector<uint8_t> serializeValue(const T& value) {
    std::vector<uint8_t> result;

    // Convert value to byte sequence
    result.insert(result.end(),
                  reinterpret_cast<const uint8_t*>(&value),
                  reinterpret_cast<const uint8_t*>(&value) + sizeof(value));

    return result;
}

T deserializeValue(const std::vector<uint8_t>& data) {
    if (data.size() != sizeof(T)) {
        throw std::runtime_error("Size mismatch in deserialization");
    }

    T result;
    std::memcpy(&result, data.data(), sizeof(T));
    return result;
}
```

**Serialization best practices**:
- ✅ **Size validation** before deserialization
- ✅ **memcpy for safety** - avoids alignment issues
- ✅ **Endianness awareness** for cross-platform compatibility

### 3. Alignment-Safe Punning

```cpp
// From: include/rad_ml/neural/advanced_reed_solomon.hpp
template<typename T>
std::vector<element_t> convert_to_elements(const T& data) const {
    // Ensure proper alignment for element type
    alignas(element_t) uint8_t buffer[sizeof(T)];

    // Copy data to aligned buffer
    std::memcpy(buffer, &data, sizeof(T));

    // Safe reinterpretation with proper alignment
    const element_t* elements = reinterpret_cast<const element_t*>(buffer);

    size_t element_count = sizeof(T) / sizeof(element_t);
    return std::vector<element_t>(elements, elements + element_count);
}
```

## 🔬 Deep Dive: Alignment and Safety

### Understanding Memory Alignment

Different types have different alignment requirements:

```cpp
struct AlignmentExample {
    char c;      // 1-byte alignment
    short s;     // 2-byte alignment
    int i;       // 4-byte alignment
    double d;    // 8-byte alignment
    void* p;     // Pointer alignment (4 or 8 bytes)
};

// Compiler adds padding:
// [c][pad][s s][i i i i][d d d d d d d d][p p p p p p p p]
```

**Alignment violations can cause**:
- ❌ **Undefined behavior** on some architectures
- ❌ **Performance penalties** due to unaligned access
- ❌ **Bus errors** on strict alignment architectures

### Safe Alignment Techniques

```cpp
template<typename From, typename To>
To safe_reinterpret(const From& from) {
    static_assert(sizeof(From) == sizeof(To), "Types must be same size");
    static_assert(alignof(From) >= alignof(To), "Target alignment must be compatible");

    // Use aligned storage for safety
    alignas(To) char buffer[sizeof(To)];
    std::memcpy(buffer, &from, sizeof(From));

    return *reinterpret_cast<const To*>(buffer);
}
```

## 🧪 Testing Type Punning Safety

### Validation Framework

```cpp
template<typename T>
void test_type_punning_safety() {
    T original_value = get_test_value<T>();

    // Test round-trip conversion
    auto bytes = reinterpret_cast<const uint8_t*>(&original_value);
    T reconstructed;
    std::memcpy(&reconstructed, bytes, sizeof(T));

    // Verify bit-perfect reconstruction
    assert(std::memcmp(&original_value, &reconstructed, sizeof(T)) == 0);

    // Test with different values
    for (int i = 0; i < 1000; ++i) {
        T test_value = generate_random_value<T>();

        // Round-trip through byte representation
        uint8_t buffer[sizeof(T)];
        std::memcpy(buffer, &test_value, sizeof(T));

        T recovered;
        std::memcpy(&recovered, buffer, sizeof(T));

        // Should be identical
        assert(std::memcmp(&test_value, &recovered, sizeof(T)) == 0);
    }
}
```

### Cross-Platform Validation

```cpp
void test_endianness_handling() {
    uint32_t test_value = 0x12345678;

    // Convert to bytes
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);

    // Verify expected byte order
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        assert(bytes[0] == 0x78);  // LSB first
        assert(bytes[1] == 0x56);
        assert(bytes[2] == 0x34);
        assert(bytes[3] == 0x12);  // MSB last
    #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        assert(bytes[0] == 0x12);  // MSB first
        assert(bytes[1] == 0x34);
        assert(bytes[2] == 0x56);
        assert(bytes[3] == 0x78);  // LSB last
    #endif
}
```

## 🚨 Common Pitfalls and How to Avoid Them

### 1. Alignment Violations

❌ **Wrong**:
```cpp
// Dangerous - may cause unaligned access
char buffer[sizeof(double)];
double* d_ptr = reinterpret_cast<double*>(buffer);
*d_ptr = 3.14159;  // May crash on some architectures
```

✅ **Correct**:
```cpp
// Safe - properly aligned
alignas(double) char buffer[sizeof(double)];
double* d_ptr = reinterpret_cast<double*>(buffer);
*d_ptr = 3.14159;  // Always safe
```

### 2. Size Mismatches

❌ **Wrong**:
```cpp
// Dangerous - different sizes
int32_t value = 42;
int64_t* big_ptr = reinterpret_cast<int64_t*>(&value);  // UB!
```

✅ **Correct**:
```cpp
// Safe - same size conversion
int32_t value = 42;
uint32_t* unsigned_ptr = reinterpret_cast<uint32_t*>(&value);  // OK
```

### 3. Lifetime Issues

❌ **Wrong**:
```cpp
uint8_t* get_bytes() {
    int value = 42;
    return reinterpret_cast<uint8_t*>(&value);  // Dangling pointer!
}
```

✅ **Correct**:
```cpp
std::vector<uint8_t> get_bytes(int value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    return std::vector<uint8_t>(bytes, bytes + sizeof(int));
}
```

## 🎯 Best Practices Summary

### When to Use Each Technique

| Technique | Use Case | Safety Level |
|-----------|----------|--------------|
| **Union** | Same-size type conversion | ✅ Safest |
| **reinterpret_cast** | Pointer/address conversion | 🟡 Careful use |
| **memcpy** | Unaligned or unsafe conversion | ✅ Always safe |
| **std::bit_cast** (C++20) | Type punning | ✅ Safest (when available) |

### Safety Checklist

1. ✅ **Verify size compatibility** with `static_assert`
2. ✅ **Check alignment requirements** before casting
3. ✅ **Use unions for same-size conversions**
4. ✅ **Use memcpy for safety-critical code**
5. ✅ **Test on target hardware** to verify behavior
6. ✅ **Handle endianness** for cross-platform code

## 🔗 Related Topics

- 📖 **Previous**: [Memory Representation Mastery](./02_MEMORY_REPRESENTATION.md) - Understanding memory layout
- 📖 **Next**: [Stuck Bit Detection Algorithms](./04_STUCK_BIT_DETECTION.md) - Advanced error detection
- 🔧 **Implementation**: [Fault Injection Testing](./09_FAULT_INJECTION.md) - Testing type punning safety
- 📊 **Performance**: [Compile-Time Bit Manipulation](./10_COMPILE_TIME_OPTIMIZATION.md) - Zero-cost abstractions

## 💡 Key Takeaways

1. **Unions are safest** for same-size type conversions
2. **reinterpret_cast for addresses** and pointer conversions
3. **Always check alignment** to avoid undefined behavior
4. **memcpy is universally safe** but may be slower
5. **Test thoroughly** on all target platforms
6. **Size mismatches are dangerous** - use static_assert
7. **Lifetime management** is critical for pointer-based punning

---

📖 **Continue Learning**: Advance to [Stuck Bit Detection Algorithms](./04_STUCK_BIT_DETECTION.md) to see how type punning enables sophisticated error detection.
