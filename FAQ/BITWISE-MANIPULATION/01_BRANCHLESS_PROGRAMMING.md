# Branchless Programming Fundamentals

## 🎯 Learning Objectives

After studying this module, you'll understand:
- Why branches are dangerous in radiation environments
- How to eliminate conditional statements using bitwise operations
- The mathematical principles behind branchless algorithms
- Performance implications of branchless code

## 🚨 The Problem: Why Branches Are Dangerous

### Branch Prediction Units and Radiation

Modern CPUs use **Branch Prediction Units (BPUs)** to guess which way conditional branches will go. These units are complex hardware components that:

1. **Store prediction tables** in SRAM structures
2. **Use complex algorithms** to predict branch outcomes
3. **Are vulnerable to radiation** like any other circuit

When a **Single Event Upset (SEU)** hits the BPU:
```
Normal execution:     if (x > 0) → predicted TRUE → execute path A
After SEU corruption: if (x > 0) → predicted FALSE → execute path B (WRONG!)
```

### Real-World Impact

📖 **Reference**: See [Comprehensive Overview](./COMPREHENSIVE_BITWISE_RADIATION_HARDENING.md#branchless-memory-protection) for full context.

In space missions, BPU corruption can cause:
- **Control flow hijacking**: Program jumps to wrong code paths
- **Data corruption**: Wrong calculations due to mispredicted branches
- **Mission failure**: Critical systems making wrong decisions

## 🔧 The Solution: Branchless Programming

### Core Principle: Mask-Based Selection

The framework uses **bitmasks** to eliminate branches. Here's the fundamental technique:

```cpp
// From: include/rad_ml/math/branchless_ops.hpp
template <typename T>
static T min(T a, T b) {
    // Step 1: Create a mask based on comparison
    T mask = -(a <= b);  // All 1s if true, all 0s if false

    // Step 2: Use mask to select value
    return (mask & a) | (~mask & b);
}
```

### 🧮 Mathematical Breakdown

Let's trace through `min(5, 10)`:

1. **Comparison**: `5 <= 10` → `true` → `1`
2. **Negation**: `-(1)` → `0xFFFFFFFF` (all 1s in two's complement)
3. **Selection**:
   ```
   mask & a     = 0xFFFFFFFF & 5 = 5
   ~mask & b    = 0x00000000 & 10 = 0
   result       = 5 | 0 = 5 ✓
   ```

For `min(10, 5)`:
1. **Comparison**: `10 <= 5` → `false` → `0`
2. **Negation**: `-(0)` → `0x00000000` (all 0s)
3. **Selection**:
   ```
   mask & a     = 0x00000000 & 10 = 0
   ~mask & b    = 0xFFFFFFFF & 5 = 5
   result       = 0 | 5 = 5 ✓
   ```

## 🎨 Advanced Branchless Patterns

### 1. Absolute Value Without Branches

```cpp
template <typename T>
static T abs(T x) {
    // Extract sign bit (0 if positive, all 1s if negative)
    T mask = x >> (sizeof(T) * 8 - 1);

    // XOR with mask and subtract mask
    return (x ^ mask) - mask;
}
```

**How it works**:
- For positive numbers: `mask = 0`, so `(x ^ 0) - 0 = x`
- For negative numbers: `mask = 0xFFFFFFFF`, so `(x ^ mask) - mask` performs two's complement negation

### 2. Generic Selection (Branchless Ternary)

```cpp
template <typename T, typename C>
static T select(C condition, T if_true, T if_false) {
    // Convert condition to mask
    T mask = -static_cast<T>(condition != 0);
    return (mask & if_true) | (~mask & if_false);
}
```

**Usage example**:
```cpp
// Instead of: result = (x > 0) ? positive_action() : negative_action();
result = select(x > 0, positive_action(), negative_action());
```

### 3. Clamping Values

```cpp
template <typename T>
static T clamp(T x, T low, T high) {
    // First clamp to upper bound
    T mask1 = -(x <= high);
    T result = (mask1 & x) | (~mask1 & high);

    // Then clamp to lower bound
    T mask2 = -(result >= low);
    return (mask2 & result) | (~mask2 & low);
}
```

## 🔬 Deep Dive: Two's Complement Magic

### Understanding the Mask Generation

The key insight is how `-(boolean)` creates perfect bitmasks:

```cpp
bool condition = (a <= b);
// condition is either 0 or 1

int mask = -condition;
// If condition = 0: -0 = 0x00000000 (all zeros)
// If condition = 1: -1 = 0xFFFFFFFF (all ones in two's complement)
```

### Why This Works

In **two's complement** representation:
- `0` stays `0` when negated
- `1` becomes `-1`, which is `0xFFFFFFFF` (all bits set)

This creates perfect **selection masks**:
- All 0s: `mask & value = 0` (value is masked out)
- All 1s: `mask & value = value` (value passes through)

## 📊 Performance Analysis

### Instruction Count Comparison

**Traditional Branching**:
```assembly
cmp  eax, ebx        ; Compare a and b
jle  .take_a         ; Conditional jump (vulnerable!)
mov  eax, ebx        ; Take b
jmp  .done           ; Unconditional jump
.take_a:
mov  eax, eax        ; Take a (no-op)
.done:
```

**Branchless Version**:
```assembly
cmp  eax, ebx        ; Compare a and b
sbb  ecx, ecx        ; Set mask based on comparison
and  eax, ecx        ; Mask a
not  ecx             ; Invert mask
and  ebx, ecx        ; Mask b
or   eax, ebx        ; Combine results
```

### Performance Characteristics

| Aspect | Branching | Branchless |
|--------|-----------|------------|
| **Predictable Performance** | ❌ Depends on prediction accuracy | ✅ Always same instruction count |
| **Radiation Tolerance** | ❌ Vulnerable to BPU corruption | ✅ No dependency on BPU |
| **Cache Performance** | ❌ Branch mispredictions flush pipeline | ✅ Straight-line execution |
| **Best Case Speed** | 🟡 Faster with perfect prediction | 🟡 Slightly slower |
| **Worst Case Speed** | ❌ Much slower with mispredictions | ✅ Consistent performance |

## 🧪 Testing Branchless Code

### Validation Strategy

🔧 **Implementation**: See the fault injection testing in [Fault Injection Testing](./09_FAULT_INJECTION.md)

```cpp
// Test that branchless min produces same results as standard min
void test_branchless_min() {
    for (int a = -100; a <= 100; ++a) {
        for (int b = -100; b <= 100; ++b) {
            int standard_result = std::min(a, b);
            int branchless_result = BranchlessOps::min(a, b);
            assert(standard_result == branchless_result);
        }
    }
}
```

### Radiation Simulation Testing

```cpp
// Simulate BPU corruption and verify branchless code is unaffected
void test_radiation_tolerance() {
    // Branchless code should produce identical results even with
    // simulated BPU corruption (since it doesn't use branches)

    for (int corruption_level = 0; corruption_level < 100; ++corruption_level) {
        simulate_bpu_corruption(corruption_level);

        // Branchless operations should be unaffected
        assert(BranchlessOps::min(5, 10) == 5);
        assert(BranchlessOps::max(5, 10) == 10);
    }
}
```

## 🎯 Best Practices

### When to Use Branchless Programming

✅ **Good candidates**:
- Simple comparisons and selections
- Mathematical operations
- Bit manipulation
- Critical path code in radiation environments

❌ **Poor candidates**:
- Complex control flow
- Exception handling
- Early returns from functions
- Code where readability is paramount

### Code Style Guidelines

1. **Comment the magic**: Always explain the bitmask generation
2. **Test thoroughly**: Verify against reference implementations
3. **Measure performance**: Profile on target hardware
4. **Consider readability**: Use wrapper functions for complex operations

## 🔗 Related Topics

- 📖 **Next**: [Memory Representation Mastery](./02_MEMORY_REPRESENTATION.md) - Understanding how data is stored
- 🔧 **Implementation**: [Type Punning and reinterpret_cast](./03_TYPE_PUNNING.md) - Safe memory reinterpretation
- 📊 **Performance**: [Compile-Time Bit Manipulation](./10_COMPILE_TIME_OPTIMIZATION.md) - Zero-cost abstractions

## 💡 Key Takeaways

1. **Branches are vulnerability points** in radiation environments
2. **Bitmasks enable selection** without conditional jumps
3. **Two's complement arithmetic** provides the mathematical foundation
4. **Consistent performance** is often more valuable than peak performance
5. **Testing is critical** to verify correctness and radiation tolerance

---

📖 **Continue Learning**: Move on to [Memory Representation Mastery](./02_MEMORY_REPRESENTATION.md) to understand how the developer manipulates data at the bit level.
