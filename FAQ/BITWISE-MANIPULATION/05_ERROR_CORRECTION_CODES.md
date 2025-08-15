# Error Correction Code Implementation

## 🎯 Learning Objectives

After studying this module, you'll understand:
- Reed-Solomon codes and Galois Field mathematics
- Systematic encoding for efficient data storage
- Syndrome calculation for error detection
- Berlekamp-Massey algorithm for error location
- Forney algorithm for error magnitude calculation

## 🧠 Error Correction Fundamentals

### Why ECC is Critical for Space Systems

Error Correction Codes (ECC) can detect and automatically correct errors without retransmission. The framework implements sophisticated ECC techniques including:

1. **Reed-Solomon codes**: For burst error correction and symbol-level protection
2. **Hamming codes**: For single-bit error correction with parity checking
3. **Systematic codes**: Data remains readable even without decoding
4. **Galois Field arithmetic**: Mathematical foundation for finite field operations

📖 **Reference**: See [Stuck Bit Detection Algorithms](./04_STUCK_BIT_DETECTION.md) for complementary error detection techniques.

### Error Types in Radiation Environments

```cpp
// Different error patterns require different correction strategies
enum class ErrorPattern {
    SINGLE_BIT,      // SEU - correctable with simple parity
    ADJACENT_BITS,   // MBU - requires multi-bit correction
    BYTE_ERROR,      // Full byte corruption - symbol-level correction
    BURST_ERROR,     // Consecutive bits - interleaving + RS codes
    STUCK_BITS       // Permanent errors - detection only
};
```

## 🔧 Reed-Solomon Implementation

### Core Architecture

The framework implements a complete Reed-Solomon encoder/decoder with proper Galois Field arithmetic:

```cpp
// From: include/rad_ml/neural/advanced_reed_solomon.hpp
template<typename T, uint8_t SymbolSize = 8, uint8_t ECCSymbols = 8>
class AdvancedReedSolomon {
    // Use GF(256) for 8-bit symbols
    using GF = std::conditional_t<SymbolSize == 8, GF256, GF16>;
    using element_t = typename GF::element_t;

    static constexpr size_t data_symbols = sizeof(T) / sizeof(element_t);
    static constexpr size_t total_symbols = data_symbols + ECCSymbols;

public:
    AdvancedReedSolomon() {
        // Precompute generator polynomial for efficiency
        generator_poly_ = field_.rs_generator_poly(ECCSymbols);
    }
};
```

**Key Design Decisions**:
- **Template-based**: Generic for any data type `T`
- **Configurable symbols**: Adjustable symbol size and ECC count
- **Systematic encoding**: Original data remains accessible
- **Precomputed polynomials**: Generator polynomial calculated once

### 1. Galois Field Mathematics

The mathematical foundation uses finite field arithmetic:

```cpp
// From: include/rad_ml/neural/galois_field.hpp
template <uint8_t m, uint16_t Poly>
class GaloisField {
    using element_t = std::conditional_t<(m <= 8), uint8_t, uint16_t>;

    static constexpr element_t field_size = (1 << m);  // 2^m elements
    static constexpr element_t primitive_poly = Poly;   // Defines field structure

public:
    // Addition in GF(2^m) is XOR
    constexpr element_t add(element_t a, element_t b) const {
        return a ^ b;
    }

    // Multiplication using lookup tables for efficiency
    element_t multiply(element_t a, element_t b) const {
        if (a == 0 || b == 0) return 0;

        // Log-antilog method: log(a*b) = log(a) + log(b)
        return exp_table[(log_table[a] + log_table[b]) % (field_size - 1)];
    }
};
```

**Mathematical Properties**:
- **Finite Field**: GF(2^8) has exactly 256 elements (0-255)
- **Primitive Polynomial**: Defines multiplication rules (e.g., 0x11D for GF(256))
- **Lookup Tables**: Precomputed exp/log tables for fast multiplication
- **Additive Identity**: 0 ⊕ a = a for all elements
- **Multiplicative Identity**: 1 ⊗ a = a for all non-zero elements

### 2. Systematic Encoding Process

```cpp
std::vector<uint8_t> encode(const T& data) const {
    // Convert data to field elements
    std::vector<element_t> message = convert_to_elements(data);

    // Systematic encoding: [data | parity]
    std::vector<element_t> codeword = message;
    codeword.resize(total_symbols, 0);

    // Compute ECC symbols using polynomial division
    auto ecc = compute_ecc_symbols(message);

    // Place ECC symbols at end (systematic form)
    std::copy(ecc.begin(), ecc.end(), codeword.begin() + data_symbols);

    return convert_from_elements(codeword);
}
```

**Systematic Encoding Benefits**:
- **Data Accessibility**: Original data readable without decoding
- **Efficiency**: No need to decode for error-free data
- **Compatibility**: Works with existing data formats

### 3. ECC Symbol Calculation

```cpp
std::vector<element_t> compute_ecc_symbols(const std::vector<element_t>& message) const {
    // Polynomial division: remainder of x^n * message(x) / generator(x)
    std::vector<element_t> remainder(ECCSymbols, 0);

    // Process each message symbol
    for (size_t i = 0; i < data_symbols; ++i) {
        element_t feedback = field_.add(message[i], remainder[0]);

        // Skip multiplication if feedback is zero (optimization)
        if (feedback != 0) {
            // Multiply by generator polynomial
            for (size_t j = 1; j < ECCSymbols; ++j) {
                remainder[j-1] = field_.add(remainder[j],
                    field_.multiply(feedback, generator_poly_[j]));
            }
            remainder[ECCSymbols-1] = field_.multiply(feedback,
                generator_poly_[ECCSymbols]);
        } else {
            // Shift remainder
            for (size_t j = 0; j < ECCSymbols - 1; ++j) {
                remainder[j] = remainder[j+1];
            }
            remainder[ECCSymbols-1] = 0;
        }
    }

    return remainder;
}
```

**Algorithm Explanation**:
1. **Polynomial Division**: Divide data polynomial by generator polynomial
2. **Remainder Calculation**: ECC symbols are the remainder
3. **Feedback Optimization**: Skip zero multiplications for efficiency
4. **Systematic Form**: Remainder becomes parity symbols

## 🎨 Error Detection and Correction

### 1. Syndrome Calculation

```cpp
std::vector<element_t> rs_calc_syndromes(const std::vector<element_t>& msg, uint8_t nsym) const {
    std::vector<element_t> syndromes(nsym + 1, 0);

    // Evaluate message polynomial at α^i for i=0..nsym
    for (uint8_t i = 0; i <= nsym; ++i) {
        syndromes[i] = eval_poly(msg, exp_table[i]);
    }

    return syndromes;
}
```

**Syndrome Properties**:
- **Error Detection**: All syndromes zero = no errors
- **Error Location**: Syndrome pattern indicates error positions
- **Error Count**: Number of non-zero syndromes ≤ 2 × number of errors

### 2. Berlekamp–Massey Algorithm

The Berlekamp–Massey algorithm computes the error locator polynomial Λ(x) by identifying the minimal linear recurrence that fits the syndrome sequence. In autoregressive/LFSR terms, Λ is the feedback polynomial. For j ≥ T (number of errors):

```
S_j = −(Λ₁ S_{j−1} + Λ₂ S_{j−2} + … + Λ_T S_{j−T})
```

In GF(2^m), subtraction equals addition, so the minus sign has no effect in implementation. After Λ(x) is found, the error evaluator polynomial is constructed as the truncated product Ω(x) = [S(x) · Λ(x)] mod x^{nsym+1}.

```cpp
std::tuple<std::vector<element_t>, std::vector<element_t>> rs_find_error_locator(
    const std::vector<element_t>& syndromes, uint8_t nsym) const {

    // Initialize error locator polynomial Λ(x) = 1
    std::vector<element_t> err_loc = {1};
    std::vector<element_t> old_loc = {1};

    for (uint8_t i = 0; i < nsym; ++i) {
        // Compute discrepancy
        element_t delta = syndromes[i + 1];
        for (size_t j = 1; j < err_loc.size(); ++j) {
            delta = field_.add(delta, field_.multiply(
                err_loc[err_loc.size() - 1 - j], syndromes[i + 1 - j]));
        }

        // Update error locator polynomial
        if (delta != 0) {
            // Berlekamp-Massey update rule
            auto new_loc = update_error_locator(err_loc, old_loc, delta, i);
            old_loc = err_loc;
            err_loc = new_loc;
        }
    }

    // Calculate error evaluator polynomial
    auto err_eval = calculate_error_evaluator(syndromes, err_loc, nsym);

    return {err_loc, err_eval};
}
```

**Algorithm Purpose**:
- **Find Error Locator**: Polynomial whose roots indicate error positions
- **Iterative Refinement**: Builds polynomial incrementally
- **Minimal Polynomial**: Finds shortest polynomial satisfying syndrome equations

### 3. Chien Search for Error Locations

Evaluate Λ at α^{−i} for i = 0..n−1; a zero indicates an error at position (n−1−i). This linear-time scan in n with a small factor deg(Λ) efficiently finds all error locations.

```cpp
std::vector<size_t> rs_find_errors(const std::vector<element_t>& err_loc, size_t msg_len) const {
    std::vector<size_t> error_positions;

    // Test each possible position α^i
    for (size_t i = 0; i < msg_len; ++i) {
        element_t test_value = eval_poly(err_loc, exp_table[i]);

        // If polynomial evaluates to zero, we found an error position
        if (test_value == 0) {
            error_positions.push_back(msg_len - 1 - i);  // Convert to position
        }
    }

    return error_positions;
}
```

### 4. Forney Algorithm for Error Values

For each error position j, compute magnitude using Forney’s formula:

```
E_j = − Ω(α^{−j}) / (α^{−j} · Λ'(α^{−j}))
```

Only odd coefficients contribute to Λ'(x) in characteristic 2, and the leading minus can be omitted in GF(2^m).

```cpp
std::vector<element_t> rs_correct_errors_at_positions(
    const std::vector<element_t>& msg_in, const std::vector<size_t>& err_pos,
    const std::vector<element_t>& err_loc, const std::vector<element_t>& err_eval) const {

    std::vector<element_t> msg = msg_in;

    // Calculate error magnitude for each position
    for (size_t i = 0; i < err_pos.size(); ++i) {
        size_t pos = err_pos[i];

        // Error position in field representation
        element_t x_inv = exp_table[(field_size - 1 - pos) % (field_size - 1)];

        // Evaluate error evaluator at position
        element_t err_eval_at_pos = eval_poly(err_eval, x_inv);

        // Calculate error locator derivative
        element_t err_loc_deriv = 0;
        for (size_t j = 1; j < err_loc.size(); j += 2) {
            err_loc_deriv = field_.add(err_loc_deriv,
                field_.multiply(err_loc[j], field_.pow(x_inv, j - 1)));
        }

        // Calculate error magnitude using Forney formula
        element_t err_mag = field_.divide(err_eval_at_pos,
            field_.multiply(x_inv, err_loc_deriv));

        // Correct the error
        msg[pos] = field_.add(msg[pos], err_mag);
    }

    return msg;
}
```

## 🔬 Advanced Techniques

### 1. Bit Interleaving for Burst Protection

```cpp
std::vector<uint8_t> interleave(const std::vector<uint8_t>& data) const {
    std::vector<uint8_t> result(data.size());

    // Determine interleaving parameters
    size_t block_count = (data.size() + 7) / 8;

    // Process each bit position across all bytes
    for (size_t bit = 0; bit < 8; ++bit) {
        for (size_t block = 0; block < block_count; ++block) {
            size_t src_idx = block;
            size_t dst_idx = bit * block_count + block;

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

    return result;
}
```

**Burst Error Mitigation**:
```
Original:    [AAAAAAAA][BBBBBBBB][CCCCCCCC][DDDDDDDD]
After burst: [XXXXXXXX][BBBBBBBB][CCCCCCCC][DDDDDDDD]  ← 8 consecutive errors

Interleaved: [A₀B₀C₀D₀][A₁B₁C₁D₁][A₂B₂C₂D₂][A₃B₃C₃D₃]...
After burst: [XXXXXXXX][A₁B₁C₁D₁][A₂B₂C₂D₂][A₃B₃C₃D₃]...  ← Distributed errors
```

### 2. Multi-Level Protection

```cpp
// From: include/rad_ml/neural/multi_bit_protection.hpp
template<typename T>
class MultiBitProtection {
public:
    // Layer 1: Simple parity for single-bit errors
    bool checkHamming() const {
        // Calculate parity bits
        uint32_t syndrome = 0;
        // ... Hamming code implementation

        if (syndrome == 0) {
            return true;  // No error
        } else if (syndrome <= 32) {
            // Single bit error - correct it
            correct_single_bit_error(syndrome);
            return true;
        }

        return false;  // Multiple errors - use Reed-Solomon
    }

    // Layer 2: Reed-Solomon for multi-bit errors
    bool checkReedSolomon() const {
        // Calculate syndromes and attempt correction
        // ... Reed-Solomon implementation
    }
};
```

### 3. Adaptive Error Correction

```cpp
std::optional<T> decode(const std::vector<uint8_t>& encoded_data) const {
    // Convert to field elements
    std::vector<element_t> codeword = convert_to_elements(encoded_data);

    // Calculate syndromes
    auto syndromes = field_.rs_calc_syndromes(codeword, ECCSymbols);

    // Check if all syndromes are zero (no errors)
    bool has_errors = false;
    for (size_t i = 1; i < syndromes.size(); ++i) {
        if (syndromes[i] != 0) {
            has_errors = true;
            break;
        }
    }

    if (!has_errors) {
        // No errors - return data directly
        return convert_elements_to_data<T>(
            std::vector<element_t>(codeword.begin(),
                                 codeword.begin() + data_symbols));
    }

    // Errors detected - attempt correction
    auto corrected = field_.rs_correct_errors(codeword, ECCSymbols);
    if (corrected) {
        return convert_elements_to_data<T>(
            std::vector<element_t>(corrected->begin(),
                                 corrected->begin() + data_symbols));
    }

    return std::nullopt;  // Uncorrectable
}
```

## 📊 Performance Analysis

### Error Correction Capability

| ECC Type | Symbol Size | Correction Capability | Overhead |
|----------|-------------|----------------------|----------|
| **Hamming(7,4)** | 1 bit | 1 bit error | 75% |
| **Reed-Solomon(255,223)** | 8 bits | 16 symbol errors | 14.3% |
| **Reed-Solomon(15,11)** | 4 bits | 2 symbol errors | 36.4% |
| **Advanced RS** | 8 bits | Configurable | Variable |

### Memory Overhead Calculation

```cpp
constexpr double overhead_percent() const {
    return (static_cast<double>(total_symbols * sizeof(element_t)) /
            sizeof(T) - 1.0) * 100.0;
}

constexpr size_t correction_capability() const {
    return ECCSymbols / 2;  // Reed-Solomon can correct t errors with 2t symbols
}
```

**Example for 32-bit float with 8 ECC symbols**:
- **Data size**: 4 bytes
- **ECC size**: 8 bytes
- **Total size**: 12 bytes
- **Overhead**: 200%
- **Correction**: Up to 4 symbol errors

## 🧪 Testing and Validation

### 1. Systematic Error Injection

```cpp
std::vector<uint8_t> apply_bit_errors(
    const std::vector<uint8_t>& data,
    double error_rate,
    uint64_t seed = 0) const {

    std::vector<uint8_t> result = data;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Apply bit errors based on probability
    for (size_t i = 0; i < result.size(); ++i) {
        for (int bit = 0; bit < 8; ++bit) {
            if (dist(rng) < error_rate) {
                result[i] ^= (1 << bit);
            }
        }
    }

    return result;
}
```

### 2. Burst Error Simulation

```cpp
std::vector<uint8_t> apply_burst_errors(
    const std::vector<uint8_t>& data,
    double error_rate,
    uint8_t burst_size = 3) const {

    // Create spatially correlated errors
    size_t num_bursts = static_cast<size_t>(error_rate * data.size() / 2) + 1;

    for (size_t i = 0; i < num_bursts; ++i) {
        // Select random starting position
        size_t byte_idx = byte_dist(rng);
        int bit_idx = bit_dist(rng);

        // Apply burst of consecutive errors
        for (int j = 0; j < burst_size; ++j) {
            int current_bit = (bit_idx + j) % 8;
            size_t current_byte = byte_idx + (bit_idx + j) / 8;
            if (current_byte < result.size()) {
                result[current_byte] ^= (1 << current_bit);
            }
        }
    }

    return result;
}
```

### 3. Correction Validation

```cpp
void validate_error_correction() {
    AdvancedReedSolomon<float> rs;

    float original = 3.14159f;
    auto encoded = rs.encode(original);

    // Test different error patterns
    for (double error_rate = 0.01; error_rate <= 0.20; error_rate += 0.01) {
        auto corrupted = rs.apply_bit_errors(encoded, error_rate, 42);
        auto decoded = rs.decode(corrupted);

        if (decoded) {
            assert(std::abs(*decoded - original) < 1e-6);
        }
    }
}
```

## 🎯 Best Practices

### Implementation Guidelines

1. **Use Systematic Codes**: Keep data accessible without decoding
2. **Precompute Polynomials**: Calculate generator polynomials once
3. **Optimize Hot Paths**: Use lookup tables for field operations
4. **Handle Edge Cases**: Zero elements and boundary conditions
5. **Validate Corrections**: Verify syndrome recalculation after correction

### Performance Optimization

1. **Lookup Tables**: Precompute exp/log tables for multiplication
2. **Early Termination**: Skip processing when syndromes are zero
3. **Memory Layout**: Organize data for cache efficiency
4. **Template Specialization**: Optimize for common symbol sizes

## 🔗 Related Topics

- 📖 **Previous**: [Stuck Bit Detection Algorithms](./04_STUCK_BIT_DETECTION.md) - Complementary error detection
- 📖 **Next**: [Memory Scrubbing Strategies](./06_MEMORY_SCRUBBING.md) - Continuous error monitoring
- 🔧 **Implementation**: [Galois Field Mathematics](./07_GALOIS_FIELD_MATH.md) - Mathematical foundation
- 📊 **Performance**: [Compile-Time Bit Manipulation](./10_COMPILE_TIME_OPTIMIZATION.md) - Zero-cost abstractions

## 💡 Key Takeaways

1. **Reed-Solomon codes provide powerful symbol-level correction** for burst errors
2. **Galois Field arithmetic** enables efficient finite field operations
3. **Systematic encoding** keeps data readable without decoding overhead
4. **Syndrome calculation** detects errors without false positives
5. **Berlekamp-Massey algorithm** finds minimal error locator polynomials
6. **Forney algorithm** calculates exact error magnitudes for correction
7. **Bit interleaving** transforms burst errors into correctable patterns
8. **Multi-level protection** combines different ECC techniques effectively

---

📖 **Continue Learning**: Advance to [Memory Scrubbing Strategies](./06_MEMORY_SCRUBBING.md) to see how ECC integrates with continuous memory monitoring.
