/**
 * @file multi_bit_protection.hpp
 * @brief Protection against multi-bit upsets in neural networks
 *
 * This file defines protection mechanisms for handling multi-bit upsets
 * that can occur in neural network parameters in radiation environments.
 */

#ifndef RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP
#define RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "../core/error/status_code.hpp"
#include "../core/redundancy/space_enhanced_tmr.hpp"
#include "../core/redundancy/tmr.hpp"
#include "advanced_reed_solomon.hpp"
#include "galois_field.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Types of multi-bit upsets that can occur
 */
enum class MultibitUpsetType {
    SINGLE_BIT,     ///< Single bit flip (SEU)
    ADJACENT_BITS,  ///< Adjacent bits in the same word
    ROW_UPSET,      ///< Bits in the same row (in a memory layout)
    COLUMN_UPSET,   ///< Bits in the same column (in a memory layout)
    RANDOM_MULTI    ///< Random multiple bit flips
};

/**
 * @brief Error correction coding schemes
 */
enum class ECCCodingScheme {
    NONE,         ///< No ECC
    HAMMING,      ///< Hamming code (single bit correction)
    SECDED,       ///< SEC-DED (Single Error Correction, Double Error Detection)
    REED_SOLOMON  ///< Reed-Solomon codes (multiple error correction)
};

/**
 * @brief Reed-Solomon correction tiers
 *
 * Following Tour of C++ philosophy: use enum class for type safety,
 * compile-time selection via templates where possible.
 */
enum class RSCorrectionTier : uint8_t {
    LIGHT = 4,     ///< t=2 errors (4 ECC symbols) - fast, for LEO
    STANDARD = 6,  ///< t=3 errors (6 ECC symbols) - balanced
    HEAVY = 8      ///< t=4 errors (8 ECC symbols) - robust, for SAA/GEO
};

/**
 * @brief Type trait: Check if type is suitable for RS protection
 *
 * Tour of C++: Use type traits for compile-time safety.
 * RS requires trivially copyable types with known size.
 */
template <typename T>
struct is_rs_protectable {
    static constexpr bool value = std::is_trivially_copyable_v<T> &&
                                  (sizeof(T) <= 247);  // RS(255,k) constraint: k ≤ 255 - ECCSymbols
};

template <typename T>
inline constexpr bool is_rs_protectable_v = is_rs_protectable<T>::value;

/**
 * @brief Reed-Solomon Protection Backend
 *
 * Tour of C++ principles applied:
 * - RAII: RS encoder initialized on construction, no manual resource management
 * - Type safety: static_assert ensures only valid types are protected
 * - Value semantics: Codeword stored as std::vector, no raw pointers
 * - Const correctness: Decode is logically const (mutable codeword for correction)
 * - Zero-cost abstraction: Template specialization avoids runtime tier selection
 * - noexcept: Mark non-throwing operations for optimizer hints
 *
 * @tparam T Data type to protect (must be trivially copyable)
 * @tparam Tier RS correction tier (LIGHT, STANDARD, HEAVY)
 */
template <typename T, RSCorrectionTier Tier = RSCorrectionTier::STANDARD>
class RSProtectionBackend {
    static_assert(is_rs_protectable_v<T>,
                  "T must be trivially copyable and fit within RS block size");

   public:
    // Type aliases for clarity
    using value_type = T;
    using codeword_type = std::vector<uint8_t>;

    // Compile-time constants
    static constexpr uint8_t ecc_symbols = static_cast<uint8_t>(Tier);
    static constexpr size_t data_bytes = sizeof(T);
    static constexpr uint8_t correction_capability = ecc_symbols / 2;

   private:
    // The RS encoder/decoder - uses GF(2^8) for 8-bit symbols
    AdvancedReedSolomon<T, 8, ecc_symbols> rs_;

    // Stored codeword (data + ECC symbols)
    // Mutable: decode may update this for error correction
    mutable codeword_type codeword_;

    // Original value for fast access when no errors
    mutable T cached_value_;

    // Flag indicating if codeword needs re-encoding
    mutable bool dirty_ = true;

   public:
    /**
     * @brief Default constructor
     *
     * RAII: RS encoder is fully initialized, ready to use.
     */
    RSProtectionBackend() noexcept : cached_value_{}
    {
        codeword_.reserve(data_bytes + ecc_symbols);
    }

    /**
     * @brief Construct with initial value
     *
     * @param value Initial value to protect
     */
    explicit RSProtectionBackend(const T& value) : cached_value_(value) { encode(value); }

    // Rule of Five: Default implementations are correct
    RSProtectionBackend(const RSProtectionBackend&) = default;
    RSProtectionBackend(RSProtectionBackend&&) noexcept = default;
    RSProtectionBackend& operator=(const RSProtectionBackend&) = default;
    RSProtectionBackend& operator=(RSProtectionBackend&&) noexcept = default;
    ~RSProtectionBackend() = default;

    /**
     * @brief Encode a value with RS protection
     *
     * @param value Value to protect
     */
    void encode(const T& value)
    {
        cached_value_ = value;
        codeword_ = rs_.encode(value);
        dirty_ = false;
    }

    /**
     * @brief Attempt to decode and correct errors
     *
     * @return Corrected value if successful, std::nullopt if uncorrectable
     *
     * Tour of C++: Use std::optional for operations that may fail.
     * Logically const - correction is an implementation detail.
     */
    [[nodiscard]] std::optional<T> decode() const
    {
        if (codeword_.empty()) {
            return std::nullopt;
        }

        auto result = rs_.decode(codeword_);
        if (result) {
            cached_value_ = *result;
            dirty_ = false;
        }
        return result;
    }

    /**
     * @brief Check if errors are present (detects both correctable and uncorrectable)
     *
     * This method detects if the stored codeword has been corrupted,
     * regardless of whether the error is correctable.
     *
     * @return true if any errors detected, false if clean
     */
    [[nodiscard]] bool has_error() const noexcept
    {
        if (codeword_.empty()) return true;

        // Re-encode the cached value and compare with stored codeword
        // If they differ, the codeword has been corrupted
        auto expected = rs_.encode(cached_value_);
        if (expected.size() != codeword_.size()) return true;

        for (size_t i = 0; i < codeword_.size(); ++i) {
            if (codeword_[i] != expected[i]) {
                return true;  // Codeword has been modified = error present
            }
        }
        return false;  // Codeword matches expected = no error
    }

    /**
     * @brief Get cached value (fast path, no error check)
     *
     * Use when you know the value is clean or want raw access.
     */
    [[nodiscard]] const T& cached_value() const noexcept { return cached_value_; }

    /**
     * @brief Get value with error correction attempt
     *
     * @return Corrected value, or cached value if correction fails
     */
    [[nodiscard]] T get() const
    {
        auto result = decode();
        return result.value_or(cached_value_);
    }

    /**
     * @brief Access the raw codeword (for testing/debugging)
     */
    [[nodiscard]] const codeword_type& codeword() const noexcept { return codeword_; }

    /**
     * @brief Inject bit error at specific position (for testing)
     *
     * @param byte_pos Byte position in codeword
     * @param bit_pos Bit position within byte
     */
    void inject_error(size_t byte_pos, uint8_t bit_pos) noexcept
    {
        if (byte_pos < codeword_.size() && bit_pos < 8) {
            codeword_[byte_pos] ^= (1u << bit_pos);
        }
    }

    /**
     * @brief Get correction capability
     */
    [[nodiscard]] static constexpr uint8_t max_correctable_errors() noexcept
    {
        return correction_capability;
    }
};

/**
 * @brief Template class for protecting values against multi-bit upsets
 *
 * This class implements various protection mechanisms for values that
 * might be affected by single or multi-bit upsets. It provides methods
 * for error detection and correction using various coding schemes.
 *
 * @tparam T Data type to protect (typically float or int)
 */
template <typename T>
class MultibitProtection {
   public:
    /**
     * @brief Default constructor
     */
    MultibitProtection() : value_(), coding_scheme_(ECCCodingScheme::NONE), valid_(true)
    {
        // Initialize ECC
        updateECC();
    }

    /**
     * @brief Constructor with initial value
     *
     * @param value Initial value
     * @param coding_scheme ECC coding scheme to use
     */
    MultibitProtection(T value, ECCCodingScheme coding_scheme = ECCCodingScheme::HAMMING)
        : value_(value), coding_scheme_(coding_scheme), valid_(true)
    {
        // Initialize ECC
        updateECC();
    }

    /**
     * @brief Get stored value with error checking/correction
     *
     * @return Stored value (corrected if possible)
     */
    T getValue() const
    {
        // Check for errors first
        if (hasError()) {
            // Try to correct errors
            if (correctErrors()) {
                return value_;
            }

            // If we can't correct, at least return the original value
            return value_;
        }

        return value_;
    }

    /**
     * @brief Set a new value
     *
     * @param value New value to store
     */
    void setValue(T value)
    {
        value_ = value;
        valid_ = true;
        updateECC();
    }

    /**
     * @brief Check if the stored value has an error
     *
     * @return True if an error is detected
     */
    bool hasError() const
    {
        // If already marked as invalid, return true
        if (!valid_) return true;

        // Check using appropriate ECC scheme
        switch (coding_scheme_) {
            case ECCCodingScheme::NONE:
                return false;  // Can't detect errors

            case ECCCodingScheme::HAMMING:
            case ECCCodingScheme::SECDED:
                return checkHammingParity();

            case ECCCodingScheme::REED_SOLOMON:
                return checkReedSolomon();

            default:
                return false;
        }
    }

    /**
     * @brief Try to correct errors in the stored value
     *
     * @return True if errors were successfully corrected
     */
    bool correctErrors() const
    {
        if (!hasError()) return true;

        // Attempt correction using appropriate ECC scheme
        switch (coding_scheme_) {
            case ECCCodingScheme::NONE:
                return false;  // Can't correct

            case ECCCodingScheme::HAMMING:
                return correctHammingCode();

            case ECCCodingScheme::SECDED:
                return correctSECDED();

            case ECCCodingScheme::REED_SOLOMON:
                return correctReedSolomon();

            default:
                return false;
        }
    }

    /**
     * @brief Mark the value as invalid
     */
    void markInvalid() { valid_ = false; }

    /**
     * @brief Check if the value is currently valid
     *
     * @return True if valid
     */
    bool isValid() const { return valid_ && !hasError(); }

    /**
     * @brief Apply bit interleaving to protect against adjacent bit upsets
     *
     * @return Interleaved value
     */
    T applyBitInterleaving() const
    {
        // For non-integral types, perform bit manipulation through a union
        union {
            T value;
            uint32_t bits;
        } original, interleaved;

        original.value = value_;
        interleaved.bits = 0;

        // Simple bit interleaving - separate adjacent bits
        for (int i = 0; i < 32; ++i) {
            // Move bits to non-adjacent positions
            // Even bits go to the first half, odd bits to the second half
            if (i % 2 == 0) {
                interleaved.bits |= ((original.bits >> i) & 1) << (i / 2);
            }
            else {
                interleaved.bits |= ((original.bits >> i) & 1) << (16 + i / 2);
            }
        }

        return interleaved.value;
    }

    /**
     * @brief Undo bit interleaving
     *
     * @param interleaved_value Interleaved value
     * @return Original (deinterleaved) value
     */
    static T undoBitInterleaving(T interleaved_value)
    {
        union {
            T value;
            uint32_t bits;
        } original, interleaved;

        interleaved.value = interleaved_value;
        original.bits = 0;

        // Undo the interleaving
        for (int i = 0; i < 16; ++i) {
            original.bits |= ((interleaved.bits >> i) & 1) << (i * 2);
            original.bits |= ((interleaved.bits >> (i + 16)) & 1) << (i * 2 + 1);
        }

        return original.value;
    }

    /**
     * @brief Static method to apply multi-bit errors to a value
     *
     * @param value Original value
     * @param error_rate Error rate (0.0-1.0)
     * @param upset_type Type of multi-bit upset to simulate
     * @param seed Random seed for reproducibility
     * @return Value with simulated bit errors
     */
    static T applyMultiBitErrors(T value, double error_rate,
                                 MultibitUpsetType upset_type = MultibitUpsetType::SINGLE_BIT,
                                 uint64_t seed = 0)
    {
        // Skip if error rate is zero
        if (error_rate <= 0.0) return value;

        // Create random number generator
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Union for bit manipulation
        union {
            T value;
            uint32_t bits;
            uint8_t bytes[sizeof(T)];
        } data;

        data.value = value;

        // Apply different types of upsets
        switch (upset_type) {
            case MultibitUpsetType::SINGLE_BIT: {
                // Flip a single bit with probability error_rate
                if (dist(rng) < error_rate) {
                    std::uniform_int_distribution<unsigned> bit_dist(0, sizeof(T) * 8 - 1);
                    unsigned bit_pos = bit_dist(rng);

                    // Flip the bit
                    data.bits ^= (1u << bit_pos);
                }
                break;
            }

            case MultibitUpsetType::ADJACENT_BITS: {
                // Flip 2-3 adjacent bits with probability error_rate
                if (dist(rng) < error_rate) {
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

            case MultibitUpsetType::ROW_UPSET: {
                // Simulate a row upset in memory (multiple bits in "row")
                if (dist(rng) < error_rate) {
                    // Choose a "row" (byte in this case)
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

            case MultibitUpsetType::COLUMN_UPSET: {
                // Simulate a column upset (same bit position in multiple bytes)
                if (dist(rng) < error_rate) {
                    // Choose a "column" (bit position)
                    std::uniform_int_distribution<unsigned> bit_dist(0, 7);
                    unsigned bit_pos = bit_dist(rng);

                    // Flip this bit in multiple bytes
                    unsigned num_bytes = 1 + static_cast<unsigned>(error_rate * (sizeof(T) - 1));
                    std::uniform_int_distribution<unsigned> byte_dist(0, sizeof(T) - 1);

                    for (unsigned i = 0; i < num_bytes; ++i) {
                        unsigned byte_idx = byte_dist(rng);
                        data.bytes[byte_idx] ^= (1u << bit_pos);
                    }
                }
                break;
            }

            case MultibitUpsetType::RANDOM_MULTI: {
                // Randomly flip multiple bits throughout the value
                unsigned num_flips = static_cast<unsigned>(error_rate * 8);

                for (unsigned i = 0; i < num_flips; ++i) {
                    if (dist(rng) < error_rate) {
                        std::uniform_int_distribution<unsigned> bit_dist(0, sizeof(T) * 8 - 1);
                        unsigned bit_pos = bit_dist(rng);

                        // Calculate byte and bit within byte
                        unsigned byte_idx = bit_pos / 8;
                        unsigned bit_in_byte = bit_pos % 8;

                        // Flip the bit
                        data.bytes[byte_idx] ^= (1u << bit_in_byte);
                    }
                }
                break;
            }
        }

        return data.value;
    }

    /**
     * @brief Assignment operator
     *
     * @param value New value
     * @return Reference to this object
     */
    MultibitProtection& operator=(const T& value)
    {
        setValue(value);
        return *this;
    }

    /**
     * @brief Conversion operator to T
     *
     * @return Protected value
     */
    operator T() const { return getValue(); }

   private:
    // Value storage
    mutable T value_;

    // ECC coding scheme
    ECCCodingScheme coding_scheme_;

    // ECC data for Hamming/SECDED (fixed size)
    mutable std::array<uint8_t, 32> ecc_data_;

    // Real Reed-Solomon backend (lazy-initialized)
    // Tour of C++: std::optional for lazy initialization without heap allocation overhead
    mutable std::optional<RSProtectionBackend<T, RSCorrectionTier::STANDARD>> rs_backend_;

    // Validity flag
    mutable bool valid_;

    /**
     * @brief Ensure RS backend is initialized (lazy RAII)
     */
    void ensureRSBackend() const
    {
        if (!rs_backend_) {
            rs_backend_.emplace(value_);
        }
    }

    /**
     * @brief Update ECC data for the current value
     */
    void updateECC()
    {
        // Generate ECC based on coding scheme
        switch (coding_scheme_) {
            case ECCCodingScheme::NONE:
                // No ECC
                break;

            case ECCCodingScheme::HAMMING:
            case ECCCodingScheme::SECDED:
                generateHammingCode();
                break;

            case ECCCodingScheme::REED_SOLOMON:
                generateReedSolomon();
                break;

            default:
                break;
        }
    }

    /**
     * @brief Generate Hamming code for error detection/correction
     */
    void generateHammingCode()
    {
        // Implementation for 32-bit values (assuming float or int32)
        // For simplicity, assume sizeof(T) == 4
        static_assert(sizeof(T) == 4, "Hamming code implementation assumes 4-byte values");

        union {
            T value;
            uint32_t bits;
        } data;

        data.value = value_;

        // Clear current ECC
        std::fill(ecc_data_.begin(), ecc_data_.begin() + 8, 0);

        // Calculate parity bits for Hamming code
        // We use a simplified Hamming(39,32) code with 7 parity bits
        for (int i = 0; i < 32; ++i) {
            if ((data.bits >> i) & 1) {
                // Position i+1 (1-indexed) contributes to parity bits where its bits are set
                int pos = i + 1;
                for (int j = 0; j < 6; ++j) {
                    if ((pos >> j) & 1) {
                        ecc_data_[j] ^= 1;
                    }
                }
            }
        }

        // Calculate overall parity for SEC-DED
        ecc_data_[6] = 0;
        for (int i = 0; i < 6; ++i) {
            ecc_data_[6] ^= ecc_data_[i];
        }
        for (int i = 0; i < 32; ++i) {
            ecc_data_[6] ^= ((data.bits >> i) & 1);
        }
    }

    /**
     * @brief Check Hamming parity
     *
     * @return True if an error is detected
     */
    bool checkHammingParity() const
    {
        union {
            T value;
            uint32_t bits;
        } data;

        data.value = value_;

        // Calculate current parity
        std::array<uint8_t, 7> current_parity;
        std::fill(current_parity.begin(), current_parity.end(), 0);

        // Calculate parity bits from the current value
        for (int i = 0; i < 32; ++i) {
            if ((data.bits >> i) & 1) {
                // Position i+1 (1-indexed) contributes to parity bits where its bits are set
                int pos = i + 1;
                for (int j = 0; j < 6; ++j) {
                    if ((pos >> j) & 1) {
                        current_parity[j] ^= 1;
                    }
                }
            }
        }

        // Calculate overall parity
        current_parity[6] = 0;
        for (int i = 0; i < 6; ++i) {
            current_parity[6] ^= current_parity[i];
        }
        for (int i = 0; i < 32; ++i) {
            current_parity[6] ^= ((data.bits >> i) & 1);
        }

        // Compare with stored parity
        for (int i = 0; i < 7; ++i) {
            if (current_parity[i] != ecc_data_[i]) {
                return true;  // Error detected
            }
        }

        return false;  // No error
    }

    /**
     * @brief Correct errors using Hamming code
     *
     * @return True if errors were successfully corrected
     */
    bool correctHammingCode() const
    {
        union {
            T value;
            uint32_t bits;
        } data;

        data.value = value_;

        // Calculate syndrome
        int syndrome = 0;
        std::array<uint8_t, 6> current_parity;
        std::fill(current_parity.begin(), current_parity.end(), 0);

        // Calculate parity bits from the current value
        for (int i = 0; i < 32; ++i) {
            if ((data.bits >> i) & 1) {
                // Position i+1 (1-indexed) contributes to parity bits where its bits are set
                int pos = i + 1;
                for (int j = 0; j < 6; ++j) {
                    if ((pos >> j) & 1) {
                        current_parity[j] ^= 1;
                    }
                }
            }
        }

        // Calculate syndrome (error position)
        for (int i = 0; i < 6; ++i) {
            if (current_parity[i] != ecc_data_[i]) {
                syndrome |= (1 << i);
            }
        }

        // If syndrome is non-zero, correct the error
        if (syndrome > 0 && syndrome <= 32) {
            // Flip the bit at the error position (0-indexed in our data)
            data.bits ^= (1u << (syndrome - 1));
            value_ = data.value;
            return true;
        }

        return false;  // Can't correct or no error
    }

    /**
     * @brief Correct errors using SEC-DED
     *
     * @return True if errors were successfully corrected
     */
    bool correctSECDED() const
    {
        union {
            T value;
            uint32_t bits;
        } data;

        data.value = value_;

        // Calculate syndrome
        int syndrome = 0;
        std::array<uint8_t, 7> current_parity;
        std::fill(current_parity.begin(), current_parity.end(), 0);

        // Calculate parity bits from the current value
        for (int i = 0; i < 32; ++i) {
            if ((data.bits >> i) & 1) {
                // Position i+1 (1-indexed) contributes to parity bits where its bits are set
                int pos = i + 1;
                for (int j = 0; j < 6; ++j) {
                    if ((pos >> j) & 1) {
                        current_parity[j] ^= 1;
                    }
                }
            }
        }

        // Calculate overall parity
        current_parity[6] = 0;
        for (int i = 0; i < 6; ++i) {
            current_parity[6] ^= current_parity[i];
        }
        for (int i = 0; i < 32; ++i) {
            current_parity[6] ^= ((data.bits >> i) & 1);
        }

        // Calculate syndrome
        for (int i = 0; i < 6; ++i) {
            if (current_parity[i] != ecc_data_[i]) {
                syndrome |= (1 << i);
            }
        }

        // Check if overall parity error exists
        bool overall_parity_error = (current_parity[6] != ecc_data_[6]);

        // SEC-DED logic:
        // 1. If syndrome is zero and overall parity is correct: no error
        // 2. If syndrome is non-zero and overall parity is wrong: single error, correctable
        // 3. If syndrome is zero and overall parity is wrong: error in parity bit, ignore
        // 4. If syndrome is non-zero and overall parity is correct: double error, uncorrectable

        if (syndrome == 0 && !overall_parity_error) {
            return true;  // No error
        }
        else if (syndrome > 0 && overall_parity_error) {
            // Single bit error, correct it
            if (syndrome <= 32) {
                // Flip the bit at the error position (0-indexed in our data)
                data.bits ^= (1u << (syndrome - 1));
                value_ = data.value;
                return true;
            }
        }
        else if (syndrome == 0 && overall_parity_error) {
            // Error in parity bit, not in data
            return true;
        }
        else {
            // Double error, uncorrectable
            return false;
        }

        return false;
    }

    /**
     * @brief Generate Reed-Solomon codes for error correction
     *
     * Tour of C++: Real implementation using GF(2^8) arithmetic.
     * Uses layered decoders (Peterson → brute-force → BM) for robust
     * multi-bit upset correction.
     */
    void generateReedSolomon()
    {
        // Initialize RS backend if needed, then encode
        ensureRSBackend();
        rs_backend_->encode(value_);
    }

    /**
     * @brief Check Reed-Solomon codes for error detection
     *
     * Uses syndrome calculation on the RS codeword.
     *
     * @return True if an error is detected
     */
    bool checkReedSolomon() const
    {
        ensureRSBackend();
        return rs_backend_->has_error();
    }

    /**
     * @brief Correct errors using Reed-Solomon codes
     *
     * Uses the layered decoder strategy:
     * - Peterson decoder for single errors (O(n))
     * - Brute-force solver for 2-3 errors (O(n²), O(n³))
     * - Berlekamp-Massey fallback for 4+ errors
     *
     * Can correct up to t=3 symbol errors (STANDARD tier).
     *
     * @return True if errors were successfully corrected
     */
    bool correctReedSolomon() const
    {
        ensureRSBackend();

        auto result = rs_backend_->decode();
        if (result) {
            value_ = *result;
            return true;
        }

        return false;  // Uncorrectable (> t errors)
    }
};

/**
 * @brief Checksum-based protection for neural network weights
 *
 * This class implements a simple checksum protection mechanism.
 * It only detects errors but doesn't provide correction.
 */
template <typename T>
class ChecksumProtection : public MultibitProtection<T> {
   public:
    /**
     * @brief Default constructor
     */
    ChecksumProtection() : MultibitProtection<T>(T{}, ECCCodingScheme::NONE) { updateChecksum(); }

    /**
     * @brief Constructor with initial value
     */
    ChecksumProtection(T value) : MultibitProtection<T>(value, ECCCodingScheme::NONE)
    {
        updateChecksum();
    }

    /**
     * @brief Set a new value
     */
    void setValue(T value)
    {
        MultibitProtection<T>::setValue(value);
        updateChecksum();
    }

    /**
     * @brief Check if the stored value has an error
     */
    bool hasError() const { return checksum_ != calculateChecksum(); }

    /**
     * @brief Try to correct errors (not possible with simple checksum)
     */
    bool correctErrors() const
    {
        return false;  // Cannot correct with just checksums
    }

   private:
    uint32_t checksum_ = 0;

    void updateChecksum() { checksum_ = calculateChecksum(); }

    uint32_t calculateChecksum() const
    {
        union {
            T value;
            uint8_t bytes[sizeof(T)];
        } data;

        data.value = this->getValue();

        uint32_t sum = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            sum = (sum << 1) ^ data.bytes[i];
        }

        return sum;
    }
};

/**
 * @brief Triple Modular Redundancy protection for neural network weights
 *
 * This class uses TMR to protect values against radiation-induced errors.
 */
template <typename T>
class TripleModularRedundancy : public MultibitProtection<T> {
   public:
    /**
     * @brief Default constructor
     */
    TripleModularRedundancy() : MultibitProtection<T>()
    {
        tmr_ = core::redundancy::TripleModularRedundancy<T>();
    }

    /**
     * @brief Constructor with initial value
     */
    TripleModularRedundancy(T value) : MultibitProtection<T>(value)
    {
        tmr_ = core::redundancy::TripleModularRedundancy<T>(value);
    }

    /**
     * @brief Get the protected value
     */
    T getValue() const { return tmr_.get(); }

    /**
     * @brief Set a new value
     */
    void setValue(T value)
    {
        MultibitProtection<T>::setValue(value);
        tmr_.set(value);
    }

    /**
     * @brief Check if the stored value has an error
     */
    bool hasError() const { return tmr_.verify() != 0; }

    /**
     * @brief Try to correct errors
     */
    bool correctErrors() const { return tmr_.repair() == 0; }

   private:
    mutable core::redundancy::TripleModularRedundancy<T> tmr_;
};

/**
 * @brief Adaptive TMR protection that adjusts to radiation levels
 *
 * This class implements TMR that can be enabled/disabled adaptively
 * based on radiation levels to save power/computation.
 */
template <typename T>
class AdaptiveTMRProtection : public MultibitProtection<T> {
   public:
    /**
     * @brief Default constructor
     */
    AdaptiveTMRProtection()
        : MultibitProtection<T>(), protection_enabled_(false), radiation_threshold_(0.3)
    {
        tmr_ = core::redundancy::TripleModularRedundancy<T>();
    }

    /**
     * @brief Constructor with initial value
     */
    AdaptiveTMRProtection(T value, double radiation_threshold = 0.3)
        : MultibitProtection<T>(value),
          protection_enabled_(false),
          radiation_threshold_(radiation_threshold)
    {
        tmr_ = core::redundancy::TripleModularRedundancy<T>(value);
    }

    /**
     * @brief Get the protected value
     */
    T getValue() const
    {
        if (protection_enabled_) {
            return tmr_.get();
        }
        else {
            return MultibitProtection<T>::getValue();
        }
    }

    /**
     * @brief Set a new value
     */
    void setValue(T value)
    {
        MultibitProtection<T>::setValue(value);
        tmr_.set(value);
    }

    /**
     * @brief Set the radiation threshold for activating protection
     */
    void setRadiationThreshold(double threshold) { radiation_threshold_ = threshold; }

    /**
     * @brief Update protection based on current radiation level
     */
    void updateProtectionStatus(double radiation_level)
    {
        protection_enabled_ = (radiation_level >= radiation_threshold_);
    }

    /**
     * @brief Check if the stored value has an error
     */
    bool hasError() const
    {
        if (protection_enabled_) {
            return tmr_.verify() != 0;
        }
        else {
            return MultibitProtection<T>::hasError();
        }
    }

    /**
     * @brief Try to correct errors
     */
    bool correctErrors() const
    {
        if (protection_enabled_) {
            return tmr_.repair() == 0;
        }
        else {
            return MultibitProtection<T>::correctErrors();
        }
    }

   private:
    mutable core::redundancy::TripleModularRedundancy<T> tmr_;
    bool protection_enabled_;
    double radiation_threshold_;
};

/**
 * @brief Space-optimized protection for neural network weights
 *
 * This class implements enhanced TMR with optimizations for
 * space applications, minimizing memory and power usage.
 */
template <typename T>
class SpaceOptimizedProtection : public MultibitProtection<T> {
   public:
    /**
     * @brief Default constructor
     */
    SpaceOptimizedProtection() : MultibitProtection<T>()
    {
        tmr_ = core::redundancy::SpaceEnhancedTMR<T>();
    }

    /**
     * @brief Constructor with initial value
     */
    SpaceOptimizedProtection(T value) : MultibitProtection<T>(value)
    {
        tmr_ = core::redundancy::SpaceEnhancedTMR<T>(value);
    }

    /**
     * @brief Get the protected value
     */
    T getValue() const
    {
        T value{};
        tmr_.get(value);
        return value;
    }

    /**
     * @brief Set a new value
     */
    void setValue(T value)
    {
        MultibitProtection<T>::setValue(value);
        tmr_.set(value);
    }

    /**
     * @brief Check if the stored value has an error
     */
    bool hasError() const
    {
        core::error::StatusCode result = tmr_.verify();
        return result != core::error::StatusCode::SUCCESS;
    }

    /**
     * @brief Try to correct errors
     */
    bool correctErrors() const
    {
        core::error::StatusCode result = tmr_.repair();
        return result == core::error::StatusCode::SUCCESS;
    }

   private:
    mutable core::redundancy::SpaceEnhancedTMR<T> tmr_;
};

/**
 * @brief Reed-Solomon Protected Value with Compile-Time Tier Selection
 *
 * Tour of C++ philosophy:
 * - Zero-cost abstraction: Tier selected at compile time, no runtime overhead
 * - RAII: Protection initialized on construction
 * - Value semantics: Acts like a regular value with protection baked in
 * - Type safety: static_assert ensures only valid types and tiers
 *
 * Usage:
 *   RSProtectedValue<float, RSCorrectionTier::HEAVY> critical_weight(3.14f);
 *   float v = critical_weight.get();  // Corrects up to 4 symbol errors
 *
 * @tparam T Data type to protect
 * @tparam Tier Correction tier (LIGHT=t2, STANDARD=t3, HEAVY=t4)
 */
template <typename T, RSCorrectionTier Tier = RSCorrectionTier::STANDARD>
class RSProtectedValue {
    static_assert(is_rs_protectable_v<T>,
                  "T must be trivially copyable and fit within RS block constraints");

   public:
    using value_type = T;
    using backend_type = RSProtectionBackend<T, Tier>;

    static constexpr uint8_t correction_capability = backend_type::correction_capability;

    /**
     * @brief Default constructor
     */
    RSProtectedValue() noexcept = default;

    /**
     * @brief Construct with initial value
     */
    explicit RSProtectedValue(const T& value) : backend_(value) {}

    // Rule of Five: Default all - backend handles resources
    RSProtectedValue(const RSProtectedValue&) = default;
    RSProtectedValue(RSProtectedValue&&) noexcept = default;
    RSProtectedValue& operator=(const RSProtectedValue&) = default;
    RSProtectedValue& operator=(RSProtectedValue&&) noexcept = default;
    ~RSProtectedValue() = default;

    /**
     * @brief Assign a new value
     */
    RSProtectedValue& operator=(const T& value)
    {
        backend_.encode(value);
        return *this;
    }

    /**
     * @brief Get value with automatic error correction
     *
     * Tour of C++: Implicit conversion enables natural usage.
     */
    [[nodiscard]] T get() const { return backend_.get(); }

    /**
     * @brief Implicit conversion to T
     */
    operator T() const { return get(); }

    /**
     * @brief Set a new value
     */
    void set(const T& value) { backend_.encode(value); }

    /**
     * @brief Check if errors are present
     */
    [[nodiscard]] bool has_error() const noexcept { return backend_.has_error(); }

    /**
     * @brief Attempt error correction
     * @return Corrected value if successful
     */
    [[nodiscard]] std::optional<T> try_correct() const { return backend_.decode(); }

    /**
     * @brief Inject error for testing (at byte_pos, bit_pos)
     */
    void inject_error(size_t byte_pos, uint8_t bit_pos) noexcept
    {
        backend_.inject_error(byte_pos, bit_pos);
    }

    /**
     * @brief Get underlying backend (for advanced usage)
     */
    [[nodiscard]] const backend_type& backend() const noexcept { return backend_; }

   private:
    backend_type backend_;
};

/**
 * @brief Hybrid TMR + RS Protection
 *
 * Combines TMR (catastrophic protection) with RS (efficient MBU correction).
 * RS handles most errors cheaply; TMR catches what RS can't.
 *
 * Tour of C++ philosophy:
 * - Composition over inheritance
 * - Zero-cost when no errors (fast path uses cached value)
 * - RAII for both protection mechanisms
 *
 * @tparam T Data type to protect
 * @tparam Tier RS correction tier
 */
template <typename T, RSCorrectionTier Tier = RSCorrectionTier::STANDARD>
class HybridRSTMRProtection {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

   public:
    using value_type = T;

    /**
     * @brief Default constructor
     */
    HybridRSTMRProtection() noexcept : tmr_() {}

    /**
     * @brief Construct with initial value
     */
    explicit HybridRSTMRProtection(const T& value) : rs_(value), tmr_(value) {}

    /**
     * @brief Get value with layered error correction
     *
     * Strategy:
     * 1. Try RS decode (fast, handles up to t errors)
     * 2. Verify against TMR majority
     * 3. Fallback to TMR voting if RS fails
     */
    [[nodiscard]] T get() const
    {
        // Fast path: Try RS correction first
        auto rs_result = rs_.try_correct();

        if (rs_result) {
            // Verify RS result against TMR
            T tmr_value = tmr_.get();
            if (*rs_result == tmr_value) {
                return *rs_result;  // Agreement = high confidence
            }
            // Disagreement: trust TMR majority voting
        }

        // Fallback: TMR voting
        return tmr_.get();
    }

    /**
     * @brief Set a new value in both protection schemes
     */
    void set(const T& value)
    {
        rs_.set(value);
        tmr_.set(value);
    }

    /**
     * @brief Assign a new value
     */
    HybridRSTMRProtection& operator=(const T& value)
    {
        set(value);
        return *this;
    }

    /**
     * @brief Implicit conversion to T
     */
    operator T() const { return get(); }

    /**
     * @brief Check if either protection detects errors
     *
     * Note: TMR is self-correcting via majority voting, so we primarily
     * rely on RS syndrome detection for error reporting.
     */
    [[nodiscard]] bool has_error() const noexcept
    {
        // RS syndrome detection is the primary error indicator
        // TMR self-corrects via get(), so we don't need separate verification
        return rs_.has_error();
    }

    /**
     * @brief Get RS correction statistics
     */
    [[nodiscard]] static constexpr uint8_t rs_correction_capability() noexcept
    {
        return RSProtectedValue<T, Tier>::correction_capability;
    }

   private:
    RSProtectedValue<T, Tier> rs_;
    core::redundancy::TripleModularRedundancy<T> tmr_;
};

// Convenient type aliases for common use cases
template <typename T>
using RSLight = RSProtectedValue<T, RSCorrectionTier::LIGHT>;

template <typename T>
using RSStandard = RSProtectedValue<T, RSCorrectionTier::STANDARD>;

template <typename T>
using RSHeavy = RSProtectedValue<T, RSCorrectionTier::HEAVY>;

template <typename T>
using HybridProtection = HybridRSTMRProtection<T, RSCorrectionTier::STANDARD>;

}  // namespace neural
}  // namespace rad_ml

#endif  // RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP
