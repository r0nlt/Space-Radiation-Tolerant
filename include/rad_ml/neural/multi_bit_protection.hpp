/**
 * @file multi_bit_protection.hpp
 * @brief Protection against multi-bit upsets in neural networks
 * 
 * This file defines protection mechanisms for handling multi-bit upsets
 * that can occur in neural network parameters in radiation environments.
 */

#ifndef RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP
#define RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP

#include <bitset>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <type_traits>

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
    NONE,           ///< No ECC
    HAMMING,        ///< Hamming code (single bit correction)
    SECDED,         ///< SEC-DED (Single Error Correction, Double Error Detection)
    REED_SOLOMON    ///< Reed-Solomon codes (multiple error correction)
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
    MultibitProtection() 
        : value_(), coding_scheme_(ECCCodingScheme::NONE), valid_(true) {
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
        : value_(value), coding_scheme_(coding_scheme), valid_(true) {
        // Initialize ECC
        updateECC();
    }
    
    /**
     * @brief Get stored value with error checking/correction
     * 
     * @return Stored value (corrected if possible)
     */
    T getValue() const {
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
    void setValue(T value) {
        value_ = value;
        valid_ = true;
        updateECC();
    }
    
    /**
     * @brief Check if the stored value has an error
     * 
     * @return True if an error is detected
     */
    bool hasError() const {
        // If already marked as invalid, return true
        if (!valid_) return true;
        
        // Check using appropriate ECC scheme
        switch (coding_scheme_) {
            case ECCCodingScheme::NONE:
                return false; // Can't detect errors
                
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
    bool correctErrors() const {
        if (!hasError()) return true;
        
        // Attempt correction using appropriate ECC scheme
        switch (coding_scheme_) {
            case ECCCodingScheme::NONE:
                return false; // Can't correct
                
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
    void markInvalid() {
        valid_ = false;
    }
    
    /**
     * @brief Check if the value is currently valid
     * 
     * @return True if valid
     */
    bool isValid() const {
        return valid_ && !hasError();
    }
    
    /**
     * @brief Apply bit interleaving to protect against adjacent bit upsets
     * 
     * @return Interleaved value
     */
    T applyBitInterleaving() const {
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
                interleaved.bits |= ((original.bits >> i) & 1) << (i/2);
            } else {
                interleaved.bits |= ((original.bits >> i) & 1) << (16 + i/2);
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
    static T undoBitInterleaving(T interleaved_value) {
        union {
            T value;
            uint32_t bits;
        } original, interleaved;
        
        interleaved.value = interleaved_value;
        original.bits = 0;
        
        // Undo the interleaving
        for (int i = 0; i < 16; ++i) {
            original.bits |= ((interleaved.bits >> i) & 1) << (i*2);
            original.bits |= ((interleaved.bits >> (i+16)) & 1) << (i*2+1);
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
    static T applyMultiBitErrors(
        T value, 
        double error_rate,
        MultibitUpsetType upset_type = MultibitUpsetType::SINGLE_BIT,
        uint64_t seed = 0
    ) {
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
    MultibitProtection& operator=(const T& value) {
        setValue(value);
        return *this;
    }
    
    /**
     * @brief Conversion operator to T
     * 
     * @return Protected value
     */
    operator T() const {
        return getValue();
    }
    
private:
    // Value storage
    mutable T value_;
    
    // ECC coding scheme
    ECCCodingScheme coding_scheme_;
    
    // ECC data (sized for largest ECC scheme)
    mutable std::array<uint8_t, 32> ecc_data_;
    
    // Validity flag
    mutable bool valid_;
    
    /**
     * @brief Update ECC data for the current value
     */
    void updateECC() {
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
    void generateHammingCode() {
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
    bool checkHammingParity() const {
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
                return true; // Error detected
            }
        }
        
        return false; // No error
    }
    
    /**
     * @brief Correct errors using Hamming code
     * 
     * @return True if errors were successfully corrected
     */
    bool correctHammingCode() const {
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
        
        return false; // Can't correct or no error
    }
    
    /**
     * @brief Correct errors using SEC-DED
     * 
     * @return True if errors were successfully corrected
     */
    bool correctSECDED() const {
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
            return true; // No error
        } else if (syndrome > 0 && overall_parity_error) {
            // Single bit error, correct it
            if (syndrome <= 32) {
                // Flip the bit at the error position (0-indexed in our data)
                data.bits ^= (1u << (syndrome - 1));
                value_ = data.value;
                return true;
            }
        } else if (syndrome == 0 && overall_parity_error) {
            // Error in parity bit, not in data
            return true;
        } else {
            // Double error, uncorrectable
            return false;
        }
        
        return false;
    }
    
    /**
     * @brief Generate Reed-Solomon codes for error correction
     * 
     * This is a simplified version of Reed-Solomon for demonstration purposes.
     * A real implementation would use proper finite field arithmetic.
     */
    void generateReedSolomon() {
        // Simplified Reed-Solomon implementation
        // In a real implementation, this would use Galois field arithmetic
        
        union {
            T value;
            uint8_t bytes[sizeof(T)];
        } data;
        
        data.value = value_;
        
        // Create a simple check sequence
        // In a real RS implementation, these would be calculated using proper encoding
        ecc_data_[0] = 0;
        ecc_data_[1] = 0;
        ecc_data_[2] = 0;
        ecc_data_[3] = 0;
        
        // Simple data-dependent checksums
        for (size_t i = 0; i < sizeof(T); ++i) {
            ecc_data_[0] ^= data.bytes[i];
            ecc_data_[1] ^= (data.bytes[i] << (i % 4));
            ecc_data_[2] ^= (data.bytes[i] >> (i % 4));
            ecc_data_[3] ^= (data.bytes[i] * (i + 1));
        }
        
        // Store additional hash of the value for more robust detection
        uint32_t hash = 0x5555;
        for (size_t i = 0; i < sizeof(T); ++i) {
            hash = ((hash << 5) + hash) + data.bytes[i];
        }
        
        // Store hash in ecc_data
        ecc_data_[4] = (hash & 0xFF);
        ecc_data_[5] = ((hash >> 8) & 0xFF);
        ecc_data_[6] = ((hash >> 16) & 0xFF);
        ecc_data_[7] = ((hash >> 24) & 0xFF);
    }
    
    /**
     * @brief Check Reed-Solomon codes for error detection
     * 
     * @return True if an error is detected
     */
    bool checkReedSolomon() const {
        union {
            T value;
            uint8_t bytes[sizeof(T)];
        } data;
        
        data.value = value_;
        
        // Recreate the checks
        uint8_t check0 = 0;
        uint8_t check1 = 0;
        uint8_t check2 = 0;
        uint8_t check3 = 0;
        
        for (size_t i = 0; i < sizeof(T); ++i) {
            check0 ^= data.bytes[i];
            check1 ^= (data.bytes[i] << (i % 4));
            check2 ^= (data.bytes[i] >> (i % 4));
            check3 ^= (data.bytes[i] * (i + 1));
        }
        
        // Check simple checksums
        if (check0 != ecc_data_[0] || 
            check1 != ecc_data_[1] || 
            check2 != ecc_data_[2] || 
            check3 != ecc_data_[3]) {
            return true; // Error detected
        }
        
        // Check hash
        uint32_t hash = 0x5555;
        for (size_t i = 0; i < sizeof(T); ++i) {
            hash = ((hash << 5) + hash) + data.bytes[i];
        }
        
        uint8_t hash0 = (hash & 0xFF);
        uint8_t hash1 = ((hash >> 8) & 0xFF);
        uint8_t hash2 = ((hash >> 16) & 0xFF);
        uint8_t hash3 = ((hash >> 24) & 0xFF);
        
        if (hash0 != ecc_data_[4] || 
            hash1 != ecc_data_[5] || 
            hash2 != ecc_data_[6] || 
            hash3 != ecc_data_[7]) {
            return true; // Error detected
        }
        
        return false; // No error
    }
    
    /**
     * @brief Correct errors using Reed-Solomon codes
     * 
     * This is a simplified version that can only correct very limited errors.
     * A real RS implementation would use proper decoding algorithms.
     * 
     * @return True if errors were successfully corrected
     */
    bool correctReedSolomon() const {
        union {
            T value;
            uint8_t bytes[sizeof(T)];
        } data;
        
        data.value = value_;
        
        // Recreate the checks
        uint8_t check0 = 0;
        uint8_t check1 = 0;
        uint8_t check2 = 0;
        uint8_t check3 = 0;
        
        for (size_t i = 0; i < sizeof(T); ++i) {
            check0 ^= data.bytes[i];
            check1 ^= (data.bytes[i] << (i % 4));
            check2 ^= (data.bytes[i] >> (i % 4));
            check3 ^= (data.bytes[i] * (i + 1));
        }
        
        // Simplified error correction for demonstration
        // In a real implementation, this would use proper RS decoding
        
        // Check if only one byte appears corrupted
        for (size_t i = 0; i < sizeof(T); ++i) {
            // Try to reconstruct the correct byte
            for (int j = 0; j < 256; ++j) {
                uint8_t trial = static_cast<uint8_t>(j);
                uint8_t orig = data.bytes[i];
                
                // Temporarily replace the byte
                data.bytes[i] = trial;
                
                // Recalculate checks
                uint8_t new_check0 = 0;
                uint8_t new_check1 = 0;
                uint8_t new_check2 = 0;
                uint8_t new_check3 = 0;
                
                for (size_t k = 0; k < sizeof(T); ++k) {
                    new_check0 ^= data.bytes[k];
                    new_check1 ^= (data.bytes[k] << (k % 4));
                    new_check2 ^= (data.bytes[k] >> (k % 4));
                    new_check3 ^= (data.bytes[k] * (k + 1));
                }
                
                // If all checks match, we've found a correction
                if (new_check0 == ecc_data_[0] && 
                    new_check1 == ecc_data_[1] && 
                    new_check2 == ecc_data_[2] && 
                    new_check3 == ecc_data_[3]) {
                    
                    // Update the value and return
                    value_ = data.value;
                    return true;
                }
                
                // Restore the original byte for next try
                data.bytes[i] = orig;
            }
        }
        
        return false; // Couldn't correct
    }
};

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_MULTI_BIT_PROTECTION_HPP 