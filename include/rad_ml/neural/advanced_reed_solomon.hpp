/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
/**
 * @file advanced_reed_solomon.hpp
 * @brief Advanced Reed-Solomon error correction for neural networks
 * 
 * This file implements a proper Reed-Solomon encoder/decoder for neural network
 * parameters, using Galois Field arithmetic for reliability in radiation
 * environments.
 */

#ifndef RAD_ML_NEURAL_ADVANCED_REED_SOLOMON_HPP
#define RAD_ML_NEURAL_ADVANCED_REED_SOLOMON_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <limits>
#include <bitset>
#include <algorithm>

#include "galois_field.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Advanced Reed-Solomon encoder/decoder
 * 
 * This class implements a proper Reed-Solomon codec for protecting data
 * against radiation-induced errors. It uses the Galois Field implementation
 * for correct finite field arithmetic.
 * 
 * @tparam T Data type to protect (must be trivially copyable)
 * @tparam SymbolSize Size of Reed-Solomon symbols in bits (usually 8)
 * @tparam ECCSymbols Number of error correction symbols to add
 */
template<typename T, uint8_t SymbolSize = 8, uint8_t ECCSymbols = 8>
class AdvancedReedSolomon {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static_assert(SymbolSize > 0 && SymbolSize <= 16, "Symbol size must be between 1 and 16 bits");
    static_assert(ECCSymbols > 0, "Error correction symbol count must be positive");
    
public:
    // Use GF256 as the default field for 8-bit symbols
    using GF = std::conditional_t<SymbolSize == 8,
                                GF256,
                                std::conditional_t<SymbolSize == 4,
                                                 GF16,
                                                 GF1024>>;
    
    using element_t = typename GF::element_t;
    
    static constexpr size_t data_size = sizeof(T);
    static constexpr size_t symbol_bytes = (SymbolSize <= 8) ? 1 : 2;
    static constexpr size_t data_symbols = data_size / symbol_bytes + ((data_size % symbol_bytes) ? 1 : 0);
    static constexpr size_t total_symbols = data_symbols + ECCSymbols;
    static constexpr size_t total_size = total_symbols * symbol_bytes;

    /**
     * @brief Constructor initializes the Galois Field
     */
    AdvancedReedSolomon() {
        // Precompute generator polynomial for efficiency
        generator_poly_ = field_.rs_generator_poly(ECCSymbols);
    }
    
    /**
     * @brief Encode data with Reed-Solomon ECC
     * 
     * @param data Data to encode
     * @return Encoded data with ECC symbols
     */
    std::vector<uint8_t> encode(const T& data) const {
        // Convert data to field elements
        std::vector<element_t> message = convert_to_elements(data);
        
        // Pad message to required length
        message.resize(total_symbols, 0);
        
        // Systematic encoding: first copy the data
        std::vector<element_t> codeword = message;
        
        // Compute ECC symbols
        auto ecc = compute_ecc_symbols(message);
        
        // Place ECC symbols at the end of the codeword
        std::copy(ecc.begin(), ecc.end(), codeword.begin() + data_symbols);
        
        // Convert back to bytes
        return convert_from_elements(codeword);
    }
    
    /**
     * @brief Decode data with Reed-Solomon error correction
     * 
     * @param encoded_data Encoded data with ECC symbols
     * @return Decoded data if correctable, or std::nullopt if uncorrectable
     */
    std::optional<T> decode(const std::vector<uint8_t>& encoded_data) const {
        if (encoded_data.size() < total_symbols * symbol_bytes) {
            return std::nullopt; // Not enough data
        }
        
        // Convert encoded data to field elements
        std::vector<element_t> codeword;
        codeword.reserve(total_symbols);
        
        for (size_t i = 0; i < total_symbols; ++i) {
            if (i * symbol_bytes >= encoded_data.size()) break;
            
            element_t symbol = 0;
            for (size_t j = 0; j < symbol_bytes && (i * symbol_bytes + j) < encoded_data.size(); ++j) {
                symbol |= static_cast<element_t>(encoded_data[i * symbol_bytes + j]) << (j * 8);
            }
            
            codeword.push_back(symbol);
        }
        
        // Try to correct errors
        auto corrected = field_.rs_correct_errors(codeword, ECCSymbols);
        if (!corrected) {
            return std::nullopt; // Uncorrectable errors
        }
        
        // Extract the data portion (systematic encoding)
        std::vector<element_t> data_symbols_vec(corrected->begin(), corrected->begin() + data_symbols);
        
        // Convert back to T
        return convert_elements_to_data<T>(data_symbols_vec);
    }
    
    /**
     * @brief Calculate the protection overhead as a percentage
     * 
     * @return Percentage overhead
     */
    constexpr double overhead_percent() const {
        return (static_cast<double>(total_size) / data_size - 1.0) * 100.0;
    }
    
    /**
     * @brief Get the maximum number of symbols that can be corrected
     * 
     * @return Error correction capability
     */
    constexpr size_t correction_capability() const {
        return ECCSymbols / 2;
    }
    
    /**
     * @brief Check if a message can be corrected without modifying it
     * 
     * @param encoded_data Encoded data with ECC symbols
     * @return True if data is valid or correctable, false otherwise
     */
    bool is_correctable(const std::vector<uint8_t>& encoded_data) const {
        if (encoded_data.size() < total_symbols * symbol_bytes) {
            return false; // Not enough data
        }
        
        // Convert encoded data to field elements
        std::vector<element_t> codeword;
        codeword.reserve(total_symbols);
        
        for (size_t i = 0; i < total_symbols; ++i) {
            if (i * symbol_bytes >= encoded_data.size()) break;
            
            element_t symbol = 0;
            for (size_t j = 0; j < symbol_bytes && (i * symbol_bytes + j) < encoded_data.size(); ++j) {
                symbol |= static_cast<element_t>(encoded_data[i * symbol_bytes + j]) << (j * 8);
            }
            
            codeword.push_back(symbol);
        }
        
        // Calculate syndromes
        auto syndromes = field_.rs_calc_syndromes(codeword, ECCSymbols);
        
        // Check if all syndromes are zero (no errors)
        for (size_t i = 1; i < syndromes.size(); ++i) {
            if (syndromes[i] != 0) {
                // Errors detected, check if they're correctable
                auto [err_loc, err_eval] = field_.rs_find_error_locator(syndromes, ECCSymbols);
                auto err_pos = field_.rs_find_errors(err_loc, codeword.size());
                
                // If we can locate all errors and there are at most t errors, it's correctable
                return !err_pos.empty() && err_pos.size() <= correction_capability();
            }
        }
        
        return true; // No errors detected
    }
    
    /**
     * @brief Apply bit interleaving to improve multi-bit upset protection
     * 
     * @param data Input data
     * @return Interleaved data
     */
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
    
    /**
     * @brief Undo bit interleaving
     * 
     * @param interleaved_data Interleaved data
     * @return Original data
     */
    std::vector<uint8_t> deinterleave(const std::vector<uint8_t>& interleaved_data) const {
        if (interleaved_data.empty()) return {};
        
        // Determine block count
        size_t block_count = (interleaved_data.size() + 7) / 8;
        std::vector<uint8_t> result(interleaved_data.size());
        
        // Process each bit position across all bytes
        for (size_t bit = 0; bit < 8; ++bit) {
            for (size_t block = 0; block < block_count; ++block) {
                size_t src_idx = bit * block_count + block;
                size_t dst_idx = block;
                
                if (src_idx < interleaved_data.size() && dst_idx < result.size()) {
                    // Extract bit from source byte
                    bool bit_value = (interleaved_data[src_idx / 8] >> (src_idx % 8)) & 1;
                    
                    // Set bit in destination byte
                    if (bit_value) {
                        result[dst_idx] |= (1 << bit);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Apply random bit errors for testing error correction
     * 
     * @param data Data to corrupt
     * @param error_rate Probability of each bit being flipped
     * @param seed Random seed for reproducibility
     * @return Corrupted data
     */
    std::vector<uint8_t> apply_bit_errors(
        const std::vector<uint8_t>& data, 
        double error_rate,
        uint64_t seed = 0
    ) const {
        if (data.empty() || error_rate <= 0.0) return data;
        
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
    
    /**
     * @brief Apply multi-bit errors with spatial correlation
     * 
     * @param data Data to corrupt
     * @param error_rate Base probability for errors
     * @param burst_size Maximum size of error bursts (1-8 bits)
     * @param seed Random seed for reproducibility
     * @return Corrupted data
     */
    std::vector<uint8_t> apply_burst_errors(
        const std::vector<uint8_t>& data,
        double error_rate,
        uint8_t burst_size = 3,
        uint64_t seed = 0
    ) const {
        if (data.empty() || error_rate <= 0.0 || burst_size == 0) return data;
        
        std::vector<uint8_t> result = data;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<size_t> byte_dist(0, data.size() - 1);
        std::uniform_int_distribution<int> bit_dist(0, 7);
        std::uniform_int_distribution<int> size_dist(1, std::min(8, static_cast<int>(burst_size)));
        
        // Number of bursts is scaled by error rate and data size
        size_t num_bursts = static_cast<size_t>(error_rate * data.size() / 2) + 1;
        
        for (size_t i = 0; i < num_bursts; ++i) {
            // Select random starting position
            size_t byte_idx = byte_dist(rng);
            int bit_idx = bit_dist(rng);
            int burst = size_dist(rng);
            
            // Apply burst error
            for (int j = 0; j < burst; ++j) {
                int current_bit = (bit_idx + j) % 8;
                size_t current_byte = byte_idx + (bit_idx + j) / 8;
                if (current_byte < result.size()) {
                    result[current_byte] ^= (1 << current_bit);
                }
            }
        }
        
        return result;
    }

private:
    GF field_;
    std::vector<element_t> generator_poly_;
    
    /**
     * @brief Convert data to a vector of field elements
     * 
     * @param data Data to convert
     * @return Vector of field elements
     */
    std::vector<element_t> convert_to_elements(const T& data) const {
        std::vector<element_t> elements;
        elements.reserve(data_symbols);
        
        // Access raw bytes of data
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&data);
        
        // Convert bytes to field elements
        for (size_t i = 0; i < data_size; i += symbol_bytes) {
            element_t element = 0;
            
            for (size_t j = 0; j < symbol_bytes && (i + j) < data_size; ++j) {
                element |= static_cast<element_t>(bytes[i + j]) << (j * 8);
            }
            
            elements.push_back(element);
        }
        
        return elements;
    }
    
    /**
     * @brief Convert field elements to raw bytes
     * 
     * @param elements Vector of field elements
     * @return Vector of bytes
     */
    std::vector<uint8_t> convert_from_elements(const std::vector<element_t>& elements) const {
        std::vector<uint8_t> bytes;
        bytes.reserve(elements.size() * symbol_bytes);
        
        for (const auto& element : elements) {
            for (size_t j = 0; j < symbol_bytes; ++j) {
                bytes.push_back(static_cast<uint8_t>((element >> (j * 8)) & 0xFF));
            }
        }
        
        return bytes;
    }
    
    /**
     * @brief Compute ECC symbols for a message
     * 
     * @param message Message to protect (without ECC symbols)
     * @return Vector of ECC symbols
     */
    std::vector<element_t> compute_ecc_symbols(const std::vector<element_t>& message) const {
        // Implementation of systematic Reed-Solomon encoding
        // We compute the remainder when dividing x^(n-k) * message(x) by generator(x)
        
        // Initialize remainder with highest-order terms of message
        std::vector<element_t> remainder(ECCSymbols, 0);
        
        // Process each message symbol
        for (size_t i = 0; i < data_symbols; ++i) {
            element_t feedback = field_.add(message[i], remainder[0]);
            
            // Skip multiplications if feedback is zero
            if (feedback != 0) {
                for (size_t j = 1; j < ECCSymbols; ++j) {
                    if (j < generator_poly_.size()) {
                        remainder[j-1] = field_.add(remainder[j], field_.multiply(feedback, generator_poly_[j]));
                    } else {
                        remainder[j-1] = remainder[j];
                    }
                }
                
                // Last element special case
                if (generator_poly_.size() > ECCSymbols) {
                    remainder[ECCSymbols-1] = 0;
                } else {
                    remainder[ECCSymbols-1] = field_.multiply(feedback, generator_poly_[ECCSymbols]);
                }
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
    
    /**
     * @brief Convert field elements back to data type T
     * 
     * @tparam U Data type to convert to
     * @param elements Field elements
     * @return Data value of type U
     */
    template<typename U>
    std::optional<U> convert_elements_to_data(const std::vector<element_t>& elements) const {
        if (elements.size() < data_symbols) {
            return std::nullopt; // Not enough data
        }
        
        U result{};
        uint8_t* result_bytes = reinterpret_cast<uint8_t*>(&result);
        
        // Convert field elements to bytes
        for (size_t i = 0; i < data_symbols && i < elements.size(); ++i) {
            for (size_t j = 0; j < symbol_bytes; ++j) {
                size_t byte_idx = i * symbol_bytes + j;
                if (byte_idx < sizeof(U)) {
                    result_bytes[byte_idx] = static_cast<uint8_t>((elements[i] >> (j * 8)) & 0xFF);
                }
            }
        }
        
        return result;
    }
};

// Common Reed-Solomon configurations
template<typename T>
using RS4Bit8Sym = AdvancedReedSolomon<T, 4, 8>;  // 4-bit symbols, 8 ECC symbols (can correct 4 symbols)

template<typename T>
using RS8Bit8Sym = AdvancedReedSolomon<T, 8, 8>;  // 8-bit symbols, 8 ECC symbols (can correct 4 symbols)

template<typename T>
using RS8Bit16Sym = AdvancedReedSolomon<T, 8, 16>; // 8-bit symbols, 16 ECC symbols (can correct 8 symbols)

template<typename T>
using RS8Bit32Sym = AdvancedReedSolomon<T, 8, 32>; // 8-bit symbols, 32 ECC symbols (can correct 16 symbols)

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_ADVANCED_REED_SOLOMON_HPP 
