#include "rad_ml/testing/protection_techniques.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <random>
#include <iostream>

namespace rad_ml {
namespace testing {

// Hamming code implementation for single-bit error correction
class HammingCode {
public:
    static constexpr size_t DATA_BITS = 4;  // Number of data bits
    static constexpr size_t PARITY_BITS = 3;  // Number of parity bits
    static constexpr size_t TOTAL_BITS = DATA_BITS + PARITY_BITS;  // Total bits in codeword

    // Encode 4 bits of data into a 7-bit Hamming codeword
    static uint8_t encode(uint8_t data) {
        // Extract data bits
        uint8_t d1 = (data >> 0) & 1;
        uint8_t d2 = (data >> 1) & 1;
        uint8_t d3 = (data >> 2) & 1;
        uint8_t d4 = (data >> 3) & 1;

        // Calculate parity bits
        uint8_t p1 = d1 ^ d2 ^ d4;
        uint8_t p2 = d1 ^ d3 ^ d4;
        uint8_t p3 = d2 ^ d3 ^ d4;

        // Construct codeword
        return (p1 << 0) | (p2 << 1) | (d1 << 2) | (p3 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6);
    }

    // Decode a 7-bit Hamming codeword back to 4 bits of data
    static std::pair<uint8_t, bool> decode(uint8_t codeword) {
        // Extract received bits
        uint8_t p1 = (codeword >> 0) & 1;
        uint8_t p2 = (codeword >> 1) & 1;
        uint8_t d1 = (codeword >> 2) & 1;
        uint8_t p3 = (codeword >> 3) & 1;
        uint8_t d2 = (codeword >> 4) & 1;
        uint8_t d3 = (codeword >> 5) & 1;
        uint8_t d4 = (codeword >> 6) & 1;

        // Calculate syndrome
        uint8_t s1 = p1 ^ d1 ^ d2 ^ d4;
        uint8_t s2 = p2 ^ d1 ^ d3 ^ d4;
        uint8_t s3 = p3 ^ d2 ^ d3 ^ d4;

        // Error position (0 if no error)
        uint8_t error_pos = (s3 << 2) | (s2 << 1) | s1;

        if (error_pos != 0) {
            // Correct the error
            codeword ^= (1 << (error_pos - 1));
            
            // Re-extract corrected bits
            d1 = (codeword >> 2) & 1;
            d2 = (codeword >> 4) & 1;
            d3 = (codeword >> 5) & 1;
            d4 = (codeword >> 6) & 1;
        }

        // Reconstruct data
        uint8_t data = (d1 << 0) | (d2 << 1) | (d3 << 2) | (d4 << 3);
        
        return {data, error_pos != 0};
    }
};

// Reed-Solomon code implementation for burst error correction
class ReedSolomonCode {
public:
    static constexpr size_t SYMBOL_SIZE = 8;  // 8 bits per symbol
    static constexpr size_t CODE_LENGTH = 255;  // Maximum codeword length
    static constexpr size_t DATA_SYMBOLS = 223;  // Number of data symbols
    static constexpr size_t PARITY_SYMBOLS = CODE_LENGTH - DATA_SYMBOLS;  // Number of parity symbols

    // Galois Field arithmetic
    static uint8_t gf_multiply(uint8_t a, uint8_t b) {
        uint8_t p = 0;
        for (int i = 0; i < 8; i++) {
            if (b & 1) p ^= a;
            bool carry = a & 0x80;
            a <<= 1;
            if (carry) a ^= 0x1D;  // Primitive polynomial x^8 + x^4 + x^3 + x + 1
        }
        return p;
    }

    // Encode data using Reed-Solomon
    static std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> codeword(CODE_LENGTH);
        
        // Copy data
        std::copy(data.begin(), data.end(), codeword.begin());
        
        // Calculate parity symbols
        for (size_t i = DATA_SYMBOLS; i < CODE_LENGTH; i++) {
            uint8_t parity = 0;
            for (size_t j = 0; j < DATA_SYMBOLS; j++) {
                parity ^= gf_multiply(codeword[j], gf_multiply(1, i - j));
            }
            codeword[i] = parity;
        }
        
        return codeword;
    }

    // Decode Reed-Solomon codeword
    static std::pair<std::vector<uint8_t>, bool> decode(const std::vector<uint8_t>& codeword) {
        std::vector<uint8_t> data(DATA_SYMBOLS);
        std::copy(codeword.begin(), codeword.begin() + DATA_SYMBOLS, data.begin());
        
        // Calculate syndromes
        std::array<uint8_t, PARITY_SYMBOLS> syndromes;
        bool has_errors = false;
        
        for (size_t i = 0; i < PARITY_SYMBOLS; i++) {
            uint8_t syndrome = 0;
            for (size_t j = 0; j < CODE_LENGTH; j++) {
                syndrome ^= gf_multiply(codeword[j], gf_multiply(1, i * j));
            }
            syndromes[i] = syndrome;
            if (syndrome != 0) has_errors = true;
        }
        
        if (has_errors) {
            // Berlekamp-Massey algorithm for error location
            std::array<uint8_t, PARITY_SYMBOLS> lambda = {1};
            std::array<uint8_t, PARITY_SYMBOLS> b = {1};
            size_t L = 0;
            size_t m = 1;
            uint8_t delta = 1;
            
            for (size_t n = 0; n < PARITY_SYMBOLS; n++) {
                uint8_t d = syndromes[n];
                for (size_t i = 1; i <= L; i++) {
                    d ^= gf_multiply(lambda[i], syndromes[n - i]);
                }
                
                if (d == 0) {
                    m++;
                } else if (2 * L <= n) {
                    std::array<uint8_t, PARITY_SYMBOLS> temp = lambda;
                    for (size_t i = 0; i < PARITY_SYMBOLS; i++) {
                        lambda[i] ^= gf_multiply(d, b[i]);
                    }
                    L = n + 1 - L;
                    b = temp;
                    m = 1;
                    delta = d;
                } else {
                    for (size_t i = 0; i < PARITY_SYMBOLS; i++) {
                        lambda[i] ^= gf_multiply(d, b[i]);
                    }
                    m++;
                }
            }
            
            // Error correction
            for (size_t i = 0; i < CODE_LENGTH; i++) {
                uint8_t eval = 0;
                for (size_t j = 0; j <= L; j++) {
                    eval ^= gf_multiply(lambda[j], gf_multiply(1, i * j));
                }
                if (eval == 0) {
                    // Error found at position i
                    if (i < DATA_SYMBOLS) {
                        data[i] ^= 1;  // Correct the error
                    }
                }
            }
        }
        
        return {data, has_errors};
    }
};

ProtectionResult applyProtectionTechnique(
    ProtectionTechnique technique,
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events) {
    
    ProtectionResult result;
    result.corrections_successful = 0;
    result.total_errors = 0;
    result.seu_rate = 0.0;
    result.let_threshold = 0.0;
    result.cross_section = 0.0;
    result.mtbf = 0.0;
    result.ber = 0.0;
    
    // Apply protection based on technique
    switch (technique) {
        case ProtectionTechnique::TMR:
            applyTMR(memory, events, result);
            break;
        case ProtectionTechnique::EDAC:
            applyEDAC(memory, events, result);
            break;
        case ProtectionTechnique::SCRUBBING:
            applyScrubbing(memory, events, result);
            break;
        case ProtectionTechnique::NONE:
            // No protection, just count errors
            for (const auto& event : events) {
                if (event.type != RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT) {
                    result.total_errors++;
                }
            }
            break;
    }
    
    // Calculate metrics
    calculateMetrics(result, events);
    
    return result;
}

void applyTMR(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result) {
    
    // Create two copies of memory for TMR
    std::vector<uint8_t> copy1 = memory;
    std::vector<uint8_t> copy2 = memory;
    
    // Random number generator for error injection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> bit_dis(0, 7);
    std::uniform_int_distribution<> loc_dis(0, memory.size() - 1);
    
    // Track original values for verification
    std::vector<uint8_t> original = memory;
    
    // Apply errors to the copies
    for (const auto& event : events) {
        if (event.type != RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT) {
            result.total_errors++;
            
            // Determine error location and pattern
            size_t base_loc = event.memory_offset % memory.size();
            int base_bit = event.bits_affected % 8;
            
            // 70% chance to use the same location (simulating localized radiation effects)
            // 30% chance for completely random location
            size_t loc1 = (dis(gen) < 0.7) ? base_loc : loc_dis(gen);
            size_t loc2 = (dis(gen) < 0.7) ? base_loc : loc_dis(gen);
            size_t loc3 = (dis(gen) < 0.7) ? base_loc : loc_dis(gen);
            
            // Similar for bits - 70% chance to affect same bit
            int bit1 = (dis(gen) < 0.7) ? base_bit : bit_dis(gen);
            int bit2 = (dis(gen) < 0.7) ? base_bit : bit_dis(gen);
            int bit3 = (dis(gen) < 0.7) ? base_bit : bit_dis(gen);
            
            // Apply errors independently with 30% chance each
            if (dis(gen) < 0.3) {
                memory[loc1] ^= (1 << bit1);
            }
            if (dis(gen) < 0.3) {
                copy1[loc2] ^= (1 << bit2);
            }
            if (dis(gen) < 0.3) {
                copy2[loc3] ^= (1 << bit3);
            }
        }
    }
    
    // Perform bit-level majority voting with error tracking
    int corrected_bits = 0;
    int total_errors_found = 0;
    
    for (size_t i = 0; i < memory.size(); i++) {
        uint8_t corrected_byte = 0;
        uint8_t original_byte = original[i];
        
        for (int bit = 0; bit < 8; bit++) {
            // Get each bit from all copies
            int bit0 = (memory[i] >> bit) & 1;
            int bit1 = (copy1[i] >> bit) & 1;
            int bit2 = (copy2[i] >> bit) & 1;
            
            // Original bit value
            int orig_bit = (original_byte >> bit) & 1;
            
            // Count votes
            int votes_for_one = bit0 + bit1 + bit2;
            
            // Determine majority
            int majority_bit = (votes_for_one >= 2) ? 1 : 0;
            
            // Check if any copy is in error
            if (bit0 != orig_bit || bit1 != orig_bit || bit2 != orig_bit) {
                total_errors_found++;
                
                // If majority matches original, we successfully corrected
                if (majority_bit == orig_bit) {
                    corrected_bits++;
                }
            }
            
            // Set the bit according to majority vote
            corrected_byte |= (majority_bit << bit);
        }
        
        // Update memory with corrected value
        memory[i] = corrected_byte;
    }
    
    // Update result statistics
    result.corrections_successful = corrected_bits;
    result.total_errors = total_errors_found;
    
    // Verify correction success
    for (size_t i = 0; i < memory.size(); i++) {
        if (memory[i] != original[i]) {
            std::cout << "WARNING: TMR failed to correct error at position " << i 
                      << ". Original: " << (int)original[i] 
                      << ", Current: " << (int)memory[i] << std::endl;
        }
    }
}

void applyEDAC(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result) {
    
    // Store original data for verification
    std::vector<uint8_t> original_data = memory;
    
    // Use Hamming code for each byte
    HammingCode hamming;
    std::vector<uint8_t> encoded_memory;
    
    // Encode all bytes using Hamming code
    for (size_t i = 0; i < memory.size(); i++) {
        // Split byte into two 4-bit chunks and encode each
        uint8_t lower = memory[i] & 0x0F;
        uint8_t upper = (memory[i] >> 4) & 0x0F;
        
        // Store encoded versions
        encoded_memory.push_back(hamming.encode(lower));
        encoded_memory.push_back(hamming.encode(upper));
    }
    
    // Random number generator for error injection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> bit_dis(0, 6);  // Hamming(7,4) code
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // Apply radiation effects to encoded data
    for (const auto& event : events) {
        if (event.type != RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT) {
            // Calculate position in encoded memory (each byte becomes two encoded bytes)
            size_t encoded_loc = (event.memory_offset % memory.size()) * 2;
            
            // For each event, we might affect multiple bits
            int num_bits_affected = 1;
            if (event.type == RadiationSimulator::RadiationEffectType::MULTI_BIT_UPSET) {
                num_bits_affected = 2;
            } else if (event.type == RadiationSimulator::RadiationEffectType::SINGLE_EVENT_LATCHUP) {
                num_bits_affected = 3;
            }
            
            // Apply the errors
            for (int i = 0; i < num_bits_affected; i++) {
                // 70% chance to affect first half, 30% chance for second half
                size_t target_loc = encoded_loc + (dis(gen) < 0.7 ? 0 : 1);
                int bit_pos = bit_dis(gen);
                
                encoded_memory[target_loc] ^= (1 << bit_pos);
                result.total_errors++;
            }
        }
    }
    
    // Decode and correct errors
    for (size_t i = 0; i < memory.size(); i++) {
        // Decode both halves of the byte
        auto [lower, lower_corrected] = hamming.decode(encoded_memory[i*2]);
        auto [upper, upper_corrected] = hamming.decode(encoded_memory[i*2 + 1]);
        
        // Reconstruct byte
        uint8_t corrected = (upper << 4) | lower;
        
        // Verify correction was accurate
        if (corrected == original_data[i]) {
            if (lower_corrected || upper_corrected) {
                result.corrections_successful++;
            }
        } else {
            std::cout << "WARNING: EDAC failed to correct error at position " << i 
                      << ". Original: " << (int)original_data[i] 
                      << ", Current: " << (int)corrected << std::endl;
        }
        
        memory[i] = corrected;
    }
}

void applyScrubbing(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result) {
    
    // Create a backup of memory
    std::vector<uint8_t> backup = memory;
    
    // Apply radiation effects
    for (const auto& event : events) {
        if (event.type != RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT) {
            result.total_errors++;
            
            size_t loc = event.memory_offset % memory.size();
            memory[loc] ^= (1 << (event.bits_affected % 8));
        }
    }
    
    // Perform scrubbing by comparing with backup
    for (size_t i = 0; i < memory.size(); i++) {
        if (memory[i] != backup[i]) {
            memory[i] = backup[i];
            result.corrections_successful++;
        }
    }
}

void calculateMetrics(
    ProtectionResult& result,
    const std::vector<RadiationSimulator::RadiationEvent>& events) {
    
    // Calculate SEU rate (errors per bit per second)
    double total_bits = events.size() * 8.0; // Assuming 8 bits per byte
    double total_time = 1.0; // Assuming 1 second simulation
    result.seu_rate = result.total_errors / (total_bits * total_time);
    
    // Calculate LET threshold (MeV-cm²/mg)
    result.let_threshold = 0.0; // Not applicable for this simulation
    
    // Calculate cross section (cm²)
    result.cross_section = result.total_errors / (total_bits * events.size());
    
    // Calculate MTBF (hours)
    result.mtbf = 1.0 / (result.seu_rate * 3600.0);
    
    // Calculate BER
    result.ber = static_cast<double>(result.total_errors - result.corrections_successful) / total_bits;
}

// Stub implementation for protection techniques
void applyProtection() {
    // Placeholder implementation
}

} // namespace testing
} // namespace rad_ml 