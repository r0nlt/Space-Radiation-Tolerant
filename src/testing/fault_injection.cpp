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
 * @file fault_injection.cpp
 * @brief Implementation of systematic fault injection functionality
 */

#include "../../include/rad_ml/testing/fault_injection.hpp"
#include <iostream>

namespace rad_ml {
namespace testing {

SystematicFaultInjector::SystematicFaultInjector() {
    // Seed with random device for non-deterministic sequence
    std::random_device rd;
    gen = std::mt19937(rd());
}

void SystematicFaultInjector::setSeed(unsigned int seed) {
    gen = std::mt19937(seed);
}

std::string SystematicFaultInjector::patternToString(FaultPattern pattern) {
    switch (pattern) {
        case SINGLE_BIT:     return "SINGLE_BIT";
        case ADJACENT_BITS:  return "ADJACENT_BITS";
        case BYTE_ERROR:     return "BYTE_ERROR";
        case WORD_ERROR:     return "WORD_ERROR";
        case STUCK_AT_ZERO:  return "STUCK_AT_ZERO";
        case STUCK_AT_ONE:   return "STUCK_AT_ONE";
        case ROW_COLUMN:     return "ROW_COLUMN";
        case BURST_ERROR:    return "BURST_ERROR";
        default:             return "UNKNOWN_PATTERN";
    }
}

std::vector<int> SystematicFaultInjector::getBitsToFlip(FaultPattern pattern, 
                                                       int total_bits, 
                                                       int starting_bit) {
    std::vector<int> bits_to_flip;
    
    // If starting_bit is negative, pick a random bit
    if (starting_bit < 0) {
        std::uniform_int_distribution<> dis(0, total_bits - 1);
        starting_bit = dis(gen);
    }
    
    // Ensure starting_bit is within valid range
    starting_bit = std::min(starting_bit, total_bits - 1);
    starting_bit = std::max(starting_bit, 0);
    
    // Random values generator for number of bits, etc.
    std::uniform_int_distribution<> num_bits_dis(2, 3);
    std::uniform_int_distribution<> bit_dis(0, total_bits - 1);
    
    switch (pattern) {
        case SINGLE_BIT:
            // Single bit flip - just one bit at the starting position
            bits_to_flip.push_back(starting_bit);
            break;
            
        case ADJACENT_BITS:
            // 2-3 adjacent bits starting from the starting position
            {
                int num_adjacent = num_bits_dis(gen);
                for (int i = 0; i < num_adjacent; i++) {
                    int bit = (starting_bit + i) % total_bits;
                    bits_to_flip.push_back(bit);
                }
            }
            break;
            
        case BYTE_ERROR:
            // Full byte (8 bits) centered around starting position
            {
                int byte_start = (starting_bit / 8) * 8; // Round to byte boundary
                for (int i = 0; i < 8; i++) {
                    if (byte_start + i < total_bits) {
                        bits_to_flip.push_back(byte_start + i);
                    }
                }
            }
            break;
            
        case WORD_ERROR:
            // Full 32-bit word centered around starting position
            {
                int word_start = (starting_bit / 32) * 32; // Round to word boundary
                for (int i = 0; i < 32; i++) {
                    if (word_start + i < total_bits) {
                        bits_to_flip.push_back(word_start + i);
                    }
                }
            }
            break;
            
        case STUCK_AT_ZERO:
        case STUCK_AT_ONE:
            // Just affect the starting bit
            bits_to_flip.push_back(starting_bit);
            break;
            
        case ROW_COLUMN:
            // Row/column pattern (common in memory)
            // For simplicity, simulate as every 8th bit in both directions
            {
                // Row - bits in the same byte
                int byte_start = (starting_bit / 8) * 8;
                for (int i = 0; i < 8; i++) {
                    if (byte_start + i < total_bits) {
                        bits_to_flip.push_back(byte_start + i);
                    }
                }
                
                // Column - every 8th bit
                for (int i = 0; i < total_bits; i += 8) {
                    if (i != byte_start && i + (starting_bit % 8) < total_bits) {
                        bits_to_flip.push_back(i + (starting_bit % 8));
                    }
                }
            }
            break;
            
        case BURST_ERROR:
            // Burst of errors - randomly distributed but clustered
            {
                std::normal_distribution<double> burst_dis(starting_bit, total_bits / 10.0);
                int burst_size = 5 + bit_dis(gen) % 10; // 5-14 bits
                
                for (int i = 0; i < burst_size; i++) {
                    int bit = static_cast<int>(burst_dis(gen)) % total_bits;
                    if (bit < 0) bit += total_bits;
                    bits_to_flip.push_back(bit);
                }
            }
            break;
            
        default:
            // Unknown pattern - just flip the starting bit
            bits_to_flip.push_back(starting_bit);
            break;
    }
    
    return bits_to_flip;
}

// Stub implementation
void injectFault() {
    // Placeholder implementation
}

} // namespace testing
} // namespace rad_ml 
