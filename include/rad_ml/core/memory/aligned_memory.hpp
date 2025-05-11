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
 * @file aligned_memory.hpp
 * @brief Alignment-based memory protection strategies
 * 
 * This file defines memory structures that use alignment to
 * reduce the chance of multi-value corruption from single radiation events.
 */

#ifndef RAD_ML_ALIGNED_MEMORY_HPP
#define RAD_ML_ALIGNED_MEMORY_HPP

#include <array>
#include <memory>
#include <type_traits>
#include <cstdint>
#include "../redundancy/enhanced_voting.hpp"

namespace rad_ml {
namespace core {
namespace memory {

/**
 * Memory protection using physical alignment to disperse copies
 * 
 * This class stores redundant copies with specific memory alignment and
 * padding to reduce the chance of a single radiation event affecting
 * multiple copies due to spatial locality.
 */
#if __cplusplus >= 202002L
template<redundancy::VotableType T, size_t Alignment = 64>
requires (Alignment >= alignof(T) && (Alignment & (Alignment - 1)) == 0)
#else
template<typename T, size_t Alignment = 64>
#endif
class AlignedProtectedMemory {
    // Static assertions for C++17 since we can't use requires
#if __cplusplus < 202002L
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    static_assert(Alignment >= alignof(T), "Alignment must be at least the natural alignment of T");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
#endif

public:
    /**
     * Constructor initializes with triplication
     * 
     * @param value Initial value to store
     */
    AlignedProtectedMemory(const T& value)
        : scrubbing_enabled(true) {
        // Initialize copies with the same value
        for (size_t i = 0; i < 3; ++i) {
            copies[i] = value;
        }
    }
    
    /**
     * Get value with automatic voting
     * 
     * @return Corrected value from voting
     */
    T get() const {
        using namespace rad_ml::core::redundancy;
        
        // Quick return for matching values
        if (copies[0].value == copies[1].value && copies[1].value == copies[2].value) {
            return copies[0].value;
        }
        
        // Extract the actual values from the AlignedValue structures
        T v1 = copies[0].value;
        T v2 = copies[1].value;
        T v3 = copies[2].value;
        
        auto [pattern, confidence] = EnhancedVoting::detectFaultPatternWithConfidence(
            v1, v2, v3);
            
        T result = EnhancedVoting::adaptiveVote(
            v1, v2, v3, pattern);
            
        // Optionally auto-scrub when reading
        if (scrubbing_enabled && confidence < 1.0f) {
            const_cast<AlignedProtectedMemory*>(this)->scrub(result);
        }
            
        return result;
    }
    
    /**
     * Set value with replication
     * 
     * @param value New value to store
     */
    void set(const T& value) {
        for (size_t i = 0; i < 3; ++i) {
            copies[i] = value;
        }
    }
    
    /**
     * Enable or disable automatic scrubbing
     * 
     * @param enable Whether to enable scrubbing
     */
    void enableScrubbing(bool enable) {
        scrubbing_enabled = enable;
    }
    
    /**
     * Perform explicit memory scrubbing
     * 
     * @param correct_value Optional known-correct value (if not provided, will use voting)
     * @return True if any copies were corrected
     */
    bool scrub(const T& correct_value) {
        bool any_corrected = false;
        
        for (size_t i = 0; i < 3; ++i) {
            if (copies[i] != correct_value) {
                copies[i] = correct_value;
                any_corrected = true;
            }
        }
        
        return any_corrected;
    }
    
    /**
     * Perform memory scrubbing using voting
     * 
     * @return True if any copies were corrected
     */
    bool scrub() {
        using namespace rad_ml::core::redundancy;
        
        // No need to scrub if all copies match
        if (copies[0] == copies[1] && copies[1] == copies[2]) {
            return false;
        }
        
        // Determine correct value through voting
        T corrected = EnhancedVoting::adaptiveVote(
            copies[0], copies[1], copies[2],
            EnhancedVoting::detectFaultPattern(copies[0], copies[1], copies[2]));
        
        // Apply correction
        return scrub(corrected);
    }
    
    /**
     * Direct access to a specific copy (for testing/debugging)
     * 
     * @param index Copy index (0-2)
     * @return Reference to the specified copy
     */
    T& getRawCopy(size_t index) {
        return copies[index % 3];
    }
    
    /**
     * Explicitly corrupt a specific copy (for testing)
     * 
     * @param index Copy to corrupt (0-2)
     * @param value Corrupted value to inject
     */
    void corruptCopy(size_t index, const T& value) {
        copies[index % 3] = value;
    }

private:
    // Structure to enforce alignment and padding between copies
    struct alignas(Alignment) AlignedValue {
        T value;
        // Padding to fill the aligned block and increase physical separation
        std::byte padding[Alignment - (sizeof(T) % Alignment)];
        
        // Conversion operators for ease of use
        operator T&() { return value; }
        operator const T&() const { return value; }
        
        // Assignment operator
        AlignedValue& operator=(const T& other) {
            value = other;
            return *this;
        }
        
        // Comparison operators
        bool operator==(const T& other) const { return value == other; }
        bool operator!=(const T& other) const { return value != other; }
    };
    
    // Three copies, each aligned to reduce chance of multi-copy corruption
    AlignedValue copies[3];
    bool scrubbing_enabled;
};

/**
 * Memory protection with interleaved bits to spread errors
 * 
 * This class stores redundant copies with bit-level interleaving
 * to ensure that adjacent bit errors don't corrupt the same logical bit
 * across multiple copies.
 */
#if __cplusplus >= 202002L
template<redundancy::VotableType T>
requires std::is_integral_v<T>
#else
template<typename T>
#endif
class InterleavedBitMemory {
    // Static assertion for C++17
#if __cplusplus < 202002L
    static_assert(std::is_integral<T>::value, "Only integral types are supported");
#endif

public:
    /**
     * Constructor initializes with triplication and interleaving
     * 
     * @param value Initial value to store
     */
    InterleavedBitMemory(const T& value) {
        set(value);
    }
    
    /**
     * Get value with automatic error correction
     * 
     * @return Corrected value
     */
    T get() const {
        // De-interleave bits
        T copy1 = deinterleave(interleaved_data, 0);
        T copy2 = deinterleave(interleaved_data, 1);
        T copy3 = deinterleave(interleaved_data, 2);
        
        // Use voting to correct errors
        using namespace rad_ml::core::redundancy;
        return EnhancedVoting::adaptiveVote(
            copy1, copy2, copy3, 
            EnhancedVoting::detectFaultPattern(copy1, copy2, copy3));
    }
    
    /**
     * Set value with interleaved bit storage
     * 
     * @param value New value to store
     */
    void set(const T& value) {
        // Store three copies with interleaved bits
        interleaved_data = 0;
        interleave(value, 0);
        interleave(value, 1);
        interleave(value, 2);
    }
    
    /**
     * Perform memory scrubbing
     * 
     * @return True if any bits were corrected
     */
    bool scrub() {
        T original = get();
        set(original);
        return true; // Always return true as we rewrite the interleaved data
    }

private:
    // For a 32-bit integer, we need 96 bits of storage
    using StorageType = typename std::conditional<
        sizeof(T) <= 2,
        uint64_t,
        typename std::conditional<
            sizeof(T) <= 4,
            unsigned __int128,  // 128-bit integer
            unsigned __int128[2] // 256-bit for 64-bit integers
        >::type
    >::type;
    
    StorageType interleaved_data;
    
    /**
     * Interleave bits from a value into the storage
     * 
     * @param value Value to interleave
     * @param copy_idx Which copy (0, 1, or 2) for bit position
     */
    void interleave(T value, int copy_idx) {
        for (size_t i = 0; i < sizeof(T) * 8; ++i) {
            // Extract bit
            StorageType bit = ((static_cast<StorageType>(value) >> i) & 1);
            
            // Calculate position in interleaved data
            size_t pos = i * 3 + copy_idx;
            
            // Set bit
            interleaved_data |= (bit << pos);
        }
    }
    
    /**
     * Extract a value from interleaved storage
     * 
     * @param data Interleaved data
     * @param copy_idx Which copy to extract (0, 1, or 2)
     * @return The extracted value
     */
    T deinterleave(StorageType data, int copy_idx) const {
        T result = 0;
        
        for (size_t i = 0; i < sizeof(T) * 8; ++i) {
            // Calculate position in interleaved data
            size_t pos = i * 3 + copy_idx;
            
            // Extract bit
            T bit = ((data >> pos) & 1);
            
            // Set bit in result
            result |= (bit << i);
        }
        
        return result;
    }
};

} // namespace memory
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_ALIGNED_MEMORY_HPP 
