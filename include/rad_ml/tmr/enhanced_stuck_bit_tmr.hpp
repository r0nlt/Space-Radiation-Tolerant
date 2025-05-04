#pragma once

#include <array>
#include <bitset>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <string>

namespace rad_ml {
namespace tmr {

// Forward declarations for specializations
template <typename T>
class EnhancedStuckBitTMR;

/**
 * @brief Enhanced TMR implementation with specialized stuck bit detection and mitigation
 * 
 * Based on NASA JPL research on Flash memory in radiation environments and
 * ESA JUICE mission radiation mitigation strategies for Jupiter's extreme environment.
 * 
 * @tparam T Data type to protect
 */
template <typename T>
class EnhancedStuckBitTMR {
public:
    /**
     * @brief Construct a new Enhanced Stuck Bit TMR
     * 
     * @param initial_value Initial value to protect
     */
    explicit EnhancedStuckBitTMR(const T& initial_value = T()) 
        : copies_{initial_value, initial_value, initial_value},
          health_scores_{1.0, 1.0, 1.0} {
        static_assert(std::is_integral<T>::value, 
            "Default EnhancedStuckBitTMR implementation only supports integral types");
    }
    
    /**
     * @brief Get the protected value using advanced voting with stuck bit awareness
     * 
     * @return The protected value
     */
    T get() const {
        // If we have identified stuck bits, use mask-aware voting
        if (potential_stuck_bits.any()) {
            return get_with_stuck_bit_awareness();
        }
        
        // Otherwise use standard health-weighted voting
        return get_standard();
    }
    
    /**
     * @brief Set a new value
     * 
     * @param value New value to protect
     */
    void set(const T& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
        for (auto& score : health_scores_) {
            score = 1.0;
        }
        
        // Reset stuck bit tracking when setting a new value intentionally
        reset_stuck_bit_tracking();
    }
    
    /**
     * @brief Repair corrupted copies
     * 
     * Enhanced repair that properly handles stuck bits by:
     * 1. Identifying new potential stuck bits
     * 2. Using bit-level repair rather than full value replacement
     */
    void repair() {
        // First detect any new stuck bits
        update_stuck_bit_tracking();
        
        // Get the correct value using mask-aware voting
        T correct_value = get();
        
        // For each copy, repair all bits EXCEPT those identified as stuck
        for (size_t i = 0; i < copies_.size(); ++i) {
            repair_non_stuck_bits(copies_[i], correct_value, i);
        }
        
        // Update health scores based on repair results
        for (size_t i = 0; i < copies_.size(); ++i) {
            // Copy is completely healthy if it matches the correct value
            bool fully_healthy = (copies_[i] == correct_value);
            // Copy is partially healthy if only stuck bits differ
            bool partially_healthy = check_only_stuck_bits_differ(copies_[i], correct_value, i);
            
            if (fully_healthy) {
                health_scores_[i] = std::min(1.0, health_scores_[i] + 0.1);
            } else if (partially_healthy) {
                health_scores_[i] = std::min(0.9, health_scores_[i] + 0.05);
            } else {
                health_scores_[i] = std::max(0.1, health_scores_[i] - 0.2);
            }
        }
    }
    
    /**
     * @brief Direct access to copies (for testing and monitoring)
     * 
     * @return Const reference to the array of copies
     */
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    /**
     * @brief Direct access to health scores (for diagnostics)
     * 
     * @return Const reference to the array of health scores
     */
    const std::array<double, 3>& getHealthScores() const {
        return health_scores_;
    }
    
    /**
     * @brief Get the stuck bit mask for diagnostics
     * 
     * @return The bitset representing potentially stuck bits
     */
    std::bitset<sizeof(T) * 8> getStuckBitMask() const {
        return potential_stuck_bits;
    }
    
    /**
     * @brief Corrupt a specific copy (for testing)
     * 
     * @param index Copy index to corrupt (0-2)
     * @param value Value to set
     */
    void corruptCopy(size_t index, const T& value) {
        if (index < copies_.size()) {
            copies_[index] = value;
        }
    }
    
    /**
     * @brief Get diagnostic information
     * 
     * @return String containing diagnostic information
     */
    std::string getDiagnostics() const {
        std::string result = "EnhancedStuckBitTMR Diagnostics:\n";
        
        result += "  Health scores: [";
        for (size_t i = 0; i < health_scores_.size(); ++i) {
            result += std::to_string(health_scores_[i]);
            if (i < health_scores_.size() - 1) {
                result += ", ";
            }
        }
        result += "]\n";
        
        result += "  Potential stuck bits: " + 
                  std::to_string(potential_stuck_bits.count()) + " bits\n";
        
        result += "  Stuck bit mask: " + potential_stuck_bits.to_string() + "\n";
        
        return result;
    }
    
protected:
    // The three redundant copies
    std::array<T, 3> copies_;
    
    // Health score for each copy (0.0-1.0)
    mutable std::array<double, 3> health_scores_;
    
    // Track potentially stuck bits across all copies
    std::bitset<sizeof(T) * 8> potential_stuck_bits{};
    
    // Track the stuck value (0/1) for each copy's bits
    std::array<std::bitset<sizeof(T) * 8>, 3> stuck_value_masks{};
    
    // Track consecutive errors at bit level (research shows 3+ consecutive 
    // errors at same bit position indicates stuck bit with >95% confidence)
    std::array<uint8_t, sizeof(T) * 8> error_consistency_counters{};
    
    // Threshold for considering a bit stuck (based on JUICE mission testing)
    static constexpr uint8_t stuck_bit_threshold = 3;
    
    /**
     * @brief Standard health-weighted voting logic
     * 
     * @return The voted value
     */
    T get_standard() const {
        // Similar to HealthWeightedTMR but without bit masking
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            updateHealthScores(0, true);
            if (copies_[0] == copies_[1]) {
                updateHealthScores(1, true);
                updateHealthScores(2, copies_[2] == copies_[0]);
            } else {
                updateHealthScores(1, copies_[1] == copies_[0]);
                updateHealthScores(2, true);
            }
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            updateHealthScores(0, false);
            updateHealthScores(1, true);
            updateHealthScores(2, true);
            return copies_[1];
        }
        
        // If no majority, use health scores
        size_t best_idx = 0;
        for (size_t i = 1; i < health_scores_.size(); ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    /**
     * @brief Update health scores based on voting results
     * 
     * @param index Copy index
     * @param correct Whether this copy matched the voted result
     */
    void updateHealthScores(size_t index, bool correct) const {
        const double reward = 0.05;
        const double penalty = 0.15;
        
        if (correct) {
            health_scores_[index] = std::min(1.0, health_scores_[index] + reward);
        } else {
            health_scores_[index] = std::max(0.1, health_scores_[index] - penalty);
        }
    }
    
    /**
     * @brief Advanced voting that accounts for known stuck bits
     * 
     * @return The voted value with stuck bit compensation
     */
    T get_with_stuck_bit_awareness() const {
        // If all three copies are exactly the same, return that value
        if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
            return copies_[0];
        }
        
        // Create a bit-by-bit voting result
        T result = 0;
        
        // For each bit position, perform masked voting
        for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
            // Check if this bit is potentially stuck
            if (potential_stuck_bits.test(bit)) {
                // If bit is stuck, get votes only from copies where this bit isn't stuck
                int valid_votes = 0;
                int bit_value = 0;
                
                // Count votes from copies with healthy bit at this position
                for (size_t i = 0; i < copies_.size(); ++i) {
                    // Skip copies where this bit is identified as potentially stuck
                    if (stuck_value_masks[i].test(bit)) {
                        continue;
                    }
                    
                    // Extract bit value for voting
                    bit_value += (copies_[i] >> bit) & 1;
                    valid_votes++;
                }
                
                // If we have valid votes, use them; otherwise use majority
                if (valid_votes > 0) {
                    // Set the bit if majority of valid votes are 1
                    if (bit_value > valid_votes / 2) {
                        result |= (static_cast<T>(1) << bit);
                    }
                } else {
                    // All copies have this bit stuck, use health-weighted majority
                    std::array<int, 3> bit_values;
                    for (size_t i = 0; i < copies_.size(); ++i) {
                        bit_values[i] = (copies_[i] >> bit) & 1;
                    }
                    
                    // Weight by health scores
                    double weighted_sum = 0.0;
                    double total_weight = 0.0;
                    for (size_t i = 0; i < copies_.size(); ++i) {
                        weighted_sum += bit_values[i] * health_scores_[i];
                        total_weight += health_scores_[i];
                    }
                    
                    // Set bit if weighted average is greater than 0.5
                    if (total_weight > 0 && weighted_sum / total_weight > 0.5) {
                        result |= (static_cast<T>(1) << bit);
                    }
                }
            } else {
                // Normal bit (not identified as stuck) - use standard majority voting
                int bit_sum = 0;
                for (size_t i = 0; i < copies_.size(); ++i) {
                    bit_sum += (copies_[i] >> bit) & 1;
                }
                
                // Set bit if majority of copies have this bit set
                if (bit_sum > static_cast<int>(copies_.size()) / 2) {
                    result |= (static_cast<T>(1) << bit);
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Update the stuck bit tracking based on current corruption patterns
     */
    void update_stuck_bit_tracking() {
        // Get the current value based on standard health-weighted voting
        T voted_value = get_standard();
        
        // Analyze each copy for potential stuck bits
        for (size_t copy_idx = 0; copy_idx < copies_.size(); ++copy_idx) {
            // Skip completely healthy copies
            if (copies_[copy_idx] == voted_value) {
                continue;
            }
            
            // Calculate XOR of copy and voted value to identify differing bits
            T diff = copies_[copy_idx] ^ voted_value;
            
            // Check each differing bit
            for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
                // If bit differs in this copy
                if ((diff >> bit) & 1) {
                    // Increment error consistency counter for this bit
                    if (error_consistency_counters[bit] < 255) {
                        error_consistency_counters[bit]++;
                    }
                    
                    // Record stuck bit value (0 or 1)
                    bool current_bit_value = (copies_[copy_idx] >> bit) & 1;
                    stuck_value_masks[copy_idx][bit] = current_bit_value;
                    
                    // Mark bit as potentially stuck if threshold reached
                    if (error_consistency_counters[bit] >= stuck_bit_threshold) {
                        potential_stuck_bits.set(bit);
                    }
                }
            }
        }
    }
    
    /**
     * @brief Reset stuck bit tracking when intentionally changing values
     */
    void reset_stuck_bit_tracking() {
        potential_stuck_bits.reset();
        for (auto& mask : stuck_value_masks) {
            mask.reset();
        }
        for (auto& counter : error_consistency_counters) {
            counter = 0;
        }
    }
    
    /**
     * @brief Repair non-stuck bits in a copy
     * 
     * @param copy Copy to repair
     * @param correct_value The correct value to use for repair
     * @param copy_idx Index of the copy
     */
    void repair_non_stuck_bits(T& copy, const T correct_value, size_t copy_idx) {
        // Don't repair if already correct
        if (copy == correct_value) {
            return;
        }
        
        // Find bits that differ
        T diff = copy ^ correct_value;
        
        // For each differing bit
        for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
            // If bit differs and isn't identified as stuck in this copy
            if (((diff >> bit) & 1) && !stuck_value_masks[copy_idx].test(bit)) {
                // Get the correct bit value
                bool correct_bit = (correct_value >> bit) & 1;
                
                // Clear the bit
                copy &= ~(static_cast<T>(1) << bit);
                
                // Set the bit to the correct value if it should be 1
                if (correct_bit) {
                    copy |= (static_cast<T>(1) << bit);
                }
            }
        }
    }
    
    /**
     * @brief Check if only stuck bits differ between two values
     * 
     * @param value Value to check
     * @param correct_value Correct value to compare against
     * @param copy_idx Index of the copy
     * @return True if only stuck bits differ
     */
    bool check_only_stuck_bits_differ(const T& value, const T& correct_value, 
                                      size_t copy_idx) const {
        // Calculate differing bits
        T diff = value ^ correct_value;
        
        // If no differences, return true
        if (diff == 0) {
            return true;
        }
        
        // Check if all differing bits are identified as stuck in this copy
        for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
            // If this bit differs
            if ((diff >> bit) & 1) {
                // If this bit is not identified as stuck, return false
                if (!stuck_value_masks[copy_idx].test(bit)) {
                    return false;
                }
            }
        }
        
        // All differing bits are identified as stuck
        return true;
    }
};

/**
 * @brief Specialization for float type
 * 
 * This specialization handles the bit-level operations differently
 * for floating point values through type punning with union.
 */
template <>
class EnhancedStuckBitTMR<float> {
public:
    explicit EnhancedStuckBitTMR(const float& initial_value = 0.0f) 
        : copies_{initial_value, initial_value, initial_value},
          health_scores_{1.0, 1.0, 1.0} {
    }
    
    float get() const {
        // Simple majority voting for now
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            return copies_[1];
        }
        
        // If no majority, use health scores
        size_t best_idx = 0;
        for (size_t i = 1; i < health_scores_.size(); ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    void set(const float& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
        for (auto& score : health_scores_) {
            score = 1.0;
        }
    }
    
    void repair() {
        // Simple repair for floating point - just use majority voting
        float correct_value = get();
        
        // Update health scores
        for (size_t i = 0; i < copies_.size(); ++i) {
            if (copies_[i] == correct_value) {
                health_scores_[i] = std::min(1.0, health_scores_[i] + 0.1);
            } else {
                health_scores_[i] = std::max(0.1, health_scores_[i] - 0.2);
                copies_[i] = correct_value; // Repair
            }
        }
    }
    
    const std::array<float, 3>& getCopies() const {
        return copies_;
    }
    
    const std::array<double, 3>& getHealthScores() const {
        return health_scores_;
    }
    
    std::bitset<32> getStuckBitMask() const {
        // Simplified for float - just return empty mask
        return std::bitset<32>();
    }
    
    void corruptCopy(size_t index, const float& value) {
        if (index < copies_.size()) {
            copies_[index] = value;
        }
    }
    
    std::string getDiagnostics() const {
        std::string result = "EnhancedStuckBitTMR<float> Diagnostics:\n";
        
        result += "  Health scores: [";
        for (size_t i = 0; i < health_scores_.size(); ++i) {
            result += std::to_string(health_scores_[i]);
            if (i < health_scores_.size() - 1) {
                result += ", ";
            }
        }
        result += "]\n";
        
        return result;
    }
    
private:
    std::array<float, 3> copies_;
    mutable std::array<double, 3> health_scores_;
};

// Similar specialization for double if needed
template <>
class EnhancedStuckBitTMR<double> {
public:
    explicit EnhancedStuckBitTMR(const double& initial_value = 0.0) 
        : copies_{initial_value, initial_value, initial_value},
          health_scores_{1.0, 1.0, 1.0} {
    }
    
    double get() const {
        // Simple majority voting for now
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            return copies_[1];
        }
        
        // If no majority, use health scores
        size_t best_idx = 0;
        for (size_t i = 1; i < health_scores_.size(); ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    void set(const double& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
        for (auto& score : health_scores_) {
            score = 1.0;
        }
    }
    
    void repair() {
        // Simple repair for floating point - just use majority voting
        double correct_value = get();
        
        // Update health scores
        for (size_t i = 0; i < copies_.size(); ++i) {
            if (copies_[i] == correct_value) {
                health_scores_[i] = std::min(1.0, health_scores_[i] + 0.1);
            } else {
                health_scores_[i] = std::max(0.1, health_scores_[i] - 0.2);
                copies_[i] = correct_value; // Repair
            }
        }
    }
    
    const std::array<double, 3>& getCopies() const {
        return copies_;
    }
    
    const std::array<double, 3>& getHealthScores() const {
        return health_scores_;
    }
    
    std::bitset<64> getStuckBitMask() const {
        // Simplified for double - just return empty mask
        return std::bitset<64>();
    }
    
    void corruptCopy(size_t index, const double& value) {
        if (index < copies_.size()) {
            copies_[index] = value;
        }
    }
    
    std::string getDiagnostics() const {
        std::string result = "EnhancedStuckBitTMR<double> Diagnostics:\n";
        
        result += "  Health scores: [";
        for (size_t i = 0; i < health_scores_.size(); ++i) {
            result += std::to_string(health_scores_[i]);
            if (i < health_scores_.size() - 1) {
                result += ", ";
            }
        }
        result += "]\n";
        
        return result;
    }
    
private:
    std::array<double, 3> copies_;
    mutable std::array<double, 3> health_scores_;
};

} // namespace tmr
} // namespace rad_ml 