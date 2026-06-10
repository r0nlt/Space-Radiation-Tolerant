#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "rad_ml/neural/advanced_reed_solomon.hpp"
#include "rad_ml/neural/radiation_environment.hpp"
#include "rad_ml/radiation/space_mission.hpp"

namespace rad_ml {
namespace neural {

// Use existing definitions from multi_bit_protection.hpp to avoid redefinition
// Note: MultibitUpsetType, ECCCodingScheme are defined in multi_bit_protection.hpp
// Note: AdaptiveProtectionLevel is defined in protected_neural_network.hpp

// Local enum for adaptive protection (avoids conflicts with multi_bit_protection.hpp)
enum class AdaptiveMultibitUpsetType { SINGLE_BIT, ADJACENT_BITS, RANDOM_MULTI };

// Local AdaptiveProtectionLevel for adaptive protection (independent of
// protected_neural_network.hpp)
enum class AdaptiveProtectionLevel {
    NONE,       // No protection
    MINIMAL,    // Basic parity-based protection
    MODERATE,   // TMR or Hamming code protection
    HIGH,       // Reed-Solomon with moderate parameter settings
    VERY_HIGH,  // Reed-Solomon with strong parameter settings
    ADAPTIVE    // Dynamically adjusted based on radiation conditions
};

/** Result of extended Hamming(8,4) SECDED decode for one 4-bit nibble */
struct HammingDecodeResult {
    uint8_t value = 0;
    bool corrected = false;      ///< A single-bit error (including overall parity) was fixed
    bool uncorrectable = false;  ///< Double-bit error detected; value is not trusted
};

// Note: Do NOT use 'using AdaptiveProtectionLevel = ...' here as it conflicts with
// protected_neural_network.hpp

// Type trait for protectable types
template <typename T>
struct is_protectable {
    static constexpr bool value =
        std::is_trivially_copyable_v<T> &&
        (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>);
};

// Weight criticality structure
template <typename T>
struct AdaptiveWeightCriticality {
    T weight;                       // The weight value
    float sensitivity;              // Sensitivity score (higher = more critical)
    AdaptiveProtectionLevel level;  // Selected protection level

    // Allow comparison for sorting
    bool operator<(const AdaptiveWeightCriticality& other) const
    {
        return sensitivity < other.sensitivity;
    }
};

// Note: Using AdvancedReedSolomon from advanced_reed_solomon.hpp

// Multi-bit upset handler for adaptive protection (local implementation)
template <typename T>
class AdaptiveMultibitHandler {
   public:
    std::vector<uint8_t> apply_multi_bit_upset(std::vector<uint8_t> data,
                                               AdaptiveMultibitUpsetType type,
                                               double seu_probability, std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        switch (type) {
            case AdaptiveMultibitUpsetType::SINGLE_BIT:
                for (auto& byte : data) {
                    for (int bit = 0; bit < 8; ++bit) {
                        if (dist(rng) < seu_probability) {
                            byte ^= (1 << bit);
                        }
                    }
                }
                break;

            case AdaptiveMultibitUpsetType::ADJACENT_BITS:
                for (auto& byte : data) {
                    if (dist(rng) < seu_probability) {
                        // Flip 2-3 adjacent bits
                        int start_bit = std::uniform_int_distribution<int>(0, 5)(rng);
                        int num_bits = std::uniform_int_distribution<int>(2, 3)(rng);
                        for (int i = 0; i < num_bits && start_bit + i < 8; ++i) {
                            byte ^= (1 << (start_bit + i));
                        }
                    }
                }
                break;

            case AdaptiveMultibitUpsetType::RANDOM_MULTI:
                for (auto& byte : data) {
                    if (dist(rng) < seu_probability) {
                        // Flip random number of bits
                        int num_bits = std::uniform_int_distribution<int>(1, 4)(rng);
                        for (int i = 0; i < num_bits; ++i) {
                            int bit_pos = std::uniform_int_distribution<int>(0, 7)(rng);
                            byte ^= (1 << bit_pos);
                        }
                    }
                }
                break;
        }

        return data;
    }
};

// Basic Adaptive Protected Neural Network Interface (local to avoid conflict with
// protected_neural_network.hpp)
template <typename T>
class AdaptiveProtectedNetwork {
   public:
    virtual ~AdaptiveProtectedNetwork() = default;

    virtual std::vector<T> forward(const std::vector<T>& input) = 0;
    virtual std::vector<T> get_all_weights() const = 0;
    virtual void replace_weight(const T& old_weight, const T& new_weight) = 0;
    virtual void set_weight_protection(const T& weight, AdaptiveProtectionLevel level) = 0;
};

// Example implementation for testing
template <typename T>
class SimpleAdaptiveNetwork : public AdaptiveProtectedNetwork<T> {
   private:
    std::vector<T> weights_;

   public:
    SimpleAdaptiveNetwork(const std::vector<T>& weights) : weights_(weights) {}

    std::vector<T> forward(const std::vector<T>& input) override
    {
        // Simplified forward pass for testing
        std::vector<T> output;
        for (size_t i = 0; i < input.size() && i < weights_.size(); ++i) {
            output.push_back(input[i] * weights_[i]);
        }
        return output;
    }

    std::vector<T> get_all_weights() const override { return weights_; }

    void replace_weight(const T& old_weight, const T& new_weight) override
    {
        for (auto& weight : weights_) {
            if (weight == old_weight) {
                weight = new_weight;
                break;
            }
        }
    }

    void set_weight_protection(const T& weight, AdaptiveProtectionLevel level) override
    {
        // Simplified implementation - in real system would store protection metadata
        (void)weight;
        (void)level;
    }
};

template <typename T>
class AdaptiveProtection {
   public:
    struct ProtectionStats {
        size_t total_weights = 0;         // Total number of weights
        size_t protected_weights = 0;     // Number of protected weights
        size_t corrections = 0;           // Number of corrections applied
        size_t uncorrectable_errors = 0;  // Number of uncorrectable errors
        size_t total_bits = 0;            // Total number of bits
        size_t flipped_bits = 0;          // Number of bits flipped

        double protection_overhead = 0.0;  // Memory overhead for protection
        double seu_rate = 0.0;             // SEU rate in errors/bit/day

        // Reset statistics
        void reset()
        {
            corrections = 0;
            uncorrectable_errors = 0;
            flipped_bits = 0;
        }

        // Get correction ratio (corrected/total errors)
        double correction_ratio() const
        {
            if (corrections + uncorrectable_errors == 0) return 1.0;
            return static_cast<double>(corrections) / (corrections + uncorrectable_errors);
        }
    };

    // Updated constructors without RNG initialization
    AdaptiveProtection()
        : radiation_env_(SpaceMission::LEO_EQUATORIAL),
          error_model_(AdaptiveMultibitUpsetType::SINGLE_BIT),
          protection_level_(AdaptiveProtectionLevel::MODERATE),
          stats_()
    {
    }

    AdaptiveProtection(const RadiationEnvironment& env,
                       AdaptiveProtectionLevel level = AdaptiveProtectionLevel::MODERATE)
        : radiation_env_(env),
          error_model_(AdaptiveMultibitUpsetType::SINGLE_BIT),
          protection_level_(level),
          stats_()
    {
    }

    // Master seed for RNG - provides better entropy on embedded systems
    // Set this once at startup for deterministic behavior across threads
    static void set_master_seed(uint64_t seed)
    {
        master_seed_.store(seed, std::memory_order_release);
        master_seed_set_.store(true, std::memory_order_release);
    }

    static bool has_master_seed() { return master_seed_set_.load(std::memory_order_acquire); }

    // Environment and configuration methods
    void set_environment(const RadiationEnvironment& env) { radiation_env_ = env; }

    const RadiationEnvironment& get_environment() const { return radiation_env_; }

    void set_protection_level(AdaptiveProtectionLevel level) { protection_level_ = level; }

    AdaptiveProtectionLevel get_protection_level() const { return protection_level_; }

    void set_error_model(AdaptiveMultibitUpsetType model) { error_model_ = model; }

    AdaptiveMultibitUpsetType get_error_model() const { return error_model_; }

    const ProtectionStats& get_stats() const { return stats_; }

    void reset_stats() { stats_.reset(); }

    // Main protection interface
    template <typename U = T>
    U protect_value(const U& value, float criticality = 1.0)
    {
        if constexpr (!is_protectable<U>::value) {
            return value;  // Cannot protect this type
        }

        AdaptiveProtectionLevel effective_level = get_effective_protection_level(criticality);

        switch (effective_level) {
            case AdaptiveProtectionLevel::NONE:
                return value;

            case AdaptiveProtectionLevel::MINIMAL: {
                // Parity protection
                auto parity_protected = add_parity_protection(value);
                return parity_protected.data;  // Return original data
            }

            case AdaptiveProtectionLevel::MODERATE: {
                // Hamming code protection
                U result = apply_hamming_protection(value);
                return result;
            }

            case AdaptiveProtectionLevel::HIGH: {
                // Reed-Solomon with 8 symbols using AdvancedReedSolomon
                neural::RS8Bit8Sym<U> rs;
                auto encoded = rs.encode(value);

                // Store encoded data for later recovery (thread-safe)
                // Store encoded data for later recovery (position-based key)
                size_t storage_key = compute_storage_key(&value);
                {
                    std::lock_guard<std::mutex> lock(rs_storage_mutex_);
                    rs_encoded_storage_[storage_key] = encoded;
                }

                return value;
            }

            case AdaptiveProtectionLevel::VERY_HIGH: {
                // Reed-Solomon with 16 symbols using AdvancedReedSolomon
                neural::RS8Bit16Sym<U> rs;
                auto encoded = rs.encode(value);

                // Store encoded data for later recovery (thread-safe, position-based key)
                size_t storage_key = compute_storage_key(&value);
                {
                    std::lock_guard<std::mutex> lock(rs_storage_mutex_);
                    rs_encoded_storage_[storage_key] = encoded;
                }

                // Return original value (encoded data stored separately)
                return value;
            }

            case AdaptiveProtectionLevel::ADAPTIVE:
                // Use moderate protection for adaptive
                return protect_value<U>(value, 5.0);

            default:
                return value;
        }
    }

    template <typename U = T>
    std::tuple<U, bool> recover_value(const U& value, float criticality = 1.0)
    {
        if constexpr (!is_protectable<U>::value) {
            return {value, false};
        }

        AdaptiveProtectionLevel effective_level = get_effective_protection_level(criticality);

        switch (effective_level) {
            case AdaptiveProtectionLevel::NONE:
                return {value, false};

            case AdaptiveProtectionLevel::MINIMAL: {
                // Parity protection recovery
                auto parity_protected = add_parity_protection(value);
                bool error_detected = !parity_protected.verify_parity();
                return {parity_protected.data, error_detected};
            }

            case AdaptiveProtectionLevel::MODERATE: {
                // Hamming code recovery - use stored encoded data from protect_value
                auto [recovered, was_corrected] = recover_with_hamming(value);
                return {recovered, was_corrected};
            }

            case AdaptiveProtectionLevel::HIGH: {
                // Reed-Solomon recovery using AdvancedReedSolomon
                neural::RS8Bit8Sym<U> rs;

                // Retrieve stored encoded data (position-based key)
                size_t storage_key = compute_storage_key(&value);
                std::vector<uint8_t> encoded;
                {
                    std::lock_guard<std::mutex> lock(rs_storage_mutex_);
                    auto it = rs_encoded_storage_.find(storage_key);
                    if (it != rs_encoded_storage_.end()) {
                        encoded = it->second;
                    }
                    else {
                        // If not found, encode current value (fallback)
                        encoded = rs.encode(value);
                    }
                }

                // Decode with error correction
                auto decoded = rs.decode(encoded);
                if (decoded) {
                    return {*decoded, false};
                }
                else {
                    return {value, true};  // Error detected but uncorrectable
                }
            }

            case AdaptiveProtectionLevel::VERY_HIGH: {
                // Reed-Solomon recovery using AdvancedReedSolomon
                neural::RS8Bit16Sym<U> rs;

                // Retrieve stored encoded data (position-based key)
                size_t storage_key = compute_storage_key(&value);
                std::vector<uint8_t> encoded;
                {
                    std::lock_guard<std::mutex> lock(rs_storage_mutex_);
                    auto it = rs_encoded_storage_.find(storage_key);
                    if (it != rs_encoded_storage_.end()) {
                        encoded = it->second;
                    }
                    else {
                        // If not found, encode current value (fallback)
                        encoded = rs.encode(value);
                    }
                }

                // Decode with error correction
                auto decoded = rs.decode(encoded);
                if (decoded) {
                    return {*decoded, false};
                }
                else {
                    return {value, true};  // Error detected but uncorrectable
                }
            }

            case AdaptiveProtectionLevel::ADAPTIVE:
                // Use moderate protection for adaptive
                return recover_value<U>(value, 5.0);

            default:
                return {value, false};
        }
    }

    // Updated apply_radiation_effects with thread-local RNG
    template <typename U = T>
    U apply_radiation_effects(const U& value, double seu_probability = -1.0)
    {
        if (seu_probability < 0) {
            seu_probability = get_current_seu_probability();
        }

        if (seu_probability <= 0) {
            return value;
        }

        std::vector<uint8_t> bytes(sizeof(U));
        std::memcpy(bytes.data(), &value, sizeof(U));

        // Use thread-local RNG for thread safety
        // Prefer master seed if set (better for embedded systems with limited entropy)
        thread_local std::mt19937_64 local_rng(get_thread_seed());

        AdaptiveMultibitHandler<U> mbu;
        bytes = mbu.apply_multi_bit_upset(bytes, error_model_, seu_probability, local_rng);

        U result;
        std::memcpy(&result, bytes.data(), sizeof(U));

        auto bit_flips = count_bit_differences(value, result);
        stats_.total_bits += sizeof(U) * 8;
        stats_.flipped_bits += bit_flips;
        stats_.seu_rate = static_cast<double>(stats_.flipped_bits) / stats_.total_bits;

        return result;
    }

    void adapt_to_environment()
    {
        // Get current SEU rate
        double current_seu_rate = get_current_seu_probability();

        // Adjust protection level based on SEU rate
        if (current_seu_rate > 1e-3) {  // High radiation
            protection_level_ = AdaptiveProtectionLevel::VERY_HIGH;
        }
        else if (current_seu_rate > 1e-4) {  // Medium radiation
            protection_level_ = AdaptiveProtectionLevel::HIGH;
        }
        else if (current_seu_rate > 1e-5) {  // Low radiation
            protection_level_ = AdaptiveProtectionLevel::MODERATE;
        }
        else {  // Very low radiation
            protection_level_ = AdaptiveProtectionLevel::MINIMAL;
        }
    }

    template <typename U = T>
    std::vector<AdaptiveWeightCriticality<U>> identify_critical_weights(
        AdaptiveProtectedNetwork<U>& network, const std::vector<std::vector<U>>& input_samples,
        const std::vector<std::vector<U>>& output_samples)
    {
        std::vector<AdaptiveWeightCriticality<U>> criticalities;
        auto weights = network.get_all_weights();

        for (const auto& weight : weights) {
            // Calculate sensitivity based on weight magnitude and position
            float sensitivity = std::abs(static_cast<float>(weight));

            // Adjust based on position (weights in later layers are more critical)
            // This is a simplified approach - in practice, use gradient-based analysis

            AdaptiveProtectionLevel level = AdaptiveProtectionLevel::NONE;
            if (sensitivity > 10.0) {
                level = AdaptiveProtectionLevel::VERY_HIGH;
            }
            else if (sensitivity > 5.0) {
                level = AdaptiveProtectionLevel::HIGH;
            }
            else if (sensitivity > 1.0) {
                level = AdaptiveProtectionLevel::MODERATE;
            }
            else if (sensitivity > 0.1) {
                level = AdaptiveProtectionLevel::MINIMAL;
            }

            criticalities.push_back({weight, sensitivity, level});
        }

        return criticalities;
    }

    template <typename U = T>
    void apply_optimized_protection(AdaptiveProtectedNetwork<U>& network,
                                    const std::vector<AdaptiveWeightCriticality<U>>& criticalities,
                                    double budget = 0.5)
    {
        // Sort criticalities by sensitivity (highest first)
        std::vector<AdaptiveWeightCriticality<U>> sorted_criticalities = criticalities;
        std::sort(sorted_criticalities.begin(), sorted_criticalities.end(),
                  std::greater<AdaptiveWeightCriticality<U>>());

        // Calculate total sensitivity
        double total_sensitivity = 0.0;
        for (const auto& crit : sorted_criticalities) {
            total_sensitivity += crit.sensitivity;
        }

        // Apply protection based on budget and criticality
        size_t protected_count = 0;
        double current_budget = 0.0;

        for (const auto& crit : sorted_criticalities) {
            double weight_budget = crit.sensitivity / total_sensitivity;

            if (current_budget + weight_budget <= budget) {
                // Apply protection to this weight
                network.set_weight_protection(crit.weight, crit.level);
                protected_count++;
                current_budget += weight_budget;
            }
            else {
                break;  // Budget exceeded
            }
        }

        // Update statistics
        stats_.total_weights = sorted_criticalities.size();
        stats_.protected_weights = protected_count;
    }

    // Test-friendly methods for verification
    template <typename U>
    bool compute_parity(const U& value) const
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        bool parity = false;

        for (size_t i = 0; i < sizeof(U); ++i) {
            uint8_t byte = bytes[i];

            // XOR all bits in the byte
            for (int bit = 0; bit < 8; ++bit) {
                parity ^= ((byte >> bit) & 1) != 0;
            }
        }

        return parity;
    }

    template <typename U>
    U add_parity_bit(const U& value, bool parity) const
    {
        // For simplicity, we just store the parity in the MSB
        // In a real implementation, we would use a more sophisticated approach

        U result = value;
        if (parity) {
            // Set the MSB
            const size_t msb_byte = sizeof(U) - 1;
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
            bytes[msb_byte] |= 0x80;  // Set MSB
        }

        return result;
    }

    template <typename U>
    bool extract_parity_bit(const U& value) const
    {
        // Corresponding to the add_parity_bit function
        const size_t msb_byte = sizeof(U) - 1;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        return (bytes[msb_byte] & 0x80) != 0;
    }

    template <typename U>
    U remove_parity_bit(const U& value) const
    {
        // Corresponding to the add_parity_bit function
        U result = value;
        const size_t msb_byte = sizeof(U) - 1;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
        bytes[msb_byte] &= 0x7F;  // Clear MSB

        return result;
    }

    /**
     * Apply Hamming protection - stores encoded data separately for full byte protection
     * Uses two Hamming(8,4) SECDED codes per byte to protect all 8 bits
     */
    template <typename U>
    U apply_hamming_protection(const U& value) const
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);

        // Encode all bytes and store in hamming_storage_
        std::vector<uint16_t> encoded(sizeof(U));
        for (size_t i = 0; i < sizeof(U); ++i) {
            encoded[i] = hamming_encode_byte_full(bytes[i]);
        }

        // Store encoded data using position-based key
        size_t storage_key = compute_storage_key(&value);
        {
            std::lock_guard<std::mutex> lock(hamming_storage_mutex_);
            hamming_encoded_storage_[storage_key] = std::move(encoded);
    }

        return value;  // Return original value (protection is in storage)
    }

    /**
     * Recover with Hamming SECDED error correction
     * Uses stored encoded data to correct single-bit errors and detect double-bit errors
     */
    template <typename U>
    std::tuple<U, bool> recover_with_hamming(const U& value) const
    {
        size_t storage_key = compute_storage_key(&value);
        std::vector<uint16_t> encoded;

        {
            std::lock_guard<std::mutex> lock(hamming_storage_mutex_);
            auto it = hamming_encoded_storage_.find(storage_key);
            if (it == hamming_encoded_storage_.end()) {
                // No stored encoding - cannot correct, return original
                return {value, false};
            }
            encoded = it->second;
        }

        if (encoded.size() != sizeof(U)) {
            return {value, false};  // Size mismatch
        }

        const uint8_t* value_bytes = reinterpret_cast<const uint8_t*>(&value);
        U result;
        uint8_t* result_bytes = reinterpret_cast<uint8_t*>(&result);
        bool event_detected = false;

        // Decode each byte
        for (size_t i = 0; i < sizeof(U); ++i) {
            auto [decoded_byte, was_corrected, uncorrectable] = hamming_decode_byte_full(encoded[i]);
            if (uncorrectable) {
                result_bytes[i] = value_bytes[i];
                event_detected = true;
            }
            else {
                result_bytes[i] = decoded_byte;
                if (was_corrected) {
                    event_detected = true;
                }
            }
        }

        return {result, event_detected};
    }

    /**
     * Compute a storage key based on pointer address
     * Uses just the pointer address for consistent lookup
     */
    size_t compute_storage_key(const void* ptr) const
    {
        // Use pointer address directly - same address = same key
        return reinterpret_cast<size_t>(ptr);
    }

    /**
     * Encode a single nibble (4 bits) using Hamming(7,4) core (7 data/parity bits)
     * Layout: p1 p2 d1 p3 d2 d3 d4
     */
    static uint8_t hamming_encode_nibble_7_4(uint8_t nibble)
    {
        uint8_t d1 = (nibble >> 0) & 1;
        uint8_t d2 = (nibble >> 1) & 1;
        uint8_t d3 = (nibble >> 2) & 1;
        uint8_t d4 = (nibble >> 3) & 1;

        uint8_t p1 = d1 ^ d2 ^ d4;
        uint8_t p2 = d1 ^ d3 ^ d4;
        uint8_t p3 = d2 ^ d3 ^ d4;

        return (p1 << 0) | (p2 << 1) | (d1 << 2) | (p3 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6);
    }

    static uint8_t extract_nibble_from_codeword_7_4(uint8_t cw7)
    {
        uint8_t d1 = (cw7 >> 2) & 1;
        uint8_t d2 = (cw7 >> 4) & 1;
        uint8_t d3 = (cw7 >> 5) & 1;
        uint8_t d4 = (cw7 >> 6) & 1;
        return static_cast<uint8_t>((d1 << 0) | (d2 << 1) | (d3 << 2) | (d4 << 3));
    }

    /**
     * Encode a single nibble using extended Hamming(8,4) SECDED
     * @param nibble Lower 4 bits of input
     * @return 8-bit codeword (bit 7 = overall parity of bits 0-6)
     */
    static uint8_t hamming_encode_nibble(uint8_t nibble)
    {
        uint8_t cw7 = hamming_encode_nibble_7_4(nibble);
        uint8_t p0 = 0;
        for (int i = 0; i < 7; ++i) {
            p0 ^= static_cast<uint8_t>((cw7 >> i) & 1);
        }
        return static_cast<uint8_t>(cw7 | (p0 << 7));
    }

    /**
     * Decode an 8-bit Hamming(8,4) SECDED codeword
     * Corrects single-bit errors; detects (does not miscorrect) double-bit errors
     */
    static HammingDecodeResult hamming_decode_nibble(uint8_t codeword)
    {
        uint8_t overall = 0;
        for (int i = 0; i < 8; ++i) {
            overall ^= static_cast<uint8_t>((codeword >> i) & 1);
        }

        uint8_t cw7 = codeword & 0x7F;
        uint8_t p1 = (cw7 >> 0) & 1;
        uint8_t p2 = (cw7 >> 1) & 1;
        uint8_t d1 = (cw7 >> 2) & 1;
        uint8_t p3 = (cw7 >> 3) & 1;
        uint8_t d2 = (cw7 >> 4) & 1;
        uint8_t d3 = (cw7 >> 5) & 1;
        uint8_t d4 = (cw7 >> 6) & 1;

        uint8_t s1 = static_cast<uint8_t>(p1 ^ d1 ^ d2 ^ d4);
        uint8_t s2 = static_cast<uint8_t>(p2 ^ d1 ^ d3 ^ d4);
        uint8_t s3 = static_cast<uint8_t>(p3 ^ d2 ^ d3 ^ d4);
        uint8_t syndrome = static_cast<uint8_t>((s3 << 2) | (s2 << 1) | s1);

        if (syndrome == 0 && overall == 0) {
            return {extract_nibble_from_codeword_7_4(cw7), false, false};
        }

        if (syndrome != 0 && overall == 1) {
            if (syndrome <= 7) {
                cw7 ^= static_cast<uint8_t>(1u << (syndrome - 1));
            }
            return {extract_nibble_from_codeword_7_4(cw7), true, false};
        }

        if (syndrome == 0 && overall == 1) {
            // Error confined to overall parity bit (position 8); data bits are intact
            return {extract_nibble_from_codeword_7_4(cw7), true, false};
        }

        // syndrome != 0 && overall == 0: double-bit error
        return {extract_nibble_from_codeword_7_4(cw7), false, true};
    }

    /**
     * Encode a full byte using two Hamming(8,4) SECDED codes
     * @param data Full 8-bit byte to encode
     * @return 16-bit codeword (low/high bytes each hold one 8-bit SECDED nibble code)
     */
    static uint16_t hamming_encode_byte_full(uint8_t data)
    {
        uint8_t low_nibble = data & 0x0F;
        uint8_t high_nibble = (data >> 4) & 0x0F;

        uint8_t encoded_low = hamming_encode_nibble(low_nibble);
        uint8_t encoded_high = hamming_encode_nibble(high_nibble);

        return static_cast<uint16_t>(encoded_low) | (static_cast<uint16_t>(encoded_high) << 8);
    }

    /**
     * Decode a 16-bit SECDED Hamming-encoded byte
     * @return Tuple of (decoded byte, single-bit corrected, double-bit detected)
     */
    static std::tuple<uint8_t, bool, bool> hamming_decode_byte_full(uint16_t codeword)
    {
        auto low = hamming_decode_nibble(static_cast<uint8_t>(codeword & 0xFF));
        auto high = hamming_decode_nibble(static_cast<uint8_t>((codeword >> 8) & 0xFF));

        uint8_t data = static_cast<uint8_t>(low.value | (high.value << 4));
        return {data, low.corrected || high.corrected, low.uncorrectable || high.uncorrectable};
    }

    // Legacy single-nibble API (lower 4 bits only)
    static uint8_t hamming_encode_byte(uint8_t data)
    {
        return hamming_encode_nibble(data & 0x0F);
    }

    static std::tuple<uint8_t, bool> hamming_decode_byte(uint8_t codeword)
    {
        auto result = hamming_decode_nibble(codeword);
        return {result.value, result.corrected};
    }

   private:
    RadiationEnvironment radiation_env_;
    AdaptiveMultibitUpsetType error_model_;
    AdaptiveProtectionLevel protection_level_;
    ProtectionStats stats_;

    // Storage for Reed-Solomon encoded values (position-based key to avoid hash collisions)
    mutable std::unordered_map<size_t, std::vector<uint8_t>> rs_encoded_storage_;
    mutable std::mutex rs_storage_mutex_;  // Thread-safe access to RS storage

    // Storage for Hamming encoded values (position-based key for full byte protection)
    mutable std::unordered_map<size_t, std::vector<uint16_t>> hamming_encoded_storage_;
    mutable std::mutex hamming_storage_mutex_;  // Thread-safe access to Hamming storage

    // Master seed for thread-local RNGs (better entropy for embedded systems)
    inline static std::atomic<uint64_t> master_seed_{0};
    inline static std::atomic<bool> master_seed_set_{false};
    inline static std::atomic<uint64_t> thread_counter_{0};

    /**
     * Get a seed for thread-local RNG
     * Uses master seed + thread ID if set, otherwise falls back to random_device
     */
    static uint64_t get_thread_seed()
    {
        if (master_seed_set_.load(std::memory_order_acquire)) {
            // Combine master seed with a unique thread counter for per-thread variation
            uint64_t base_seed = master_seed_.load(std::memory_order_acquire);
            uint64_t thread_offset = thread_counter_.fetch_add(1, std::memory_order_relaxed);
            // Mix using a simple hash to avoid correlation between threads
            return base_seed ^ (thread_offset * 0x9e3779b97f4a7c15ULL);
        }
        else {
            // Fall back to random_device (may have limited entropy on embedded)
            try {
                return std::random_device{}();
            }
            catch (...) {
                // If random_device fails (some embedded systems), use time-based fallback
                return static_cast<uint64_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count());
            }
        }
    }

    // Safe parity-protected value structure
    template <typename U>
    struct ParityProtectedValue {
        U data;
        bool parity_bit;

        ParityProtectedValue(const U& value) : data(value)
        {
            parity_bit = compute_parity_safe(value);
        }

        bool verify_parity() const { return parity_bit == compute_parity_safe(data); }
    };

    // Safe parity computation (doesn't modify data)
    template <typename U>
    static bool compute_parity_safe(const U& value)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        bool parity = false;

        for (size_t i = 0; i < sizeof(U); ++i) {
            uint8_t byte = bytes[i];
            for (int bit = 0; bit < 8; ++bit) {
                parity ^= ((byte >> bit) & 1) != 0;
            }
        }
        return parity;
    }

    // Safe parity protection that doesn't corrupt data
    template <typename U>
    ParityProtectedValue<U> add_parity_protection(const U& value) const
    {
        return ParityProtectedValue<U>(value);
    }

    // Thread-safe random bit flip
    template <typename U>
    U flip_random_bit(const U& value)
    {  // Removed const for thread safety
        // Use thread-local RNG for thread safety
        // Prefer master seed if set (better for embedded systems with limited entropy)
        thread_local std::mt19937_64 local_rng(get_thread_seed());

        U result = value;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);

        std::uniform_int_distribution<size_t> byte_dist(0, sizeof(U) - 1);
        std::uniform_int_distribution<int> bit_dist(0, 7);

        size_t byte_idx = byte_dist(local_rng);
        int bit_idx = bit_dist(local_rng);

        bytes[byte_idx] ^= (1 << bit_idx);
        return result;
    }

    double get_current_seu_probability() const
    {
        // Create a default orbital position for SEU calculation
        OrbitalPosition pos{0.0, 0.0, 400.0};  // Default LEO position
        return radiation_env_.calculateSEUProbability(pos);
    }

    AdaptiveProtectionLevel get_effective_protection_level(float criticality) const
    {
        if (protection_level_ != AdaptiveProtectionLevel::ADAPTIVE) {
            return protection_level_;
        }

        // For adaptive mode, select based on criticality
        if (criticality > 10.0) {
            return AdaptiveProtectionLevel::VERY_HIGH;
        }
        else if (criticality > 5.0) {
            return AdaptiveProtectionLevel::HIGH;
        }
        else if (criticality > 1.0) {
            return AdaptiveProtectionLevel::MODERATE;
        }
        else if (criticality > 0.1) {
            return AdaptiveProtectionLevel::MINIMAL;
        }
        else {
            return AdaptiveProtectionLevel::NONE;
        }
    }

    template <typename U>
    size_t count_bit_differences(const U& a, const U& b) const
    {
        const uint8_t* bytes_a = reinterpret_cast<const uint8_t*>(&a);
        const uint8_t* bytes_b = reinterpret_cast<const uint8_t*>(&b);

        size_t differences = 0;

        for (size_t i = 0; i < sizeof(U); ++i) {
            uint8_t diff = bytes_a[i] ^ bytes_b[i];

            // Count bits in the difference
            for (int bit = 0; bit < 8; ++bit) {
                if ((diff >> bit) & 1) {
                    differences++;
                }
            }
        }

        return differences;
    }

    template <typename U>
    double calculate_network_error(AdaptiveProtectedNetwork<U>& network,
                                   const std::vector<std::vector<U>>& inputs,
                                   const std::vector<std::vector<U>>& expected) const
    {
        double total_error = 0.0;
        size_t total_outputs = 0;

        for (size_t i = 0; i < inputs.size() && i < expected.size(); ++i) {
            auto output = network.forward(inputs[i]);
            for (size_t j = 0; j < output.size() && j < expected[i].size(); ++j) {
                double diff = static_cast<double>(output[j] - expected[i][j]);
                total_error += diff * diff;
                total_outputs++;
            }
        }

        return total_outputs > 0 ? total_error / total_outputs : 0.0;
    }
};

}  // namespace neural
}  // namespace rad_ml
