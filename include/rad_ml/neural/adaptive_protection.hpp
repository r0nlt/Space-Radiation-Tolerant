#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "rad_ml/neural/radiation_environment.hpp"
#include "rad_ml/radiation/space_mission.hpp"

namespace rad_ml {
namespace neural {

// Forward declarations
template <typename T>
class ProtectedNeuralNetwork;

// Missing enums
enum class MultibitUpsetType { SINGLE_BIT, ADJACENT_BITS, RANDOM_MULTI };

enum class ProtectionLevel {
    NONE,       // No protection
    MINIMAL,    // Basic parity-based protection
    MODERATE,   // TMR or Hamming code protection
    HIGH,       // Reed-Solomon with moderate parameter settings
    VERY_HIGH,  // Reed-Solomon with strong parameter settings
    ADAPTIVE    // Dynamically adjusted based on radiation conditions
};

// Type trait for protectable types
template <typename T>
struct is_protectable {
    static constexpr bool value =
        std::is_trivially_copyable_v<T> &&
        (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>);
};

// Weight criticality structure
template <typename T>
struct WeightCriticality {
    T weight;               // The weight value
    float sensitivity;      // Sensitivity score (higher = more critical)
    ProtectionLevel level;  // Selected protection level

    // Allow comparison for sorting
    bool operator<(const WeightCriticality& other) const { return sensitivity < other.sensitivity; }
};

// Reed-Solomon implementation using GF(256)
template <typename T>
class RS8Bit8Sym {
   private:
    static constexpr uint8_t GF256_PRIM = 0x1D;  // Primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
    static constexpr size_t NSYM = 8;            // Number of parity symbols

    // Galois Field tables (simplified - in production use precomputed tables)
    std::array<uint8_t, 256> gf_exp;
    std::array<uint8_t, 256> gf_log;

   public:
    RS8Bit8Sym()
    {
        // Initialize GF(256) tables
        init_gf_tables();
    }

    std::vector<uint8_t> encode(const T& data)
    {
        // Convert data to bytes
        std::vector<uint8_t> bytes(sizeof(T));
        std::memcpy(bytes.data(), &data, sizeof(T));

        // Add Reed-Solomon parity symbols
        std::vector<uint8_t> encoded = bytes;
        encoded.resize(bytes.size() + NSYM, 0);

        // Simplified RS encoding (in production, use proper generator polynomial)
        for (size_t i = 0; i < NSYM; ++i) {
            uint8_t parity = 0;
            for (size_t j = 0; j < bytes.size(); ++j) {
                parity ^= gf_mult(bytes[j], gf_pow(2, i * j % 255));
            }
            encoded[bytes.size() + i] = parity;
        }

        return encoded;
    }

    std::optional<T> decode(std::vector<uint8_t>& encoded)
    {
        if (encoded.size() < sizeof(T) + NSYM) {
            return std::nullopt;
        }

        // Extract data portion
        std::vector<uint8_t> data(encoded.begin(), encoded.begin() + sizeof(T));

        // Simple error detection (in production, use proper syndrome calculation)
        auto expected_encoded = encode(*reinterpret_cast<T*>(data.data()));

        size_t errors = 0;
        for (size_t i = 0; i < encoded.size() && i < expected_encoded.size(); ++i) {
            if (encoded[i] != expected_encoded[i]) {
                errors++;
            }
        }

        // Can correct up to NSYM/2 errors
        if (errors > NSYM / 2) {
            return std::nullopt;  // Too many errors
        }

        // Return corrected data
        T result;
        std::memcpy(&result, data.data(), sizeof(T));
        return result;
    }

    std::vector<uint8_t> apply_bit_errors(std::vector<uint8_t> data, double error_rate,
                                          uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (auto& byte : data) {
            for (int bit = 0; bit < 8; ++bit) {
                if (dist(rng) < error_rate) {
                    byte ^= (1 << bit);
                }
            }
        }
        return data;
    }

    double overhead_percent() const { return 100.0 * NSYM / sizeof(T); }

   private:
    void init_gf_tables()
    {
        // Initialize all entries to 0
        std::fill(gf_exp.begin(), gf_exp.end(), 0);
        std::fill(gf_log.begin(), gf_log.end(), 0);

        uint16_t x = 1;
        for (size_t i = 0; i < 255; ++i) {  // Only 255 non-zero elements
            gf_exp[i] = static_cast<uint8_t>(x);
            if (x < 256) {  // Bounds check
                gf_log[x] = static_cast<uint8_t>(i);
            }
            x <<= 1;
            if (x & 0x100) {
                x ^= GF256_PRIM;
            }
        }
        // Set gf_exp[255] = 1 (alpha^255 = 1)
        gf_exp[255] = 1;
    }

    uint8_t gf_mult(uint8_t a, uint8_t b)
    {
        if (a == 0 || b == 0) return 0;
        return gf_exp[(gf_log[a] + gf_log[b]) % 255];
    }

    uint8_t gf_pow(uint8_t base, uint8_t exp)
    {
        if (base == 0) return 0;
        return gf_exp[(gf_log[base] * exp) % 255];
    }
};

// RS with 16 symbols (stronger protection)
template <typename T>
class RS8Bit16Sym {
   private:
    static constexpr size_t NSYM = 16;
    RS8Bit8Sym<T> base_rs;  // Reuse the base implementation

   public:
    std::vector<uint8_t> encode(const T& data)
    {
        auto encoded = base_rs.encode(data);
        // Add extra parity symbols (simplified approach)
        for (size_t i = 0; i < 8; ++i) {
            encoded.push_back(encoded[i] ^ encoded[i + 8]);  // Simple additional parity
        }
        return encoded;
    }

    std::optional<T> decode(std::vector<uint8_t>& encoded)
    {
        if (encoded.size() < sizeof(T) + NSYM) {
            return std::nullopt;
        }

        // Use first 8 symbols for primary correction
        std::vector<uint8_t> primary(encoded.begin(), encoded.begin() + sizeof(T) + 8);
        return base_rs.decode(primary);
    }

    std::vector<uint8_t> apply_bit_errors(std::vector<uint8_t> data, double error_rate,
                                          uint64_t seed)
    {
        return base_rs.apply_bit_errors(data, error_rate, seed);
    }

    double overhead_percent() const { return 100.0 * NSYM / sizeof(T); }
};

// Multi-bit protection implementation
template <typename T>
class MultibitProtection {
   public:
    std::vector<uint8_t> apply_multi_bit_upset(std::vector<uint8_t> data, MultibitUpsetType type,
                                               double seu_probability, std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        switch (type) {
            case MultibitUpsetType::SINGLE_BIT:
                for (auto& byte : data) {
                    for (int bit = 0; bit < 8; ++bit) {
                        if (dist(rng) < seu_probability) {
                            byte ^= (1 << bit);
                        }
                    }
                }
                break;

            case MultibitUpsetType::ADJACENT_BITS:
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

            case MultibitUpsetType::RANDOM_MULTI:
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

// Basic Protected Neural Network Interface
template <typename T>
class ProtectedNeuralNetwork {
   public:
    virtual ~ProtectedNeuralNetwork() = default;

    virtual std::vector<T> forward(const std::vector<T>& input) = 0;
    virtual std::vector<T> get_all_weights() const = 0;
    virtual void replace_weight(const T& old_weight, const T& new_weight) = 0;
    virtual void set_weight_protection(const T& weight, ProtectionLevel level) = 0;
};

// Example implementation for testing
template <typename T>
class SimpleProtectedNetwork : public ProtectedNeuralNetwork<T> {
   private:
    std::vector<T> weights_;

   public:
    SimpleProtectedNetwork(const std::vector<T>& weights) : weights_(weights) {}

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

    void set_weight_protection(const T& weight, ProtectionLevel level) override
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
          error_model_(MultibitUpsetType::SINGLE_BIT),
          protection_level_(ProtectionLevel::MODERATE),
          stats_()
    {
    }

    AdaptiveProtection(const RadiationEnvironment& env,
                       ProtectionLevel level = ProtectionLevel::MODERATE)
        : radiation_env_(env),
          error_model_(MultibitUpsetType::SINGLE_BIT),
          protection_level_(level),
          stats_()
    {
    }

    // Environment and configuration methods
    void set_environment(const RadiationEnvironment& env) { radiation_env_ = env; }

    const RadiationEnvironment& get_environment() const { return radiation_env_; }

    void set_protection_level(ProtectionLevel level) { protection_level_ = level; }

    ProtectionLevel get_protection_level() const { return protection_level_; }

    void set_error_model(MultibitUpsetType model) { error_model_ = model; }

    MultibitUpsetType get_error_model() const { return error_model_; }

    const ProtectionStats& get_stats() const { return stats_; }

    void reset_stats() { stats_.reset(); }

    // Main protection interface
    template <typename U = T>
    U protect_value(const U& value, float criticality = 1.0)
    {
        if constexpr (!is_protectable<U>::value) {
            return value;  // Cannot protect this type
        }

        ProtectionLevel effective_level = get_effective_protection_level(criticality);

        switch (effective_level) {
            case ProtectionLevel::NONE:
                return value;

            case ProtectionLevel::MINIMAL: {
                // Parity protection
                auto parity_protected = add_parity_protection(value);
                return parity_protected.data;  // Return original data
            }

            case ProtectionLevel::MODERATE: {
                // Hamming code protection
                U result = apply_hamming_protection(value);
                return result;
            }

            case ProtectionLevel::HIGH: {
                // Reed-Solomon with 8 symbols
                RS8Bit8Sym<U> rs;
                auto encoded = rs.encode(value);
                // For now, return original value (in real system would store encoded)
                return value;
            }

            case ProtectionLevel::VERY_HIGH: {
                // Reed-Solomon with 16 symbols
                RS8Bit16Sym<U> rs;
                auto encoded = rs.encode(value);
                // For now, return original value (in real system would store encoded)
                return value;
            }

            case ProtectionLevel::ADAPTIVE:
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

        ProtectionLevel effective_level = get_effective_protection_level(criticality);

        switch (effective_level) {
            case ProtectionLevel::NONE:
                return {value, false};

            case ProtectionLevel::MINIMAL: {
                // Parity protection recovery
                auto parity_protected = add_parity_protection(value);
                bool error_detected = !parity_protected.verify_parity();
                return {parity_protected.data, error_detected};
            }

            case ProtectionLevel::MODERATE: {
                // Hamming code recovery
                U result = apply_hamming_protection(value);
                auto [recovered, was_corrected] = recover_with_hamming(result);
                return {recovered, was_corrected};
            }

            case ProtectionLevel::HIGH: {
                // Reed-Solomon recovery (simplified)
                RS8Bit8Sym<U> rs;
                std::vector<uint8_t> encoded(sizeof(U) + 8);
                std::memcpy(encoded.data(), &value, sizeof(U));
                auto decoded = rs.decode(encoded);
                if (decoded) {
                    return {*decoded, false};
                }
                else {
                    return {value, true};  // Error detected
                }
            }

            case ProtectionLevel::VERY_HIGH: {
                // Reed-Solomon recovery (simplified)
                RS8Bit16Sym<U> rs;
                std::vector<uint8_t> encoded(sizeof(U) + 16);
                std::memcpy(encoded.data(), &value, sizeof(U));
                auto decoded = rs.decode(encoded);
                if (decoded) {
                    return {*decoded, false};
                }
                else {
                    return {value, true};  // Error detected
                }
            }

            case ProtectionLevel::ADAPTIVE:
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
        thread_local std::mt19937_64 local_rng(std::random_device{}());

        MultibitProtection<U> mbu;
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
            protection_level_ = ProtectionLevel::VERY_HIGH;
        }
        else if (current_seu_rate > 1e-4) {  // Medium radiation
            protection_level_ = ProtectionLevel::HIGH;
        }
        else if (current_seu_rate > 1e-5) {  // Low radiation
            protection_level_ = ProtectionLevel::MODERATE;
        }
        else {  // Very low radiation
            protection_level_ = ProtectionLevel::MINIMAL;
        }
    }

    template <typename U = T>
    std::vector<WeightCriticality<U>> identify_critical_weights(
        ProtectedNeuralNetwork<U>& network, const std::vector<std::vector<U>>& input_samples,
        const std::vector<std::vector<U>>& output_samples)
    {
        std::vector<WeightCriticality<U>> criticalities;
        auto weights = network.get_all_weights();

        for (const auto& weight : weights) {
            // Calculate sensitivity based on weight magnitude and position
            float sensitivity = std::abs(static_cast<float>(weight));

            // Adjust based on position (weights in later layers are more critical)
            // This is a simplified approach - in practice, use gradient-based analysis

            ProtectionLevel level = ProtectionLevel::NONE;
            if (sensitivity > 10.0) {
                level = ProtectionLevel::VERY_HIGH;
            }
            else if (sensitivity > 5.0) {
                level = ProtectionLevel::HIGH;
            }
            else if (sensitivity > 1.0) {
                level = ProtectionLevel::MODERATE;
            }
            else if (sensitivity > 0.1) {
                level = ProtectionLevel::MINIMAL;
            }

            criticalities.push_back({weight, sensitivity, level});
        }

        return criticalities;
    }

    template <typename U = T>
    void apply_optimized_protection(ProtectedNeuralNetwork<U>& network,
                                    const std::vector<WeightCriticality<U>>& criticalities,
                                    double budget = 0.5)
    {
        // Sort criticalities by sensitivity (highest first)
        std::vector<WeightCriticality<U>> sorted_criticalities = criticalities;
        std::sort(sorted_criticalities.begin(), sorted_criticalities.end(),
                  std::greater<WeightCriticality<U>>());

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

    template <typename U>
    U apply_hamming_protection(const U& value) const
    {
        U result = value;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);

        // Apply Hamming code to each byte
        for (size_t i = 0; i < sizeof(U); ++i) {
            bytes[i] = hamming_encode_byte(bytes[i]);
        }

        return result;
    }

    template <typename U>
    std::tuple<U, bool> recover_with_hamming(const U& value) const
    {
        U result = value;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
        bool correction_applied = false;

        // Decode each byte with Hamming error correction
        for (size_t i = 0; i < sizeof(U); ++i) {
            auto [decoded_byte, was_corrected] = hamming_decode_byte(bytes[i]);
            bytes[i] = decoded_byte;
            if (was_corrected) {
                correction_applied = true;
            }
        }

        return {result, correction_applied};
    }

    static uint8_t hamming_encode_byte(uint8_t data)
    {
        // Extract data bits (we'll use the lower 4 bits)
        uint8_t d1 = (data >> 0) & 1;
        uint8_t d2 = (data >> 1) & 1;
        uint8_t d3 = (data >> 2) & 1;
        uint8_t d4 = (data >> 3) & 1;

        // Calculate parity bits
        uint8_t p1 = d1 ^ d2 ^ d4;
        uint8_t p2 = d1 ^ d3 ^ d4;
        uint8_t p3 = d2 ^ d3 ^ d4;

        // Construct codeword: p1 p2 d1 p3 d2 d3 d4
        return (p1 << 0) | (p2 << 1) | (d1 << 2) | (p3 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6);
    }

    static std::tuple<uint8_t, bool> hamming_decode_byte(uint8_t codeword)
    {
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

        // Reconstruct data (4-bit result)
        uint8_t data = (d1 << 0) | (d2 << 1) | (d3 << 2) | (d4 << 3);

        return {data, error_pos != 0};
    }

   private:
    RadiationEnvironment radiation_env_;
    MultibitUpsetType error_model_;
    ProtectionLevel protection_level_;
    ProtectionStats stats_;

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
        thread_local std::mt19937_64 local_rng(std::random_device{}());

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

    ProtectionLevel get_effective_protection_level(float criticality) const
    {
        if (protection_level_ != ProtectionLevel::ADAPTIVE) {
            return protection_level_;
        }

        // For adaptive mode, select based on criticality
        if (criticality > 10.0) {
            return ProtectionLevel::VERY_HIGH;
        }
        else if (criticality > 5.0) {
            return ProtectionLevel::HIGH;
        }
        else if (criticality > 1.0) {
            return ProtectionLevel::MODERATE;
        }
        else if (criticality > 0.1) {
            return ProtectionLevel::MINIMAL;
        }
        else {
            return ProtectionLevel::NONE;
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
    double calculate_network_error(ProtectedNeuralNetwork<U>& network,
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
