/**
 * @file galois_field.hpp
 * @brief Galois Field implementation for Reed-Solomon error correction
 * 
 * This file implements the Galois Field arithmetic operations necessary for
 * properly implementing Reed-Solomon error correction coding. It provides
 * efficient finite field arithmetic using lookup tables.
 */

#ifndef RAD_ML_NEURAL_GALOIS_FIELD_HPP
#define RAD_ML_NEURAL_GALOIS_FIELD_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <tuple>
#include <optional>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <initializer_list>
#include <utility>

namespace rad_ml {
namespace neural {

/**
 * @brief Galois Field template for Reed-Solomon error correction
 * 
 * This class implements a Galois Field GF(2^m) for use in Reed-Solomon
 * error correction coding. It provides efficient finite field arithmetic
 * using lookup tables.
 * 
 * @tparam m Galois Field order (2^m elements)
 * @tparam Poly Primitive polynomial defining the field
 */
template<uint8_t m, uint16_t Poly>
class GaloisField {
public:
    using element_t = std::conditional_t<(m <= 8), uint8_t, uint16_t>;
    
    static constexpr element_t field_size = (1 << m);
    static constexpr element_t field_mask = field_size - 1;
    static constexpr element_t primitive_poly = Poly;

    /**
     * @brief Constructor initializes lookup tables
     */
    GaloisField() {
        // Initialize exp and log tables for fast multiplication
        initialize_tables();
    }
    
    /**
     * @brief Addition in GF(2^m) is XOR
     * 
     * @param a First operand
     * @param b Second operand
     * @return Sum in GF(2^m)
     */
    constexpr element_t add(element_t a, element_t b) const {
        return a ^ b;
    }
    
    /**
     * @brief Subtraction in GF(2^m) is identical to addition (XOR)
     * 
     * @param a First operand
     * @param b Second operand
     * @return Difference in GF(2^m)
     */
    constexpr element_t subtract(element_t a, element_t b) const {
        return a ^ b;
    }
    
    /**
     * @brief Multiplication in GF(2^m) using lookup tables
     * 
     * @param a First operand
     * @param b Second operand
     * @return Product in GF(2^m)
     */
    element_t multiply(element_t a, element_t b) const {
        // Handle special cases
        if (a == 0 || b == 0) return 0;
        
        // Use log-antilog method
        return exp_table[(log_table[a] + log_table[b]) % (field_size - 1)];
    }
    
    /**
     * @brief Division in GF(2^m) using lookup tables
     * 
     * @param a Numerator
     * @param b Denominator (must be non-zero)
     * @return Quotient in GF(2^m)
     */
    element_t divide(element_t a, element_t b) const {
        if (b == 0) {
            throw std::domain_error("Division by zero in Galois Field");
        }
        
        if (a == 0) return 0;
        
        return exp_table[(log_table[a] + field_size - 1 - log_table[b]) % (field_size - 1)];
    }
    
    /**
     * @brief Exponentiation in GF(2^m)
     * 
     * @param a Base element
     * @param power Power to raise to
     * @return a^power in GF(2^m)
     */
    element_t pow(element_t a, unsigned int power) const {
        if (a == 0) return (power == 0) ? 1 : 0;
        
        if (power == 0) return 1;
        
        // Use log-antilog method
        return exp_table[(static_cast<unsigned int>(log_table[a]) * power) % (field_size - 1)];
    }
    
    /**
     * @brief Find the multiplicative inverse of an element
     * 
     * @param a Element to invert
     * @return Multiplicative inverse in GF(2^m)
     */
    element_t inverse(element_t a) const {
        if (a == 0) {
            throw std::domain_error("Cannot invert zero in Galois Field");
        }
        
        return exp_table[field_size - 1 - log_table[a]];
    }
    
    /**
     * @brief Evaluate a polynomial at a point using Horner's method
     * 
     * @param poly Polynomial coefficients (highest degree first)
     * @param x Point to evaluate at
     * @return Polynomial value at x
     */
    element_t eval_poly(const std::vector<element_t>& poly, element_t x) const {
        element_t result = 0;
        
        for (const auto& coeff : poly) {
            result = add(multiply(result, x), coeff);
        }
        
        return result;
    }
    
    /**
     * @brief Generate Reed-Solomon generator polynomial
     * 
     * @param nsym Number of error correction symbols
     * @return Generator polynomial coefficients
     */
    std::vector<element_t> rs_generator_poly(uint8_t nsym) const {
        // Start with g(x) = (x - α^0)
        std::vector<element_t> g = {1};
        
        // Multiply by (x - α^i) for i=1..nsym
        for (uint8_t i = 0; i < nsym; ++i) {
            // Multiply g(x) by (x - α^i)
            std::vector<element_t> g_new(g.size() + 1, 0);
            
            // Multiply g(x) by x
            std::copy(g.begin(), g.end(), g_new.begin());
            
            // Multiply g(x) by -α^i and add
            for (size_t j = 0; j < g.size(); ++j) {
                g_new[j+1] = add(g_new[j+1], multiply(g[j], exp_table[i]));
            }
            
            g = std::move(g_new);
        }
        
        return g;
    }

    /**
     * @brief Calculate Reed-Solomon syndromes for error detection
     * 
     * @param msg Message with ecc symbols
     * @param nsym Number of ecc symbols
     * @return Syndrome values (first value is always 0)
     */
    std::vector<element_t> rs_calc_syndromes(const std::vector<element_t>& msg, uint8_t nsym) const {
        std::vector<element_t> synd(nsym + 1, 0);
        
        // Evaluate message polynomial at α^i for i=0..nsym
        for (uint8_t i = 0; i <= nsym; ++i) {
            synd[i] = eval_poly(msg, exp_table[i]);
        }
        
        return synd;
    }
    
    /**
     * @brief Find error locator and evaluator polynomials using Berlekamp-Massey algorithm
     * 
     * @param syndromes Syndromes from rs_calc_syndromes
     * @param nsym Number of ecc symbols
     * @return Tuple of error locator and evaluator polynomials
     */
    std::tuple<std::vector<element_t>, std::vector<element_t>> 
    rs_find_error_locator(const std::vector<element_t>& syndromes, uint8_t nsym) const {
        // Berlekamp-Massey algorithm to find the error locator polynomial
        std::vector<element_t> err_loc = {1}; // Initialize error locator polynomial
        std::vector<element_t> old_loc = {1}; // Previous iteration
        
        for (uint8_t i = 0; i < nsym; ++i) {
            // Compute discrepancy
            element_t delta = syndromes[i + 1];
            for (size_t j = 1; j < err_loc.size(); ++j) {
                delta = add(delta, multiply(err_loc[err_loc.size() - 1 - j], syndromes[i + 1 - j]));
            }
            
            // Shift old_loc and multiply by delta
            std::vector<element_t> new_loc = old_loc;
            new_loc.insert(new_loc.begin(), 0); // Multiply by x
            
            if (delta != 0) {
                for (size_t j = 0; j < new_loc.size(); ++j) {
                    new_loc[j] = add(err_loc[j], multiply(delta, new_loc[j]));
                }
            }
            
            if (2 * old_loc.size() <= i + 1 && delta != 0) {
                old_loc = err_loc;
                for (auto& el : old_loc) {
                    el = multiply(el, delta);
                }
                err_loc = new_loc;
            } else {
                old_loc = new_loc;
            }
        }
        
        // Calculate error evaluator polynomial using the formula:
        // Ω(x) = Syndromes(x) * Λ(x) mod x^(nsym+1)
        std::vector<element_t> err_eval(nsym);
        
        for (uint8_t i = 0; i < nsym; ++i) {
            element_t tmp = 0;
            for (size_t j = 0; j < std::min(i + 1, err_loc.size()); ++j) {
                tmp = add(tmp, multiply(err_loc[j], syndromes[i - j + 1]));
            }
            err_eval[i] = tmp;
        }
        
        return {err_loc, err_eval};
    }
    
    /**
     * @brief Find error positions using Chien search
     * 
     * @param err_loc Error locator polynomial
     * @param msg_len Message length
     * @return Vector of error positions
     */
    std::vector<size_t> rs_find_errors(const std::vector<element_t>& err_loc, size_t msg_len) const {
        std::vector<size_t> err_pos;
        
        // Number of errors = degree of error locator polynomial
        size_t num_errors = err_loc.size() - 1;
        
        if (num_errors > msg_len) {
            return {}; // Error count exceeds message length - uncorrectable
        }
        
        // Chien search: evaluate error locator polynomial at all positions
        for (size_t i = 0; i < msg_len; ++i) {
            element_t eval = 0;
            element_t x_inv = exp_table[(field_size - 1 - i) % (field_size - 1)]; // α^(-i)
            
            // Evaluate using Horner's method
            for (const auto& coeff : err_loc) {
                eval = add(multiply(eval, x_inv), coeff);
            }
            
            if (eval == 0) {
                // Found an error location
                err_pos.push_back(msg_len - 1 - i);
            }
        }
        
        // Check if we found the correct number of errors
        if (err_pos.size() != num_errors) {
            return {}; // Number of roots doesn't match error count - uncorrectable
        }
        
        return err_pos;
    }
    
    /**
     * @brief Correct errors in a message using Forney algorithm
     * 
     * @param msg_in Message with errors
     * @param err_pos Error positions
     * @param err_loc Error locator polynomial
     * @param err_eval Error evaluator polynomial
     * @return Corrected message
     */
    std::vector<element_t> rs_correct_errors_at_positions(
        const std::vector<element_t>& msg_in,
        const std::vector<size_t>& err_pos,
        const std::vector<element_t>& err_loc,
        const std::vector<element_t>& err_eval
    ) const {
        std::vector<element_t> msg = msg_in;
        
        // Forney algorithm to calculate error magnitudes
        for (size_t i = 0; i < err_pos.size(); ++i) {
            // Get error position
            size_t pos = err_pos[i];
            
            // Calculate error magnitude using Forney algorithm
            element_t x_inv = exp_table[(field_size - 1 - pos) % (field_size - 1)]; // α^(-j)
            
            // Calculate error evaluator at position
            element_t err_eval_at_pos = 0;
            for (size_t j = 0; j < err_eval.size(); ++j) {
                err_eval_at_pos = add(err_eval_at_pos, multiply(err_eval[j], pow(x_inv, j)));
            }
            
            // Calculate error locator derivative at position
            element_t err_loc_deriv = 0;
            for (size_t j = 1; j < err_loc.size(); j += 2) {
                err_loc_deriv = add(err_loc_deriv, multiply(err_loc[j], pow(x_inv, j-1)));
            }
            
            // Calculate error magnitude
            element_t err_mag = divide(err_eval_at_pos, multiply(x_inv, err_loc_deriv));
            
            // Correct the error
            msg[pos] = add(msg[pos], err_mag);
        }
        
        return msg;
    }
    
    /**
     * @brief Complete Reed-Solomon error correction process
     * 
     * @param msg Message with errors
     * @param nsym Number of ecc symbols
     * @return Corrected message or std::nullopt if uncorrectable
     */
    std::optional<std::vector<element_t>> rs_correct_errors(
        const std::vector<element_t>& msg,
        uint8_t nsym
    ) const {
        // Calculate syndromes
        auto syndromes = rs_calc_syndromes(msg, nsym);
        
        // Check if message has errors
        bool has_errors = false;
        for (size_t i = 1; i < syndromes.size(); ++i) {
            if (syndromes[i] != 0) {
                has_errors = true;
                break;
            }
        }
        
        if (!has_errors) {
            return msg; // No errors to correct
        }
        
        // Find error locator and evaluator polynomials
        auto [err_loc, err_eval] = rs_find_error_locator(syndromes, nsym);
        
        // Find error positions
        auto err_pos = rs_find_errors(err_loc, msg.size());
        
        if (err_pos.empty()) {
            return std::nullopt; // Uncorrectable errors
        }
        
        // Correct errors
        return rs_correct_errors_at_positions(msg, err_pos, err_loc, err_eval);
    }
    
    /**
     * @brief Generate a pseudorandom element for testing
     * 
     * @param rng Random number generator
     * @return Random element
     */
    template<typename RNG>
    element_t random_element(RNG& rng) const {
        std::uniform_int_distribution<element_t> dist(0, field_mask);
        return dist(rng);
    }

private:
    std::array<element_t, field_size> exp_table;  // α^i lookup
    std::array<element_t, field_size> log_table;  // log_α(i) lookup
    
    /**
     * @brief Initialize lookup tables for multiplication and division
     */
    void initialize_tables() {
        // Initialize exp and log tables for efficient multiplication
        element_t x = 1;
        
        for (element_t i = 0; i < field_size - 1; ++i) {
            exp_table[i] = x;
            
            // Multiply by α in GF(2^m)
            x = multiply_no_lut(x, 2);
            if (x >= field_size) {
                x ^= primitive_poly;
            }
        }
        
        // Set the last element
        exp_table[field_size - 1] = exp_table[0];
        
        // Generate log table
        log_table[0] = 0; // log(0) is undefined, set to 0 for convenience
        
        for (element_t i = 0; i < field_size - 1; ++i) {
            log_table[exp_table[i]] = i;
        }
    }
    
    /**
     * @brief Multiplication without using lookup tables
     * 
     * This is used only for initializing the tables.
     * 
     * @param a First operand
     * @param b Second operand
     * @return Product in GF(2^m)
     */
    element_t multiply_no_lut(element_t a, element_t b) const {
        element_t result = 0;
        
        for (size_t i = 0; i < m; ++i) {
            if (b & (1 << i)) {
                result ^= a;
            }
            
            // Multiply a by x
            bool overflow = (a & (1 << (m - 1))) != 0;
            a <<= 1;
            
            if (overflow) {
                a ^= (primitive_poly & field_mask);
            }
        }
        
        return result & field_mask;
    }
};

// Common Galois Fields for Reed-Solomon codes
using GF16 = GaloisField<4, 0x13>;    // GF(2^4) with polynomial x^4 + x + 1
using GF256 = GaloisField<8, 0x11d>;  // GF(2^8) with polynomial x^8 + x^4 + x^3 + x^2 + 1
using GF1024 = GaloisField<10, 0x409>; // GF(2^10) with polynomial x^10 + x^3 + 1

} // namespace neural
} // namespace rad_ml

#endif // RAD_ML_NEURAL_GALOIS_FIELD_HPP 