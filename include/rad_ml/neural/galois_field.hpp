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

#ifndef RADML_RS_BM_NOHEAP
#define RADML_RS_BM_NOHEAP 0
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

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
template <uint8_t m, uint16_t Poly>
class GaloisField {
   public:
    using element_t = std::conditional_t<(m <= 8), uint8_t, uint16_t>;

    static constexpr uint32_t field_size = (1ULL << m);
    static constexpr element_t field_mask = static_cast<element_t>(field_size - 1);
    static constexpr uint16_t primitive_poly = Poly;

    /**
     * @brief Constructor initializes lookup tables
     */
    GaloisField()
    {
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
    constexpr element_t add(element_t a, element_t b) const { return a ^ b; }

    /**
     * @brief Subtraction in GF(2^m) is identical to addition (XOR)
     *
     * @param a First operand
     * @param b Second operand
     * @return Difference in GF(2^m)
     */
    constexpr element_t subtract(element_t a, element_t b) const { return a ^ b; }

    /**
     * @brief Multiplication in GF(2^m) using lookup tables
     *
     * @param a First operand
     * @param b Second operand
     * @return Product in GF(2^m)
     */
    element_t multiply(element_t a, element_t b) const
    {
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
    element_t divide(element_t a, element_t b) const
    {
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
    element_t pow(element_t a, unsigned int power) const
    {
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
    element_t inverse(element_t a) const
    {
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
    element_t eval_poly(const std::vector<element_t>& poly, element_t x) const
    {
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
    std::vector<element_t> rs_generator_poly(uint8_t nsym) const
    {
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
                g_new[j + 1] = add(g_new[j + 1], multiply(g[j], exp_table[i]));
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
     * @return Syndrome values (size nsym+1). Indexing convention used by this codec:
     *         syndromes[i] = r(α^i) for i = 0..nsym. Valid codewords have
     *         syndromes[0]..syndromes[nsym-1] = 0 (roots of g(x)); syndromes[nsym]
     *         may be nonzero even on valid codewords and is not used for detection.
     */
    std::vector<element_t> rs_calc_syndromes(const std::vector<element_t>& msg, uint8_t nsym) const
    {
        std::vector<element_t> synd(nsym + 1, 0);

        // Evaluate message polynomial at α^i for i=0..nsym
        for (uint8_t i = 0; i <= nsym; ++i) {
            synd[i] = eval_poly(msg, exp_table[i]);
        }

        return synd;
    }

    /**
     * @brief Compute Λ(x) and Ω(x) via the Berlekamp–Massey (BM) decoding step
     *
     * This routine performs the core Reed–Solomon decoding step that turns a
     * sequence of syndromes into:
     *  - Λ(x): error locator polynomial (roots correspond to error locations)
     *  - Ω(x): error evaluator polynomial (used to compute error magnitudes)
     *
     * Conceptual model
     * -----------------
     * The BM algorithm finds the shortest linear recurrence that the syndrome
     * sequence satisfies. In field terms, it finds Λ(x) = 1 + Λ₁x + … + Λ_T x^T
     * such that for j ≥ T the syndromes obey the recurrence
     *
     *   S_j = −(Λ₁ S_{j−1} + Λ₂ S_{j−2} + … + Λ_T S_{j−T}).
     *
     * In GF(2^m), subtraction equals addition (XOR), so the minus sign can be
     * dropped in implementation. The degree T is the number of symbol errors.
     *
     * Implementation sketch
     * ---------------------
     * We iterate through the syndrome indices, maintain the current locator
     * polynomial `err_loc` and a backup `old_loc`. At each step we compute the
     * discrepancy Δ (delta). If Δ ≠ 0, we update `err_loc` using a shifted copy
     * of `old_loc` (corresponds to multiplying by x) scaled by Δ; when a new
     * maximum span is reached, we also scale and swap `old_loc` per BM rules.
     *
     * After Λ(x) is determined, the evaluator is formed by the truncated product
     *
     *   Ω(x) = (S(x) · Λ(x)) mod x^{nsym},
     *
     * where S(x) = S₁x + S₂x² + … collects the nonzero syndromes.
     *
     * @param syndromes Syndromes from rs_calc_syndromes (size nsym+1; BM uses syndromes[1..nsym])
     * @param nsym Number of ECC symbols (designed correction capacity is nsym/2)
     * @return Tuple {Λ(x), Ω(x)} as vectors of coefficients (highest degree first)
     */
    std::tuple<std::vector<element_t>, std::vector<element_t>> rs_find_error_locator(
        const std::vector<element_t>& syndromes, uint8_t nsym) const
    {
        // Berlekamp-Massey algorithm (standard lowest-degree-first convention)
        // Λ(x) = Λ_0 + Λ_1*x + Λ_2*x^2 + ..., stored as [Λ_0, Λ_1, Λ_2, ...]
        // Initial: Λ = [1] (constant 1)

        std::vector<element_t> C = {1};  // Current error locator Λ(x)
        std::vector<element_t> B = {1};  // Previous error locator (backup)
        size_t L = 0;                    // Current number of assumed errors
        element_t b_prev = 1;            // Previous discrepancy
        size_t shift = 1;                // Number of iterations since L changed

        for (uint8_t n = 0; n < nsym; ++n) {
            // Compute discrepancy: d = S_{n+1} + Σ_{i=1}^{L} C_i * S_{n+1-i}
            // d = S_{n+1} + sum_{i=1}^{L} C_i S_{n+1-i}; bound by L (Massey), coeffs by C.size().
            element_t d = syndromes[n + 1];
            for (size_t i = 1; i <= L; ++i) {
                if (i < C.size() && n + 1 >= i) {
                    d = add(d, multiply(C[i], syndromes[n + 1 - i]));
                }
            }

            if (d == 0) {
                // No change needed
                shift++;
            }
            else {
                // T(x) = C(x) - (d/b) * x^shift * B(x)
                std::vector<element_t> T = C;
                element_t coeff = divide(d, b_prev);

                // Ensure T is large enough
                if (T.size() < B.size() + shift) {
                    T.resize(B.size() + shift, 0);
                }

                // T = C - coeff * x^shift * B
                for (size_t i = 0; i < B.size(); ++i) {
                    T[i + shift] = add(T[i + shift], multiply(coeff, B[i]));
                }

                if (2 * L <= n) {
                    // Increase L
                    L = n + 1 - L;
                    B = C;
                    b_prev = d;
                    shift = 1;
                }
                else {
                    shift++;
                }

                C = T;
            }
        }

        // Trim trailing zeros from C
        while (C.size() > 1 && C.back() == 0) {
            C.pop_back();
        }

        // Compute error evaluator Ω(x) = S(x) * Λ(x) mod x^nsym
        // where S(x) = S_1 + S_2*x + S_3*x^2 + ...
        std::vector<element_t> omega(nsym, 0);
        for (size_t i = 0; i < nsym; ++i) {
            for (size_t j = 0; j < C.size() && j <= i; ++j) {
                // S(x) has S_1 at x^0, S_2 at x^1, etc., so syndromes[j+1]
                if (i - j + 1 < syndromes.size()) {
                    omega[i] = add(omega[i], multiply(C[j], syndromes[i - j + 1]));
                }
            }
        }

        return {C, omega};
    }

    /**
     * @brief Same as rs_find_error_locator but uses fixed stack buffers (no heap in BM).
     *
     * Intended for parity testing against rs_find_error_locator and eventual bare-metal use.
     * Throws std::length_error if an internal polynomial would exceed its static capacity
     * (conservative bound; increase k_rs_bm_poly_cap if needed for larger nsym).
     */
    std::tuple<std::vector<element_t>, std::vector<element_t>> rs_find_error_locator_noheap(
        const std::vector<element_t>& syndromes, uint8_t nsym) const
    {
        struct PolyBuf {
            std::array<element_t, k_rs_bm_poly_cap> data{};
            size_t len = 0;

            static PolyBuf one()
            {
                PolyBuf p;
                p.data[0] = 1;
                p.len = 1;
                return p;
            }

            void copy_from(const PolyBuf& o)
            {
                const size_t prev = len;
                for (size_t i = 0; i < o.len; ++i) {
                    data[i] = o.data[i];
                }
                // Clear slots we no longer own (fixed buffer); avoids stale coeffs if len grows again.
                for (size_t i = o.len; i < prev; ++i) {
                    data[i] = 0;
                }
                len = o.len;
            }

            void ensure_len_at_least(size_t new_len, element_t fill)
            {
                if (new_len > k_rs_bm_poly_cap) {
                    throw std::length_error("rs_find_error_locator_noheap: polynomial capacity exceeded");
                }
                while (len < new_len) {
                    data[len++] = fill;
                }
            }

            void trim_trailing_zeros()
            {
                while (len > 1 && data[len - 1] == 0) {
                    --len;
                }
            }

            std::vector<element_t> to_vector() const
            {
                return std::vector<element_t>(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(len));
            }
        };

        PolyBuf C = PolyBuf::one();
        PolyBuf B = PolyBuf::one();
        size_t L = 0;
        element_t b_prev = 1;
        size_t shift = 1;

        for (uint8_t n = 0; n < nsym; ++n) {
            // d = S_{n+1} + sum_{i=1}^{L} C_i S_{n+1-i}; bound recurrence by L (Massey), coeffs by len.
            element_t d = syndromes[n + 1];
            for (size_t i = 1; i <= L; ++i) {
                if (i < C.len && n + 1 >= i) {
                    d = add(d, multiply(C.data[i], syndromes[n + 1 - i]));
                }
            }

            if (d == 0) {
                shift++;
            } else {
                PolyBuf T;
                T.copy_from(C);
                element_t coeff = divide(d, b_prev);

                const size_t need_len = B.len + shift;
                T.ensure_len_at_least(need_len, 0);

                for (size_t i = 0; i < B.len; ++i) {
                    T.data[i + shift] = add(T.data[i + shift], multiply(coeff, B.data[i]));
                }

                if (2 * L <= n) {
                    L = n + 1 - L;
                    B.copy_from(C);
                    b_prev = d;
                    shift = 1;
                } else {
                    shift++;
                }

                C.copy_from(T);
            }
        }

        C.trim_trailing_zeros();

        std::vector<element_t> omega(nsym, 0);
        for (size_t i = 0; i < nsym; ++i) {
            for (size_t j = 0; j < C.len && j <= i; ++j) {
                if (i - j + 1 < syndromes.size()) {
                    omega[i] = add(omega[i], multiply(C.data[j], syndromes[i - j + 1]));
                }
            }
        }

        return {C.to_vector(), omega};
    }

    /**
     * @brief Find error positions using Chien search
     *
     * Chien search evaluates the locator polynomial Λ(x) at successive inverses
     * of field elements. For RS codes, if Λ(α^{-k}) = 0, there's an error at
     * array position (n-1-k).
     *
     * @param err_loc Error locator polynomial Λ(x) (coefficients LOWEST degree first)
     *                Λ(x) = err_loc[0] + err_loc[1]*x + err_loc[2]*x^2 + ...
     * @param msg_len Message length (n)
     * @return Vector of error positions (0-based array indices)
     */
    std::vector<size_t> rs_find_errors(const std::vector<element_t>& err_loc, size_t msg_len) const
    {
        std::vector<size_t> err_pos;

        // Number of errors = degree of error locator polynomial
        size_t num_errors = err_loc.size() - 1;

        if (num_errors == 0 || num_errors > msg_len) {
            return {};
        }

        // Chien search: evaluate Λ(α^{-k}) for k = 0, 1, ..., n-1
        for (size_t k = 0; k < msg_len; ++k) {
            // Compute α^{-k} = α^{field_size - 1 - k}
            element_t x_inv = (k == 0) ? 1 : exp_table[(field_size - 1 - k) % (field_size - 1)];

            // Evaluate Λ(x_inv) = Σ err_loc[i] * x_inv^i
            element_t eval = 0;
            element_t x_pow = 1;  // x_inv^0 = 1

            for (size_t i = 0; i < err_loc.size(); ++i) {
                eval = add(eval, multiply(err_loc[i], x_pow));
                x_pow = multiply(x_pow, x_inv);
            }

            if (eval == 0) {
                // α^{-k} is a root, so error at position (n-1-k)
                size_t pos = msg_len - 1 - k;
                err_pos.push_back(pos);
            }
        }

        // Verify we found the expected number of roots
        if (err_pos.size() != num_errors) {
            return {};  // Mismatch - uncorrectable
        }

        return err_pos;
    }

    /**
     * @brief Correct errors using the Forney algorithm
     *
     * Forney's formula: E = X * Ω(X^{-1}) / Λ'(X^{-1})
     * where X = α^{n-1-pos} for array position pos.
     *
     * @param msg_in Message with errors
     * @param err_pos Error positions (0-based array indices)
     * @param err_loc Error locator Λ(x) (LOWEST degree first)
     * @param err_eval Error evaluator Ω(x) (LOWEST degree first)
     * @return Corrected message
     */
    std::vector<element_t> rs_correct_errors_at_positions(
        const std::vector<element_t>& msg_in, const std::vector<size_t>& err_pos,
        const std::vector<element_t>& err_loc, const std::vector<element_t>& err_eval) const
    {
        std::vector<element_t> msg = msg_in;
        size_t n = msg.size();

        for (size_t pos : err_pos) {
            // For array position pos, the position exponent is j = n-1-pos
            // X = α^j, X^{-1} = α^{-j}
            size_t j = n - 1 - pos;
            element_t X = exp_table[j % (field_size - 1)];
            element_t X_inv = (j == 0) ? 1 : exp_table[(field_size - 1 - j) % (field_size - 1)];

            // Evaluate Ω(X^{-1}) = Σ omega[i] * X_inv^i
            element_t omega_val = 0;
            element_t X_inv_pow = 1;
            for (size_t i = 0; i < err_eval.size(); ++i) {
                omega_val = add(omega_val, multiply(err_eval[i], X_inv_pow));
                X_inv_pow = multiply(X_inv_pow, X_inv);
            }

            // Evaluate Λ'(X^{-1})
            // For Λ(x) = Σ Λ_i * x^i, the formal derivative is Λ'(x) = Σ i*Λ_i * x^{i-1}
            // In GF(2^m), coefficients multiply by i mod 2, so only odd i survive:
            // Λ'(x) = Λ_1 + Λ_3*x^2 + Λ_5*x^4 + ...
            element_t lambda_deriv = 0;
            X_inv_pow = 1;  // Start at X_inv^0 for i=1 term (which contributes Λ_1 * X_inv^0)
            for (size_t i = 1; i < err_loc.size(); i += 2) {  // Only odd indices
                lambda_deriv = add(lambda_deriv, multiply(err_loc[i], X_inv_pow));
                // Next odd index is i+2, need X_inv^{(i+2)-1} = X_inv^{i+1}
                // Currently at X_inv^{i-1}, need to multiply by X_inv^2
                X_inv_pow = multiply(X_inv_pow, multiply(X_inv, X_inv));
            }

            // Forney formula: E = X * Ω(X^{-1}) / Λ'(X^{-1})
            if (lambda_deriv == 0) {
                continue;  // Can't correct this position
            }

            element_t err_mag = divide(multiply(X, omega_val), lambda_deriv);

            // Correct the error
            msg[pos] = add(msg[pos], err_mag);
        }

        return msg;
    }

    /**
     * @brief Complete Reed–Solomon decoding pipeline
     *
     * Steps:
     *  1) Compute syndromes S₁..S_nsym.
     *  2) Run Berlekamp–Massey to obtain Λ(x) and Ω(x).
     *  3) Use Chien search to locate error positions (roots of Λ).
     *  4) Apply Forney algorithm to compute magnitudes and correct the codeword.
     *
     * Returns std::nullopt if the error pattern is beyond the code's capability
     * (e.g., root count mismatch or derivative zero).
     *
     * @param msg Message with errors
     * @param nsym Number of ECC symbols
     * @return Corrected message or std::nullopt if uncorrectable
     */
    std::optional<std::vector<element_t>> rs_correct_errors(const std::vector<element_t>& msg,
                                                            uint8_t nsym) const
    {
        // Calculate syndromes
        auto syndromes = rs_calc_syndromes(msg, nsym);

        // Nonzero in syndromes[0]..syndromes[nsym-1] indicates errors.
        // syndromes[nsym] is not a root of g(x) and may be nonzero on valid codewords.
        bool has_errors = false;
        for (size_t i = 0; i < static_cast<size_t>(nsym); ++i) {
            if (syndromes[i] != 0) {
                has_errors = true;
                break;
            }
        }

        if (!has_errors) {
            return msg;  // No errors to correct
        }

        // Try simple single-error correction first (Peterson decoder)
        // For RS with polynomial c(x) = c_0*x^{n-1} + ... + c_{n-1}:
        // Error e at array position i contributes e*α^{j*(n-1-i)} to S_j
        // So S_2/S_1 = α^{n-1-i}, meaning i = n-1 - log_α(S_2/S_1)
        if (syndromes[1] != 0 && syndromes[2] != 0) {
            element_t ratio = divide(syndromes[2], syndromes[1]);
            size_t n = msg.size();

            // Find exponent k where α^k = ratio, then position = n-1-k
            for (size_t k = 0; k < field_size - 1; ++k) {
                if (exp_table[k] == ratio) {
                    // Check if position is valid
                    if (k >= n) {
                        break;  // Invalid position, try multi-error
                    }

                    size_t pos = n - 1 - k;  // Actual array position

                    // Compute magnitude: e = S_1 / α^{n-1-pos} = S_1 / α^k = S_1 * α^{-k}
                    element_t alpha_neg_k = exp_table[(field_size - 1 - k) % (field_size - 1)];
                    element_t magnitude = multiply(syndromes[1], alpha_neg_k);

                    // Match S_1..S_{nsym-1} to one error (c(α^j)=0 there). S_nsym includes c(α^{nsym})
                    // and must not be checked — same reason as has_errors above.
                    bool matches_single_error = true;
                    for (uint8_t sj = 1; sj < nsym && matches_single_error; ++sj) {
                        element_t expected =
                            multiply(magnitude, exp_table[(sj * k) % (field_size - 1)]);
                        if (syndromes[sj] != expected) {
                            matches_single_error = false;
                        }
                    }
                    if (!matches_single_error) {
                        break;
                    }

                    // Single error confirmed - correct it
                    std::vector<element_t> corrected = msg;
                    corrected[pos] = add(corrected[pos], magnitude);
                    return corrected;
                }
            }
        }

        // Try 2-error correction using brute-force position search
        // For short codewords, this is practical (O(n²) pairs to check)
        if (nsym >= 4 && msg.size() <= 32) {
            size_t n = msg.size();

            // Try all pairs of positions
            for (size_t p1 = 0; p1 < n; ++p1) {
                for (size_t p2 = p1 + 1; p2 < n; ++p2) {
                    // Position exponents: k1 = n-1-p1, k2 = n-1-p2
                    size_t k1 = n - 1 - p1;
                    size_t k2 = n - 1 - p2;

                    element_t X1 = exp_table[k1 % (field_size - 1)];
                    element_t X2 = exp_table[k2 % (field_size - 1)];
                    element_t X1_2 = multiply(X1, X1);
                    element_t X2_2 = multiply(X2, X2);

                    // Solve: e1*X1 + e2*X2 = S1
                    //        e1*X1² + e2*X2² = S2
                    // Cramer's rule: det = X1*X2² - X2*X1² = X1*X2*(X2 - X1)
                    element_t det = multiply(multiply(X1, X2), add(X2, X1));

                    if (det == 0) continue;  // Singular (X1 == X2)

                    // e1 = (S1*X2² - S2*X2) / det = (S1*X2² + S2*X2) / det in GF(2^m)
                    // e2 = (S2*X1 - S1*X1²) / det = (S2*X1 + S1*X1²) / det in GF(2^m)
                    element_t e1 =
                        divide(add(multiply(syndromes[1], X2_2), multiply(syndromes[2], X2)), det);
                    element_t e2 =
                        divide(add(multiply(syndromes[2], X1), multiply(syndromes[1], X1_2)), det);

                    // Skip if either error magnitude is 0 (not a valid 2-error case)
                    if (e1 == 0 || e2 == 0) continue;

                    // Apply candidate correction and verify ALL syndromes become zero
                    std::vector<element_t> candidate = msg;
                    candidate[p1] = add(candidate[p1], e1);
                    candidate[p2] = add(candidate[p2], e2);

                    // Recompute syndromes for the corrected message
                    auto new_syndromes = rs_calc_syndromes(candidate, nsym);

                    // Match has_errors / BM: require S_0..S_{nsym-1} zero (same indices as before)
                    bool all_zero = true;
                    for (uint8_t i = 0; i < nsym && all_zero; ++i) {
                        if (new_syndromes[i] != 0) all_zero = false;
                    }

                    if (all_zero) {
                        return candidate;
                    }
                }
            }

            // Try 3-error correction (O(n³) - still practical for small n; cap matches 2-error path)
            if (nsym >= 6 && n <= 32) {
                for (size_t p1 = 0; p1 < n; ++p1) {
                    for (size_t p2 = p1 + 1; p2 < n; ++p2) {
                        for (size_t p3 = p2 + 1; p3 < n; ++p3) {
                            // Compute X values for each position
                            size_t k1 = n - 1 - p1, k2 = n - 1 - p2, k3 = n - 1 - p3;
                            element_t X1 = exp_table[k1 % (field_size - 1)];
                            element_t X2 = exp_table[k2 % (field_size - 1)];
                            element_t X3 = exp_table[k3 % (field_size - 1)];

                            // Solve 3x3 system: Vandermonde-like matrix
                            // [X1  X2  X3 ][e1]   [S1]
                            // [X1² X2² X3²][e2] = [S2]
                            // [X1³ X2³ X3³][e3]   [S3]
                            element_t X1_2 = multiply(X1, X1), X2_2 = multiply(X2, X2),
                                      X3_2 = multiply(X3, X3);
                            element_t X1_3 = multiply(X1_2, X1), X2_3 = multiply(X2_2, X2),
                                      X3_3 = multiply(X3_2, X3);

                            // Compute 2x2 minors (cofactors) for column expansion
                            // M_ij = det of 2x2 submatrix with row i and col j removed
                            element_t M11 = add(multiply(X2_2, X3_3), multiply(X3_2, X2_3));
                            element_t M21 = add(multiply(X2, X3_3), multiply(X3, X2_3));
                            element_t M31 = add(multiply(X2, X3_2), multiply(X3, X2_2));

                            // Full 3x3 determinant by cofactor expansion along column 1:
                            // det = X1*M11 + X1²*M21 + X1³*M31 (signs are +1 in GF(2))
                            element_t det = add(add(multiply(X1, M11), multiply(X1_2, M21)),
                                                multiply(X1_3, M31));
                            if (det == 0) continue;

                            element_t S1 = syndromes[1], S2 = syndromes[2], S3 = syndromes[3];

                            // e1: replace col 1 with [S1,S2,S3], expand along col 1
                            element_t e1_num =
                                add(add(multiply(S1, M11), multiply(S2, M21)), multiply(S3, M31));
                            element_t e1 = divide(e1_num, det);

                            // For e2 and e3, compute different minors
                            element_t M12 = add(multiply(X1_2, X3_3), multiply(X3_2, X1_3));
                            element_t M22 = add(multiply(X1, X3_3), multiply(X3, X1_3));
                            element_t M32 = add(multiply(X1, X3_2), multiply(X3, X1_2));

                            element_t M13 = add(multiply(X1_2, X2_3), multiply(X2_2, X1_3));
                            element_t M23 = add(multiply(X1, X2_3), multiply(X2, X1_3));
                            element_t M33 = add(multiply(X1, X2_2), multiply(X2, X1_2));

                            // e2: replace col 2 with [S1,S2,S3], expand along col 2
                            element_t e2_num =
                                add(add(multiply(S1, M12), multiply(S2, M22)), multiply(S3, M32));
                            element_t e2 = divide(e2_num, det);

                            // e3: replace col 3 with [S1,S2,S3], expand along col 3
                            element_t e3_num =
                                add(add(multiply(S1, M13), multiply(S2, M23)), multiply(S3, M33));
                            element_t e3 = divide(e3_num, det);

                            if (e1 == 0 || e2 == 0 || e3 == 0) continue;

                            // Apply and verify
                            std::vector<element_t> candidate = msg;
                            candidate[p1] = add(candidate[p1], e1);
                            candidate[p2] = add(candidate[p2], e2);
                            candidate[p3] = add(candidate[p3], e3);

                            auto new_syndromes = rs_calc_syndromes(candidate, nsym);
                            bool all_zero = true;
                            for (uint8_t i = 0; i < nsym && all_zero; ++i) {
                                if (new_syndromes[i] != 0) all_zero = false;
                            }
                            if (all_zero) return candidate;
                        }
                    }
                }
            }
        }

    full_decode:
        // Fall back to full Berlekamp-Massey for multiple errors
#if RADML_RS_BM_NOHEAP
        auto [err_loc, err_eval] = rs_find_error_locator_noheap(syndromes, nsym);
#else
        auto [err_loc, err_eval] = rs_find_error_locator(syndromes, nsym);
#endif

        // Find error positions
        auto err_pos = rs_find_errors(err_loc, msg.size());

        if (err_pos.empty()) {
            return std::nullopt;  // Uncorrectable errors
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
    template <typename RNG>
    element_t random_element(RNG& rng) const
    {
        std::uniform_int_distribution<element_t> dist(0, field_mask);
        return dist(rng);
    }

   private:
    /// Max coefficients for no-heap BM temporary polynomials (shift + degree bound).
    static constexpr size_t k_rs_bm_poly_cap = 256;

    std::array<element_t, static_cast<size_t>(field_size)> exp_table;  // α^i lookup
    std::array<element_t, static_cast<size_t>(field_size)> log_table;  // log_α(i) lookup

    /**
     * @brief Initialize lookup tables for multiplication and division
     */
    void initialize_tables()
    {
        // Initialize exp and log tables for efficient multiplication
        element_t x = 1;

        // Clear tables first
        for (uint32_t i = 0; i < field_size; ++i) {
            exp_table[i] = 0;
            log_table[i] = 0;
        }

        for (uint32_t i = 0; i < field_size - 1; ++i) {
            exp_table[i] = x;

            // Multiply by α in GF(2^m)
            x = multiply_no_lut(x, 2);
            if (x >= field_size) {
                x ^= (primitive_poly & field_mask);
            }
        }

        // Set the last element
        exp_table[field_size - 1] = exp_table[0];

        // Generate log table
        log_table[0] = 0;  // log(0) is undefined, set to 0 for convenience

        for (uint32_t i = 0; i < field_size - 1; ++i) {
            log_table[exp_table[i]] = static_cast<element_t>(i);
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
    element_t multiply_no_lut(element_t a, element_t b) const
    {
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
using GF16 = GaloisField<4, 0x13>;      // GF(2^4) with polynomial x^4 + x + 1
using GF256 = GaloisField<8, 0x11d>;    // GF(2^8) with polynomial x^8 + x^4 + x^3 + x^2 + 1
using GF1024 = GaloisField<10, 0x409>;  // GF(2^10) with polynomial x^10 + x^3 + 1

}  // namespace neural
}  // namespace rad_ml

#endif  // RAD_ML_NEURAL_GALOIS_FIELD_HPP
