/**
 * One-off probe: syndrome indexing + RS(8,16) k=4 decode
 */
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "../../include/rad_ml/neural/advanced_reed_solomon.hpp"
#include "../../include/rad_ml/neural/galois_field.hpp"

using rad_ml::neural::AdvancedReedSolomon;
using rad_ml::neural::GF256;

int main()
{
    using RS = AdvancedReedSolomon<uint8_t, 8, 16>;
    RS rs;
    GF256 gf;
    constexpr uint8_t nsym = 16;
    constexpr size_t n_sym = RS::total_symbols;

    const uint8_t original = 0x42;
    std::vector<uint8_t> enc = rs.encode(original);
    std::vector<uint8_t> codew(n_sym, 0);
    for (size_t i = 0; i < n_sym && i < enc.size(); ++i) {
        codew[i] = enc[i];
    }

    auto clean_synd = gf.rs_calc_syndromes(codew, nsym);
    std::cout << "Clean codeword syndromes (size " << clean_synd.size() << "):\n  ";
    for (size_t i = 0; i < clean_synd.size(); ++i) {
        std::cout << "[" << i << "]=" << static_cast<int>(clean_synd[i]) << " ";
    }
    std::cout << "\n  index0_is_zero=" << (clean_synd[0] == 0 ? "yes" : "no")
              << " (pad vs r(alpha^0))\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> err_mag(1, 255);
    int pass = 0;
    constexpr int trials = 50;
    for (int tr = 0; tr < trials; ++tr) {
        std::vector<uint8_t> corrupted = codew;
        std::set<size_t> used;
        constexpr int k = 4;
        for (int e = 0; e < k; ++e) {
            size_t pos = rng() % n_sym;
            while (!used.insert(pos).second) {
                pos = (pos + 1) % n_sym;
            }
            corrupted[pos] = static_cast<uint8_t>(corrupted[pos] ^
                                                  static_cast<uint8_t>(err_mag(rng)));
        }
        auto dec = rs.decode(corrupted);
        if (dec.has_value() && *dec == original) {
            ++pass;
        }
    }
    std::cout << "RS(8,16) k=4 decode: " << pass << "/" << trials << " passed\n";

    // BM with shifted vs direct syndromes on first corrupted trial
    std::vector<uint8_t> corrupted = codew;
    corrupted[0] ^= 0x11;
    corrupted[3] ^= 0x22;
    corrupted[7] ^= 0x33;
    corrupted[12] ^= 0x44;
    auto synd = gf.rs_calc_syndromes(corrupted, nsym);

    auto bm_direct = gf.rs_find_error_locator(synd, nsym);
    std::vector<uint8_t> padded(static_cast<size_t>(nsym) + 1, 0);
    padded[0] = 0;
    for (uint8_t i = 1; i <= nsym; ++i) {
        padded[i] = synd[i];
    }
    auto bm_padded = gf.rs_find_error_locator(padded, nsym);

    const auto& loc_direct = std::get<0>(bm_direct);
    const auto& loc_padded = std::get<0>(bm_padded);
    std::cout << "BM locator degree (direct syndromes): " << (loc_direct.empty() ? 0 : loc_direct.size() - 1)
              << "\n";
    std::cout << "BM locator degree (padded [0]=0, [1..nsym]=synd): "
              << (loc_padded.empty() ? 0 : loc_padded.size() - 1) << "\n";

    auto dec4 = rs.decode(corrupted);
    std::cout << "Fixed 4-error decode: " << (dec4.has_value() && *dec4 == original ? "PASS" : "FAIL")
              << "\n";
    return 0;
}
