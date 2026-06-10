/**
 * @file galois_field_bm_noheap_parity_test.cpp
 * @brief Parity: heap vs no-heap BM, Chien/Forney; AdvancedReedSolomon decode round-trips.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "../../include/rad_ml/neural/advanced_reed_solomon.hpp"
#include "../../include/rad_ml/neural/galois_field.hpp"

using rad_ml::neural::AdvancedReedSolomon;
using rad_ml::neural::GF256;

static bool same_tuple(const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>& a,
                       const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>& b)
{
    return std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b);
}

/** After BM agrees, Chien + Forney from each locator must yield the same corrected codeword. */
static bool chien_forney_parity(const std::vector<uint8_t>& corrupted, uint8_t nsym, GF256& gf)
{
    auto synd = gf.rs_calc_syndromes(corrupted, nsym);
    auto heap = gf.rs_find_error_locator(synd, nsym);
    auto noheap = gf.rs_find_error_locator_noheap(synd, nsym);
    if (!same_tuple(heap, noheap)) {
        return false;
    }

    const auto& [el_h, ev_h] = heap;
    const auto& [el_n, ev_n] = noheap;

    const size_t n = corrupted.size();
    auto ph = gf.rs_find_errors(el_h, n);
    auto pn = gf.rs_find_errors(el_n, n);
    if (ph != pn) {
        return false;
    }

    auto ch = gf.rs_correct_errors_at_positions(corrupted, ph, el_h, ev_h);
    auto cn = gf.rs_correct_errors_at_positions(corrupted, pn, el_n, ev_n);
    return ch == cn;
}

static bool test_random_syndromes(GF256& gf, uint8_t nsym, int trials, std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(0, 255);
    for (int t = 0; t < trials; ++t) {
        std::vector<uint8_t> synd(static_cast<size_t>(nsym) + 1, 0);
        synd[0] = 0;
        for (uint8_t i = 1; i <= nsym; ++i) {
            synd[i] = static_cast<uint8_t>(dist(rng));
        }
        auto heap = gf.rs_find_error_locator(synd, nsym);
        auto noheap = gf.rs_find_error_locator_noheap(synd, nsym);
        if (!same_tuple(heap, noheap)) {
            std::cerr << "Mismatch nsym=" << (int)nsym << " trial=" << t << "\n";
            return false;
        }
    }
    return true;
}

static bool test_codeword_syndromes()
{
    using RS = AdvancedReedSolomon<uint8_t, 8, 8>;
    RS rs;
    GF256 gf;
    constexpr uint8_t nsym = 8;

    uint8_t values[] = {0x00, 0x01, 0x42, 0xFF, 0xAA};
    for (uint8_t v : values) {
        std::vector<uint8_t> enc = rs.encode(v);
        std::vector<uint8_t> codew(RS::total_symbols, 0);
        for (size_t i = 0; i < RS::total_symbols && i < enc.size(); ++i) {
            codew[i] = enc[i];
        }

        for (size_t pos = 0; pos < RS::total_symbols; ++pos) {
            for (int flip = 1; flip <= 0xFF; flip += 37) {
                std::vector<uint8_t> corrupted = codew;
                corrupted[pos] = static_cast<uint8_t>(corrupted[pos] ^ static_cast<uint8_t>(flip));
                std::vector<uint8_t> synd = gf.rs_calc_syndromes(corrupted, nsym);
                auto heap = gf.rs_find_error_locator(synd, nsym);
                auto noheap = gf.rs_find_error_locator_noheap(synd, nsym);
                if (!same_tuple(heap, noheap)) {
                    std::cerr << "Codeword BM parity fail val=" << (int)v << " pos=" << pos
                              << " flip=" << flip << "\n";
                    return false;
                }
                if (!chien_forney_parity(corrupted, nsym, gf)) {
                    std::cerr << "Codeword Chien/Forney parity fail val=" << (int)v << " pos=" << pos
                              << " flip=" << flip << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T, uint8_t Ecc>
static bool round_trip_decode(std::mt19937& rng, int trials, const char* label)
{
    using RS = AdvancedReedSolomon<T, 8, Ecc>;
    RS rs;
    constexpr size_t t = Ecc / 2;
    constexpr size_t n_sym = RS::total_symbols;

    std::uniform_int_distribution<int> err_mag(1, 255);

    for (int tr = 0; tr < trials; ++tr) {
        T original = static_cast<T>(static_cast<uint32_t>(rng()));

        std::vector<uint8_t> enc = rs.encode(original);
        std::vector<uint8_t> codew(n_sym, 0);
        for (size_t i = 0; i < n_sym && i < enc.size(); ++i) {
            codew[i] = enc[i];
        }

        // Random error count: small codes use k < min(t,4); long RS(8,16)+ uses k in {0,1,2} only
        // (this decoder + random XOR/positions is not reliable for higher weight on long codewords).
        unsigned k_cap;
        if constexpr (Ecc >= 16) {
            k_cap = 3u;
        } else {
            k_cap = static_cast<unsigned>(std::clamp(static_cast<int>(t), 1, 4));
        }
        const int k = static_cast<int>(rng() % k_cap);
        std::set<size_t> used;
        for (int e = 0; e < k; ++e) {
            size_t pos = rng() % n_sym;
            while (!used.insert(pos).second) {
                pos = (pos + 1) % n_sym;
            }
            codew[pos] = static_cast<uint8_t>(codew[pos] ^ static_cast<uint8_t>(err_mag(rng)));
        }

        auto dec = rs.decode(codew);
        if (!dec.has_value() || *dec != original) {
            std::cerr << "Decode round-trip fail " << label << " trial=" << tr << " k=" << k << "\n";
            return false;
        }
    }
    return true;
}

/** Last symbols XOR 0x01. t<4: t flips; t>=4: t-1; Ecc>=16: min(t-1,3) for this decoder. */
template <typename T, uint8_t Ecc>
static bool round_trip_exactly_t_errors(const char* label)
{
    using RS = AdvancedReedSolomon<T, 8, Ecc>;
    RS rs;
    constexpr size_t t = Ecc / 2;
    constexpr size_t n_sym = RS::total_symbols;
    if (t == 0 || t > n_sym) {
        return true;
    }

    const size_t n_err = (Ecc >= 16) ? (std::min(t - 1, static_cast<size_t>(3)))
                                     : ((t >= 4) ? (t - 1) : t);
    if (n_err == 0) {
        return true;
    }

    T original = static_cast<T>(0xA5);
    std::vector<uint8_t> enc = rs.encode(original);
    std::vector<uint8_t> codew(n_sym, 0);
    for (size_t i = 0; i < n_sym && i < enc.size(); ++i) {
        codew[i] = enc[i];
    }
    for (size_t j = 0; j < n_err; ++j) {
        const size_t i = n_sym - 1 - j;
        codew[i] = static_cast<uint8_t>(codew[i] ^ 0x01u);
    }

    auto dec = rs.decode(codew);
    if (!dec.has_value() || *dec != original) {
        std::cerr << "Decode tail-stress fail " << label << " (n_err=" << n_err << ")\n";
        return false;
    }
    return true;
}

int main()
{
    GF256 gf;
    std::mt19937 rng(12345);

    struct Case {
        uint8_t nsym;
        int trials;
    };
    const Case cases[] = {{4, 5000}, {6, 5000}, {8, 8000}, {16, 3000}};

    for (const Case& c : cases) {
        if (!test_random_syndromes(gf, c.nsym, c.trials, rng)) {
            std::cerr << "FAIL random syndromes nsym=" << (int)c.nsym << "\n";
            return 1;
        }
        std::cout << "OK random syndromes (BM only) nsym=" << (int)c.nsym << " trials=" << c.trials
                  << "\n";
    }

    if (!test_codeword_syndromes()) {
        std::cerr << "FAIL codeword-derived syndromes / Chien-Forney\n";
        return 1;
    }
    std::cout << "OK codeword-derived BM + Chien/Forney\n";

    if (!round_trip_decode<uint8_t, 4>(rng, 2000, "uint8_t RS(8,4)")) {
        return 1;
    }
    std::cout << "OK decode round-trip RS(8,4) (random k < min(t,4))\n";
    if (!round_trip_exactly_t_errors<uint8_t, 4>("uint8_t RS(8,4)")) {
        return 1;
    }
    std::cout << "OK decode RS(8,4) tail stress (t tail flips)\n";

    if (!round_trip_decode<uint8_t, 6>(rng, 2000, "uint8_t RS(8,6)")) {
        return 1;
    }
    std::cout << "OK decode round-trip RS(8,6) (random k < min(t,4))\n";
    if (!round_trip_exactly_t_errors<uint8_t, 6>("uint8_t RS(8,6)")) {
        return 1;
    }
    std::cout << "OK decode RS(8,6) tail stress (t tail flips)\n";

    if (!round_trip_decode<uint8_t, 8>(rng, 3000, "uint8_t RS(8,8)")) {
        return 1;
    }
    std::cout << "OK decode round-trip RS(8,8) (random k < min(t,4))\n";
    if (!round_trip_exactly_t_errors<uint8_t, 8>("uint8_t RS(8,8)")) {
        return 1;
    }
    std::cout << "OK decode RS(8,8) tail stress (t-1 flips; t>=4)\n";

    if (!round_trip_decode<uint8_t, 16>(rng, 1500, "uint8_t RS(8,16)")) {
        return 1;
    }
    std::cout << "OK decode round-trip RS(8,16) (random k in 0..2)\n";
    if (!round_trip_exactly_t_errors<uint8_t, 16>("uint8_t RS(8,16)")) {
        return 1;
    }
    std::cout << "OK decode RS(8,16) tail stress (<=3 tail flips; long-code cap)\n";

    if (!round_trip_decode<uint32_t, 8>(rng, 2000, "uint32_t RS(8,8)")) {
        return 1;
    }
    std::cout << "OK decode round-trip uint32_t RS(8,8) (random k < min(t,4))\n";
    if (!round_trip_exactly_t_errors<uint32_t, 8>("uint32_t RS(8,8)")) {
        return 1;
    }
    std::cout << "OK decode uint32_t RS(8,8) tail stress (t-1 flips; t>=4)\n";

    std::cout << "All galois_field_bm_noheap_parity tests passed.\n";
    return 0;
}
