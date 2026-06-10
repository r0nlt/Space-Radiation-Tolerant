/**
 * @file syndrome_index_probe.cpp
 * @brief Configurable RS syndrome / BM decode probe (manual diagnostic, not CTest)
 *
 * Usage:
 *   syndrome_index_probe [--trials N] [--errors K] [--seed N] [--value V]
 *                        [--ecc 4|6|8|16] [--fixed pos:mask,...] [--help]
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../../include/rad_ml/neural/advanced_reed_solomon.hpp"
#include "../../include/rad_ml/neural/galois_field.hpp"

using rad_ml::neural::AdvancedReedSolomon;
using rad_ml::neural::GF256;

struct ProbeConfig {
    int trials = 50;
    int error_count = 4;
    unsigned seed = 42;
    uint8_t value = 0x42;
    uint8_t ecc = 16;
    std::vector<std::pair<size_t, uint8_t>> fixed_flips;
    bool show_help = false;
};

static void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog
              << " [--trials N] [--errors K] [--seed N] [--value V]\n"
                 "       [--ecc 4|6|8|16] [--fixed pos:mask,...] [--help]\n"
                 "\n"
                 "  --trials N     Random decode trials (default: 50)\n"
                 "  --errors K     Symbol errors per random trial (default: 4)\n"
                 "  --seed N       RNG seed (default: 42)\n"
                 "  --value V      Data byte to encode (hex, default: 0x42)\n"
                 "  --ecc N        ECC symbols: 4, 6, 8, or 16 (default: 16)\n"
                 "  --fixed LIST   Fixed XOR pattern pos:mask,... for BM degree check\n"
                 "                 (default: 0:0x11,3:0x22,7:0x33,12:0x44)\n";
}

static bool parse_hex_byte(const std::string& s, uint8_t& out)
{
    char* end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 0);
    if (end == s.c_str() || *end != '\0' || v > 0xFF) {
        return false;
    }
    out = static_cast<uint8_t>(v);
    return true;
}

static bool parse_fixed_list(const std::string& list, std::vector<std::pair<size_t, uint8_t>>& out)
{
    out.clear();
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto colon = item.find(':');
        if (colon == std::string::npos) {
            return false;
        }
        char* end = nullptr;
        const unsigned long pos = std::strtoul(item.substr(0, colon).c_str(), &end, 0);
        if (end == item.c_str() || pos > 255) {
            return false;
        }
        uint8_t mask = 0;
        if (!parse_hex_byte(item.substr(colon + 1), mask)) {
            return false;
        }
        out.emplace_back(static_cast<size_t>(pos), mask);
    }
    return !out.empty();
}

static ProbeConfig parse_args(int argc, char** argv)
{
    ProbeConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cfg.show_help = true;
            return cfg;
        }
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--trials") {
            cfg.trials = std::stoi(need_value("--trials"));
        } else if (arg == "--errors") {
            cfg.error_count = std::stoi(need_value("--errors"));
        } else if (arg == "--seed") {
            cfg.seed = static_cast<unsigned>(std::stoul(need_value("--seed")));
        } else if (arg == "--value") {
            if (!parse_hex_byte(need_value("--value"), cfg.value)) {
                throw std::runtime_error("Invalid --value (use hex, e.g. 0x42)");
            }
        } else if (arg == "--ecc") {
            cfg.ecc = static_cast<uint8_t>(std::stoul(need_value("--ecc")));
        } else if (arg == "--fixed") {
            if (!parse_fixed_list(need_value("--fixed"), cfg.fixed_flips)) {
                throw std::runtime_error("Invalid --fixed (use pos:mask,pos:mask,...)");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (cfg.fixed_flips.empty()) {
        cfg.fixed_flips = {{0, 0x11}, {3, 0x22}, {7, 0x33}, {12, 0x44}};
    }
    return cfg;
}

template <uint8_t Ecc>
static int run_probe(const ProbeConfig& cfg)
{
    using RS = AdvancedReedSolomon<uint8_t, 8, Ecc>;
    RS rs;
    GF256 gf;
    constexpr uint8_t nsym = Ecc;
    constexpr size_t n_sym = RS::total_symbols;

    std::vector<uint8_t> enc = rs.encode(cfg.value);
    std::vector<uint8_t> codew(n_sym, 0);
    for (size_t i = 0; i < n_sym && i < enc.size(); ++i) {
        codew[i] = enc[i];
    }

    auto clean_synd = gf.rs_calc_syndromes(codew, nsym);
    std::cout << "RS(8," << static_cast<int>(Ecc) << ") value=0x" << std::hex
              << static_cast<int>(cfg.value) << std::dec << "\n";
    std::cout << "Clean codeword syndromes (size " << clean_synd.size() << "):\n  ";
    for (size_t i = 0; i < clean_synd.size(); ++i) {
        std::cout << "[" << i << "]=" << static_cast<int>(clean_synd[i]) << " ";
    }
    std::cout << "\n  index0_is_zero=" << (clean_synd[0] == 0 ? "yes" : "no")
              << " (on valid codeword, r(alpha^0)=0)\n";

    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<int> err_mag(1, 255);
    int pass = 0;
    for (int tr = 0; tr < cfg.trials; ++tr) {
        std::vector<uint8_t> corrupted = codew;
        std::set<size_t> used;
        for (int e = 0; e < cfg.error_count; ++e) {
            size_t pos = rng() % n_sym;
            while (!used.insert(pos).second) {
                pos = (pos + 1) % n_sym;
            }
            corrupted[pos] = static_cast<uint8_t>(corrupted[pos] ^
                                                  static_cast<uint8_t>(err_mag(rng)));
        }
        auto dec = rs.decode(corrupted);
        if (dec.has_value() && *dec == cfg.value) {
            ++pass;
        }
    }
    std::cout << "Random decode k=" << cfg.error_count << ": " << pass << "/" << cfg.trials
              << " passed (seed=" << cfg.seed << ")\n";

    std::vector<uint8_t> corrupted = codew;
    for (const auto& [pos, mask] : cfg.fixed_flips) {
        if (pos < n_sym) {
            corrupted[pos] = static_cast<uint8_t>(corrupted[pos] ^ mask);
        }
    }
    auto synd = gf.rs_calc_syndromes(corrupted, nsym);

    auto bm_direct = gf.rs_find_error_locator(synd, nsym);
    std::vector<uint8_t> padded(static_cast<size_t>(nsym) + 1, 0);
    for (uint8_t i = 0; i < nsym; ++i) {
        padded[i + 1] = synd[i];
    }
    auto bm_padded = gf.rs_find_error_locator(padded, nsym);

    const auto& loc_direct = std::get<0>(bm_direct);
    const auto& loc_padded = std::get<0>(bm_padded);
    std::cout << "BM locator degree (direct syndromes[0..nsym-1]): "
              << (loc_direct.empty() ? 0 : loc_direct.size() - 1) << "\n";
    std::cout << "BM locator degree (legacy padded [1..nsym]): "
              << (loc_padded.empty() ? 0 : loc_padded.size() - 1) << "\n";

    auto dec_fixed = rs.decode(corrupted);
    const bool ok = dec_fixed.has_value() && *dec_fixed == cfg.value;
    std::cout << "Fixed-pattern decode (" << cfg.fixed_flips.size() << " errors): "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

static int dispatch_probe(const ProbeConfig& cfg)
{
    switch (cfg.ecc) {
        case 4:
            return run_probe<4>(cfg);
        case 6:
            return run_probe<6>(cfg);
        case 8:
            return run_probe<8>(cfg);
        case 16:
            return run_probe<16>(cfg);
        default:
            std::cerr << "Unsupported --ecc " << static_cast<int>(cfg.ecc)
                      << " (use 4, 6, 8, or 16)\n";
            return 1;
    }
}

int main(int argc, char** argv)
{
    try {
        const ProbeConfig cfg = parse_args(argc, argv);
        if (cfg.show_help) {
            print_usage(argv[0]);
            return 0;
        }
        if (cfg.trials < 0 || cfg.error_count < 0) {
            std::cerr << "--trials and --errors must be non-negative\n";
            return 1;
        }
        return dispatch_probe(cfg);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
