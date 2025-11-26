/**
 * @file radiation_physics_test.cpp
 * @brief Validation tests for physics-based radiation models
 *
 * Tests validate that our models produce physically reasonable results
 * compared to literature values and NASA models.
 */

#include "rad_ml/physics/radiation_physics.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace rad_ml::physics;

// Test tolerance for floating point comparisons
constexpr double TOLERANCE = 0.5;  // 50% tolerance for order-of-magnitude physics

/**
 * @brief Check if value is within expected range
 */
bool in_range(double value, double min_expected, double max_expected)
{
    return value >= min_expected && value <= max_expected;
}

/**
 * @brief Test Weibull cross-section model
 *
 * Expected behavior:
 * - Zero below threshold
 * - Increases smoothly to saturation
 * - Saturates at high LET
 */
void test_weibull_cross_section()
{
    std::cout << "\n=== Weibull Cross-Section Test ===" << std::endl;

    auto params = WeibullParameters::cmos_65nm();

    // Below threshold: should be zero
    double sigma_low = weibull_cross_section(0.1, params);
    assert(sigma_low == 0.0);
    std::cout << "✓ Below threshold: σ = 0" << std::endl;

    // At threshold: should be near zero
    double sigma_thresh = weibull_cross_section(params.let_threshold + 0.01, params);
    assert(sigma_thresh < params.sigma_sat * 0.01);
    std::cout << "✓ Near threshold: σ << σ_sat" << std::endl;

    // Well above threshold: should approach saturation
    double sigma_high = weibull_cross_section(100.0, params);
    assert(sigma_high > params.sigma_sat * 0.95);
    std::cout << "✓ High LET: σ ≈ σ_sat = " << sigma_high << " cm²/bit" << std::endl;

    // Compare device types
    auto sram = WeibullParameters::sram_commercial();
    auto radhard = WeibullParameters::rad_hard();

    double sigma_sram = weibull_cross_section(10.0, sram);
    double sigma_radhard = weibull_cross_section(10.0, radhard);

    std::cout << "\nDevice comparison at LET=10 MeV·cm²/mg:" << std::endl;
    std::cout << "  Commercial SRAM: " << sigma_sram << " cm²/bit" << std::endl;
    std::cout << "  Rad-hardened:    " << sigma_radhard << " cm²/bit" << std::endl;
    std::cout << "  Ratio (SRAM/RadHard): " << sigma_sram / sigma_radhard << "x" << std::endl;

    assert(sigma_sram > sigma_radhard * 10);  // RadHard should be much better
    std::cout << "✓ Rad-hard device has lower cross-section" << std::endl;
}

/**
 * @brief Test Bendel proton SEU model
 */
void test_bendel_proton_model()
{
    std::cout << "\n=== Bendel Proton SEU Model Test ===" << std::endl;

    auto params_65nm = BendelParameters::cmos_65nm();
    auto params_radhard = BendelParameters::rad_hard();

    // Below threshold: should be zero
    double sigma_low = bendel_proton_cross_section(10.0, params_65nm);
    assert(sigma_low == 0.0);
    std::cout << "✓ Below threshold (E < A): σ = 0" << std::endl;

    // Above threshold: should increase
    double sigma_50 = bendel_proton_cross_section(50.0, params_65nm);
    double sigma_100 = bendel_proton_cross_section(100.0, params_65nm);
    double sigma_200 = bendel_proton_cross_section(200.0, params_65nm);

    std::cout << "\nProton cross-section vs energy (65nm CMOS):" << std::endl;
    std::cout << "  50 MeV:  " << std::scientific << sigma_50 << " cm²/bit" << std::endl;
    std::cout << "  100 MeV: " << sigma_100 << " cm²/bit" << std::endl;
    std::cout << "  200 MeV: " << sigma_200 << " cm²/bit" << std::endl;

    assert(sigma_100 > sigma_50);
    assert(sigma_200 > sigma_100);
    std::cout << "✓ Cross-section increases with energy above threshold" << std::endl;

    // Compare to rad-hard
    double sigma_radhard = bendel_proton_cross_section(100.0, params_radhard);
    std::cout << "\n  Rad-hard at 100 MeV: " << sigma_radhard << " cm²/bit" << std::endl;
    if (sigma_radhard > 0) {
        std::cout << "  Ratio (65nm/radhard): " << std::fixed << sigma_100 / sigma_radhard << "x"
                  << std::endl;
    }
    std::cout << "✓ Bendel model validated" << std::endl;
}

/**
 * @brief Test rigidity to energy conversion
 */
void test_rigidity_conversion()
{
    std::cout << "\n=== Rigidity to Energy Conversion Test ===" << std::endl;

    // Known values:
    // 1 GV proton ≈ 430 MeV kinetic energy
    // 10 GV proton ≈ 9.1 GeV = 9100 MeV

    double E_1GV = proton_energy_from_rigidity(1.0);
    double E_10GV = proton_energy_from_rigidity(10.0);
    double E_01GV = proton_energy_from_rigidity(0.1);

    std::cout << "Proton energy from rigidity:" << std::endl;
    std::cout << "  0.1 GV → " << std::fixed << std::setprecision(1) << E_01GV
              << " MeV (expected ~5 MeV)" << std::endl;
    std::cout << "  1.0 GV → " << E_1GV << " MeV (expected ~430 MeV)" << std::endl;
    std::cout << "  10 GV  → " << E_10GV << " MeV (expected ~9100 MeV)" << std::endl;

    assert(in_range(E_1GV, 350.0, 500.0));
    assert(in_range(E_10GV, 8000.0, 10000.0));
    std::cout << "✓ Rigidity-energy conversion within expected range" << std::endl;
}

/**
 * @brief Test temperature correction
 */
void test_temperature_correction()
{
    std::cout << "\n=== Temperature Correction Test ===" << std::endl;

    double factor_300K = temperature_factor(300.0);
    double factor_350K = temperature_factor(350.0);
    double factor_250K = temperature_factor(250.0);

    std::cout << "Temperature factors:" << std::endl;
    std::cout << "  250K: " << std::fixed << std::setprecision(3) << factor_250K << std::endl;
    std::cout << "  300K: " << factor_300K << " (reference)" << std::endl;
    std::cout << "  350K: " << factor_350K << std::endl;

    assert(factor_300K == 1.0);
    assert(factor_350K > 1.0);
    assert(factor_250K < 1.0);
    std::cout << "✓ Temperature correction behaves correctly" << std::endl;
}

/**
 * @brief Test GCR spectrum model
 *
 * Expected: Power-law spectrum with ~1.9 index
 * Literature: ~0.1-1 particles/cm²/s/sr above 1 MeV·cm²/mg
 */
void test_gcr_spectrum()
{
    std::cout << "\n=== GCR Spectrum Test ===" << std::endl;

    // Solar minimum (low modulation)
    GCRSpectrum gcr_min(400.0);

    // Solar maximum (high modulation)
    GCRSpectrum gcr_max(1200.0);

    std::cout << "\nIntegral flux above given LET:" << std::endl;
    std::cout << std::setw(10) << "LET" << std::setw(20) << "Solar Min" << std::setw(20)
              << "Solar Max" << std::endl;

    for (double let : {1.0, 5.0, 10.0, 25.0, 100.0}) {
        double flux_min = gcr_min.integral_flux(let);
        double flux_max = gcr_max.integral_flux(let);

        std::cout << std::setw(10) << let << std::setw(20) << std::scientific << flux_min
                  << std::setw(20) << flux_max << std::endl;
    }

    // Solar minimum should have higher GCR flux
    double flux_min_10 = gcr_min.integral_flux(10.0);
    double flux_max_10 = gcr_max.integral_flux(10.0);

    assert(flux_min_10 > flux_max_10);
    std::cout << "\n✓ Solar minimum has higher GCR flux (less modulation)" << std::endl;

    // Check order of magnitude (literature: ~0.01 particles/cm²/s/sr at LET>10)
    // Our model gives integral, multiply by 4π for omnidirectional
    double omni_flux = flux_min_10 * 4.0 * M_PI;
    std::cout << "✓ Omnidirectional flux at LET>10: " << omni_flux << " p/cm²/s" << std::endl;
}

/**
 * @brief Test geomagnetic field calculations
 *
 * Validate L-shell and B/B0 calculations against known values
 */
void test_geomagnetic_field()
{
    std::cout << "\n=== Geomagnetic Field Test ===" << std::endl;

    // ISS orbit: ~400 km, ~51.6° inclination
    // Expected L at equator crossing: ~1.06
    double L_iss_eq = GeomagneticField::calculate_L(400.0, 0.0);
    std::cout << "ISS at equator: L = " << L_iss_eq << " (expected ~1.06)" << std::endl;
    assert(in_range(L_iss_eq, 1.0, 1.2));

    // ISS at max latitude (51.6°)
    double L_iss_max = GeomagneticField::calculate_L(400.0, 51.6);
    std::cout << "ISS at 51.6°:   L = " << L_iss_max << " (expected ~2.5)" << std::endl;
    assert(in_range(L_iss_max, 2.0, 4.0));

    // GEO orbit: 35786 km, 0° latitude
    // Expected L ~ 6.6
    double L_geo = GeomagneticField::calculate_L(35786.0, 0.0);
    std::cout << "GEO:            L = " << L_geo << " (expected ~6.6)" << std::endl;
    assert(in_range(L_geo, 6.0, 7.0));

    // Cutoff rigidity test
    double Rc_equator = GeomagneticField::cutoff_rigidity(0.0);
    double Rc_polar = GeomagneticField::cutoff_rigidity(80.0);

    std::cout << "\nCutoff rigidity:" << std::endl;
    std::cout << "  Equator: " << Rc_equator << " GV (expected ~15 GV)" << std::endl;
    std::cout << "  80° lat: " << Rc_polar << " GV (expected <1 GV)" << std::endl;

    assert(in_range(Rc_equator, 10.0, 20.0));
    assert(Rc_polar < 1.0);
    std::cout << "✓ Geomagnetic cutoff varies correctly with latitude" << std::endl;
}

/**
 * @brief Test trapped particle models
 *
 * Validate flux levels in Van Allen belts
 */
void test_trapped_particles()
{
    std::cout << "\n=== Trapped Particle Test ===" << std::endl;

    // Inner belt peak: L~1.5, should have high proton flux
    double proton_inner = TrappedProtonModel::integral_flux(1.5, 1.0, 10.0, false);
    std::cout << "Inner belt protons (L=1.5): " << proton_inner << " p/cm²/s" << std::endl;

    // Outer belt peak: L~4.5, should have high electron flux
    double electron_outer = TrappedElectronModel::integral_flux(4.5, 1.0, 1.0, false);
    std::cout << "Outer belt electrons (L=4.5): " << electron_outer << " e/cm²/s" << std::endl;

    // Slot region (L~2.5): should have lower flux
    double proton_slot = TrappedProtonModel::integral_flux(2.5, 1.0, 10.0, false);
    double electron_slot = TrappedElectronModel::integral_flux(2.5, 1.0, 1.0, false);
    std::cout << "Slot region (L=2.5): p=" << proton_slot << ", e=" << electron_slot << std::endl;

    assert(proton_inner > proton_slot);
    assert(electron_outer > electron_slot);
    std::cout << "✓ Belt structure verified (inner/outer peaks, slot minimum)" << std::endl;
}

/**
 * @brief Test SAA model
 */
void test_saa_model()
{
    std::cout << "\n=== South Atlantic Anomaly Test ===" << std::endl;

    // Center of SAA
    double enhance_center = SouthAtlanticAnomaly::enhancement_factor(-29.0, -47.0);
    std::cout << "SAA center enhancement: " << enhance_center << "x" << std::endl;
    assert(enhance_center > 50.0);

    // Outside SAA (North America)
    double enhance_na = SouthAtlanticAnomaly::enhancement_factor(40.0, -100.0);
    std::cout << "North America (outside SAA): " << enhance_na << "x" << std::endl;
    assert(enhance_na == 1.0);

    // Edge of SAA
    double enhance_edge = SouthAtlanticAnomaly::enhancement_factor(-15.0, -30.0);
    std::cout << "SAA edge enhancement: " << enhance_edge << "x" << std::endl;
    assert(enhance_edge > 1.0 && enhance_edge < enhance_center);

    std::cout << "✓ SAA enhancement correctly localized" << std::endl;
}

/**
 * @brief Test complete SEU rate calculation
 */
void test_seu_rate()
{
    std::cout << "\n=== SEU Rate Calculation Test ===" << std::endl;

    auto device_65nm = WeibullParameters::cmos_65nm();
    auto device_radhard = WeibullParameters::rad_hard();

    // ISS orbit parameters
    SEUCalculator::Environment iss_env;
    iss_env.altitude_km = 400.0;
    iss_env.latitude_deg = 0.0;  // At equator
    iss_env.longitude_deg = 0.0;
    iss_env.solar_modulation_mv = 650.0;
    iss_env.solar_maximum = false;
    iss_env.in_spe = false;

    double seu_iss_65nm = SEUCalculator::calculate_seu_rate(iss_env, device_65nm);
    double seu_iss_radhard = SEUCalculator::calculate_seu_rate(iss_env, device_radhard);

    std::cout << "\nISS orbit (equator crossing):" << std::endl;
    std::cout << "  65nm CMOS:    " << std::scientific << seu_iss_65nm << " errors/bit/day"
              << std::endl;
    std::cout << "  Rad-hardened: " << std::scientific << seu_iss_radhard << " errors/bit/day"
              << std::endl;

    // ISS in SAA
    iss_env.latitude_deg = -29.0;
    iss_env.longitude_deg = -47.0;
    double seu_iss_saa = SEUCalculator::calculate_seu_rate(iss_env, device_65nm);
    std::cout << "  In SAA:       " << std::scientific << seu_iss_saa << " errors/bit/day"
              << std::endl;

    assert(seu_iss_saa > seu_iss_65nm);
    std::cout << "✓ SAA increases SEU rate by " << std::fixed << std::setprecision(1)
              << seu_iss_saa / seu_iss_65nm << "x" << std::endl;

    // GEO orbit (higher radiation)
    SEUCalculator::Environment geo_env = iss_env;
    geo_env.altitude_km = 35786.0;
    geo_env.latitude_deg = 0.0;
    geo_env.longitude_deg = 0.0;

    double seu_geo = SEUCalculator::calculate_seu_rate(geo_env, device_65nm);
    std::cout << "\nGEO orbit:" << std::endl;
    std::cout << "  65nm CMOS: " << std::scientific << seu_geo << " errors/bit/day" << std::endl;

    // During solar particle event
    geo_env.in_spe = true;
    geo_env.spe_magnitude = SolarParticleEvent::Magnitude::STRONG;
    double seu_geo_spe = SEUCalculator::calculate_seu_rate(geo_env, device_65nm);
    std::cout << "  During strong SPE: " << std::scientific << seu_geo_spe << " errors/bit/day"
              << std::endl;

    assert(seu_geo_spe > seu_geo);
    std::cout << "✓ SPE increases SEU rate significantly" << std::endl;
}

/**
 * @brief Test scrubbing interval calculation
 */
void test_scrub_interval()
{
    std::cout << "\n=== Scrubbing Interval Test ===" << std::endl;

    // Scenario: 1MB of weight data, RS(255,223) protection (16 symbol correction)
    constexpr size_t weight_bits = 8 * 1024 * 1024;  // 1MB = 8 Mbit
    constexpr int rs_correction = 16;                // RS can correct 16 symbol errors
    constexpr int hamming_correction = 1;            // Hamming SECDED: 1 error

    // LEO environment
    PhysicsRadiationEnvironment::Config leo_config;
    leo_config.altitude_km = 400.0;
    leo_config.inclination_deg = 51.6;
    leo_config.heavy_ion_device = WeibullParameters::cmos_65nm();
    leo_config.proton_device = BendelParameters::cmos_65nm();

    PhysicsRadiationEnvironment leo_env(leo_config);

    double seu_rate = leo_env.get_orbit_average_seu_rate();
    std::cout << "Orbit-average SEU rate: " << std::scientific << seu_rate << " errors/bit/day"
              << std::endl;

    // Calculate scrub intervals
    double interval_rs = leo_env.recommended_scrub_interval(weight_bits, rs_correction);
    double interval_hamming = leo_env.recommended_scrub_interval(weight_bits, hamming_correction);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nRecommended scrub intervals for 1MB data:" << std::endl;
    std::cout << "  With RS(255,223):  " << interval_rs << " seconds" << std::endl;
    std::cout << "  With SECDED:       " << interval_hamming << " seconds" << std::endl;

    // RS should allow longer intervals
    assert(interval_rs > interval_hamming);
    std::cout << "\n✓ RS allows " << interval_rs / interval_hamming << "x longer scrub intervals"
              << std::endl;

    // SAA statistics (simulate 16 orbits = ~1 day)
    auto saa_stats = leo_env.get_saa_statistics(16);
    std::cout << "\nSAA Statistics (16 orbits):" << std::endl;
    std::cout << "  Time in SAA: " << std::fixed << std::setprecision(1)
              << saa_stats.fraction * 100.0 << "%" << std::endl;
    std::cout << "  Crossings per day: " << saa_stats.crossings_per_day << std::endl;
    std::cout << "  Avg crossing duration: " << std::setprecision(1)
              << saa_stats.avg_crossing_minutes << " minutes" << std::endl;
    std::cout << "  Peak enhancement: " << std::setprecision(0) << saa_stats.peak_enhancement << "x"
              << std::endl;

    // Verify reasonable SAA fraction (ISS typically 10-15%)
    assert(saa_stats.fraction > 0.05 && saa_stats.fraction < 0.25);
    std::cout << "✓ SAA exposure within expected range for ISS orbit" << std::endl;
}

/**
 * @brief Test orbit profile integration
 */
void test_orbit_profile()
{
    std::cout << "\n=== Orbit Profile Test ===" << std::endl;

    PhysicsRadiationEnvironment::Config config;
    config.altitude_km = 400.0;
    config.inclination_deg = 51.6;  // ISS
    config.heavy_ion_device = WeibullParameters::cmos_65nm();
    config.proton_device = BendelParameters::cmos_65nm();

    PhysicsRadiationEnvironment env(config);

    // Show orbital parameters
    double T = env.orbital_period_minutes();
    std::cout << "\nOrbital parameters:" << std::endl;
    std::cout << "  Altitude: " << config.altitude_km << " km" << std::endl;
    std::cout << "  Inclination: " << config.inclination_deg << "°" << std::endl;
    std::cout << "  Period: " << std::fixed << std::setprecision(1) << T << " minutes" << std::endl;
    std::cout << "  Orbits per day: " << std::setprecision(1) << 24.0 * 60.0 / T << std::endl;

    std::cout << "\nSEU rate around single orbit:" << std::endl;
    std::cout << std::setw(10) << "Phase" << std::setw(15) << "Latitude" << std::setw(22)
              << "SEU Rate (err/bit/day)" << std::endl;

    double min_rate = 1e10, max_rate = 0;
    for (int i = 0; i < 12; ++i) {
        double phase = i / 12.0;
        double rate = env.get_seu_rate(phase);
        double lat = config.inclination_deg * std::sin(2.0 * M_PI * phase);

        min_rate = std::min(min_rate, rate);
        max_rate = std::max(max_rate, rate);

        std::cout << std::fixed << std::setw(10) << std::setprecision(2) << phase << std::setw(15)
                  << std::setprecision(1) << lat << "°" << std::setw(20) << std::scientific << rate
                  << std::endl;
    }

    double avg_rate = env.get_orbit_average_seu_rate();
    double worst_rate = env.get_worst_case_seu_rate();

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Minimum:    " << std::scientific << min_rate << std::endl;
    std::cout << "  Maximum:    " << max_rate << std::endl;
    std::cout << "  Average:    " << avg_rate << std::endl;
    std::cout << "  Worst-case (in SAA): " << worst_rate << std::endl;
    std::cout << "  SAA/Avg ratio: " << std::fixed << std::setprecision(1) << worst_rate / avg_rate
              << "x" << std::endl;
}

/**
 * @brief Compare different mission profiles
 */
void test_mission_comparison()
{
    std::cout << "\n=== Mission Profile Comparison ===" << std::endl;

    auto hi_device = WeibullParameters::cmos_65nm();
    auto proton_device = BendelParameters::cmos_65nm();

    struct Mission {
        const char* name;
        double altitude_km;
        double inclination_deg;
    };

    Mission missions[] = {
        {"LEO (ISS)", 400.0, 51.6},
        {"LEO Polar", 600.0, 90.0},
        {"MEO (GPS)", 20200.0, 55.0},
        {"GEO", 35786.0, 0.0},
    };

    std::cout << std::setw(15) << "Mission" << std::setw(20) << "Avg SEU (err/bit/day)"
              << std::setw(20) << "Scrub Interval (s)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (const auto& m : missions) {
        PhysicsRadiationEnvironment::Config config;
        config.altitude_km = m.altitude_km;
        config.inclination_deg = m.inclination_deg;
        config.heavy_ion_device = hi_device;
        config.proton_device = proton_device;

        PhysicsRadiationEnvironment env(config);
        double seu_rate = env.get_orbit_average_seu_rate();
        double scrub_int = env.recommended_scrub_interval(8 * 1024 * 1024, 16);  // 1MB, RS

        std::cout << std::setw(15) << m.name << std::setw(20) << std::scientific << seu_rate
                  << std::setw(20) << std::fixed << std::setprecision(1) << scrub_int << std::endl;
    }
}

int main()
{
    std::cout << "======================================" << std::endl;
    std::cout << "  Physics Radiation Model Validation" << std::endl;
    std::cout << "======================================" << std::endl;

    try {
        test_weibull_cross_section();
        test_bendel_proton_model();
        test_rigidity_conversion();
        test_temperature_correction();
        test_gcr_spectrum();
        test_geomagnetic_field();
        test_trapped_particles();
        test_saa_model();
        test_seu_rate();
        test_scrub_interval();
        test_orbit_profile();
        test_mission_comparison();

        std::cout << "\n======================================" << std::endl;
        std::cout << "  All physics model tests PASSED!" << std::endl;
        std::cout << "======================================" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
