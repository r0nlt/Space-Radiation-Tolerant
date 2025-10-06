/**
 * Radiation Intensity Aggregation Test
 *
 * Verifies EnhancedPhysicsRadiationSimulator::calculateRadiationIntensity aggregates
 * environment fields sensibly and responds to changes (fluxes, solar activity, distance,
 * SAA, atmosphere depth, magnetic field strength).
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <rad_ml/physics/enhanced_physics_radiation_simulator.hpp>
#include <random>
#include <vector>

using namespace rad_ml;

static bool approxEqual(double a, double b, double rel = 1e-2)
{
    return std::fabs(a - b) <= rel * std::max(1.0, std::fabs(b));
}

static sim::RadiationEnvironment makeEnv()
{
    sim::RadiationEnvironment env{};
    env.altitude = 500.0;               // km
    env.distance_from_sun = 1.0;        // AU
    env.gcr_intensity = 0.2;            // relative
    env.solar_activity = 0.3;           // 0..1
    env.trapped_proton_flux = 1.0e5;    // protons/cm^2/s
    env.trapped_electron_flux = 2.0e5;  // electrons/cm^2/s
    env.saa_region = false;
    env.atmosphere_depth = 5.0;         // g/cm^2
    env.magnetic_field_strength = 1.0;  // ~Earth
    env.temperature = {200.0, 300.0, 24.0};
    return env;
}

int main()
{
    std::cout << "Radiation Intensity Aggregation Test\n";
    std::cout << "====================================\n";

    sim::EnhancedPhysicsRadiationSimulator sim;

    // Baseline (mixed sources)
    auto env = makeEnv();
    double I0 = sim.calculateRadiationIntensity(env);
    std::cout << "Baseline intensity: " << I0 << "\n";
    assert(std::isfinite(I0) && I0 > 0.0);

    // Higher trapped flux increases intensity
    env.trapped_proton_flux *= 2.0;
    env.trapped_electron_flux *= 2.0;
    double I_flux = sim.calculateRadiationIntensity(env);
    std::cout << "High trapped flux intensity: " << I_flux << "\n";
    assert(I_flux > I0);

    // Higher solar activity and closer distance increases SPE component
    env.solar_activity = 0.9;
    env.distance_from_sun = 0.5;  // 1/r^2 → 4x
    double I_solar = sim.calculateRadiationIntensity(env);
    std::cout << "High solar activity, close distance intensity: " << I_solar << "\n";
    assert(I_solar > I_flux);

    // SAA region boosts intensity
    env.saa_region = true;
    double I_saa = sim.calculateRadiationIntensity(env);
    std::cout << "SAA region intensity: " << I_saa << "\n";
    assert(I_saa > I_solar);

    // Atmosphere depth attenuates
    env.atmosphere_depth = 100.0;  // heavy attenuation
    double I_atm = sim.calculateRadiationIntensity(env);
    std::cout << "High atmosphere depth intensity: " << I_atm << "\n";
    assert(I_atm < I_saa);

    // Stronger magnetic field attenuates
    env.magnetic_field_strength = 2.0;
    double I_mag = sim.calculateRadiationIntensity(env);
    std::cout << "High magnetic field intensity: " << I_mag << "\n";
    assert(I_mag < I_atm);

    // === Isolated component checks ===
    // Isolate SPE: zero trapped and GCR, disable SAA, no atmosphere, unit magnetic field
    env = makeEnv();
    env.trapped_proton_flux = 0.0;
    env.trapped_electron_flux = 0.0;
    env.gcr_intensity = 0.0;
    env.saa_region = false;
    env.atmosphere_depth = 0.0;
    env.magnetic_field_strength = 1.0;
    env.solar_activity = 0.5;
    env.distance_from_sun = 1.0;
    double I_spe_1 = sim.calculateRadiationIntensity(env);
    double expected_spe_1 = 0.5 * 5.0e4 * 1.0;  // s * 5e4 * 1/r^2
    std::cout << "Isolated SPE (s=0.5, r=1): I=" << I_spe_1 << ", expected=" << expected_spe_1
              << "\n";
    assert(approxEqual(I_spe_1, expected_spe_1, 1e-3));

    env.solar_activity = 1.0;
    env.distance_from_sun = 2.0;  // 1/4 factor
    double I_spe_2 = sim.calculateRadiationIntensity(env);
    double expected_spe_2 = 1.0 * 5.0e4 * (1.0 / 4.0);
    std::cout << "Isolated SPE (s=1.0, r=2): I=" << I_spe_2 << ", expected=" << expected_spe_2
              << "\n";
    assert(approxEqual(I_spe_2, expected_spe_2, 1e-3));

    // SAA factor check with GCR-only
    env.trapped_proton_flux = 0.0;
    env.trapped_electron_flux = 0.0;
    env.solar_activity = 0.0;
    env.distance_from_sun = 1.0;
    env.gcr_intensity = 1.0;
    env.atmosphere_depth = 0.0;
    env.magnetic_field_strength = 1.0;
    env.saa_region = false;
    double I_gcr = sim.calculateRadiationIntensity(env);  // 1.0 * 1e4
    env.saa_region = true;
    double I_gcr_saa = sim.calculateRadiationIntensity(env);
    std::cout << "GCR baseline=" << I_gcr << ", GCR with SAA=" << I_gcr_saa << "\n";
    // Expect ~2.5x increase
    assert(approxEqual(I_gcr_saa / I_gcr, 2.5, 5e-2));

    // Atmosphere attenuation ratio check
    env.saa_region = false;
    env.atmosphere_depth = 0.0;
    double I_atm0 = sim.calculateRadiationIntensity(env);
    env.atmosphere_depth = 100.0;
    double I_atm100 = sim.calculateRadiationIntensity(env);
    double expected_atm_ratio = 1.0 / (1.0 + 100.0 / 50.0);  // 1/3
    std::cout << "Atmosphere ratio I(100)/I(0)=" << (I_atm100 / I_atm0)
              << ", expected=" << expected_atm_ratio << "\n";
    assert(approxEqual(I_atm100 / I_atm0, expected_atm_ratio, 1e-2));

    // Magnetic field attenuation ratio check
    env.atmosphere_depth = 0.0;
    env.magnetic_field_strength = 1.0;
    double I_mag1 = sim.calculateRadiationIntensity(env);
    env.magnetic_field_strength = 2.0;
    double I_mag2 = sim.calculateRadiationIntensity(env);
    double expected_mag_ratio = 1.0 / (1.0 + 0.3 * (2.0 - 1.0));  // 1/1.3
    std::cout << "Magnetic ratio I(2.0)/I(1.0)=" << (I_mag2 / I_mag1)
              << ", expected=" << expected_mag_ratio << "\n";
    assert(approxEqual(I_mag2 / I_mag1, expected_mag_ratio, 1e-2));

    // Negative flux clamping
    env.trapped_proton_flux = -1.0e6;
    env.trapped_electron_flux = 0.0;
    env.gcr_intensity = 0.0;
    env.solar_activity = 0.0;
    env.magnetic_field_strength = 1.0;
    double I_negflux = sim.calculateRadiationIntensity(env);
    std::cout << "Negative flux clamped intensity: " << I_negflux << "\n";
    assert(I_negflux >= 0.0);
    assert(approxEqual(I_negflux, 0.0, 1e-12));

    // Combined factors check (all terms composed)
    env = makeEnv();
    env.trapped_proton_flux = 1.0e5;
    env.trapped_electron_flux = 2.0e5;
    env.solar_activity = 0.4;
    env.distance_from_sun = 1.2;  // distance factor = 1/1.44
    env.gcr_intensity = 0.3;
    env.saa_region = true;              // 2.5x
    env.atmosphere_depth = 25.0;        // factor = 1/(1+0.5) = 2/3
    env.magnetic_field_strength = 1.5;  // factor = 1/(1+0.3*(0.5)) = 1/1.15

    double I_combined = sim.calculateRadiationIntensity(env);

    double proton_flux = std::max(0.0, env.trapped_proton_flux);
    double electron_flux = std::max(0.0, env.trapped_electron_flux);
    double distance_factor =
        (env.distance_from_sun > 0.0) ? 1.0 / (env.distance_from_sun * env.distance_from_sun) : 1.0;
    double spe_component = std::max(0.0, env.solar_activity) * 5.0e4 * distance_factor;
    double gcr_component = std::max(0.0, env.gcr_intensity) * 1.0e4;
    double expected_total = proton_flux + electron_flux + spe_component + gcr_component;
    expected_total *= 2.5;  // SAA
    double atmosphere_factor = 1.0 / (1.0 + std::max(0.0, env.atmosphere_depth) / 50.0);
    expected_total *= atmosphere_factor;
    double mag_factor = 1.0 / (1.0 + 0.3 * std::max(0.0, env.magnetic_field_strength - 1.0));
    expected_total *= mag_factor;

    std::cout << "Combined I=" << I_combined << ", expected=" << expected_total << "\n";
    assert(approxEqual(I_combined, expected_total, 1e-6));

    // Edge case: zero distance -> distance_factor treated as 1.0
    env = makeEnv();
    env.trapped_proton_flux = 0.0;
    env.trapped_electron_flux = 0.0;
    env.gcr_intensity = 0.0;
    env.solar_activity = 1.0;
    env.distance_from_sun = 0.0;  // non-physical; function uses 1.0 fallback
    env.saa_region = false;
    env.atmosphere_depth = 0.0;
    env.magnetic_field_strength = 1.0;
    double I_zero_r = sim.calculateRadiationIntensity(env);
    double expected_zero_r = 1.0 * 5.0e4 * 1.0;  // 1/r^2 -> 1.0 fallback
    std::cout << "Zero-distance guard I=" << I_zero_r << ", expected=" << expected_zero_r << "\n";
    assert(approxEqual(I_zero_r, expected_zero_r, 1e-6));

    // Noise robustness: multi-seed Gaussian noise and mean stability checks
    {
        const double sigma = 0.01;  // 1% relative noise
        const int trials = 200;
        std::vector<unsigned int> seeds = {1u, 42u, 1337u, 2025u, 99991u};

        auto base = makeEnv();
        base.saa_region = false;
        base.atmosphere_depth = 0.0;
        base.magnetic_field_strength = 1.0;
        double I_base = sim.calculateRadiationIntensity(base);

        for (unsigned int seed : seeds) {
            std::mt19937 rng(seed);
            std::normal_distribution<double> gauss(0.0, 1.0);

            auto perturb = [&](const sim::RadiationEnvironment& e) {
                sim::RadiationEnvironment n = e;
                auto mult = [&](double v) { return v * std::max(0.0, 1.0 + sigma * gauss(rng)); };
                n.trapped_proton_flux = std::max(0.0, mult(n.trapped_proton_flux));
                n.trapped_electron_flux = std::max(0.0, mult(n.trapped_electron_flux));
                n.gcr_intensity = std::max(0.0, mult(n.gcr_intensity));
                n.solar_activity = std::min(1.0, std::max(0.0, mult(n.solar_activity)));
                n.distance_from_sun = std::max(0.1, mult(n.distance_from_sun));
                return n;
            };

            std::vector<double> samples;
            samples.reserve(trials);
            for (int i = 0; i < trials; ++i) {
                auto nenv = perturb(base);
                samples.push_back(sim.calculateRadiationIntensity(nenv));
            }

            double sum = 0.0;
            for (double v : samples) sum += v;
            double mean = sum / samples.size();
            double var = 0.0;
            for (double v : samples) var += (v - mean) * (v - mean);
            var /= std::max(1, (int)samples.size() - 1);
            double sd = std::sqrt(var);

            std::cout << "Noise seed=" << seed << ": mean=" << mean << ", sd=" << sd
                      << ", base=" << I_base << "\n";
            // Expect mean within ~5% of base for 1% noise (nonlinearities tolerated)
            double rel_err = std::fabs(mean - I_base) / std::max(1.0, std::fabs(I_base));
            assert(rel_err < 0.05);
            assert(sd > 0.0);
        }
    }

    std::cout << "\n\xF0\x9F\x8E\x89 Radiation Intensity Aggregation Test Passed!\n";
    return 0;
}
