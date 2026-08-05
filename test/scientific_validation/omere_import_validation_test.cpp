#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_ml/physics/irpp_rate.hpp"
#include "rad_ml/physics/omere_io.hpp"

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double relative_tolerance,
                 const char* message)
{
    const double scale = std::max(std::abs(expected), 1.0e-30);
    if (!std::isfinite(actual) || std::abs(actual - expected) > relative_tolerance * scale) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void requireInvalidArgument(Callable&& callable, const char* message)
{
    try {
        callable();
    }
    catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void validateChordGeometry(
    const rad_ml::physics::RectangularSensitiveVolume& volume)
{
    const rad_ml::physics::ExactRectangularChordDistribution exact(volume, 32);
    constexpr std::size_t sampled_chord_count = 524288;
    const rad_ml::physics::RectangularChordDistribution sampled(
        volume, sampled_chord_count);
    require(sampled.sampleCount() == sampled_chord_count,
            "Sampled RPP chord distribution dropped remainder samples");
    const double analytic_mean =
        4.0 * volume.length_um * volume.width_um * volume.depth_um /
        volume.totalSurfaceAreaUm2();
    requireNear(exact.meanChordUm(), analytic_mean, 1.0e-13,
                "Exact RPP chord mean violates the Cauchy identity");
    requireNear(sampled.meanChordUm(), analytic_mean, 0.01,
                "Sampled RPP chord mean violates the Cauchy identity");
    requireNear(exact.probabilityGreaterThan(0), 1.0, 1.0e-13,
                "Exact RPP chord CDF has an invalid origin");
    requireNear(exact.probabilityGreaterThan(exact.bodyDiagonalUm()), 0.0, 1.0e-13,
                "Exact RPP chord CDF extends beyond the body diagonal");

    double previous = 1;
    for (std::size_t i = 1; i <= 20; ++i) {
        const double distance =
            exact.bodyDiagonalUm() * static_cast<double>(i) / 20.0;
        const double probability = exact.probabilityGreaterThan(distance);
        require(probability <= previous + 1.0e-12,
                "Exact RPP chord CDF is not monotone");
        previous = probability;
    }

    const double comparison_distance =
        0.5 * std::min({volume.length_um, volume.width_um, volume.depth_um});
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double exact_low_distance_probability =
        (0.25 * pi *
             (volume.length_um * volume.width_um +
              volume.length_um * volume.depth_um +
              volume.width_um * volume.depth_um) -
         (2.0 * comparison_distance / 3.0) *
             (volume.length_um + volume.width_um + volume.depth_um) +
         3.0 * comparison_distance * comparison_distance / 8.0) /
        (0.25 * pi *
         (volume.length_um * volume.width_um +
          volume.length_um * volume.depth_um +
          volume.width_um * volume.depth_um));
    requireNear(exact.probabilityGreaterThan(comparison_distance),
                exact_low_distance_probability, 1.0e-10,
                "Exact RPP chord CDF fails its closed-form low-distance branch");
    requireNear(exact.probabilityGreaterThan(comparison_distance),
                sampled.probabilityGreaterThan(comparison_distance), 0.02,
                "Exact and sampled RPP chord probabilities disagree");
}

void validateSpectrumColumns(const rad_ml::physics::OmereLetSpectrum& spectrum)
{
    rad_ml::physics::validateIrppSpectrum(spectrum, 1.0e-4);
    double integrated_flux = 0;
    for (std::size_t i = 1; i < spectrum.points.size(); ++i) {
        const auto& left = spectrum.points[i - 1];
        const auto& right = spectrum.points[i];
        integrated_flux +=
            0.5 * (left.differential_flux_per_cm2_s_per_let +
                   right.differential_flux_per_cm2_s_per_let) *
            (right.let_mev_cm2_mg - left.let_mev_cm2_mg);
    }
    requireNear(integrated_flux, spectrum.points.front().integral_flux_cm2_s,
                1.0e-4,
                "OMERE differential and integral LET columns are inconsistent");
}

void validateMalformedSpectraAreRejected()
{
    rad_ml::physics::OmereLetSpectrum malformed;
    malformed.points = {{1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}};
    requireInvalidArgument(
        [&] { rad_ml::physics::validateIrppSpectrum(malformed); },
        "IRPP accepted a degenerate LET bin");

    malformed.points = {{1.0, 1.0, 1.0}, {2.0, 2.0, 1.0}};
    requireInvalidArgument(
        [&] { rad_ml::physics::validateIrppSpectrum(malformed); },
        "IRPP accepted increasing integral LET flux");

    malformed.points = {{1.0, 2.0, 0.0}, {2.0, 0.0, 0.0}};
    requireInvalidArgument(
        [&] { rad_ml::physics::validateIrppSpectrum(malformed); },
        "IRPP accepted inconsistent integral and differential LET columns");
}

std::string filename(const std::string& path)
{
    const auto separator = path.find_last_of("/\\");
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 3 || argc == 6,
                "Usage: omere_import_validation_test <spectrum.let> <result.see> "
                "[uniform.let low_band.let high_band.let]");
        const auto spectrum = rad_ml::physics::loadOmereLetSpectrum(argv[1]);
        const auto result = rad_ml::physics::loadOmereSeeResult(argv[2]);
        validateMalformedSpectraAreRejected();
        validateSpectrumColumns(spectrum);
        for (int argument = 3; argument < argc; ++argument) {
            validateSpectrumColumns(
                rad_ml::physics::loadOmereLetSpectrum(argv[argument]));
        }

        require(spectrum.version == result.version,
                "OMERE LET and SEE files must come from the same software version");
        require(filename(result.let_file) == filename(argv[1]),
                "OMERE SEE result references a different LET spectrum");
        require(spectrum.model.find("GCR ISO 15390") != std::string::npos,
                "Expected a traceable GCR ISO 15390 spectrum");
        require(spectrum.solar_activity == "Min", "Expected solar-minimum GCR input");
        requireNear(spectrum.shielding_g_cm2, 1.0, 1.0e-12,
                    "OMERE shielding metadata was parsed incorrectly");
        requireNear(spectrum.mission_duration_years, 15.0, 1.0e-12,
                    "OMERE mission duration was parsed incorrectly");
        requireNear(result.cell_depth_um, 2.0, 1.0e-12,
                    "OMERE sensitive depth was parsed incorrectly");
        requireNear(result.let_threshold_mev_cm2_mg, 2.57, 1.0e-12,
                    "OMERE Weibull threshold was parsed incorrectly");
        requireNear(result.saturation_cross_section_cm2_per_bit, 5.72e-11, 1.0e-12,
                    "OMERE Weibull saturation was parsed incorrectly");
        requireNear(result.weibull_width, 17.9, 1.0e-12,
                    "OMERE Weibull width was parsed incorrectly");
        requireNear(result.weibull_shape, 0.97, 1.0e-12,
                    "OMERE Weibull shape was parsed incorrectly");
        requireNear(result.heavy_ion_rate_per_bit_day, 3.71e-13, 1.0e-12,
                    "OMERE heavy-ion benchmark rate was parsed incorrectly");

        // The fixture and real OMERE files use MeV.cm^2/g. Confirm conversion
        // into the framework's canonical MeV.cm^2/mg contract.
        requireNear(spectrum.points.front().let_mev_cm2_mg,
                    spectrum.points.size() > 100 ? 1.60896e-3 : 1.0, 1.0e-10,
                    "OMERE LET g-to-mg conversion is incorrect");
        if (spectrum.points.size() <= 100) {
            requireNear(spectrum.points.front().differential_flux_per_cm2_s_per_let,
                        1.0, 1.0e-12,
                        "OMERE differential-flux unit conversion is incorrect");
        }

        const bool external_benchmark = spectrum.points.size() > 100;
        // The .see text rounds S and sigma_sat to 0.97 and 5.72e-11.
        // OMERE's fit panel used 0.975 and 5.721e-11 for this benchmark.
        // Preserve those independently recorded inputs for an apples-to-apples
        // calculation instead of treating display-rounded output as input.
        const double calculation_saturation =
            external_benchmark ? 5.721e-11
                               : result.saturation_cross_section_cm2_per_bit;
        const double calculation_shape =
            external_benchmark ? 0.975 : result.weibull_shape;
        const rad_ml::physics::WeibullCrossSection<double> weibull(
            calculation_saturation, result.let_threshold_mev_cm2_mg,
            result.weibull_width, calculation_shape);
        const auto sensitive_volume =
            rad_ml::physics::squareSensitiveVolumeFromSaturation(
                calculation_saturation, result.cell_depth_um);
        validateChordGeometry({1.0, 1.0, 1.0});
        validateChordGeometry({4.0, 4.0, 0.4});
        validateChordGeometry(sensitive_volume);

        const std::vector<rad_ml::physics::IrppNumerics> numerical_settings =
            external_benchmark
                ? std::vector<rad_ml::physics::IrppNumerics>{
                      {12, 24, 1.0e-10}, {24, 48, 1.0e-12}, {32, 64, 1.0e-13}}
                : std::vector<rad_ml::physics::IrppNumerics>{{16, 24, 1.0e-10}};
        std::vector<double> converged_rates;
        converged_rates.reserve(numerical_settings.size());
        for (const auto& settings : numerical_settings) {
            const double rate = rad_ml::physics::calculateIrppRatePerBitDay(
                spectrum, weibull, result.cell_depth_um, settings);
            require(std::isfinite(rate) && rate > 0,
                    "Framework IRPP calculation must produce a positive finite rate");
            converged_rates.push_back(rate);
        }

        const double calculated_rate = converged_rates.back();
        const double relative_error =
            std::abs(calculated_rate - result.heavy_ion_rate_per_bit_day) /
            result.heavy_ion_rate_per_bit_day;
        double numerical_convergence = 0;
        if (external_benchmark) {
            numerical_convergence =
                std::abs(converged_rates.back() - converged_rates[converged_rates.size() - 2]) /
                converged_rates.back();
            for (std::size_t i = 0; i < converged_rates.size(); ++i) {
                std::cerr << "IRPP diagnostic (angular/weibull orders "
                          << numerical_settings[i].angular_order << "/"
                          << numerical_settings[i].weibull_order
                          << "): " << converged_rates[i] << " /bit/day\n";
            }
            std::cerr << "IRPP diagnostic convergence change: "
                      << 100.0 * numerical_convergence << "%\n";
            require(numerical_convergence <= 0.005,
                    "Deterministic IRPP quadrature convergence exceeds 0.5 percent");

            const rad_ml::physics::ExactRectangularChordDistribution exact_chords(
                sensitive_volume, 24);
            const double integrated_exported_flux =
                rad_ml::physics::rppEffectiveFlux(
                    spectrum, exact_chords, result.cell_depth_um, 0.0);
            requireNear(integrated_exported_flux,
                        spectrum.points.front().integral_flux_cm2_s, 1.0e-4,
                        "Differential and integral OMERE LET columns disagree");

            using SpectrumIntegration =
                rad_ml::physics::IrppNumerics::SpectrumIntegration;
            const std::vector<std::pair<const char*, SpectrumIntegration>>
                spectrum_conventions = {
                    {"differential Simpson",
                     SpectrumIntegration::DifferentialLinearSimpson},
                    {"differential trapezoid",
                     SpectrumIntegration::DifferentialLinearTrapezoid},
                    {"integral geometric midpoint",
                     SpectrumIntegration::IntegralBinGeometricMidpoint},
                    {"integral arithmetic midpoint",
                     SpectrumIntegration::IntegralBinArithmeticMidpoint}};
            std::vector<double> convention_rates;
            for (const auto& convention : spectrum_conventions) {
                rad_ml::physics::IrppNumerics settings{24, 48, 1.0e-12};
                settings.spectrum_integration = convention.second;
                const double rate = rad_ml::physics::calculateIrppRatePerBitDay(
                    spectrum, weibull, result.cell_depth_um, settings);
                convention_rates.push_back(rate);
                std::cerr << "IRPP spectrum convention (" << convention.first
                          << "): " << rate << " /bit/day\n";
            }
            require(std::abs(convention_rates[0] - convention_rates[1]) /
                                calculated_rate <=
                            0.005,
                    "Differential LET quadrature differs by more than 0.5 percent");

            // This remains an external cross-tool gate, not a claim of
            // bit-for-bit identity with OMERE's closed CREME86 implementation.
            require(relative_error <= 0.10,
                    "Framework IRPP rate differs from OMERE by more than 10 percent");
        }

        std::cout << "OMERE import validation passed\n";
        std::cout << "Version: " << spectrum.version << '\n';
        std::cout << "Spectrum points: " << spectrum.points.size() << '\n';
        std::cout << "Model: " << spectrum.model << '\n';
        std::cout << "Shielding/depth: " << spectrum.shielding_g_cm2 << " g/cm^2 / "
                  << result.cell_depth_um << " um\n";
        std::cout << "External heavy-ion rate: "
                  << result.heavy_ion_rate_per_bit_day << " /bit/day\n";
        for (std::size_t i = 0; i < converged_rates.size(); ++i) {
            std::cout << "IRPP rate (angular/weibull orders "
                      << numerical_settings[i].angular_order << "/"
                      << numerical_settings[i].weibull_order
                      << "): " << converged_rates[i] << " /bit/day\n";
        }
        std::cout << "Framework IRPP rate: " << calculated_rate << " /bit/day\n";
        std::cout << "Numerical convergence change: "
                  << 100.0 * numerical_convergence << "%\n";
        std::cout << "IRPP relative difference: " << 100.0 * relative_error << "%\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "OMERE import validation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
