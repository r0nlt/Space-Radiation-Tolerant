#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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

std::string filename(const std::string& path)
{
    const auto separator = path.find_last_of("/\\");
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 3, "Usage: omere_import_validation_test <spectrum.let> <result.see>");
        const auto spectrum = rad_ml::physics::loadOmereLetSpectrum(argv[1]);
        const auto result = rad_ml::physics::loadOmereSeeResult(argv[2]);

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
                        2.0, 1.0e-12,
                        "OMERE differential-flux unit conversion is incorrect");
        }

        const rad_ml::physics::WeibullCrossSection<double> weibull(
            result.saturation_cross_section_cm2_per_bit,
            result.let_threshold_mev_cm2_mg, result.weibull_width,
            result.weibull_shape);
        const auto sensitive_volume =
            rad_ml::physics::squareSensitiveVolumeFromSaturation(
                result.saturation_cross_section_cm2_per_bit, result.cell_depth_um);
        const rad_ml::physics::RectangularChordDistribution chord_distribution(
            sensitive_volume);
        const double analytic_mean_chord =
            4.0 * sensitive_volume.length_um * sensitive_volume.width_um *
            sensitive_volume.depth_um / sensitive_volume.totalSurfaceAreaUm2();
        requireNear(chord_distribution.meanChordUm(), analytic_mean_chord, 0.01,
                    "RPP chord sampler violates the Cauchy mean-chord identity");

        const double calculated_rate = rad_ml::physics::calculateIrppRatePerBitDay(
            spectrum, weibull, result.cell_depth_um);
        require(std::isfinite(calculated_rate) && calculated_rate > 0,
                "Framework IRPP calculation must produce a positive finite rate");
        const double relative_error =
            std::abs(calculated_rate - result.heavy_ion_rate_per_bit_day) /
            result.heavy_ion_rate_per_bit_day;
        if (spectrum.points.size() > 100) {
            // OMERE exports rounded Weibull parameters, so exact internal
            // reproduction is impossible from the .see file alone.
            require(relative_error <= 0.15,
                    "Framework IRPP rate differs from OMERE by more than 15 percent");
        }

        std::cout << "OMERE import validation passed\n";
        std::cout << "Version: " << spectrum.version << '\n';
        std::cout << "Spectrum points: " << spectrum.points.size() << '\n';
        std::cout << "Model: " << spectrum.model << '\n';
        std::cout << "Shielding/depth: " << spectrum.shielding_g_cm2 << " g/cm^2 / "
                  << result.cell_depth_um << " um\n";
        std::cout << "External heavy-ion rate: "
                  << result.heavy_ion_rate_per_bit_day << " /bit/day\n";
        std::cout << "Framework IRPP rate: " << calculated_rate << " /bit/day\n";
        std::cout << "IRPP relative difference: " << 100.0 * relative_error << "%\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "OMERE import validation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
