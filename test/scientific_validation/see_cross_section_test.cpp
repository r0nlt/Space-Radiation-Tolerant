#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_ml/physics/see_cross_section.hpp"

namespace {

using BeamRun = rad_ml::physics::HeavyIonBeamRun<double>;

std::vector<std::string> splitCsvRow(const std::string& row)
{
    std::vector<std::string> fields;
    std::stringstream stream(row);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

std::vector<BeamRun> loadRuns(const std::string& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open Versal beam dataset");

    std::string row;
    std::getline(input, row);
    std::vector<BeamRun> runs;
    while (std::getline(input, row)) {
        if (row.empty()) continue;
        const auto fields = splitCsvRow(row);
        if (fields.size() != 14) throw std::runtime_error("Malformed Versal beam row");
        if (fields[7] != "OFF") throw std::runtime_error("Raw curve requires XilSEM-off runs");
        if (fields[12] != "10.1109/TNS.2025.3531510" ||
            fields[13] != "Tables I and III") {
            throw std::runtime_error("Beam row is missing exact source provenance");
        }

        BeamRun run;
        run.facility = fields[0];
        run.ion = fields[1];
        run.let_at_active_silicon_mev_cm2_mg = std::stod(fields[3]);
        run.fluence_ions_cm2 = std::stod(fields[4]);
        run.exposed_bits = std::stoull(fields[6]);
        run.event_counts = {std::stoull(fields[8]), std::stoull(fields[9]),
                            std::stoull(fields[10]), std::stoull(fields[11])};
        runs.push_back(run);
    }
    return runs;
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double relative_tolerance, const char* message)
{
    const double scale = std::max(std::abs(expected), 1.0e-30);
    if (!std::isfinite(actual) || std::abs(actual - expected) > relative_tolerance * scale) {
        throw std::runtime_error(message);
    }
}

struct ConstantCrossSection {
    double value;
    double evaluate(double) const { return value; }
};

}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Usage: see_cross_section_test <versal-beam-data.csv>");
        const auto runs = loadRuns(argv[1]);
        require(runs.size() == 6, "Expected six published XilSEM-off runs");

        using rad_ml::physics::CrossSectionPoint;
        using rad_ml::physics::TabulatedCrossSection;
        using rad_ml::physics::UpsetMultiplicity;
        using rad_ml::physics::WeibullCrossSection;
        using rad_ml::physics::estimateCrossSection;

        const auto high_let_sbu =
            estimateCrossSection(runs.back(), UpsetMultiplicity::SingleBit);
        const auto high_let_two_bit =
            estimateCrossSection(runs.back(), UpsetMultiplicity::TwoBit);
        requireNear(high_let_sbu.cross_section_cm2_per_bit, 5.73e-11, 0.002,
                    "Published 62.4 LET SBU cross-section was not reproduced");
        requireNear(high_let_two_bit.cross_section_cm2_per_bit, 5.97e-11, 0.002,
                    "Published 62.4 LET two-bit MCU cross-section was not reproduced");

        const double zero_count_upper = rad_ml::physics::zeroCountCrossSectionUpperBound(
            runs.front(), 0.05);
        requireNear(zero_count_upper, 2.5024e-13, 0.001,
                    "Conventional 95% zero-count upper bound is incorrect");

        std::vector<CrossSectionPoint<double>> sbu_points;
        for (const auto& run : runs) {
            sbu_points.push_back(estimateCrossSection(run, UpsetMultiplicity::SingleBit));
        }
        const TabulatedCrossSection<double> tabulated(sbu_points);
        requireNear(tabulated.evaluate(62.4), high_let_sbu.cross_section_cm2_per_bit, 1.0e-12,
                    "Tabulated model must reproduce measured points exactly");

        const double midpoint = tabulated.evaluate((18.1 + 32.1) / 2.0);
        const double endpoint_average =
            0.5 * (tabulated.evaluate(18.1) + tabulated.evaluate(32.1));
        requireNear(midpoint, endpoint_average, 1.0e-12,
                    "Tabulated model must interpolate linearly");

        bool rejected_extrapolation = false;
        try {
            (void)tabulated.evaluate(80.0);
        }
        catch (const std::out_of_range&) {
            rejected_extrapolation = true;
        }
        require(rejected_extrapolation,
                "Versal curve must not silently assume unmeasured saturation");

        // The measured SBU curve drops between 32.1 and 54.3 LET because the
        // runs use different species/facilities. A monotonic Weibull cannot
        // exactly represent these raw observations.
        require(tabulated.evaluate(54.3) < tabulated.evaluate(32.1),
                "Test dataset no longer demonstrates the published non-monotonic response");

        const WeibullCrossSection<double> weibull(1.0e-10, 10.0, 10.0, 1.0);
        require(weibull.evaluate(10.0) == 0.0, "Weibull must be zero at onset");
        requireNear(weibull.evaluate(20.0), 1.0e-10 * (1.0 - std::exp(-1.0)), 1.0e-12,
                    "Weibull evaluator is incorrect");

        const double log_likelihood = rad_ml::physics::poissonLogLikelihood(
            runs, UpsetMultiplicity::SingleBit, tabulated);
        require(std::isfinite(log_likelihood), "Poisson beam likelihood must be finite");

        const std::vector<rad_ml::physics::DifferentialLetFluxPoint<double>> spectrum = {
            {10.0, 2.0}, {20.0, 2.0}};
        const double rate =
            rad_ml::physics::integrateEventRate(spectrum, ConstantCrossSection{3.0e-10}, 100);
        requireNear(rate, 6.0e-7, 1.0e-12, "Flux-times-cross-section integration is incorrect");

        std::mt19937 rng(12345);
        require(rad_ml::physics::samplePoissonEventCount(0.0, 1000.0, rng) == 0,
                "Zero event rate must generate zero events");

        std::cout << "Versal SEE cross-section tests passed\n";
        std::cout << "62.4 LET SBU: " << high_let_sbu.cross_section_cm2_per_bit
                  << " cm^2/bit\n";
        std::cout << "62.4 LET two-bit MCU: "
                  << high_let_two_bit.cross_section_cm2_per_bit << " cm^2/bit\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "Versal SEE cross-section tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
