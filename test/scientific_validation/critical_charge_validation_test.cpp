#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_ml/physics/critical_charge.hpp"

namespace {

struct BenchmarkPoint {
    double feature_size_nm;
    double critical_charge_fc;
    std::string partition;
    std::string evidence_type;
    std::string device;
    std::string source_doi;
    std::string source_locator;
};

std::vector<std::string> splitCsvRow(const std::string& row)
{
    std::vector<std::string> fields;
    std::stringstream stream(row);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<BenchmarkPoint> loadBenchmark(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open critical-charge benchmark: " + path);
    }

    std::string row;
    std::getline(input, row);
    if (row !=
        "feature_size_nm,critical_charge_fc,partition,evidence_type,device,source_doi,"
        "source_locator") {
        throw std::runtime_error("Unexpected critical-charge benchmark schema");
    }

    std::vector<BenchmarkPoint> points;
    while (std::getline(input, row)) {
        if (row.empty()) continue;
        const auto fields = splitCsvRow(row);
        if (fields.size() != 7) {
            throw std::runtime_error("Malformed critical-charge benchmark row: " + row);
        }
        points.push_back({std::stod(fields[0]), std::stod(fields[1]), fields[2], fields[3],
                          fields[4], fields[5], fields[6]});
    }
    return points;
}

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double correlation(const std::vector<double>& expected, const std::vector<double>& predicted)
{
    const double count = static_cast<double>(expected.size());
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        sum_x += expected[i];
        sum_y += predicted[i];
        sum_xx += expected[i] * expected[i];
        sum_yy += predicted[i] * predicted[i];
        sum_xy += expected[i] * predicted[i];
    }
    const double denominator =
        std::sqrt((count * sum_xx - sum_x * sum_x) * (count * sum_yy - sum_y * sum_y));
    return denominator > 0.0 ? (count * sum_xy - sum_x * sum_y) / denominator : 0.0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Usage: critical_charge_validation_test <benchmark.csv>");
        const auto points = loadBenchmark(argv[1]);
        require(points.size() == 5, "Expected all five traceable benchmark rows");

        using Model = rad_ml::physics::CriticalChargePowerLaw<double>;
        std::vector<Model::CalibrationPoint> calibration;
        std::vector<BenchmarkPoint> evaluation;

        for (const auto& point : points) {
            require(point.evidence_type == "SPICE-derived",
                    "Benchmark evidence must not be mislabeled as measured");
            require(point.source_doi == "10.1109/IRPS.2009.5173252",
                    "Every benchmark row must retain its source DOI");
            require(!point.source_locator.empty(), "Every benchmark row needs a source locator");

            if (point.partition == "calibration") {
                calibration.push_back({point.feature_size_nm, point.critical_charge_fc});
            }
            else if (point.partition == "evaluation") {
                evaluation.push_back(point);
            }
            else {
                throw std::runtime_error("Unknown benchmark partition: " + point.partition);
            }
        }

        require(calibration.size() == 2, "Expected two calibration points");
        require(evaluation.size() == 3, "Expected three held-out evaluation points");

        const Model model = Model::fit(calibration);
        std::vector<double> expected;
        std::vector<double> predicted;
        double absolute_percentage_error = 0.0;

        std::cout << "Source: DOI 10.1109/IRPS.2009.5173252, Table III (SPICE-derived)\n";
        std::cout << "Fitted Qcrit = " << model.coefficient() << " * feature_nm^"
                  << model.exponent() << '\n';
        for (const auto& point : evaluation) {
            const double prediction = model.predict(point.feature_size_nm);
            const double error_percent =
                100.0 * std::abs(prediction - point.critical_charge_fc) /
                point.critical_charge_fc;
            expected.push_back(point.critical_charge_fc);
            predicted.push_back(prediction);
            absolute_percentage_error += error_percent;
            std::cout << point.feature_size_nm << " nm: expected " << point.critical_charge_fc
                      << " fC, predicted " << prediction << " fC, error " << error_percent
                      << "%\n";
        }

        const double mean_absolute_percentage_error =
            absolute_percentage_error / static_cast<double>(evaluation.size());
        const double held_out_correlation = correlation(expected, predicted);
        std::cout << "Held-out MAPE: " << mean_absolute_percentage_error << "%\n";
        std::cout << "Held-out correlation: " << held_out_correlation << '\n';

        require(mean_absolute_percentage_error <= 20.0,
                "Held-out critical-charge MAPE exceeds 20%");
        require(held_out_correlation >= 0.99,
                "Held-out critical-charge correlation is below 0.99");

        bool rejected_invalid_input = false;
        try {
            (void)model.predict(0.0);
        }
        catch (const std::invalid_argument&) {
            rejected_invalid_input = true;
        }
        require(rejected_invalid_input, "Critical-charge model accepted an invalid feature size");

        std::cout << "Critical-charge validation passed\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "Critical-charge validation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
