#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_ml/physics/see_event_rate.hpp"
#include "rad_ml/testing/radiation_simulator.hpp"

namespace {

using Rate = rad_ml::physics::MissionEventRate<double>;

std::vector<std::string> splitCsvRow(const std::string& row)
{
    std::vector<std::string> fields;
    std::stringstream stream(row);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
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

std::vector<Rate> loadPublishedRates(const std::string& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open published GEO event-rate data");

    std::string row;
    std::getline(input, row);
    std::vector<Rate> rates;
    while (std::getline(input, row)) {
        if (row.empty()) continue;
        const auto fields = splitCsvRow(row);
        require(fields.size() == 8, "Malformed published event-rate row");
        require(fields[0] == "15-year GEO", "Unexpected mission in event-rate benchmark");
        require(fields[4] == "OMERE", "Published calculation tool must be OMERE");
        require(fields[5] == "CREME96 heavy ions and GCR",
                "Published environment model must be CREME96");
        require(fields[6] == "10.1109/TNS.2025.3531510",
                "Event-rate row is missing its source DOI");
        require(fields[7] == "Table VI", "Event-rate row is missing its table locator");

        Rate rate;
        rate.multiplicity =
            fields[2] == "SBU" ? rad_ml::physics::UpsetMultiplicity::SingleBit
                               : rad_ml::physics::UpsetMultiplicity::TwoBit;
        rate.events_per_day_per_bit = std::stod(fields[3]);
        rate.origin = rad_ml::physics::EventRateOrigin::PublishedExternal;
        rate.environment = fields[1];
        rate.source = fields[6] + ", " + fields[7];
        rates.push_back(rate);
    }
    return rates;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2,
                "Usage: versal_event_rate_validation_test <published-rates.csv>");
        const auto rates = loadPublishedRates(argv[1]);
        require(rates.size() == 4, "Expected all four rates from published Table VI");

        // These constants are independently transcribed from Table VI. The CSV
        // is rejected if a future edit changes the published benchmark.
        requireNear(rates[0].events_per_day_per_bit, 1.21e-12, 1.0e-15,
                    "Steady-GCR SBU rate differs from Table VI");
        requireNear(rates[1].events_per_day_per_bit, 1.50e-14, 1.0e-15,
                    "Steady-GCR 2-bit MCU rate differs from Table VI");
        requireNear(rates[2].events_per_day_per_bit, 1.07e-10, 1.0e-15,
                    "Worst-week SBU rate differs from Table VI");
        requireNear(rates[3].events_per_day_per_bit, 1.35e-12, 1.0e-15,
                    "Worst-week 2-bit MCU rate differs from Table VI");

        constexpr std::uint64_t vc1902_exposed_bits = 59842000;
        requireNear(rates[0].eventsPerDeviceDay(vc1902_exposed_bits), 7.240882e-5, 1.0e-12,
                    "Steady-GCR SBU device-rate conversion is incorrect");
        requireNear(rates[1].eventsPerDeviceDay(vc1902_exposed_bits), 8.9763e-7, 1.0e-12,
                    "Steady-GCR 2-bit MCU device-rate conversion is incorrect");
        requireNear(rates[2].eventsPerDeviceDay(vc1902_exposed_bits), 6.403094e-3, 1.0e-12,
                    "Worst-week SBU device-rate conversion is incorrect");
        requireNear(rates[3].eventsPerDeviceDay(vc1902_exposed_bits), 8.07867e-5, 1.0e-12,
                    "Worst-week 2-bit MCU device-rate conversion is incorrect");

        // Validate the event-source statistics at a practical synthetic rate.
        // Arrival-time sampling is tested separately from the published values
        // because their expected daily counts are intentionally very small.
        constexpr std::size_t trials = 20000;
        constexpr double seconds_per_day = 86400.0;
        constexpr double expected_per_trial = 4.0;
        std::mt19937_64 rng(0x56455253414cULL);
        double sum = 0;
        double sum_squares = 0;
        for (std::size_t trial = 0; trial < trials; ++trial) {
            const auto arrivals = rad_ml::physics::samplePoissonArrivalTimes(
                expected_per_trial / seconds_per_day, seconds_per_day, rng);
            require(std::is_sorted(arrivals.begin(), arrivals.end()),
                    "Poisson arrival times must be ordered");
            const double count = static_cast<double>(arrivals.size());
            sum += count;
            sum_squares += count * count;
        }
        const double mean = sum / static_cast<double>(trials);
        const double variance =
            (sum_squares - static_cast<double>(trials) * mean * mean) /
            static_cast<double>(trials - 1);
        require(std::abs(mean - expected_per_trial) < 0.06,
                "Poisson event-stream mean is outside its validation band");
        require(std::abs(variance - expected_per_trial) < 0.12,
                "Poisson event-stream variance is outside its validation band");

        // End-to-end integration: typed rates schedule and inject exact SBU and
        // 2-bit MCU events into mission memory through RadiationSimulator.
        auto environment =
            rad_ml::testing::RadiationSimulator::getMissionEnvironment("GEO");
        rad_ml::testing::RadiationSimulator published_simulator(environment);
        published_simulator.useMissionEventRates(
            rates[0], rates[1], vc1902_exposed_bits);
        requireNear(published_simulator.getEventRates().single_bit_flip_rate,
                    rates[0].eventsPerDeviceSecond(vc1902_exposed_bits), 1.0e-12,
                    "Published SBU rate did not reach RadiationSimulator");
        requireNear(published_simulator.getEventRates().multi_bit_upset_rate,
                    rates[1].eventsPerDeviceSecond(vc1902_exposed_bits), 1.0e-12,
                    "Published 2-bit MCU rate did not reach RadiationSimulator");

        constexpr std::size_t memory_bytes = 16384;
        constexpr std::uint64_t memory_bits = memory_bytes * 8;
        Rate accelerated_sbu;
        accelerated_sbu.multiplicity = rad_ml::physics::UpsetMultiplicity::SingleBit;
        accelerated_sbu.events_per_day_per_bit = 20.0 / memory_bits;
        accelerated_sbu.origin = rad_ml::physics::EventRateOrigin::Calculated;
        Rate accelerated_mcu = accelerated_sbu;
        accelerated_mcu.multiplicity = rad_ml::physics::UpsetMultiplicity::TwoBit;

        std::vector<std::uint8_t> memory_a(memory_bytes, 0);
        std::vector<std::uint8_t> memory_b(memory_bytes, 0);
        rad_ml::testing::RadiationSimulator simulator_a(environment);
        rad_ml::testing::RadiationSimulator simulator_b(environment);
        simulator_a.useMissionEventRates(accelerated_sbu, accelerated_mcu, memory_bits);
        simulator_b.useMissionEventRates(accelerated_sbu, accelerated_mcu, memory_bits);
        simulator_a.setSeed(0x56455253u);
        simulator_b.setSeed(0x56455253u);
        const auto campaign_duration = std::chrono::hours(24 * 10);
        const auto events_a = simulator_a.simulateEffects(
            memory_a.data(), memory_a.size(),
            std::chrono::duration_cast<std::chrono::milliseconds>(campaign_duration));
        const auto events_b = simulator_b.simulateEffects(
            memory_b.data(), memory_b.size(),
            std::chrono::duration_cast<std::chrono::milliseconds>(campaign_duration));

        require(events_a.size() == events_b.size() && memory_a == memory_b,
                "Seeded mission-rate campaigns must be reproducible");
        require(events_a.size() > 300 && events_a.size() < 500,
                "Accelerated campaign count is outside its Poisson validation band");
        std::size_t sbu_count = 0;
        std::size_t mcu_count = 0;
        std::size_t frame_step_mcu_count = 0;
        double previous_time = -1;
        for (std::size_t i = 0; i < events_a.size(); ++i) {
            const auto& event = events_a[i];
            const auto& repeated = events_b[i];
            require(event.type == repeated.type &&
                        event.memory_offset == repeated.memory_offset &&
                        event.bits_affected == repeated.bits_affected &&
                        event.time_offset_seconds == repeated.time_offset_seconds &&
                        event.bit_offsets == repeated.bit_offsets,
                    "Seeded mission event metadata must be reproducible");
            require(event.time_offset_seconds >= previous_time,
                    "Mission events must be emitted in chronological order");
            previous_time = event.time_offset_seconds;
            if (event.type ==
                rad_ml::testing::RadiationSimulator::RadiationEffectType::SINGLE_BIT_FLIP) {
                require(event.bits_affected == 1, "SBU must flip exactly one bit");
                ++sbu_count;
            } else {
                require(event.bits_affected == 2, "2-bit MCU must flip exactly two bits");
                const auto left = event.bit_offsets[0];
                const auto right = event.bit_offsets[1];
                const std::uint64_t separation = left > right ? left - right : right - left;
                require(rad_ml::physics::VersalConfigurationMcuTopology::isModeledOffset(
                            separation),
                        "Versal MCU used an offset not supported by the beam data");
                if (separation == 3200) ++frame_step_mcu_count;
                ++mcu_count;
            }
        }
        require(sbu_count > 150 && sbu_count < 250,
                "Accelerated SBU stream is outside its Poisson validation band");
        require(mcu_count > 150 && mcu_count < 250,
                "Accelerated 2-bit MCU stream is outside its Poisson validation band");
        require(frame_step_mcu_count > static_cast<std::size_t>(0.80 * mcu_count),
                "Versal frame-step MCU topology is outside its validation band");

        std::cout << "Published Versal GEO event-rate validation passed\n";
        std::cout << "Steady-GCR SBU/device/day: "
                  << rates[0].eventsPerDeviceDay(vc1902_exposed_bits) << '\n';
        std::cout << "Worst-week SBU/device/day: "
                  << rates[2].eventsPerDeviceDay(vc1902_exposed_bits) << '\n';
        std::cout << "Poisson mean/variance: " << mean << " / " << variance << '\n';
        std::cout << "Integrated accelerated SBU/2-bit MCU events: "
                  << sbu_count << " / " << mcu_count << '\n';
        std::cout << "2-bit MCUs separated by 3200 readback bits: "
                  << frame_step_mcu_count << '\n';
        std::cout << "NOTE: this confirms Table VI ingestion and campaign generation; "
                     "an exact OMERE/CREME96 recalculation still requires its spectrum and "
                     "IRPP inputs.\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "Versal event-rate validation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
