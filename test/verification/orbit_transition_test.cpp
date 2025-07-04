#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <rad_ml/core/radiation/adaptive_protection.hpp>
#include <rad_ml/mission/mission_profile.hpp>
#include <rad_ml/sim/mission_environment.hpp>
#include <rad_ml/testing/radiation_simulator.hpp>
#include <rad_ml/tmr/enhanced_tmr.hpp>
#include <random>
#include <string>
#include <vector>

using namespace rad_ml;
using namespace rad_ml::mission;
using namespace rad_ml::testing;
using namespace rad_ml::core::radiation;

// Orbit transition test simulating 4-year mission alternating LEO/GEO every 30 days
class OrbitTransitionTest : public ::testing::Test {
   protected:
    static constexpr int YEARS = 4;
    static constexpr int DAYS_PER_YEAR = 365;
    static constexpr int SAMPLES_PER_DAY = 24;    // hourly samples
    static constexpr int ORBIT_SWITCH_DAYS = 30;  // LEO↔GEO every 30 days

    MissionProfile leo_profile{MissionProfile::MissionType::LEO_EARTH_OBSERVATION};
    MissionProfile geo_profile{MissionProfile::MissionType::GEOSTATIONARY};

    using EnhancedByte = rad_ml::tmr::EnhancedTMR<uint8_t>;
    using ProtectionLevel = rad_ml::core::radiation::AdaptiveProtection::ProtectionLevel;

    rad_ml::core::radiation::AdaptiveProtection protector;

    struct DummyPayload {
        std::vector<std::unique_ptr<EnhancedByte>> cells;  // TMR-protected bytes
        std::vector<uint8_t> raw;                          // plain memory for simulator API
        std::vector<uint8_t> golden;                       // expected correct values

        DummyPayload()
        {
            const size_t N = 1024;
            cells.reserve(N);
            for (size_t i = 0; i < N; ++i) cells.emplace_back(std::make_unique<EnhancedByte>(0));
            raw.assign(N, 0);
            golden.assign(N, 0);
        }

        void workload()
        {
            for (size_t i = 0; i < cells.size(); ++i) {
                uint8_t v = cells[i]->get();
                uint8_t newV = static_cast<uint8_t>(v ^ 0x01);
                cells[i]->set(newV);
                golden[i] ^= 0x01;  // mirror expected change
            }
        }

        struct Stats {
            uint64_t detected = 0, corrected = 0, uncorrectable = 0;
        };

        Stats gatherAndReset()
        {
            Stats total;
            for (auto &c : cells) {
                // We only have string stats for EnhancedTMR; so we approximate by error count
                // difference We'll just reset copies via regenerateCopies to clear divergence (not
                // affecting stats). For now leave stats empty.
            }
            return total;
        }
    };

    std::mt19937 gen{std::random_device{}()};

    double solarActivity(int day) const
    {
        constexpr double CYCLE = 4015.0;  // ~11-year solar cycle (days)
        return 0.5 + 0.4 * std::sin(2 * M_PI * day / CYCLE);
    }

    bool randomEvent(double p)
    {
        std::uniform_real_distribution<> d(0.0, 1.0);
        return d(gen) < p;
    }
};

TEST_F(OrbitTransitionTest, FourYearMissionSimulation)
{
    // environment simulators
    RadiationSimulator leo_sim(leo_profile.getSimulationEnvironment());
    RadiationSimulator geo_sim(geo_profile.getSimulationEnvironment());

    DummyPayload payload;

    struct Metrics {
        uint64_t injected = 0;
        uint64_t corrected = 0;
        uint64_t uncorrectable = 0;
        uint64_t bitFlips = 0, multiUpsets = 0, latchups = 0, transients = 0;
        int levelChanges = 0;
        ProtectionLevel maxLevel = ProtectionLevel::MINIMAL;
    } m;
    bool inLeo = true;
    int switchCount = 0;  // count LEO↔GEO transitions

    for (int year = 0; year < YEARS; ++year) {
        for (int day = 0; day < DAYS_PER_YEAR; ++day) {
            if (day % ORBIT_SWITCH_DAYS == 0) {
                if (day != 0) {
                    inLeo = !inLeo;
                    ++switchCount;
                }
            }
            auto &sim = inLeo ? leo_sim : geo_sim;
            auto envParams = sim.getSimulationEnvironment();
            envParams.solar_activity = solarActivity(year * DAYS_PER_YEAR + day);
            sim.updateEnvironment(envParams);

            for (int hour = 0; hour < SAMPLES_PER_DAY; ++hour) {
                // update solar activity already handled via updateEnvironment above
                // Simulate radiation effects for one hour (3600 s)
                auto events = sim.simulateEffects(payload.raw.data(), payload.raw.size(),
                                                  std::chrono::milliseconds(3600 * 1000));
                m.injected += events.size();
                for (const auto &e : events) {
                    // choose target cell
                    size_t idx = e.memory_offset % payload.cells.size();
                    // expectation value
                    uint8_t expected = payload.cells[idx]->get();

                    // corrupt first copy bits depending on effect type (simple flip of LSB or
                    // multiple bits)
                    uint8_t corruptVal = payload.cells[idx]->getRawCopy(0);
                    corruptVal ^= 0x01;  // flip LSB
                    payload.cells[idx]->setRawCopy(0, corruptVal);

                    // classify
                    using E = RadiationSimulator::RadiationEffectType;
                    switch (e.type) {
                        case E::SINGLE_BIT_FLIP:
                            ++m.bitFlips;
                            break;
                        case E::MULTI_BIT_UPSET:
                            ++m.multiUpsets;
                            break;
                        case E::SINGLE_EVENT_LATCHUP:
                            ++m.latchups;
                            break;
                        case E::SINGLE_EVENT_TRANSIENT:
                            ++m.transients;
                            break;
                    }

                    uint8_t result = payload.cells[idx]->get();
                    if (result == expected)
                        ++m.corrected;
                    else
                        ++m.uncorrectable;
                }
                // Update adaptive protection with simple error counts
                protector.updateEnvironment(static_cast<uint32_t>(events.size()), 0);
                ProtectionLevel newLevel = protector.getProtectionLevel();
                if (newLevel != m.maxLevel) {
                    ++m.levelChanges;
                    if (static_cast<int>(newLevel) > static_cast<int>(m.maxLevel))
                        m.maxLevel = newLevel;
                }
                payload.workload();
            }
        }
    }

    ASSERT_GT(m.injected, 0);
    double efficiency = static_cast<double>(m.corrected) / m.injected;
    EXPECT_GT(efficiency, 0.90);
    EXPECT_GT(m.bitFlips, 0);
    EXPECT_GT(m.multiUpsets, 0);
    EXPECT_GT(m.levelChanges, 0);
    EXPECT_GE(static_cast<int>(m.maxLevel), static_cast<int>(ProtectionLevel::ENHANCED));
    std::cout << "Errors summary - injected:" << m.injected << " corrected:" << m.corrected
              << " uncorrectable:" << m.uncorrectable << " eff:" << efficiency * 100 << "%"
              << " bitFlips:" << m.bitFlips << " multiUpsets:" << m.multiUpsets
              << " latchups:" << m.latchups << " transients:" << m.transients
              << " levelChanges:" << m.levelChanges << " maxLevel:" << static_cast<int>(m.maxLevel)
              << "\n";

    ASSERT_GT(switchCount, 0);
    const double expectedSwitches = static_cast<double>(YEARS * DAYS_PER_YEAR) / ORBIT_SWITCH_DAYS;
    EXPECT_NEAR(static_cast<double>(switchCount), expectedSwitches, expectedSwitches * 0.05);

    // Optional runtime guard (approx <5 s)
    // (Measured externally via chrono)

    // Data integrity check: ensure all TMR values equal golden reference
    size_t mismatches = 0;
    for (size_t i = 0; i < payload.cells.size(); ++i) {
        if (payload.cells[i]->get() != payload.golden[i]) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u);
}
