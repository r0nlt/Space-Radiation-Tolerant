#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_ml/physics/see_cross_section.hpp"

namespace rad_ml {
namespace physics {

enum class EventRateOrigin {
    PublishedExternal,
    Calculated
};

/**
 * @brief A typed SEE rate with an explicit evidence boundary
 *
 * PublishedExternal values are reference outputs from another rate tool or
 * publication. Calculated values are produced by this framework. Keeping the
 * origin explicit prevents an imported benchmark from being presented as an
 * independently reproduced calculation.
 */
template <typename Scalar = double>
struct MissionEventRate {
    UpsetMultiplicity multiplicity = UpsetMultiplicity::SingleBit;
    Scalar events_per_day_per_bit = 0;
    EventRateOrigin origin = EventRateOrigin::Calculated;
    std::string environment;
    std::string source;

    Scalar eventsPerDeviceDay(std::uint64_t bit_count) const
    {
        validate(bit_count);
        return events_per_day_per_bit * static_cast<Scalar>(bit_count);
    }

    Scalar eventsPerDeviceSecond(std::uint64_t bit_count) const
    {
        constexpr Scalar seconds_per_day = static_cast<Scalar>(86400);
        return eventsPerDeviceDay(bit_count) / seconds_per_day;
    }

    Scalar expectedDeviceEvents(std::uint64_t bit_count, Scalar duration_seconds) const
    {
        if (!std::isfinite(duration_seconds) || duration_seconds < 0) {
            throw std::invalid_argument("Mission duration must be finite and non-negative");
        }
        return eventsPerDeviceSecond(bit_count) * duration_seconds;
    }

   private:
    void validate(std::uint64_t bit_count) const
    {
        if (!std::isfinite(events_per_day_per_bit) || events_per_day_per_bit < 0 ||
            bit_count == 0) {
            throw std::invalid_argument("Event rate and device bit count must be valid");
        }
    }
};

/**
 * @brief Generate event arrival times from a constant-rate Poisson process
 *
 * The supplied engine controls reproducibility. Returned times are seconds
 * from the beginning of the requested interval.
 */
template <typename Scalar, typename RandomEngine>
std::vector<Scalar> samplePoissonArrivalTimes(Scalar event_rate_per_second,
                                              Scalar duration_seconds,
                                              RandomEngine& engine)
{
    if (!std::isfinite(event_rate_per_second) || event_rate_per_second < 0 ||
        !std::isfinite(duration_seconds) || duration_seconds < 0) {
        throw std::invalid_argument("Poisson rate and duration must be finite and non-negative");
    }

    std::vector<Scalar> arrival_times;
    if (event_rate_per_second == 0 || duration_seconds == 0) return arrival_times;

    std::exponential_distribution<Scalar> waiting_time(event_rate_per_second);
    Scalar time = 0;
    while (true) {
        time += waiting_time(engine);
        if (!std::isfinite(time) || time > duration_seconds) break;
        arrival_times.push_back(time);
    }
    return arrival_times;
}

}  // namespace physics
}  // namespace rad_ml
