#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rad_ml {
namespace physics {

enum class UpsetMultiplicity : std::size_t {
    SingleBit = 0,
    TwoBit = 1,
    ThreeBit = 2,
    FourBit = 3
};

template <typename Scalar = double>
struct HeavyIonBeamRun {
    Scalar let_at_active_silicon_mev_cm2_mg = 0;
    Scalar fluence_ions_cm2 = 0;
    std::uint64_t exposed_bits = 0;
    std::array<std::uint64_t, 4> event_counts{};
    std::string facility;
    std::string ion;

    std::uint64_t events(UpsetMultiplicity multiplicity) const
    {
        return event_counts.at(static_cast<std::size_t>(multiplicity));
    }

    Scalar exposure_bit_ions() const
    {
        return fluence_ions_cm2 * static_cast<Scalar>(exposed_bits);
    }
};

template <typename Scalar = double>
struct CrossSectionPoint {
    Scalar let_mev_cm2_mg = 0;
    Scalar cross_section_cm2_per_bit = 0;
};

template <typename Scalar>
CrossSectionPoint<Scalar> estimateCrossSection(const HeavyIonBeamRun<Scalar>& run,
                                               UpsetMultiplicity multiplicity)
{
    if (!std::isfinite(run.let_at_active_silicon_mev_cm2_mg) ||
        run.let_at_active_silicon_mev_cm2_mg < 0 || !std::isfinite(run.fluence_ions_cm2) ||
        run.fluence_ions_cm2 <= 0 || run.exposed_bits == 0) {
        throw std::invalid_argument("Beam-run LET, fluence, and exposed bits must be valid");
    }

    return {run.let_at_active_silicon_mev_cm2_mg,
            static_cast<Scalar>(run.events(multiplicity)) / run.exposure_bit_ions()};
}

/**
 * @brief One-sided Poisson upper bound for a run with zero observed events
 *
 * alpha is the upper-tail probability (0.05 gives a conventional 95% bound).
 */
template <typename Scalar>
Scalar zeroCountCrossSectionUpperBound(const HeavyIonBeamRun<Scalar>& run, Scalar alpha)
{
    if (run.exposure_bit_ions() <= 0 || !std::isfinite(alpha) || alpha <= 0 || alpha >= 1) {
        throw std::invalid_argument("Invalid zero-count upper-bound inputs");
    }
    return -std::log(alpha) / run.exposure_bit_ions();
}

template <typename Scalar = double>
class WeibullCrossSection {
   public:
    WeibullCrossSection(Scalar saturation_cm2_per_bit, Scalar onset_let, Scalar width,
                        Scalar shape)
        : saturation_(saturation_cm2_per_bit), onset_(onset_let), width_(width), shape_(shape)
    {
        if (!positiveFinite(saturation_) || !std::isfinite(onset_) || onset_ < 0 ||
            !positiveFinite(width_) || !positiveFinite(shape_)) {
            throw std::invalid_argument("Invalid Weibull cross-section parameters");
        }
    }

    Scalar evaluate(Scalar let_mev_cm2_mg) const
    {
        if (!std::isfinite(let_mev_cm2_mg) || let_mev_cm2_mg < 0) {
            throw std::invalid_argument("LET must be finite and non-negative");
        }
        if (let_mev_cm2_mg <= onset_) {
            return 0;
        }
        const Scalar normalized = (let_mev_cm2_mg - onset_) / width_;
        return saturation_ *
               (static_cast<Scalar>(1) - std::exp(-std::pow(normalized, shape_)));
    }

    Scalar saturation() const noexcept { return saturation_; }
    Scalar onset() const noexcept { return onset_; }
    Scalar width() const noexcept { return width_; }
    Scalar shape() const noexcept { return shape_; }

   private:
    Scalar saturation_;
    Scalar onset_;
    Scalar width_;
    Scalar shape_;

    static bool positiveFinite(Scalar value) { return std::isfinite(value) && value > 0; }
};

/**
 * @brief Linear interpolation of measured cross-sections
 *
 * Evaluation outside the measured LET range is rejected. This prevents silent
 * assumptions about threshold or saturation when beam data do not establish
 * them, as in currently published 7-nm Versal data.
 */
template <typename Scalar = double>
class TabulatedCrossSection {
   public:
    explicit TabulatedCrossSection(std::vector<CrossSectionPoint<Scalar>> points)
        : points_(std::move(points))
    {
        if (points_.size() < 2) {
            throw std::invalid_argument("At least two cross-section points are required");
        }
        std::sort(points_.begin(), points_.end(),
                  [](const auto& left, const auto& right) {
                      return left.let_mev_cm2_mg < right.let_mev_cm2_mg;
                  });
        for (std::size_t i = 0; i < points_.size(); ++i) {
            const auto& point = points_[i];
            if (!std::isfinite(point.let_mev_cm2_mg) || point.let_mev_cm2_mg < 0 ||
                !std::isfinite(point.cross_section_cm2_per_bit) ||
                point.cross_section_cm2_per_bit < 0 ||
                (i > 0 && point.let_mev_cm2_mg <= points_[i - 1].let_mev_cm2_mg)) {
                throw std::invalid_argument("Invalid tabulated cross-section point");
            }
        }
    }

    Scalar evaluate(Scalar let_mev_cm2_mg) const
    {
        if (!std::isfinite(let_mev_cm2_mg) || let_mev_cm2_mg < points_.front().let_mev_cm2_mg ||
            let_mev_cm2_mg > points_.back().let_mev_cm2_mg) {
            throw std::out_of_range("LET is outside the measured cross-section range");
        }

        const auto upper = std::lower_bound(
            points_.begin(), points_.end(), let_mev_cm2_mg,
            [](const auto& point, Scalar value) { return point.let_mev_cm2_mg < value; });
        if (upper == points_.begin() || upper->let_mev_cm2_mg == let_mev_cm2_mg) {
            return upper->cross_section_cm2_per_bit;
        }

        const auto lower = upper - 1;
        const Scalar fraction = (let_mev_cm2_mg - lower->let_mev_cm2_mg) /
                                (upper->let_mev_cm2_mg - lower->let_mev_cm2_mg);
        return lower->cross_section_cm2_per_bit +
               fraction *
                   (upper->cross_section_cm2_per_bit - lower->cross_section_cm2_per_bit);
    }

    const std::vector<CrossSectionPoint<Scalar>>& points() const noexcept { return points_; }

   private:
    std::vector<CrossSectionPoint<Scalar>> points_;
};

template <typename Scalar = double>
struct DifferentialLetFluxPoint {
    Scalar let_mev_cm2_mg = 0;
    Scalar flux_per_cm2_s_per_let = 0;
};

/**
 * @brief Integrate differential LET flux times per-bit cross-section
 */
template <typename Scalar, typename CrossSectionModel>
Scalar integrateEventRate(const std::vector<DifferentialLetFluxPoint<Scalar>>& spectrum,
                          const CrossSectionModel& model, std::uint64_t bit_count)
{
    if (spectrum.size() < 2 || bit_count == 0) {
        throw std::invalid_argument("Event-rate integration needs a spectrum and exposed bits");
    }

    Scalar rate_per_bit_s = 0;
    for (std::size_t i = 1; i < spectrum.size(); ++i) {
        const auto& left = spectrum[i - 1];
        const auto& right = spectrum[i];
        if (!std::isfinite(left.let_mev_cm2_mg) || !std::isfinite(right.let_mev_cm2_mg) ||
            !std::isfinite(left.flux_per_cm2_s_per_let) ||
            !std::isfinite(right.flux_per_cm2_s_per_let) ||
            left.flux_per_cm2_s_per_let < 0 || right.flux_per_cm2_s_per_let < 0 ||
            right.let_mev_cm2_mg <= left.let_mev_cm2_mg) {
            throw std::invalid_argument("Invalid differential LET spectrum");
        }
        const Scalar left_integrand =
            left.flux_per_cm2_s_per_let * model.evaluate(left.let_mev_cm2_mg);
        const Scalar right_integrand =
            right.flux_per_cm2_s_per_let * model.evaluate(right.let_mev_cm2_mg);
        rate_per_bit_s += static_cast<Scalar>(0.5) * (left_integrand + right_integrand) *
                          (right.let_mev_cm2_mg - left.let_mev_cm2_mg);
    }
    return rate_per_bit_s * static_cast<Scalar>(bit_count);
}

template <typename Scalar, typename RandomEngine>
std::uint64_t samplePoissonEventCount(Scalar event_rate_per_s, Scalar duration_s,
                                      RandomEngine& engine)
{
    if (!std::isfinite(event_rate_per_s) || event_rate_per_s < 0 ||
        !std::isfinite(duration_s) || duration_s < 0) {
        throw std::invalid_argument("Poisson event rate and duration must be non-negative");
    }
    const Scalar expected_events = event_rate_per_s * duration_s;
    if (expected_events > static_cast<Scalar>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Expected event count exceeds Poisson sampler range");
    }
    std::poisson_distribution<std::uint64_t> distribution(expected_events);
    return distribution(engine);
}

template <typename Scalar, typename CrossSectionModel>
Scalar poissonLogLikelihood(const std::vector<HeavyIonBeamRun<Scalar>>& runs,
                            UpsetMultiplicity multiplicity, const CrossSectionModel& model)
{
    Scalar log_likelihood = 0;
    for (const auto& run : runs) {
        const Scalar expected =
            run.exposure_bit_ions() * model.evaluate(run.let_at_active_silicon_mev_cm2_mg);
        const Scalar observed = static_cast<Scalar>(run.events(multiplicity));
        if (!std::isfinite(expected) || expected < 0 || (expected == 0 && observed > 0)) {
            return -std::numeric_limits<Scalar>::infinity();
        }
        if (expected > 0) {
            log_likelihood +=
                observed * std::log(expected) - expected - std::lgamma(observed + 1);
        }
    }
    return log_likelihood;
}

}  // namespace physics
}  // namespace rad_ml
