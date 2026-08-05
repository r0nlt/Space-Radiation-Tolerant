#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rad_ml/physics/omere_io.hpp"
#include "rad_ml/physics/see_cross_section.hpp"

namespace rad_ml {
namespace physics {

struct RectangularSensitiveVolume {
    double length_um = 0;
    double width_um = 0;
    double depth_um = 0;

    double planarAreaUm2() const { return length_um * width_um; }
    double totalSurfaceAreaUm2() const
    {
        return 2.0 * (length_um * width_um + length_um * depth_um +
                      width_um * depth_um);
    }
};

/**
 * @brief Deterministic chord distribution for isotropic incidence on an RPP
 *
 * Entry faces are weighted by area and inward directions use the cosine law.
 * Halton points make the numerical result reproducible across runs.
 */
class RectangularChordDistribution {
   public:
    explicit RectangularChordDistribution(const RectangularSensitiveVolume& volume,
                                          std::size_t sample_count = 262144)
        : volume_(volume)
    {
        if (!positiveFinite(volume.length_um) || !positiveFinite(volume.width_um) ||
            !positiveFinite(volume.depth_um) || sample_count < 1024) {
            throw std::invalid_argument("Invalid RPP dimensions or chord sample count");
        }

        const std::array<double, 3> face_areas = {
            volume.width_um * volume.depth_um,
            volume.length_um * volume.depth_um,
            volume.length_um * volume.width_um};
        const double face_area_sum = face_areas[0] + face_areas[1] + face_areas[2];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            face_weights_[axis] = face_areas[axis] / face_area_sum;
        }

        // Equal samples per orientation are an importance-stratified estimate.
        // This resolves the rare long-chord tail on the small top/bottom face
        // of a deep, narrow sensitive volume. Face-area weights recover the
        // physical isotropic-entry distribution.
        const std::size_t samples_per_axis = sample_count / 3;
        for (auto& chords : chords_by_axis_) chords.reserve(samples_per_axis);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (std::size_t index = 1; index <= samples_per_axis; ++index) {
                const double cos_theta = std::sqrt(halton(index, 2));
                const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                const double phi = 2.0 * pi * halton(index, 3);
                const double tangent_a = sin_theta * std::cos(phi);
                const double tangent_b = sin_theta * std::sin(phi);
                const double position_a = halton(index, 5);
                const double position_b = halton(index, 7);

                double x = 0;
                double y = 0;
                double z = 0;
                double dx = 0;
                double dy = 0;
                double dz = 0;
                if (axis == 0) {
                    y = position_a * volume.width_um;
                    z = position_b * volume.depth_um;
                    dx = cos_theta;
                    dy = tangent_a;
                    dz = tangent_b;
                } else if (axis == 1) {
                    x = position_a * volume.length_um;
                    z = position_b * volume.depth_um;
                    dx = tangent_a;
                    dy = cos_theta;
                    dz = tangent_b;
                } else {
                    x = position_a * volume.length_um;
                    y = position_b * volume.width_um;
                    dx = tangent_a;
                    dy = tangent_b;
                    dz = cos_theta;
                }

                const double chord = std::min(
                    {distanceToBoundary(x, dx, volume.length_um),
                     distanceToBoundary(y, dy, volume.width_um),
                     distanceToBoundary(z, dz, volume.depth_um)});
                if (!positiveFinite(chord)) {
                    throw std::runtime_error("Failed to generate a finite RPP chord");
                }
                chords_by_axis_[axis].push_back(chord);
            }
            std::sort(chords_by_axis_[axis].begin(), chords_by_axis_[axis].end());
        }
    }

    double probabilityGreaterThan(double distance_um) const
    {
        if (!std::isfinite(distance_um)) return 0;
        if (distance_um <= 0) return 1;
        double probability = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto& chords = chords_by_axis_[axis];
            const auto first_greater =
                std::upper_bound(chords.begin(), chords.end(), distance_um);
            probability +=
                face_weights_[axis] * static_cast<double>(chords.end() - first_greater) /
                static_cast<double>(chords.size());
        }
        return probability;
    }

    double meanChordUm() const
    {
        double mean = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            double axis_total = 0;
            for (const double chord : chords_by_axis_[axis]) axis_total += chord;
            mean += face_weights_[axis] * axis_total /
                    static_cast<double>(chords_by_axis_[axis].size());
        }
        return mean;
    }

    std::size_t sampleCount() const noexcept
    {
        return chords_by_axis_[0].size() + chords_by_axis_[1].size() +
               chords_by_axis_[2].size();
    }

   private:
    static constexpr double pi = 3.141592653589793238462643383279502884;
    RectangularSensitiveVolume volume_;
    std::array<std::vector<double>, 3> chords_by_axis_;
    std::array<double, 3> face_weights_{};

    static bool positiveFinite(double value) { return std::isfinite(value) && value > 0; }

    static double halton(std::size_t index, std::size_t base)
    {
        double result = 0;
        double fraction = 1.0 / static_cast<double>(base);
        while (index > 0) {
            result += fraction * static_cast<double>(index % base);
            index /= base;
            fraction /= static_cast<double>(base);
        }
        return result;
    }

    static double distanceToBoundary(double position, double direction, double size)
    {
        constexpr double epsilon = 1.0e-15;
        if (direction > epsilon) return (size - position) / direction;
        if (direction < -epsilon) return -position / direction;
        return std::numeric_limits<double>::infinity();
    }
};

namespace irpp_detail {

inline std::vector<std::pair<double, double>> gaussLegendre(std::size_t order,
                                                            double lower,
                                                            double upper)
{
    if (order < 2 || !std::isfinite(lower) || !std::isfinite(upper) ||
        upper <= lower) {
        throw std::invalid_argument("Invalid Gauss-Legendre quadrature interval");
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double tolerance = 1.0e-15;
    std::vector<std::pair<double, double>> result(order);
    const std::size_t roots = (order + 1) / 2;
    const double midpoint = 0.5 * (lower + upper);
    const double half_width = 0.5 * (upper - lower);
    for (std::size_t i = 0; i < roots; ++i) {
        double root =
            std::cos(pi * (static_cast<double>(i) + 0.75) /
                     (static_cast<double>(order) + 0.5));
        double derivative = 0;
        for (std::size_t iteration = 0; iteration < 64; ++iteration) {
            double previous = 1;
            double current = root;
            for (std::size_t degree = 2; degree <= order; ++degree) {
                const double next =
                    ((2.0 * static_cast<double>(degree) - 1.0) * root * current -
                     (static_cast<double>(degree) - 1.0) * previous) /
                    static_cast<double>(degree);
                previous = current;
                current = next;
            }
            derivative = static_cast<double>(order) * (root * current - previous) /
                         (root * root - 1.0);
            const double update = current / derivative;
            root -= update;
            if (std::abs(update) <= tolerance) break;
        }
        const double weight =
            half_width * 2.0 / ((1.0 - root * root) * derivative * derivative);
        result[i] = {midpoint - half_width * root, weight};
        result[order - 1 - i] = {midpoint + half_width * root, weight};
    }
    return result;
}

}  // namespace irpp_detail

/**
 * @brief Full-range integral chord distribution of a rectangular volume
 *
 * For a direction n in one octant, lines with chord length greater than d
 * have projected area
 *
 *   -d/dd [(a-d nx)(b-d ny)(c-d nz)].
 *
 * Integrating that exact projected area over isotropic directions and
 * normalizing by the projected area at d=0 gives P(chord > d). This is the
 * integral form of the exact RPP chord construction and remains valid through
 * the body diagonal; only the smooth two-dimensional angular integral is
 * evaluated numerically.
 */
class ExactRectangularChordDistribution {
   public:
    explicit ExactRectangularChordDistribution(const RectangularSensitiveVolume& volume,
                                               std::size_t angular_order = 24)
        : volume_(volume)
    {
        if (!positiveFinite(volume.length_um) || !positiveFinite(volume.width_um) ||
            !positiveFinite(volume.depth_um) || angular_order < 8) {
            throw std::invalid_argument("Invalid RPP dimensions or angular order");
        }

        constexpr double half_pi = 1.570796326794896619231321691639751442;
        const auto theta_nodes =
            irpp_detail::gaussLegendre(angular_order, 0.0, half_pi);
        const auto phi_nodes =
            irpp_detail::gaussLegendre(angular_order, 0.0, half_pi);
        directions_.reserve(angular_order * angular_order);
        for (const auto& theta : theta_nodes) {
            const double sin_theta = std::sin(theta.first);
            const double cos_theta = std::cos(theta.first);
            for (const auto& phi : phi_nodes) {
                const Direction direction{
                    sin_theta * std::cos(phi.first),
                    sin_theta * std::sin(phi.first),
                    cos_theta,
                    theta.second * phi.second * sin_theta};
                directions_.push_back(direction);
                normalization_ += direction.weight * projectedArea(direction);
            }
        }
        if (!positiveFinite(normalization_)) {
            throw std::runtime_error("Failed to normalize exact RPP chord distribution");
        }
    }

    double probabilityGreaterThan(double distance_um) const
    {
        if (!std::isfinite(distance_um)) return 0;
        if (distance_um <= 0) return 1;
        if (distance_um >= bodyDiagonalUm()) return 0;

        double numerator = 0;
        for (const auto& direction : directions_) {
            const double x = volume_.length_um - distance_um * direction.x;
            const double y = volume_.width_um - distance_um * direction.y;
            const double z = volume_.depth_um - distance_um * direction.z;
            if (x <= 0 || y <= 0 || z <= 0) continue;
            const double surviving_projected_area =
                direction.x * y * z + direction.y * x * z + direction.z * x * y;
            numerator += direction.weight * surviving_projected_area;
        }
        return std::max(0.0, std::min(1.0, numerator / normalization_));
    }

    double meanChordUm() const
    {
        return 4.0 * volume_.length_um * volume_.width_um * volume_.depth_um /
               volume_.totalSurfaceAreaUm2();
    }

    double bodyDiagonalUm() const
    {
        return std::sqrt(volume_.length_um * volume_.length_um +
                         volume_.width_um * volume_.width_um +
                         volume_.depth_um * volume_.depth_um);
    }

   private:
    struct Direction {
        double x;
        double y;
        double z;
        double weight;
    };

    RectangularSensitiveVolume volume_;
    std::vector<Direction> directions_;
    double normalization_ = 0;

    static bool positiveFinite(double value) { return std::isfinite(value) && value > 0; }

    double projectedArea(const Direction& direction) const
    {
        return direction.x * volume_.width_um * volume_.depth_um +
               direction.y * volume_.length_um * volume_.depth_um +
               direction.z * volume_.length_um * volume_.width_um;
    }
};

inline RectangularSensitiveVolume squareSensitiveVolumeFromSaturation(
    double saturation_cross_section_cm2_per_bit, double depth_um)
{
    constexpr double square_micrometers_per_square_centimeter = 1.0e8;
    if (!std::isfinite(saturation_cross_section_cm2_per_bit) ||
        saturation_cross_section_cm2_per_bit <= 0 || !std::isfinite(depth_um) ||
        depth_um <= 0) {
        throw std::invalid_argument("Invalid saturation cross-section or sensitive depth");
    }
    const double side_um = std::sqrt(
        saturation_cross_section_cm2_per_bit *
        square_micrometers_per_square_centimeter);
    return {side_um, side_um, depth_um};
}

struct IrppNumerics {
    enum class SpectrumIntegration {
        DifferentialLinearSimpson,
        DifferentialLinearTrapezoid,
        IntegralBinGeometricMidpoint,
        IntegralBinArithmeticMidpoint
    };

    std::size_t angular_order = 24;
    std::size_t weibull_order = 48;
    double weibull_tail_probability = 1.0e-12;
    SpectrumIntegration spectrum_integration =
        SpectrumIntegration::DifferentialLinearSimpson;
};

inline double irppGeometryFactor(const RectangularSensitiveVolume& volume)
{
    return volume.totalSurfaceAreaUm2() / (4.0 * volume.planarAreaUm2());
}

inline double interpolateDifferentialFlux(const OmereLetPoint& left,
                                          const OmereLetPoint& right, double let)
{
    const double fraction =
        (let - left.let_mev_cm2_mg) /
        (right.let_mev_cm2_mg - left.let_mev_cm2_mg);
    return left.differential_flux_per_cm2_s_per_let +
           fraction * (right.differential_flux_per_cm2_s_per_let -
                       left.differential_flux_per_cm2_s_per_let);
}

inline double rppEffectiveFlux(const OmereLetSpectrum& spectrum,
                               const ExactRectangularChordDistribution& chords,
                               double sensitive_depth_um, double threshold_let,
                               IrppNumerics::SpectrumIntegration integration =
                                   IrppNumerics::SpectrumIntegration::
                                       DifferentialLinearSimpson)
{
    if (spectrum.points.size() < 2) {
        throw std::invalid_argument("IRPP calculation requires an LET spectrum");
    }
    if (!std::isfinite(sensitive_depth_um) || sensitive_depth_um <= 0 ||
        !std::isfinite(threshold_let) || threshold_let < 0) {
        throw std::invalid_argument("Invalid RPP flux inputs");
    }

    // Composite Simpson integration in each exported LET interval. OMERE's
    // differential spectrum is linear on this grid to the precision of the
    // accompanying integral column, while the chord probability is evaluated
    // continuously rather than at one representative LET.
    double effective_flux = 0;
    for (std::size_t i = 1; i < spectrum.points.size(); ++i) {
        const auto& left = spectrum.points[i - 1];
        const auto& right = spectrum.points[i];
        if (integration ==
                IrppNumerics::SpectrumIntegration::IntegralBinGeometricMidpoint ||
            integration ==
                IrppNumerics::SpectrumIntegration::IntegralBinArithmeticMidpoint) {
            const double bin_flux =
                left.integral_flux_cm2_s - right.integral_flux_cm2_s;
            if (bin_flux <= 0) continue;
            const double representative_let =
                integration ==
                        IrppNumerics::SpectrumIntegration::
                            IntegralBinGeometricMidpoint
                    ? std::sqrt(left.let_mev_cm2_mg * right.let_mev_cm2_mg)
                    : 0.5 * (left.let_mev_cm2_mg + right.let_mev_cm2_mg);
            const double required_chord =
                sensitive_depth_um * threshold_let / representative_let;
            effective_flux +=
                bin_flux * chords.probabilityGreaterThan(required_chord);
            continue;
        }

        const double midpoint_let =
            0.5 * (left.let_mev_cm2_mg + right.let_mev_cm2_mg);
        const auto integrand = [&](double particle_let) {
            if (particle_let <= 0) return 0.0;
            const double required_chord =
                sensitive_depth_um * threshold_let / particle_let;
            return interpolateDifferentialFlux(left, right, particle_let) *
                   chords.probabilityGreaterThan(required_chord);
        };
        if (integration ==
            IrppNumerics::SpectrumIntegration::DifferentialLinearTrapezoid) {
            effective_flux +=
                0.5 * (right.let_mev_cm2_mg - left.let_mev_cm2_mg) *
                (integrand(left.let_mev_cm2_mg) +
                 integrand(right.let_mev_cm2_mg));
        } else {
            effective_flux +=
                (right.let_mev_cm2_mg - left.let_mev_cm2_mg) *
                (integrand(left.let_mev_cm2_mg) + 4.0 * integrand(midpoint_let) +
                 integrand(right.let_mev_cm2_mg)) /
                6.0;
        }
    }
    return effective_flux;
}

inline double calculateIrppRatePerBitSecond(
    const OmereLetSpectrum& spectrum, const WeibullCrossSection<double>& cross_section,
    double sensitive_depth_um, const IrppNumerics& numerics = {})
{
    if (numerics.angular_order < 8 || numerics.weibull_order < 8 ||
        !std::isfinite(numerics.weibull_tail_probability) ||
        numerics.weibull_tail_probability <= 0 ||
        numerics.weibull_tail_probability >= 1) {
        throw std::invalid_argument("Invalid IRPP numerical controls");
    }

    const auto volume = squareSensitiveVolumeFromSaturation(
        cross_section.saturation(), sensitive_depth_um);
    const ExactRectangularChordDistribution chords(volume, numerics.angular_order);

    // With t=((L-L0)/W)^S, d(sigma)/dt=sigma_sat*exp(-t). This removes
    // the Weibull threshold singularity and evaluates the Petersen Stieltjes
    // integral directly, independently of the environmental LET grid.
    const double maximum_t = -std::log(numerics.weibull_tail_probability);
    const auto weibull_nodes =
        irpp_detail::gaussLegendre(numerics.weibull_order, 0.0, maximum_t);
    double normalized_rate = 0;
    for (const auto& node : weibull_nodes) {
        const double effective_threshold =
            cross_section.onset() +
            cross_section.width() *
                std::pow(node.first, 1.0 / cross_section.shape());
        normalized_rate +=
            node.second * std::exp(-node.first) *
            rppEffectiveFlux(spectrum, chords, sensitive_depth_um, effective_threshold,
                             numerics.spectrum_integration);
    }
    return irppGeometryFactor(volume) * cross_section.saturation() * normalized_rate;
}

inline double calculateIrppRatePerBitDay(
    const OmereLetSpectrum& spectrum, const WeibullCrossSection<double>& cross_section,
    double sensitive_depth_um, const IrppNumerics& numerics = {})
{
    constexpr double seconds_per_day = 86400.0;
    return seconds_per_day * calculateIrppRatePerBitSecond(
                                 spectrum, cross_section, sensitive_depth_um, numerics);
}

}  // namespace physics
}  // namespace rad_ml
