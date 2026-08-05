#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
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

        chords_um_.reserve(sample_count);
        const double x_face_area = volume.width_um * volume.depth_um;
        const double y_face_area = volume.length_um * volume.depth_um;
        const double z_face_area = volume.length_um * volume.width_um;
        const double face_area_sum = x_face_area + y_face_area + z_face_area;
        const double x_limit = x_face_area / face_area_sum;
        const double y_limit = (x_face_area + y_face_area) / face_area_sum;

        for (std::size_t index = 1; index <= sample_count; ++index) {
            const double face = halton(index, 2);
            const double cos_theta = std::sqrt(halton(index, 3));
            const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
            const double phi = 2.0 * pi * halton(index, 5);
            const double tangent_a = sin_theta * std::cos(phi);
            const double tangent_b = sin_theta * std::sin(phi);
            const double position_a = halton(index, 7);
            const double position_b = halton(index, 11);

            double x = 0;
            double y = 0;
            double z = 0;
            double dx = 0;
            double dy = 0;
            double dz = 0;
            if (face < x_limit) {
                y = position_a * volume.width_um;
                z = position_b * volume.depth_um;
                dx = cos_theta;
                dy = tangent_a;
                dz = tangent_b;
            } else if (face < y_limit) {
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
            chords_um_.push_back(chord);
        }
        std::sort(chords_um_.begin(), chords_um_.end());
    }

    double probabilityGreaterThan(double distance_um) const
    {
        if (!std::isfinite(distance_um)) return 0;
        if (distance_um <= 0) return 1;
        const auto first_greater =
            std::upper_bound(chords_um_.begin(), chords_um_.end(), distance_um);
        return static_cast<double>(chords_um_.end() - first_greater) /
               static_cast<double>(chords_um_.size());
    }

    double meanChordUm() const
    {
        double total = 0;
        for (const double chord : chords_um_) total += chord;
        return total / static_cast<double>(chords_um_.size());
    }

    std::size_t sampleCount() const noexcept { return chords_um_.size(); }

   private:
    static constexpr double pi = 3.141592653589793238462643383279502884;
    RectangularSensitiveVolume volume_;
    std::vector<double> chords_um_;

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

inline double calculateIrppRatePerBitSecond(
    const OmereLetSpectrum& spectrum, const WeibullCrossSection<double>& cross_section,
    double sensitive_depth_um, std::size_t chord_samples = 262144)
{
    if (spectrum.points.size() < 2) {
        throw std::invalid_argument("IRPP calculation requires an LET spectrum");
    }

    const auto volume = squareSensitiveVolumeFromSaturation(
        cross_section.saturation(), sensitive_depth_um);
    const RectangularChordDistribution chords(volume, chord_samples);
    const double geometry_factor =
        volume.totalSurfaceAreaUm2() / (4.0 * volume.planarAreaUm2());

    double rate_per_second = 0;
    double lower_threshold = cross_section.onset();
    double lower_cross_section = 0;
    for (const auto& threshold_point : spectrum.points) {
        const double upper_threshold = threshold_point.let_mev_cm2_mg;
        if (upper_threshold <= lower_threshold) continue;

        const double upper_cross_section = cross_section.evaluate(upper_threshold);
        const double cross_section_increment = upper_cross_section - lower_cross_section;
        if (cross_section_increment > 0) {
            const double midpoint_cross_section =
                0.5 * (lower_cross_section + upper_cross_section);
            const double fraction = midpoint_cross_section / cross_section.saturation();
            const double effective_threshold =
                cross_section.onset() +
                cross_section.width() *
                    std::pow(-std::log1p(-fraction), 1.0 / cross_section.shape());

            double effective_flux = 0;
            for (std::size_t i = 1; i < spectrum.points.size(); ++i) {
                const auto& left = spectrum.points[i - 1];
                const auto& right = spectrum.points[i];
                const double bin_flux =
                    left.integral_flux_cm2_s - right.integral_flux_cm2_s;
                if (bin_flux <= 0) continue;
                const double particle_let =
                    std::sqrt(left.let_mev_cm2_mg * right.let_mev_cm2_mg);
                const double required_chord_um =
                    sensitive_depth_um * effective_threshold / particle_let;
                effective_flux +=
                    bin_flux * chords.probabilityGreaterThan(required_chord_um);
            }
            rate_per_second +=
                geometry_factor * cross_section_increment * effective_flux;
        }
        lower_threshold = upper_threshold;
        lower_cross_section = upper_cross_section;
    }
    return rate_per_second;
}

inline double calculateIrppRatePerBitDay(
    const OmereLetSpectrum& spectrum, const WeibullCrossSection<double>& cross_section,
    double sensitive_depth_um, std::size_t chord_samples = 262144)
{
    constexpr double seconds_per_day = 86400.0;
    return seconds_per_day * calculateIrppRatePerBitSecond(
                                 spectrum, cross_section, sensitive_depth_um, chord_samples);
}

}  // namespace physics
}  // namespace rad_ml
