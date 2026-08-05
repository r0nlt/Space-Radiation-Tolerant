#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace rad_ml {
namespace physics {

/**
 * @brief Circuit-side critical-charge model
 *
 * Critical charge is the minimum charge collected at a sensitive circuit node
 * that changes its logical state. It is intentionally modeled independently
 * from incident particle energy and LET; those belong to charge deposition and
 * collection models.
 */
template <typename Scalar = double>
class CriticalChargePowerLaw {
   public:
    struct CalibrationPoint {
        Scalar feature_size_nm;
        Scalar critical_charge_fc;
    };

    CriticalChargePowerLaw(Scalar coefficient, Scalar exponent)
        : coefficient_(coefficient), exponent_(exponent)
    {
        validatePositive(coefficient_, "coefficient");
        if (!std::isfinite(exponent_)) {
            throw std::invalid_argument("Critical-charge exponent must be finite");
        }
    }

    /**
     * @brief Fit Qcrit = coefficient * feature_size_nm^exponent in log space
     */
    static CriticalChargePowerLaw fit(const std::vector<CalibrationPoint>& points)
    {
        if (points.size() < 2) {
            throw std::invalid_argument("At least two critical-charge points are required");
        }

        Scalar sum_x = 0;
        Scalar sum_y = 0;
        Scalar sum_xx = 0;
        Scalar sum_xy = 0;
        for (const auto& point : points) {
            validatePositive(point.feature_size_nm, "feature size");
            validatePositive(point.critical_charge_fc, "critical charge");
            const Scalar x = std::log(point.feature_size_nm);
            const Scalar y = std::log(point.critical_charge_fc);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        const Scalar count = static_cast<Scalar>(points.size());
        const Scalar denominator = count * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) <= static_cast<Scalar>(1.0e-15)) {
            throw std::invalid_argument("Critical-charge calibration feature sizes are degenerate");
        }

        const Scalar exponent = (count * sum_xy - sum_x * sum_y) / denominator;
        const Scalar intercept = (sum_y - exponent * sum_x) / count;
        return CriticalChargePowerLaw(std::exp(intercept), exponent);
    }

    Scalar predict(Scalar feature_size_nm) const
    {
        validatePositive(feature_size_nm, "feature size");
        return coefficient_ * std::pow(feature_size_nm, exponent_);
    }

    Scalar coefficient() const noexcept { return coefficient_; }
    Scalar exponent() const noexcept { return exponent_; }

   private:
    Scalar coefficient_;
    Scalar exponent_;

    static void validatePositive(Scalar value, const char* name)
    {
        if (!std::isfinite(value) || value <= 0) {
            throw std::invalid_argument(std::string("Critical-charge ") + name +
                                        " must be finite and positive");
        }
    }
};

}  // namespace physics
}  // namespace rad_ml
