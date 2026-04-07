#pragma once

#ifdef __has_include
#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#else
#error "Could not find Eigen/Dense"
#endif
#else
#include <Eigen/Dense>
#endif

#include <algorithm>
#include <utility>
#include <vector>

namespace rad_ml {
namespace optimization {

// Project a vector x onto the probability simplex {z | z>=0, 1^T z = 1}
// Forward uses the well-known sorting-based algorithm (O(n log n)).
// Backward returns the VJP wrt x for a given upstream gradient g (dL/dz).
struct SimplexProjection {
    static Eigen::VectorXd forward(const Eigen::VectorXd &x)
    {
        const int n = static_cast<int>(x.size());
        Eigen::VectorXd u = x;
        std::vector<std::pair<double, int>> a;
        a.reserve(n);
        for (int i = 0; i < n; ++i) a.emplace_back(u(i), i);
        std::sort(a.begin(), a.end(), [](auto &l, auto &r) { return l.first > r.first; });
        double sum = 0.0;
        int rho = -1;
        double theta = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += a[i].first;
            double t = (sum - 1.0) / (i + 1);
            if (a[i].first - t > 0.0) {
                rho = i;
                theta = t;
            }
        }
        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        for (int i = 0; i <= rho; ++i) z(a[i].second) = std::max(0.0, a[i].first - theta);
        return z;
    }

    // Vector-Jacobian product with respect to input x
    // For active set A = {i | z_i > 0}, gradient is projected so sum to zero on A
    static Eigen::VectorXd backward(const Eigen::VectorXd &x, const Eigen::VectorXd &g_upstream)
    {
        Eigen::VectorXd z = forward(x);
        const int n = static_cast<int>(z.size());
        std::vector<int> active;
        active.reserve(n);
        for (int i = 0; i < n; ++i)
            if (z(i) > 0.0) active.push_back(i);
        if (active.empty()) return Eigen::VectorXd::Zero(n);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
        double mean = 0.0;
        for (int idx : active) mean += g_upstream(idx);
        mean /= static_cast<double>(active.size());
        for (int idx : active) g(idx) = g_upstream(idx) - mean;
        return g;
    }

    // Convenience wrapper for callers using std::vector.
    static std::vector<double> forward_vector(const std::vector<double> &x)
    {
        Eigen::VectorXd x_eig(static_cast<int>(x.size()));
        for (size_t i = 0; i < x.size(); ++i) {
            x_eig(static_cast<int>(i)) = x[i];
        }

        Eigen::VectorXd z_eig = forward(x_eig);
        std::vector<double> z(static_cast<size_t>(z_eig.size()), 0.0);
        for (int i = 0; i < z_eig.size(); ++i) {
            z[static_cast<size_t>(i)] = z_eig(i);
        }
        return z;
    }

    // Convenience wrapper for callers using std::vector.
    static std::vector<double> backward_vector(const std::vector<double> &x,
                                               const std::vector<double> &g_upstream)
    {
        if (x.size() != g_upstream.size()) {
            return std::vector<double>(x.size(), 0.0);
        }

        Eigen::VectorXd x_eig(static_cast<int>(x.size()));
        Eigen::VectorXd g_eig(static_cast<int>(g_upstream.size()));
        for (size_t i = 0; i < x.size(); ++i) {
            x_eig(static_cast<int>(i)) = x[i];
            g_eig(static_cast<int>(i)) = g_upstream[i];
        }

        Eigen::VectorXd d_eig = backward(x_eig, g_eig);
        std::vector<double> d(static_cast<size_t>(d_eig.size()), 0.0);
        for (int i = 0; i < d_eig.size(); ++i) {
            d[static_cast<size_t>(i)] = d_eig(i);
        }
        return d;
    }
};

}  // namespace optimization
}  // namespace rad_ml
