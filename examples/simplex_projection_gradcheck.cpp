#include <cmath>
#include <iostream>
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

#include <rad_ml/optimization/simplex_projection.hpp>

using namespace rad_ml::optimization;

static double loss(const Eigen::VectorXd &z) { return z.squaredNorm(); }

int main()
{
    Eigen::VectorXd x(6);
    x << 0.5, -0.2, 0.1, 0.9, -0.3, 0.0;
    Eigen::VectorXd z = SimplexProjection::forward(x);
    Eigen::VectorXd g_up = 2.0 * z;  // d/dz ||z||^2
    Eigen::VectorXd g = SimplexProjection::backward(x, g_up);

    // Numerical check w.r.t. x
    const double eps = 1e-6;
    Eigen::VectorXd num(x.size());
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd xp = x;
        xp(i) += eps;
        Eigen::VectorXd xm = x;
        xm(i) -= eps;
        double Lp = loss(SimplexProjection::forward(xp));
        double Lm = loss(SimplexProjection::forward(xm));
        num(i) = (Lp - Lm) / (2.0 * eps);
    }

    double max_err = (num - g).cwiseAbs().maxCoeff();
    std::cout << "max_err = " << max_err << "\n";
    if (max_err < 1e-4) {
        std::cout << "GRADCHECK PASSED\n";
        return 0;
    }
    std::cerr << "GRADCHECK FAILED\n";
    return 1;
}
