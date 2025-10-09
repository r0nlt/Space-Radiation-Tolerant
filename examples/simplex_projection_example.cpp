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

int main()
{
    Eigen::VectorXd x(5);
    x << 0.5, -0.1, 0.2, 1.3, 0.0;
    auto z = SimplexProjection::forward(x);
    std::cout << "z = " << z.transpose() << "\n";
    Eigen::VectorXd g = Eigen::VectorXd::Ones(5);
    auto vjp = SimplexProjection::backward(x, g);
    std::cout << "vjp = " << vjp.transpose() << "\n";
    return 0;
}
