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

#include <rad_ml/optimization/diff_qp.hpp>

using namespace rad_ml::optimization;

static double loss_value(const Eigen::VectorXd &z) { return 0.5 * z.squaredNorm(); }

int main()
{
    const int n = 3;
    QPProblem prob;
    prob.P = Eigen::MatrixXd::Identity(n, n);
    prob.q = Eigen::VectorXd::LinSpaced(n, -0.2, 0.2);  // non-symmetric q
    prob.A = Eigen::RowVector3d(1.0, 1.0, 1.0);
    prob.b = Eigen::VectorXd::Ones(1);
    prob.G = -Eigen::MatrixXd::Identity(n, n);
    prob.h = Eigen::VectorXd::Zero(n);
    prob.regularization_eps = 1e-8;

    DifferentiableQPLayer layer;
    QPContext ctx;
    QPSolution sol = layer.forward(prob, &ctx, true);

    // Analytic gradient via implicit differentiation
    Eigen::VectorXd dL_dz = sol.z;  // gradient of 0.5||z||^2
    auto grads = layer.backward(prob, sol, dL_dz, &ctx);

    // Numerical gradient wrt q by central difference
    const double eps = 1e-5;
    Eigen::VectorXd num(n);
    for (int i = 0; i < n; ++i) {
        QPProblem prob_p = prob;
        QPProblem prob_m = prob;
        prob_p.q(i) += eps;
        prob_m.q(i) -= eps;
        QPSolution sp = layer.forward(prob_p);
        QPSolution sm = layer.forward(prob_m);
        double Lp = loss_value(sp.z);
        double Lm = loss_value(sm.z);
        num(i) = (Lp - Lm) / (2.0 * eps);
    }

    double max_abs_err = (num - grads.dL_dq).cwiseAbs().maxCoeff();
    std::cout << "analytical dL/dq = " << grads.dL_dq.transpose() << "\n";
    std::cout << "numerical  dL/dq = " << num.transpose() << "\n";
    std::cout << "max_abs_err = " << max_abs_err << "\n";

    // Simple success criterion
    if (max_abs_err < 1e-4) {
        std::cout << "GRADCHECK PASSED\n";
        return 0;
    }
    else {
        std::cerr << "GRADCHECK FAILED\n";
        return 1;
    }
}
