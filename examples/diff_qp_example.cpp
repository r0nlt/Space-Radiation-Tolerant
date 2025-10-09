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
#include <iostream>
#include <rad_ml/optimization/diff_qp.hpp>

using namespace rad_ml::optimization;

int main()
{
    // Simple QP: minimize 1/2 ||z||^2 + q^T z s.t. z >= 0 and sum z = 1
    const int n = 3;
    QPProblem prob;
    prob.P = Eigen::MatrixXd::Identity(n, n);
    prob.q = Eigen::VectorXd::Zero(n);
    prob.A = Eigen::RowVector3d(1.0, 1.0, 1.0);
    prob.b = Eigen::VectorXd::Ones(1);
    prob.G = -Eigen::MatrixXd::Identity(n, n);  // -z <= 0 => z >= 0
    prob.h = Eigen::VectorXd::Zero(n);
    prob.regularization_eps = 1e-8;

    DifferentiableQPLayer layer;
    QPContext ctx;
    QPSolution sol = layer.forward(prob, &ctx, true);
    std::cout << "z* = " << sol.z.transpose() << "\n";

    // Gradient of loss L = 0.5 * ||z||^2 w.r.t z is dL/dz = z
    Eigen::VectorXd dL_dz = sol.z;
    auto grads = layer.backward(prob, sol, dL_dz, &ctx);
    std::cout << "dL/dq = " << grads.dL_dq.transpose() << "\n";
    std::cout << "Active set size = " << sol.active_set_indices.size() << "\n";
    return 0;
}
