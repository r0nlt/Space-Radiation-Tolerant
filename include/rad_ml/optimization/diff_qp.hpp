// Copyright (c) 2025 Space-Labs-AI
// Differentiable Quadratic Program layer (implicit differentiation via KKT)

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
#include <vector>

namespace rad_ml {
namespace optimization {

struct QPProblem {
    // Minimize 1/2 z^T P z + q^T z
    // s.t. A z = b, G z <= h
    Eigen::MatrixXd P;  // nxn (assumed positive semidefinite; we add eps I)
    Eigen::VectorXd q;  // n
    Eigen::MatrixXd A;  // m_eq x n (may be empty)
    Eigen::VectorXd b;  // m_eq
    Eigen::MatrixXd G;  // m_ineq x n (may be empty)
    Eigen::VectorXd h;  // m_ineq
    double regularization_eps{1e-6};
};

struct QPSolution {
    Eigen::VectorXd z;                    // primal optimum
    Eigen::VectorXd lambda;               // inequality duals (for active constraints)
    Eigen::VectorXd nu;                   // equality duals
    std::vector<int> active_set_indices;  // indices of active inequalities
};

// Optional cached context to reuse KKT factorization across fwd/bwd
struct QPContext {
    Eigen::MatrixXd kkt_matrix;
    Eigen::LDLT<Eigen::MatrixXd> kkt_ldlt;
    bool valid{false};
};

// A tiny differentiable QP layer. Forward: solve; Backward: implicit diff.
class DifferentiableQPLayer {
   public:
    QPSolution forward(const QPProblem &prob, QPContext *ctx = nullptr,
                       bool reuse_factorization = true);

    // Given upstream gradient dL/dz, compute gradients w.r.t. q (and optionally P)
    // Here we expose a minimal API: gradients for q and b, h (active rows only).
    struct BackwardGrads {
        Eigen::VectorXd dL_dq;  // equals v (adjoint) from KKT solve
        Eigen::MatrixXd dL_dP;  // symmetric part contribution: 0.5*(v z^T + z v^T)
        Eigen::VectorXd dL_db;  // = -nu_adj
        Eigen::VectorXd
            dL_dh;  // active-set only; the vector length equals prob.h.size(), zeros for inactive
    };

    BackwardGrads backward(const QPProblem &prob, const QPSolution &sol,
                           const Eigen::VectorXd &dL_dz, QPContext *ctx = nullptr);
};

}  // namespace optimization
}  // namespace rad_ml
