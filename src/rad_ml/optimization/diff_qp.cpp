#include <algorithm>
#include <rad_ml/optimization/diff_qp.hpp>

namespace rad_ml {
namespace optimization {

static Eigen::VectorXi buildActiveSet(const Eigen::MatrixXd &G, const Eigen::VectorXd &h,
                                      const Eigen::VectorXd &z, double tol = 1e-7)
{
    const int m = static_cast<int>(h.size());
    std::vector<int> active;
    active.reserve(m);
    for (int i = 0; i < m; ++i) {
        const double slack = h(i) - G.row(i).dot(z);
        if (slack <= tol) {
            active.push_back(i);
        }
    }
    Eigen::VectorXi idx(active.size());
    for (int i = 0; i < static_cast<int>(active.size()); ++i) idx(i) = active[i];
    return idx;
}

QPSolution DifferentiableQPLayer::forward(const QPProblem &prob, QPContext *ctx,
                                          bool reuse_factorization)
{
    const int n = static_cast<int>(prob.q.size());
    const int meq = static_cast<int>(prob.A.rows());
    const bool has_eq = meq > 0;

    // Solve equality-constrained QP first (ignore inequalities to get candidate z)
    Eigen::MatrixXd P_reg = prob.P;
    P_reg.diagonal().array() += prob.regularization_eps;

    // KKT system for eq constraints:
    // [P  A^T][z] = [-q]
    // [A  0  ][nu]  [ b]
    Eigen::MatrixXd KKT(n + meq, n + meq);
    KKT.setZero();
    KKT.topLeftCorner(n, n) = P_reg;
    if (has_eq) {
        KKT.topRightCorner(n, meq) = prob.A.transpose();
        KKT.bottomLeftCorner(meq, n) = prob.A;
    }

    Eigen::VectorXd rhs(n + meq);
    rhs.head(n) = -prob.q;
    if (has_eq) rhs.tail(meq) = prob.b;

    Eigen::LDLT<Eigen::MatrixXd> ldlt1;
    ldlt1.compute(KKT);
    if (ldlt1.info() != Eigen::Success) {
        // Fallback: add more regularization and retry
        KKT.topLeftCorner(n, n).diagonal().array() += std::max(1e-8, prob.regularization_eps);
        ldlt1.compute(KKT);
    }
    if (ldlt1.info() != Eigen::Success) {
        throw std::runtime_error("KKT LDLT factorization failed in forward (eq-only solve)");
    }
    Eigen::VectorXd sol = ldlt1.solve(rhs);
    QPSolution out;
    out.z = sol.head(n);
    out.nu = has_eq ? sol.tail(meq) : Eigen::VectorXd();

    // Project inequalities by identifying active set at the candidate solution
    Eigen::VectorXi active_idx = buildActiveSet(prob.G, prob.h, out.z);
    out.active_set_indices.assign(active_idx.data(), active_idx.data() + active_idx.size());

    if (active_idx.size() > 0) {
        // Resolve with active inequalities as equalities
        const int ma = active_idx.size();
        Eigen::MatrixXd Gact(ma, n);
        for (int i = 0; i < ma; ++i) Gact.row(i) = prob.G.row(active_idx(i));

        const int kkt_dim = n + meq + ma;
        Eigen::MatrixXd KKT2(kkt_dim, kkt_dim);
        KKT2.setZero();
        KKT2.topLeftCorner(n, n) = P_reg;
        if (has_eq) {
            KKT2.topRightCorner(n, meq) = prob.A.transpose();
            KKT2.block(n, 0, meq, n) = prob.A;
        }
        // add G_active^T and G_active blocks
        KKT2.block(0, n + meq, n, ma) = Gact.transpose();
        KKT2.block(n + meq, 0, ma, n) = Gact;

        Eigen::VectorXd rhs2(kkt_dim);
        rhs2.setZero();
        rhs2.head(n) = -prob.q;
        if (has_eq) rhs2.segment(n, meq) = prob.b;
        rhs2.tail(ma) = prob.h(active_idx);

        Eigen::LDLT<Eigen::MatrixXd> ldlt2;
        ldlt2.compute(KKT2);
        if (ldlt2.info() != Eigen::Success) {
            // Fallback: bump Hessian regularization
            KKT2.topLeftCorner(n, n).diagonal().array() += std::max(1e-8, prob.regularization_eps);
            ldlt2.compute(KKT2);
        }
        if (ldlt2.info() != Eigen::Success) {
            throw std::runtime_error("KKT LDLT factorization failed in forward (active-set solve)");
        }
        Eigen::VectorXd sol2 = ldlt2.solve(rhs2);
        out.z = sol2.head(n);
        if (has_eq) out.nu = sol2.segment(n, meq);
        out.lambda = sol2.tail(ma);

        if (ctx && reuse_factorization) {
            ctx->kkt_matrix = KKT2;
            ctx->kkt_ldlt = ldlt2;
            ctx->valid = true;
        }
    }

    return out;
}

DifferentiableQPLayer::BackwardGrads DifferentiableQPLayer::backward(const QPProblem &prob,
                                                                     const QPSolution &sol,
                                                                     const Eigen::VectorXd &dL_dz,
                                                                     QPContext *ctx)
{
    const int n = static_cast<int>(prob.q.size());
    const int meq = static_cast<int>(prob.A.rows());
    const bool has_eq = meq > 0;
    const int ma = static_cast<int>(sol.active_set_indices.size());

    Eigen::MatrixXd P_reg = prob.P;
    P_reg.diagonal().array() += prob.regularization_eps;

    // Build active G
    Eigen::MatrixXd Gact(ma, n);
    for (int i = 0; i < ma; ++i) Gact.row(i) = prob.G.row(sol.active_set_indices[i]);

    // Solve adjoint system (transpose KKT) for v (adjoint wrt z) and multipliers
    const int kkt_dim = n + meq + ma;
    Eigen::MatrixXd KKT(kkt_dim, kkt_dim);
    KKT.setZero();
    KKT.topLeftCorner(n, n) = P_reg;
    if (has_eq) {
        KKT.topRightCorner(n, meq) = prob.A.transpose();
        KKT.block(n, 0, meq, n) = prob.A;
    }
    KKT.block(0, n + meq, n, ma) = Gact.transpose();
    KKT.block(n + meq, 0, ma, n) = Gact;

    // Right-hand side for adjoint: [dL/dz; 0; 0]
    Eigen::VectorXd rhs(kkt_dim);
    rhs.setZero();
    rhs.head(n) = dL_dz;

    Eigen::VectorXd adj;
    if (ctx && ctx->valid && ctx->kkt_matrix.rows() == kkt_dim) {
        if (ctx->kkt_ldlt.info() != Eigen::Success) {
            // Recompute if cached factorization is invalid
            ctx->kkt_ldlt.compute(ctx->kkt_matrix);
        }
        if (ctx->kkt_ldlt.info() != Eigen::Success) {
            throw std::runtime_error("Cached KKT LDLT invalid in backward");
        }
        adj = ctx->kkt_ldlt.transpose().solve(rhs);
    }
    else {
        Eigen::LDLT<Eigen::MatrixXd> ldlt;
        ldlt.compute(KKT.transpose());
        if (ldlt.info() != Eigen::Success) {
            // Try slight bump
            KKT.topLeftCorner(n, n).diagonal().array() += std::max(1e-8, prob.regularization_eps);
            ldlt.compute(KKT.transpose());
        }
        if (ldlt.info() != Eigen::Success) {
            throw std::runtime_error("KKT LDLT factorization failed in backward");
        }
        adj = ldlt.solve(rhs);
    }

    Eigen::VectorXd v = adj.head(n);  // adjoint for z
    Eigen::VectorXd nu_adj = has_eq ? adj.segment(n, meq) : Eigen::VectorXd();
    Eigen::VectorXd lam_adj = adj.tail(ma);

    BackwardGrads grads;
    // Sign convention from implicit differentiation of KKT residuals:
    // r_z = P z + q + A^T nu + G^T lambda -> \partial r/\partial q = I
    // grad = -adjoint_z
    grads.dL_dq = -v;
    grads.dL_dP = -0.5 * (v * sol.z.transpose() + sol.z * v.transpose());
    // r_nu = A z - b -> \partial r/\partial b = -I -> grad = +adjoint_nu
    grads.dL_db = has_eq ? nu_adj : Eigen::VectorXd();
    grads.dL_dh = Eigen::VectorXd::Zero(prob.h.size());
    for (int i = 0; i < ma; ++i) grads.dL_dh(sol.active_set_indices[i]) = lam_adj(i);
    return grads;
}

}  // namespace optimization
}  // namespace rad_ml
