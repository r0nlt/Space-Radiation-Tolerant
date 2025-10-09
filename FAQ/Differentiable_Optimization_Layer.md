# Differentiable Optimization Layer (Implicit KKT)

This FAQ documents the plan, implementation steps, and current progress for integrating a differentiable optimization layer into the framework.

## What is it?
A layer whose forward pass solves a convex optimization problem (here: QP) and whose backward pass uses implicit differentiation of the KKT conditions to provide gradients with respect to upstream inputs/parameters.

- Forward: solve min 1/2 z^T P z + q^T z  s.t. A z = b,  G z ≤ h
- Backward: solve one linear system (KKT adjoint) to obtain vector–Jacobian products without unrolling solver iterations.

## Why use it?
- Exact vJPs under convex regularity; no need to multiply local Jacobians of unrolled iterations.
- Efficient: reuse KKT factorization from forward in backward.
- Stable: avoids long unroll chains and large memory.

## Current status
- Library `rad_ml_opt` with `DifferentiableQPLayer` added.
- Example `diff_qp_example` shows forward optimum and gradients.
- CMake wired; builds with Eigen.

## Files
- Include: `include/rad_ml/optimization/diff_qp.hpp`
- Source: `src/rad_ml/optimization/diff_qp.cpp`
- Example: `examples/diff_qp_example.cpp`
- CMake targets: `rad_ml_opt`, `diff_qp_example`

## Build and run
```bash
cmake -S /Users/rishabnuguru/space -B /Users/rishabnuguru/space/build-release -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3
cmake --build /Users/rishabnuguru/space/build-release --target diff_qp_example -j 8
/Users/rishabnuguru/space/build-release/examples/diff_qp_example
```
Expected output (similar):
```
z* = 0.333333 0.333333 0.333333
dL/dq ≈ 0
Active set size = 0
```

### Validation: Gradient check
Run a numerical vs analytical gradient comparison for q using central differences.

```bash
cmake --build /Users/rishabnuguru/space/build-release --target diff_qp_gradcheck -j 8
ctest --test-dir /Users/rishabnuguru/space/build-release -R diff_qp_gradcheck_run --output-on-failure
```
Expected tail output:
```
analytical dL/dq = [...]
numerical  dL/dq = [...]
max_abs_err < 1e-4
GRADCHECK PASSED
```

## Simplex projection layer (in-network)

Provides a fast projection layer onto the probability simplex, useful for classification heads:

- File: `include/rad_ml/optimization/simplex_projection.hpp`
- Forward: projects any vector to nonnegative entries that sum to 1 (sorting-based algorithm)
- Backward: analytical VJP that preserves the chain rule without unrolling

### Enable in training
`ProtectedNeuralNetwork` exposes a simple toggle to activate projection on the output for loss/metrics and to backpropagate through it:

```cpp
rad_ml::neural::ProtectedNeuralNetwork<float> net(architecture);
net.setUseSimplexProjection(true);
```

Projection is applied only when the output dimension > 1.

### Tests and examples
- Example: `examples/simplex_projection_example.cpp`
- Gradcheck: `examples/simplex_projection_gradcheck.cpp` with CTest target `simplex_projection_gradcheck_run`
- Integration training test: `test/integration/simplex_projection_training_test.cpp`

Run:
```bash
cmake --build /Users/rishabnuguru/space/build-release --target simplex_projection_example simplex_projection_gradcheck simplex_projection_training_test -j 8
ctest --test-dir /Users/rishabnuguru/space/build-release -R "simplex_projection_(gradcheck_run|training_test)" --output-on-failure
```
The integration test trains a tiny network and verifies losses are finite and reasonably decreasing.

## Mermaid diagram: end-to-end data/gradient flow

```mermaid
graph TD
    X[Input features x] --> L1[Hidden layers]
    L1 --> L2[Output logits a]
    L2 -->|optional| OPT[Optimization Layer<br/>(QP solve, implicit KKT)]
    L2 -->|optional| PRJ[Simplex Projection<br/>(z = Π_Δ(a))]

    OPT --> Z1[z*]
    PRJ --> Z2[ẑ]

    Z1 --> LOSS[Loss L]
    Z2 --> LOSS

    %% Backward paths
    LOSS -. dL/da via KKT adjoint .-> KKT[(KKT^T solve)]
    KKT -. vJP .-> OPT

    LOSS -. standard backprop .-> L2
    L2 -. chain rule .-> L1
    L1 -. gradients .-> X

    classDef opt fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef proj fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef loss fill:#ffe0e0,stroke:#b71c1c,stroke-width:2px
    class OPT opt
    class PRJ proj
    class LOSS loss
```

## How gradients are computed
- Treat the solution map z*(·) as an implicit function defined by KKT.
- Differentiate KKT and solve the adjoint KKT^T system once for the vJP.
- Provided grads: dL/dq, dL/dP, dL/db, dL/dh (for active constraints).

## Design choices
- Regularization: add εI to P for numerical stability (configurable).
- Active set: detect tight Gz = h at the solution and include in the KKT for backward.
- Scope: minimal QP first; LP/conic can extend the same pattern.

## Next steps
1) [Done] Optional reuse of LDLᵀ factorization between forward/backward via `QPContext`.
2) Expose Jacobian hooks for parameters that are functions of network outputs.
3) [Done] Add tests: finite-difference gradient checks (`diff_qp_gradcheck`, `simplex_projection_gradcheck`).
4) [Done] Provide a projection-layer variant (simplex/box) as a lightweight example.

## Integration notes
- The rest of the network uses normal backprop. Insert this layer where an optimization decision is required.
- Ensure convexity and regularity (LICQ, strong convexity) for clean derivatives; otherwise, fall back to unrolled gradients.

### API update (factorization reuse)
- `forward(const QPProblem&, QPContext* ctx, bool reuse_factorization=true)`
- `backward(const QPProblem&, const QPSolution&, const Eigen::VectorXd& dL_dz, QPContext* ctx)`
- Example updated to pass and reuse `QPContext`.

### Changelog
- Added `rad_ml_opt` with `DifferentiableQPLayer` (QP).
- Added `QPContext` for factorization reuse.
- Added `diff_qp_example` and `diff_qp_gradcheck` (CTest target `diff_qp_gradcheck_run`).
- Added simplex projection layer, example, gradcheck, and training integration test.

## Troubleshooting
- Eigen not found: pass `-DEIGEN3_INCLUDE_DIR=/path/to/eigen3` to CMake.
- Near-singular KKT: increase `regularization_eps`.
- Zero/NaN grads: verify active-set detection tolerance and constraints scaling.
