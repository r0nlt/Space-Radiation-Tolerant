#include <cassert>
#include <cmath>
#include <iostream>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <random>
#include <vector>

#ifdef __has_include
#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#endif
#endif
#include <rad_ml/optimization/simplex_projection.hpp>

using rad_ml::neural::ProtectedNeuralNetwork;
using rad_ml::neural::ProtectionLevel;

static std::vector<float> one_hot(int idx, int classes)
{
    std::vector<float> v(classes, 0.0f);
    v[idx] = 1.0f;
    return v;
}

int main()
{
    // Tiny dataset: 64 samples, input 4, output 3
    const int num_samples = 64;
    const int input_size = 4;
    const int output_size = 3;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> cls(0, output_size - 1);

    std::vector<float> X;
    X.reserve(num_samples * input_size);
    std::vector<float> Y;
    Y.reserve(num_samples * output_size);
    for (int n = 0; n < num_samples; ++n) {
        for (int i = 0; i < input_size; ++i) X.push_back(nd(rng));
        auto y = one_hot(cls(rng), output_size);
        Y.insert(Y.end(), y.begin(), y.end());
    }

    auto run_once = [&](bool use_projection) {
        std::vector<size_t> arch = {static_cast<size_t>(input_size), 5u,
                                    static_cast<size_t>(output_size)};
        ProtectedNeuralNetwork<float> net(arch, ProtectionLevel::ADAPTIVE_TMR);
        net.setUseSimplexProjection(use_projection);

        ProtectedNeuralNetwork<float>::OptimizerConfig cfg;
        cfg.type = ProtectedNeuralNetwork<float>::OptimizerType::ADAM;
        cfg.learning_rate = 0.01f;

        auto hist = net.train(X, Y, /*epochs=*/6, /*batch_size=*/16, cfg, {}, {},
                              /*early_stop=*/false, /*patience=*/0, /*min_delta=*/0.0f,
                              /*verbose=*/false);

        // Finite and mildly decreasing
        for (float L : hist.train_losses) {
            if (!std::isfinite(L)) return std::make_pair(hist.train_losses.back(), false);
        }
        bool non_increasing = hist.train_losses.back() <= hist.train_losses.front() + 1e-3f;

        // Validate simplex transformation on a random output vector
        Eigen::VectorXd a(5);
        for (int i = 0; i < a.size(); ++i) a(i) = nd(rng);
        auto z = rad_ml::optimization::SimplexProjection::forward(a);
        bool simplex_ok = (std::abs(z.sum() - 1.0) < 1e-6);
        for (int i = 0; i < z.size(); ++i) simplex_ok = simplex_ok && (z(i) >= -1e-9);

        return std::make_pair(hist.train_losses.back(), non_increasing && simplex_ok);
    };

    auto [loss_proj, ok_proj] = run_once(true);
    auto [loss_base, ok_base] = run_once(false);

    if (!ok_proj || !ok_base) return 1;
    if (!(loss_proj <= loss_base + 1e-3f)) return 1;

    std::cout << "final_loss_proj=" << loss_proj << " final_loss_base=" << loss_base << "\n";
    return 0;
}
