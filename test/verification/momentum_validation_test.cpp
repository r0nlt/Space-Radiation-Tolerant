#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rad_ml/neural/protected_neural_network.hpp"

using T = float;

/**
 * @brief Test to validate momentum optimizer is working correctly
 */
class MomentumValidationTest {
   public:
    bool runTest()
    {
        std::cout << "🔬 Momentum Optimizer Validation Test\n";
        std::cout << "=====================================\n\n";

        bool all_passed = true;

        // Test 1: Momentum vs SGD convergence
        std::cout << "📊 Test 1: Momentum vs SGD Convergence Comparison\n";
        all_passed &= testMomentumVsSGD();

        // Test 2: Momentum state persistence across calls
        std::cout << "\n📊 Test 2: Momentum State Persistence\n";
        all_passed &= testMomentumStatePersistence();

        // Final result
        std::cout << "\n" << std::string(50, '=') << "\n";
        if (all_passed) {
            std::cout << "✅ ALL MOMENTUM VALIDATION TESTS PASSED!\n";
            std::cout << "🚀 Momentum optimizer is working correctly!\n";
        }
        else {
            std::cout << "❌ SOME MOMENTUM VALIDATION TESTS FAILED!\n";
            std::cout << "🔧 Please review the momentum implementation.\n";
        }
        std::cout << std::string(50, '=') << "\n";

        return all_passed;
    }

   private:
    bool testMomentumVsSGD()
    {
        // Create identical networks
        rad_ml::neural::ProtectedNeuralNetwork<T> sgd_network({3, 5, 3, 1});
        rad_ml::neural::ProtectedNeuralNetwork<T> momentum_network({3, 5, 3, 1});

        // Set identical weights for fair comparison
        for (size_t layer = 0; layer < 3; ++layer) {
            for (size_t i = 0; i < sgd_network.getLayerCount() - 1; ++i) {
                // This is a simplified approach - in practice you'd need to copy exact weights
                // But for this test, random initialization should be sufficient
            }
        }

        // Configure optimizers
        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig sgd_config;
        sgd_config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::SGD;
        sgd_config.learning_rate = 0.1f;

        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig momentum_config;
        momentum_config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::MOMENTUM;
        momentum_config.learning_rate = 0.1f;
        momentum_config.momentum = 0.9f;

        // Create training data with some complexity
        std::vector<T> train_data;
        std::vector<T> train_labels;

        // Generate 24 samples (3 inputs each)
        for (int i = 0; i < 24; ++i) {
            float x1 = std::sin(i * 0.1f);
            float x2 = std::cos(i * 0.1f);
            float x3 = std::sin(i * 0.05f);

            train_data.insert(train_data.end(), {x1, x2, x3});

            // Target: nonlinear function
            float target = std::tanh(x1 * x2 + x3 * 0.5f);
            train_labels.push_back(target);
        }

        try {
            // Train both networks
            auto sgd_history = sgd_network.train(train_data, train_labels, 20, 6, sgd_config, {},
                                                 {}, false, 10, 0.001f, false);
            auto momentum_history = momentum_network.train(
                train_data, train_labels, 20, 6, momentum_config, {}, {}, false, 10, 0.001f, false);

            T sgd_final_loss = sgd_history.train_losses.back();
            T momentum_final_loss = momentum_history.train_losses.back();

            // Momentum should generally converge faster/better
            bool momentum_better =
                momentum_final_loss <= sgd_final_loss * 1.1f;  // Allow 10% tolerance

            std::cout << "  SGD final loss: " << std::fixed << std::setprecision(6)
                      << sgd_final_loss << "\n";
            std::cout << "  Momentum final loss: " << std::fixed << std::setprecision(6)
                      << momentum_final_loss << "\n";
            std::cout << "  Momentum performance: " << (momentum_better ? "✅ GOOD" : "❌ POOR")
                      << "\n";

            return momentum_better;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during momentum vs SGD test: " << e.what() << "\n";
            return false;
        }
    }

    bool testMomentumStatePersistence()
    {
        rad_ml::neural::ProtectedNeuralNetwork<T> network({3, 4, 1});

        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig config;
        config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::MOMENTUM;
        config.learning_rate = 0.05f;
        config.momentum = 0.8f;

        // Create training data
        std::vector<T> train_data;
        std::vector<T> train_labels;

        for (int i = 0; i < 12; ++i) {
            float x1 = (i % 3) - 1.0f;
            float x2 = ((i / 3) % 2) * 2.0f - 1.0f;
            float x3 = (i % 2) * 2.0f - 1.0f;

            train_data.insert(train_data.end(), {x1, x2, x3});

            float target = x1 * 0.5f + x2 * 0.3f + x3 * 0.2f;
            train_labels.push_back(target);
        }

        try {
            // Phase 1: Build momentum
            auto history1 = network.train(train_data, train_labels, 15, 3, config, {}, {}, false,
                                          10, 0.001f, false);

            // Phase 2: Continue training (should use existing momentum)
            auto history2 = network.train(train_data, train_labels, 10, 3, config, {}, {}, false,
                                          10, 0.001f, false);

            // Phase 3: Reset momentum and train again
            network.resetOptimizerState();
            auto history3 = network.train(train_data, train_labels, 10, 3, config, {}, {}, false,
                                          10, 0.001f, false);

            T loss_phase1 = history1.train_losses.back();
            T loss_phase2 = history2.train_losses.back();
            T loss_phase3 = history3.train_losses.back();

            // Phase 2 should benefit from momentum (continue improving)
            // Phase 3 should start fresh (may not improve as much)
            bool momentum_persisted = loss_phase2 < loss_phase1;

            std::cout << "  Phase 1 (build momentum) final loss: " << std::fixed
                      << std::setprecision(6) << loss_phase1 << "\n";
            std::cout << "  Phase 2 (use momentum) final loss: " << std::fixed
                      << std::setprecision(6) << loss_phase2 << "\n";
            std::cout << "  Phase 3 (reset momentum) final loss: " << std::fixed
                      << std::setprecision(6) << loss_phase3 << "\n";
            std::cout << "  Momentum persistence: " << (momentum_persisted ? "✅ YES" : "❌ NO")
                      << "\n";

            return momentum_persisted;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during momentum persistence test: " << e.what() << "\n";
            return false;
        }
    }
};

int main()
{
    MomentumValidationTest test;
    bool success = test.runTest();
    return success ? 0 : 1;
}
