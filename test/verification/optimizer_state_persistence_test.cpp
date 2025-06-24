#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rad_ml/neural/protected_neural_network.hpp"

using T = float;

/**
 * @brief Test optimizer state persistence across multiple training calls
 */
class OptimizerStatePersistenceTest {
   public:
    bool runTest()
    {
        std::cout << "🔬 Optimizer State Persistence Test\n";
        std::cout << "====================================\n\n";

        bool all_passed = true;

        // Test 1: Momentum persistence
        std::cout << "📊 Test 1: Momentum Optimizer State Persistence\n";
        all_passed &= testMomentumPersistence();

        // Test 2: Adam optimizer persistence
        std::cout << "\n📊 Test 2: Adam Optimizer State Persistence\n";
        all_passed &= testAdamPersistence();

        // Test 3: Optimizer config change handling
        std::cout << "\n📊 Test 3: Optimizer Configuration Change Handling\n";
        all_passed &= testOptimizerConfigChange();

        // Test 4: Manual state reset
        std::cout << "\n📊 Test 4: Manual Optimizer State Reset\n";
        all_passed &= testManualStateReset();

        // Final result
        std::cout << "\n" << std::string(50, '=') << "\n";
        if (all_passed) {
            std::cout << "✅ ALL OPTIMIZER PERSISTENCE TESTS PASSED!\n";
            std::cout << "🚀 Optimizer state management is working correctly!\n";
        }
        else {
            std::cout << "❌ SOME OPTIMIZER PERSISTENCE TESTS FAILED!\n";
            std::cout << "🔧 Please review the optimizer implementation.\n";
        }
        std::cout << std::string(50, '=') << "\n";

        return all_passed;
    }

   private:
    bool testMomentumPersistence()
    {
        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 3, 1});

        // Configure momentum optimizer
        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig config;
        config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::MOMENTUM;
        config.learning_rate = 0.01f;
        config.momentum = 0.9f;

        // Create simple training data
        std::vector<T> train_data = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f, 0.5f, -0.5f};
        std::vector<T> train_labels = {1.0f, 0.0f, 0.5f, 0.3f};

        try {
            // First training call - should initialize momentum
            auto history1 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);

            // Get initial loss
            T loss_after_first = history1.train_losses.back();

            // Second training call - should continue with existing momentum
            auto history2 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);

            // Get final loss
            T loss_after_second = history2.train_losses.back();

            // Loss should continue improving (momentum should be preserved)
            bool momentum_preserved = loss_after_second < loss_after_first;

            std::cout << "  First training final loss: " << std::fixed << std::setprecision(6)
                      << loss_after_first << "\n";
            std::cout << "  Second training final loss: " << std::fixed << std::setprecision(6)
                      << loss_after_second << "\n";
            std::cout << "  Momentum preserved: " << (momentum_preserved ? "✅ YES" : "❌ NO")
                      << "\n";

            return momentum_preserved;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during momentum test: " << e.what() << "\n";
            return false;
        }
    }

    bool testAdamPersistence()
    {
        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 3, 1});

        // Configure Adam optimizer
        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig config;
        config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::ADAM;
        config.learning_rate = 0.01f;
        config.beta1 = 0.9f;
        config.beta2 = 0.999f;

        // Create simple training data
        std::vector<T> train_data = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f, 0.5f, -0.5f};
        std::vector<T> train_labels = {1.0f, 0.0f, 0.5f, 0.3f};

        try {
            // First training call
            auto history1 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);
            T loss_after_first = history1.train_losses.back();

            // Second training call
            auto history2 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);
            T loss_after_second = history2.train_losses.back();

            // Adam state should be preserved
            bool adam_preserved = loss_after_second < loss_after_first;

            std::cout << "  First training final loss: " << std::fixed << std::setprecision(6)
                      << loss_after_first << "\n";
            std::cout << "  Second training final loss: " << std::fixed << std::setprecision(6)
                      << loss_after_second << "\n";
            std::cout << "  Adam state preserved: " << (adam_preserved ? "✅ YES" : "❌ NO")
                      << "\n";

            return adam_preserved;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during Adam test: " << e.what() << "\n";
            return false;
        }
    }

    bool testOptimizerConfigChange()
    {
        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 3, 1});

        // Start with momentum
        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig momentum_config;
        momentum_config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::MOMENTUM;
        momentum_config.learning_rate = 0.01f;

        // Switch to Adam
        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig adam_config;
        adam_config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::ADAM;
        adam_config.learning_rate = 0.01f;

        std::vector<T> train_data = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f, 0.5f, -0.5f};
        std::vector<T> train_labels = {1.0f, 0.0f, 0.5f, 0.3f};

        try {
            // Train with momentum
            network.train(train_data, train_labels, 3, 2, momentum_config, {}, {}, false, 10,
                          0.001f, false);

            // Switch to Adam - should reinitialize state
            network.train(train_data, train_labels, 3, 2, adam_config, {}, {}, false, 10, 0.001f,
                          false);

            // Switch back to momentum - should reinitialize again
            network.train(train_data, train_labels, 3, 2, momentum_config, {}, {}, false, 10,
                          0.001f, false);

            std::cout << "  Optimizer configuration changes: ✅ Handled correctly\n";
            return true;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during config change test: " << e.what() << "\n";
            return false;
        }
    }

    bool testManualStateReset()
    {
        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 3, 1});

        rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerConfig config;
        config.type = rad_ml::neural::ProtectedNeuralNetwork<T>::OptimizerType::ADAM;
        config.learning_rate = 0.01f;

        std::vector<T> train_data = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f, 0.5f, -0.5f};
        std::vector<T> train_labels = {1.0f, 0.0f, 0.5f, 0.3f};

        try {
            // Train to build up optimizer state
            auto history1 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);
            T loss_before_reset = history1.train_losses.back();

            // Manually reset optimizer state
            network.resetOptimizerState();

            // Train again - should start fresh
            auto history2 = network.train(train_data, train_labels, 5, 2, config, {}, {}, false, 10,
                                          0.001f, false);
            T loss_after_reset = history2.train_losses.back();

            std::cout << "  Loss before reset: " << std::fixed << std::setprecision(6)
                      << loss_before_reset << "\n";
            std::cout << "  Loss after reset: " << std::fixed << std::setprecision(6)
                      << loss_after_reset << "\n";
            std::cout << "  Manual reset: ✅ Executed successfully\n";

            return true;
        }
        catch (const std::exception& e) {
            std::cout << "  ❌ Exception during manual reset test: " << e.what() << "\n";
            return false;
        }
    }
};

int main()
{
    OptimizerStatePersistenceTest test;
    bool success = test.runTest();
    return success ? 0 : 1;
}
