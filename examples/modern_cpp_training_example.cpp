/**
 * @file modern_cpp_training_example.cpp
 * @brief Comprehensive example of real neural network training using modern C++17/20 features
 *
 * This example demonstrates:
 * - Real backpropagation training with gradient computation
 * - Multiple optimization algorithms (SGD, Momentum, Adam, RMSprop)
 * - Batch processing with shuffling
 * - Validation and early stopping
 * - Learning rate scheduling
 * - L2 regularization
 * - Comprehensive training history tracking
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/neural/protected_neural_network.hpp"

using namespace rad_ml::neural;

/**
 * @brief Generate synthetic XOR dataset for training
 */
std::pair<std::vector<float>, std::vector<float>> generateXORDataset(size_t num_samples = 1000)
{
    std::vector<float> data;
    std::vector<float> labels;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);

    data.reserve(num_samples * 2);
    labels.reserve(num_samples * 1);

    for (size_t i = 0; i < num_samples; ++i) {
        // Generate XOR pattern with noise
        int pattern = i % 4;
        float x1, x2, y;

        switch (pattern) {
            case 0:
                x1 = 0.0f;
                x2 = 0.0f;
                y = 0.0f;
                break;  // 0 XOR 0 = 0
            case 1:
                x1 = 0.0f;
                x2 = 1.0f;
                y = 1.0f;
                break;  // 0 XOR 1 = 1
            case 2:
                x1 = 1.0f;
                x2 = 0.0f;
                y = 1.0f;
                break;  // 1 XOR 0 = 1
            case 3:
                x1 = 1.0f;
                x2 = 1.0f;
                y = 0.0f;
                break;  // 1 XOR 1 = 0
        }

        // Add noise
        x1 += noise_dist(gen);
        x2 += noise_dist(gen);

        data.push_back(x1);
        data.push_back(x2);
        labels.push_back(y);
    }

    return {data, labels};
}

/**
 * @brief Generate synthetic regression dataset
 */
std::pair<std::vector<float>, std::vector<float>> generateRegressionDataset(
    size_t num_samples = 1000)
{
    std::vector<float> data;
    std::vector<float> labels;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> input_dist(-2.0f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);

    data.reserve(num_samples * 2);
    labels.reserve(num_samples * 1);

    for (size_t i = 0; i < num_samples; ++i) {
        float x1 = input_dist(gen);
        float x2 = input_dist(gen);

        // Non-linear function: y = sin(x1) * cos(x2) + noise
        float y = std::sin(x1) * std::cos(x2) + noise_dist(gen);

        data.push_back(x1);
        data.push_back(x2);
        labels.push_back(y);
    }

    return {data, labels};
}

/**
 * @brief Demonstrate different optimizers
 */
void demonstrateOptimizers()
{
    std::cout << "\n🚀 Modern C++ Training: Optimizer Comparison\n";
    std::cout << "============================================\n";

    // Generate training and validation data
    auto [train_data, train_labels] = generateXORDataset(800);
    auto [val_data, val_labels] = generateXORDataset(200);

    // Network architecture: 2 inputs -> 8 hidden -> 4 hidden -> 1 output
    std::vector<size_t> architecture = {2, 8, 4, 1};

    // Test different optimizers with improved learning rates
    std::vector<std::pair<std::string, ProtectedNeuralNetwork<float>::OptimizerConfig>> optimizers =
        {{"SGD", {ProtectedNeuralNetwork<float>::OptimizerType::SGD, 0.01f}},
         {"SGD+Momentum", {ProtectedNeuralNetwork<float>::OptimizerType::MOMENTUM, 0.01f, 0.9f}},
         {"Adam", {ProtectedNeuralNetwork<float>::OptimizerType::ADAM, 0.003f}},
         {"RMSprop", {ProtectedNeuralNetwork<float>::OptimizerType::RMSPROP, 0.003f}}};

    for (const auto& [name, config] : optimizers) {
        std::cout << "\n⚙️ Training with " << name << " optimizer:\n";

        // Create network with radiation protection
        ProtectedNeuralNetwork<float> network(architecture, ProtectionLevel::ADAPTIVE_TMR);

        // Configure activation functions (ReLU for hidden, sigmoid for output)
        network.setActivationFunction(0, [](float x) { return std::max(0.0f, x); });  // ReLU
        network.setActivationFunction(1, [](float x) { return std::max(0.0f, x); });  // ReLU
        network.setActivationFunction(
            2, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });  // Sigmoid

        auto start_time = std::chrono::high_resolution_clock::now();

        // Train with validation and early stopping
        auto history = network.train(train_data, train_labels,
                                     100,                   // epochs
                                     32,                    // batch_size
                                     config,                // optimizer config
                                     val_data, val_labels,  // validation data
                                     true,                  // early_stopping
                                     15,                    // patience
                                     0.001f,                // min_delta
                                     false                  // verbose (we'll show summary)
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Show results
        std::cout << "  Final Training Loss: " << std::fixed << std::setprecision(6)
                  << history.train_losses.back() << "\n";
        std::cout << "  Final Training Accuracy: " << std::setprecision(2)
                  << history.train_accuracies.back() * 100 << "%\n";

        if (!history.val_losses.empty()) {
            std::cout << "  Best Validation Loss: " << std::setprecision(6) << history.best_val_loss
                      << " (Epoch " << history.best_epoch + 1 << ")\n";
        }

        std::cout << "  Training Time: " << duration.count() << "ms\n";
        std::cout << "  Epochs Completed: " << history.train_losses.size() << "\n";

        // Test final network
        auto [test_loss, test_accuracy] = network.evaluate(val_data, val_labels);
        std::cout << "  Test Accuracy: " << std::setprecision(2) << test_accuracy * 100 << "%\n";
    }
}

/**
 * @brief Demonstrate advanced training features
 */
void demonstrateAdvancedFeatures()
{
    std::cout << "\n🔬 Advanced Training Features\n";
    std::cout << "=============================\n";

    // Generate larger dataset for regression
    auto [train_data, train_labels] = generateRegressionDataset(2000);
    auto [val_data, val_labels] = generateRegressionDataset(500);

    // Larger network: 2 -> 16 -> 16 -> 8 -> 1
    std::vector<size_t> architecture = {2, 16, 16, 8, 1};

    // Create network with full TMR protection
    ProtectedNeuralNetwork<float> network(architecture, ProtectionLevel::FULL_TMR);

    // Configure Adam with L2 regularization and learning rate decay
    ProtectedNeuralNetwork<float>::OptimizerConfig config;
    config.type = ProtectedNeuralNetwork<float>::OptimizerType::ADAM;
    config.learning_rate = 0.001f;
    config.weight_decay = 0.0001f;  // L2 regularization
    config.decay = 0.001f;          // Learning rate decay
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.epsilon = 1e-8f;

    std::cout << "Training regression network with advanced features:\n";
    std::cout << "- L2 Regularization: " << config.weight_decay << "\n";
    std::cout << "- Learning Rate Decay: " << config.decay << "\n";
    std::cout << "- Early Stopping with patience=20\n";
    std::cout << "- Radiation Protection: FULL_TMR\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    auto history = network.train(train_data, train_labels,
                                 200,                   // epochs
                                 64,                    // batch_size
                                 config,                // optimizer config with regularization
                                 val_data, val_labels,  // validation data
                                 true,                  // early_stopping
                                 20,                    // patience
                                 0.0001f,               // min_delta
                                 true                   // verbose
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n📈 Training Summary:\n";
    std::cout << "  Total Training Time: " << duration.count() << "ms\n";
    std::cout << "  Epochs Completed: " << history.train_losses.size() << "\n";
    std::cout << "  Best Epoch: " << history.best_epoch + 1 << "\n";
    std::cout << "  Final Train Loss: " << std::fixed << std::setprecision(6)
              << history.train_losses.back() << "\n";
    std::cout << "  Best Val Loss: " << history.best_val_loss << "\n";

    // Show learning curve (last 10 epochs)
    std::cout << "\n📊 Learning Curve (Last 10 Epochs):\n";
    size_t start_idx = std::max(0, static_cast<int>(history.train_losses.size()) - 10);
    for (size_t i = start_idx; i < history.train_losses.size(); ++i) {
        std::cout << "  Epoch " << std::setw(3) << i + 1 << ": Train=" << std::setprecision(6)
                  << history.train_losses[i];
        if (i < history.val_losses.size()) {
            std::cout << ", Val=" << history.val_losses[i];
        }
        std::cout << "\n";
    }
}

/**
 * @brief Demonstrate radiation-aware training
 */
void demonstrateRadiationAwareTraining()
{
    std::cout << "\n☢️ Radiation-Aware Training\n";
    std::cout << "===========================\n";

    auto [train_data, train_labels] = generateXORDataset(1000);
    auto [val_data, val_labels] = generateXORDataset(200);

    std::vector<size_t> architecture = {2, 12, 8, 1};

    // Compare different protection levels
    std::vector<std::pair<std::string, ProtectionLevel>> protection_levels = {
        {"No Protection", ProtectionLevel::NONE},
        {"Checksum Only", ProtectionLevel::CHECKSUM_ONLY},
        {"Selective TMR", ProtectionLevel::SELECTIVE_TMR},
        {"Adaptive TMR", ProtectionLevel::ADAPTIVE_TMR},
        {"Full TMR", ProtectionLevel::FULL_TMR}};

    for (const auto& [name, protection] : protection_levels) {
        std::cout << "\n🛡️ Training with " << name << ":\n";

        ProtectedNeuralNetwork<float> network(architecture, protection);

        // Use consistent Adam configuration
        ProtectedNeuralNetwork<float>::OptimizerConfig config;
        config.type = ProtectedNeuralNetwork<float>::OptimizerType::ADAM;
        config.learning_rate = 0.001f;

        auto start_time = std::chrono::high_resolution_clock::now();

        auto history = network.train(train_data, train_labels,
                                     50,  // epochs
                                     32,  // batch_size
                                     config, val_data, val_labels,
                                     true,    // early_stopping
                                     10,      // patience
                                     0.001f,  // min_delta
                                     false    // verbose
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Test with radiation effects
        network.applyRadiationEffects(0.1, 12345);  // 10% radiation level
        auto [rad_loss, rad_accuracy] = network.evaluate(val_data, val_labels);

        std::cout << "  Training Time: " << duration.count() << "ms\n";
        std::cout << "  Final Accuracy: " << std::setprecision(2)
                  << history.train_accuracies.back() * 100 << "%\n";
        std::cout << "  Post-Radiation Accuracy: " << rad_accuracy * 100 << "%\n";
        std::cout << "  Radiation Tolerance: " << std::setprecision(1)
                  << (rad_accuracy / history.train_accuracies.back()) * 100 << "%\n";

        // Show error statistics
        auto [detected, corrected] = network.getErrorStats();
        std::cout << "  Errors Detected: " << detected << ", Corrected: " << corrected << "\n";
    }
}

/**
 * @brief Main demonstration function
 */
int main()
{
    std::cout << "🎯 Modern C++ Neural Network Training Framework\n";
    std::cout << "===============================================\n";
    std::cout << "Demonstrating real backpropagation training with:\n";
    std::cout << "✅ Multiple optimization algorithms (SGD, Momentum, Adam, RMSprop)\n";
    std::cout << "✅ Batch processing with data shuffling\n";
    std::cout << "✅ Validation and early stopping\n";
    std::cout << "✅ Learning rate scheduling and L2 regularization\n";
    std::cout << "✅ Radiation-tolerant training with TMR protection\n";
    std::cout << "✅ Comprehensive training history tracking\n\n";

    try {
        // Initialize logging (if available)
        // rad_ml::core::Logger::setLevel(rad_ml::core::Logger::Level::INFO);

        // Run demonstrations
        demonstrateOptimizers();
        demonstrateAdvancedFeatures();
        demonstrateRadiationAwareTraining();

        std::cout << "\n🎉 All demonstrations completed successfully!\n";
        std::cout << "\nThis framework provides production-ready neural network training\n";
        std::cout << "with modern C++17/20 features and space-grade radiation tolerance.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
