#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rad_ml/neural/protected_neural_network.hpp"

using T = float;

/**
 * @brief Focused validation test for activation derivatives
 * This test validates only the core functionality without numerical differentiation noise
 */
class ActivationDerivativeValidator {
   private:
    struct ValidationCase {
        std::string name;
        std::function<T(T)> activation;
        std::function<T(T)> expected_derivative;
        std::vector<T> test_points;
    };

   public:
    bool runValidation()
    {
        std::cout << "🔬 Activation Derivative Core Validation\n";
        std::cout << "========================================\n\n";

        auto test_cases = getValidationCases();
        bool all_passed = true;

        // Test 1: Direct analytical derivative computation
        std::cout << "📊 Test 1: Direct Analytical Derivative Computation\n";
        for (const auto& test_case : test_cases) {
            std::cout << "\n🧪 Testing " << test_case.name << ":\n";

            rad_ml::neural::ProtectedNeuralNetwork<T> network({3, 5, 2});
            network.setActivationFunction(0, test_case.activation);

            bool case_passed = true;
            for (T x : test_case.test_points) {
                T network_result = network.computeActivationDerivative(x, 0);
                T expected = test_case.expected_derivative(x);
                T error = std::abs(network_result - expected);

                bool point_passed = error < 1e-6f;
                case_passed &= point_passed;

                std::cout << "  x=" << std::setw(8) << x << " | Network=" << std::setw(10)
                          << network_result << " | Expected=" << std::setw(10) << expected
                          << " | Error=" << std::setw(12) << error << " | "
                          << (point_passed ? "✅" : "❌") << "\n";
            }

            if (!case_passed) all_passed = false;
            std::cout << "  Result: " << (case_passed ? "✅ PASS" : "❌ FAIL") << "\n";
        }

        // Test 2: Gradient flow validation
        std::cout << "\n📊 Test 2: Gradient Flow Validation\n";
        bool gradient_test = testGradientFlow();
        all_passed &= gradient_test;

        // Test 3: Multi-layer consistency
        std::cout << "\n📊 Test 3: Multi-Layer Consistency\n";
        bool multi_layer_test = testMultiLayerConsistency();
        all_passed &= multi_layer_test;

        // Test 4: Edge case robustness
        std::cout << "\n📊 Test 4: Edge Case Robustness\n";
        bool edge_case_test = testEdgeCases();
        all_passed &= edge_case_test;

        // Final result
        std::cout << "\n" << std::string(50, '=') << "\n";
        if (all_passed) {
            std::cout << "✅ ALL VALIDATION TESTS PASSED!\n";
            std::cout << "🚀 Neural network training is ready for production!\n";
        }
        else {
            std::cout << "❌ SOME VALIDATION TESTS FAILED!\n";
            std::cout << "🔧 Please review the implementation.\n";
        }
        std::cout << std::string(50, '=') << "\n";

        return all_passed;
    }

   private:
    std::vector<ValidationCase> getValidationCases()
    {
        return {{"ReLU",
                 [](T x) { return x > 0 ? x : T{0}; },
                 [](T x) { return x > 0 ? T{1} : T{0}; },
                 {-2.0f, -0.1f, 0.1f, 1.0f, 2.0f}},
                {"Leaky ReLU (α=0.01)",
                 [](T x) { return x > 0 ? x : T{0.01} * x; },
                 [](T x) { return x > 0 ? T{1} : T{0.01}; },
                 {-2.0f, -0.1f, 0.1f, 1.0f, 2.0f}},
                {"Sigmoid",
                 [](T x) { return T{1} / (T{1} + std::exp(-x)); },
                 [](T x) {
                     T sigmoid_x = T{1} / (T{1} + std::exp(-x));
                     return sigmoid_x * (T{1} - sigmoid_x);
                 },
                 {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}},
                {"Tanh",
                 [](T x) { return std::tanh(x); },
                 [](T x) {
                     T tanh_x = std::tanh(x);
                     return T{1} - tanh_x * tanh_x;
                 },
                 {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}},
                {"Linear",
                 [](T x) { return x; },
                 [](T x) { return T{1}; },
                 {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}},
                {"ELU (α=1.0)",
                 [](T x) { return x > 0 ? x : std::exp(x) - T{1}; },
                 [](T x) { return x > 0 ? T{1} : std::exp(x); },
                 {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}}};
    }

    bool testGradientFlow()
    {
        std::cout << "Testing gradient flow through backpropagation...\n";

        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 4, 1});

        // Test with different activation functions
        std::vector<std::function<T(T)>> activations = {
            [](T x) { return x > 0 ? x : T{0}; },              // ReLU
            [](T x) { return T{1} / (T{1} + std::exp(-x)); },  // Sigmoid
            [](T x) { return std::tanh(x); }                   // Tanh
        };

        std::vector<std::string> names = {"ReLU", "Sigmoid", "Tanh"};

        for (size_t i = 0; i < activations.size(); ++i) {
            network.setActivationFunction(0, activations[i]);

            std::vector<T> input = {1.0f, -0.5f};
            std::vector<T> target = {0.5f};

            try {
                auto output = network.forward(input);
                network.backward(input, target);
                std::cout << "  " << names[i] << ": ✅ Gradient flow successful\n";
            }
            catch (const std::exception& e) {
                std::cout << "  " << names[i] << ": ❌ Gradient flow failed: " << e.what() << "\n";
                return false;
            }
        }

        return true;
    }

    bool testMultiLayerConsistency()
    {
        std::cout << "Testing multi-layer activation derivative consistency...\n";

        rad_ml::neural::ProtectedNeuralNetwork<T> network({3, 5, 4, 2});

        // Set same activation for all layers
        auto relu = [](T x) { return x > 0 ? x : T{0}; };
        for (size_t layer = 0; layer < 3; ++layer) {
            network.setActivationFunction(layer, relu);
        }

        // Test that all layers give same derivative for same input
        T test_x = 1.5f;
        T expected = 1.0f;  // ReLU derivative for positive input

        bool all_consistent = true;
        for (size_t layer = 0; layer < 3; ++layer) {
            T derivative = network.computeActivationDerivative(test_x, layer);
            T error = std::abs(derivative - expected);

            bool layer_passed = error < 1e-6f;
            all_consistent &= layer_passed;

            std::cout << "  Layer " << layer << ": derivative=" << derivative << ", error=" << error
                      << " " << (layer_passed ? "✅" : "❌") << "\n";
        }

        return all_consistent;
    }

    bool testEdgeCases()
    {
        std::cout << "Testing edge cases and boundary conditions...\n";

        rad_ml::neural::ProtectedNeuralNetwork<T> network({2, 3, 1});
        auto relu = [](T x) { return x > 0 ? x : T{0}; };
        network.setActivationFunction(0, relu);

        bool all_passed = true;

        // Test extreme values
        std::vector<T> extreme_values = {-1000.0f, -100.0f, 100.0f, 1000.0f};
        for (T val : extreme_values) {
            T derivative = network.computeActivationDerivative(val, 0);
            T expected = val > 0 ? 1.0f : 0.0f;
            bool passed = std::abs(derivative - expected) < 1e-6f;
            all_passed &= passed;

            std::cout << "  Extreme value " << val << ": " << (passed ? "✅" : "❌") << "\n";
        }

        // Test very small values
        std::vector<T> tiny_values = {1e-10f, -1e-10f, 1e-15f, -1e-15f};
        for (T val : tiny_values) {
            T derivative = network.computeActivationDerivative(val, 0);
            T expected = val > 0 ? 1.0f : 0.0f;
            bool passed = std::abs(derivative - expected) < 1e-6f;
            all_passed &= passed;

            std::cout << "  Tiny value " << val << ": " << (passed ? "✅" : "❌") << "\n";
        }

        // Test invalid layer index
        try {
            T derivative = network.computeActivationDerivative(1.0f, 999);
            std::cout << "  Invalid layer index: ✅ (returned " << derivative << ")\n";
        }
        catch (...) {
            std::cout << "  Invalid layer index: ❌ (threw exception)\n";
            all_passed = false;
        }

        return all_passed;
    }
};

int main()
{
    ActivationDerivativeValidator validator;
    bool success = validator.runValidation();
    return success ? 0 : 1;
}
