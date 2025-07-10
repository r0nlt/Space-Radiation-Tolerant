/**
 * @file weight_diagnostic_test.cpp
 * @brief Diagnostic test to examine weight changes during radiation injection
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <rad_ml/research/radiation_aware_training.hpp>
#include <rad_ml/research/residual_network.hpp>

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::neural;

std::pair<std::vector<float>, std::vector<float>> createSimpleDataset(size_t samples,
                                                                      size_t input_size,
                                                                      size_t output_size)
{
    std::vector<float> data;
    std::vector<float> labels;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    for (size_t i = 0; i < samples; ++i) {
        // Generate input
        for (size_t j = 0; j < input_size; ++j) {
            data.push_back(input_dist(gen));
        }

        // Generate one-hot label
        size_t label_class = i % output_size;
        for (size_t j = 0; j < output_size; ++j) {
            labels.push_back(j == label_class ? 1.0f : 0.0f);
        }
    }

    return {data, labels};
}

TEST(WeightDiagnosticTest, RadiationWeightChanges)
{
    std::cout << "\n=== WEIGHT DIAGNOSTIC TEST ===\n" << std::endl;

    // Create a simple network
    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

    // Get initial weights
    std::cout << "=== INITIAL WEIGHTS ===\n";
    auto layers = network.getLayers();

    std::cout << "Layer 0 weights (first 5x4):\n";
    for (size_t i = 0; i < std::min(size_t(5), layers[0].weights.size()); ++i) {
        for (size_t j = 0; j < layers[0].weights[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << layers[0].weights[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nLayer 0 biases:\n";
    for (size_t i = 0; i < layers[0].biases.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << layers[0].biases[i] << " ";
    }
    std::cout << std::endl;

    // Create radiation trainer with EXTREME radiation
    RadiationAwareTraining trainer(10.0f, false, sim::Environment::EXTREME);  // 10x probability!

    // Create simple test data
    auto [data, labels] = createSimpleDataset(40, 4, 4);

    // Measure baseline accuracy
    std::cout << "\n=== BASELINE PERFORMANCE ===\n";
    size_t correct = 0;
    for (size_t i = 0; i < data.size() / 4; ++i) {
        std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
        std::vector<float> label(labels.begin() + i * 4, labels.begin() + (i + 1) * 4);
        auto prediction = network.forward(sample);

        size_t pred_idx = std::distance(prediction.begin(),
                                        std::max_element(prediction.begin(), prediction.end()));
        size_t label_idx =
            std::distance(label.begin(), std::max_element(label.begin(), label.end()));
        if (pred_idx == label_idx) correct++;
    }
    float baseline_accuracy = (float)correct / (data.size() / 4);
    std::cout << "Baseline accuracy: " << std::fixed << std::setprecision(4) << baseline_accuracy
              << std::endl;

    // Apply radiation effects manually to see what happens
    std::cout << "\n=== APPLYING RADIATION EFFECTS ===\n";

    // Get mutable layer
    auto& layer = network.getLayerMutable(0);

    // Apply bit flips manually to some weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> bit_dist(0, 31);

    int flips_applied = 0;
    for (size_t i = 0; i < layer.weights.size(); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
            float original = layer.weights[i][j];

            // Apply bit flip using bit manipulation
            union {
                float f;
                uint32_t i;
            } converter;
            converter.f = original;
            int bit_pos = bit_dist(gen);
            converter.i ^= (1u << bit_pos);
            float flipped = converter.f;

            layer.weights[i][j] = flipped;
            flips_applied++;

            if (flips_applied <= 10) {  // Show first 10 flips
                std::cout << "Weight[" << i << "][" << j << "]: " << std::fixed
                          << std::setprecision(6) << original << " -> " << flipped << " (bit "
                          << bit_pos << ")" << std::endl;
            }
        }
    }

    std::cout << "Total bit flips applied: " << flips_applied << std::endl;

    // Check weights after radiation
    std::cout << "\n=== WEIGHTS AFTER RADIATION ===\n";
    std::cout << "Layer 0 weights (first 5x4):\n";
    for (size_t i = 0; i < std::min(size_t(5), layer.weights.size()); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << layer.weights[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Measure accuracy after radiation
    std::cout << "\n=== PERFORMANCE AFTER RADIATION ===\n";
    correct = 0;
    for (size_t i = 0; i < data.size() / 4; ++i) {
        std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
        std::vector<float> label(labels.begin() + i * 4, labels.begin() + (i + 1) * 4);
        auto prediction = network.forward(sample);

        size_t pred_idx = std::distance(prediction.begin(),
                                        std::max_element(prediction.begin(), prediction.end()));
        size_t label_idx =
            std::distance(label.begin(), std::max_element(label.begin(), label.end()));
        if (pred_idx == label_idx) correct++;
    }
    float post_radiation_accuracy = (float)correct / (data.size() / 4);
    std::cout << "Post-radiation accuracy: " << std::fixed << std::setprecision(4)
              << post_radiation_accuracy << std::endl;

    float accuracy_drop = baseline_accuracy - post_radiation_accuracy;
    std::cout << "Accuracy drop: " << std::fixed << std::setprecision(4) << accuracy_drop
              << std::endl;

    if (std::abs(accuracy_drop) > 0.001f) {
        std::cout << "✅ SIGNIFICANT accuracy change detected!" << std::endl;
    }
    else {
        std::cout << "❌ No significant accuracy change" << std::endl;
    }

    std::cout << "\n=== ANALYSIS ===\n";
    std::cout << "This test shows whether bit flips to weights actually affect network performance."
              << std::endl;
    std::cout << "If accuracy doesn't change despite massive weight changes, the network may be"
              << std::endl;
    std::cout << "performing at random levels where weight corruption doesn't matter." << std::endl;
}
