/**
 * @file trivial_radiation_test.cpp
 * @brief Test radiation effects on a network trained on trivial data
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <rad_ml/research/radiation_aware_training.hpp>
#include <rad_ml/research/residual_network.hpp>

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::neural;

// Create extremely simple binary classification dataset
std::pair<std::vector<float>, std::vector<float>> createTrivialDataset()
{
    std::vector<float> data = {// Class 0: All inputs negative
                               -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f, -2.0f, -1.5f, -1.5f,
                               -1.5f, -1.5f, -0.5f, -0.5f, -0.5f, -0.5f,

                               // Class 1: All inputs positive
                               1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.5f, 1.5f, 1.5f,
                               1.5f, 0.5f, 0.5f, 0.5f, 0.5f,

                               // Repeat for more training samples
                               -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f, -2.0f, -1.5f, -1.5f,
                               -1.5f, -1.5f, -0.5f, -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f,
                               2.0f, 2.0f, 2.0f, 2.0f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, 0.5f,
                               0.5f};

    std::vector<float> labels = {
        // Class 0 (negative inputs)
        1.0f, 0.0f,  // Sample 1
        1.0f, 0.0f,  // Sample 2
        1.0f, 0.0f,  // Sample 3
        1.0f, 0.0f,  // Sample 4

        // Class 1 (positive inputs)
        0.0f, 1.0f,  // Sample 5
        0.0f, 1.0f,  // Sample 6
        0.0f, 1.0f,  // Sample 7
        0.0f, 1.0f,  // Sample 8

        // Repeat
        1.0f, 0.0f,  // Sample 9
        1.0f, 0.0f,  // Sample 10
        1.0f, 0.0f,  // Sample 11
        1.0f, 0.0f,  // Sample 12
        0.0f, 1.0f,  // Sample 13
        0.0f, 1.0f,  // Sample 14
        0.0f, 1.0f,  // Sample 15
        0.0f, 1.0f   // Sample 16
    };

    return {data, labels};
}

float measureAccuracy(ResidualNeuralNetwork<float>& network, const std::vector<float>& data,
                      const std::vector<float>& labels)
{
    size_t correct = 0;
    size_t total = data.size() / 4;  // 4 inputs per sample

    for (size_t i = 0; i < total; ++i) {
        std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
        std::vector<float> label(labels.begin() + i * 2, labels.begin() + (i + 1) * 2);

        auto prediction = network.forward(sample);

        size_t pred_idx = std::distance(prediction.begin(),
                                        std::max_element(prediction.begin(), prediction.end()));
        size_t label_idx =
            std::distance(label.begin(), std::max_element(label.begin(), label.end()));

        if (pred_idx == label_idx) correct++;
    }

    return static_cast<float>(correct) / total;
}

TEST(TrivialRadiationTest, RadiationEffectsOnTrivialDataset)
{
    std::cout << "\n=== TRIVIAL RADIATION TEST ===\n" << std::endl;
    std::cout << "Testing radiation effects on a network trained on trivial binary classification:"
              << std::endl;
    std::cout << "- Negative inputs -> Class 0" << std::endl;
    std::cout << "- Positive inputs -> Class 1" << std::endl;

    // Create a small network for binary classification
    ResidualNeuralNetwork<float> network({4, 8, 2}, ProtectionLevel::NONE);

    // Create trivial dataset
    auto [data, labels] = createTrivialDataset();

    std::cout << "\nDataset size: " << data.size() / 4 << " samples" << std::endl;

    // Train the network
    std::cout << "\n=== TRAINING NETWORK ===\n";
    float initial_accuracy = measureAccuracy(network, data, labels);
    std::cout << "Initial accuracy: " << std::fixed << std::setprecision(4) << initial_accuracy
              << std::endl;

    // Train for many epochs with high learning rate
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (size_t i = 0; i < data.size() / 4; ++i) {
            std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
            std::vector<float> label(labels.begin() + i * 2, labels.begin() + (i + 1) * 2);

            // Train one step with high learning rate
            network.train(sample, label, 1, 1, 0.1f);
        }

        if (epoch % 200 == 0) {
            float current_accuracy = measureAccuracy(network, data, labels);
            std::cout << "Epoch " << epoch << " accuracy: " << std::fixed << std::setprecision(4)
                      << current_accuracy << std::endl;
        }
    }

    // Final accuracy after training
    float trained_accuracy = measureAccuracy(network, data, labels);
    std::cout << "\nFinal trained accuracy: " << std::fixed << std::setprecision(4)
              << trained_accuracy << std::endl;

    // Only proceed if we achieved good accuracy
    if (trained_accuracy < 0.9f) {
        std::cout << "⚠️ Network didn't train well enough (accuracy < 90%). Something is wrong."
                  << std::endl;

        // Debug: Test predictions manually
        std::cout << "\n=== DEBUGGING PREDICTIONS ===\n";
        for (size_t i = 0; i < 4; ++i) {
            std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
            std::vector<float> label(labels.begin() + i * 2, labels.begin() + (i + 1) * 2);
            auto prediction = network.forward(sample);

            std::cout << "Sample " << i << " - Input: [";
            for (float f : sample) std::cout << f << " ";
            std::cout << "] Expected: [";
            for (float f : label) std::cout << f << " ";
            std::cout << "] Predicted: [";
            for (float f : prediction) std::cout << std::fixed << std::setprecision(4) << f << " ";
            std::cout << "]" << std::endl;
        }

        return;
    }

    std::cout << "✅ Network trained successfully to " << std::fixed << std::setprecision(2)
              << trained_accuracy * 100 << "%!" << std::endl;

    // Show some example predictions
    std::cout << "\n=== EXAMPLE PREDICTIONS ===\n";
    for (size_t i = 0; i < 4; ++i) {
        std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
        auto prediction = network.forward(sample);

        std::cout << "Input: [";
        for (float f : sample) std::cout << f << " ";
        std::cout << "] -> Prediction: [";
        for (float f : prediction) std::cout << std::fixed << std::setprecision(4) << f << " ";
        std::cout << "]" << std::endl;
    }

    // Now test radiation effects
    std::cout << "\n=== TESTING RADIATION EFFECTS ===\n";

    // Apply radiation effects to the first layer
    auto& layer = network.getLayerMutable(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> bit_dist(0, 31);

    int flips_applied = 0;
    std::cout << "Applying bit flips to layer 0 weights..." << std::endl;

    // Apply bit flips to some weights
    for (size_t i = 0; i < layer.weights.size(); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
            if (gen() % 2 == 0) {  // Apply to 50% of weights
                float original = layer.weights[i][j];

                // Apply bit flip
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

                if (flips_applied <= 5) {
                    std::cout << "Weight[" << i << "][" << j << "]: " << std::fixed
                              << std::setprecision(6) << original << " -> " << flipped << " (bit "
                              << bit_pos << ")" << std::endl;
                }
            }
        }
    }

    std::cout << "Total bit flips applied: " << flips_applied << std::endl;

    // Measure accuracy after radiation
    float post_radiation_accuracy = measureAccuracy(network, data, labels);
    std::cout << "\nPost-radiation accuracy: " << std::fixed << std::setprecision(4)
              << post_radiation_accuracy << std::endl;

    float accuracy_drop = trained_accuracy - post_radiation_accuracy;
    std::cout << "Accuracy drop: " << std::fixed << std::setprecision(4) << accuracy_drop
              << std::endl;
    std::cout << "Relative accuracy drop: " << std::fixed << std::setprecision(2)
              << (accuracy_drop / trained_accuracy) * 100.0f << "%" << std::endl;

    // Show some post-radiation predictions
    std::cout << "\n=== POST-RADIATION PREDICTIONS ===\n";
    for (size_t i = 0; i < 4; ++i) {
        std::vector<float> sample(data.begin() + i * 4, data.begin() + (i + 1) * 4);
        auto prediction = network.forward(sample);

        std::cout << "Input: [";
        for (float f : sample) std::cout << f << " ";
        std::cout << "] -> Prediction: [";
        for (float f : prediction) std::cout << std::fixed << std::setprecision(4) << f << " ";
        std::cout << "]" << std::endl;
    }

    if (std::abs(accuracy_drop) > 0.05f) {
        std::cout << "\n✅ SIGNIFICANT accuracy drop detected! Radiation effects are measurable."
                  << std::endl;
    }
    else {
        std::cout << "\n❌ No significant accuracy drop despite " << flips_applied << " bit flips."
                  << std::endl;
        std::cout << "This suggests either the network is very robust or there's still a "
                     "measurement issue."
                  << std::endl;
    }
}
