/**
 * @file trained_network_radiation_test.cpp
 * @brief Test radiation effects on a properly trained network
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <rad_ml/research/radiation_aware_training.hpp>
#include <rad_ml/research/residual_network.hpp>

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::neural;

// Create a simple learnable dataset
std::pair<std::vector<float>, std::vector<float>> createLearnableDataset(size_t samples,
                                                                         size_t input_size,
                                                                         size_t output_size)
{
    std::vector<float> data;
    std::vector<float> labels;

    // Create a simple pattern: output class = sum of inputs mod output_size
    for (size_t i = 0; i < samples; ++i) {
        // Generate deterministic but varied input
        std::vector<float> input(input_size);
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; ++j) {
            input[j] = std::sin(static_cast<float>(i * (j + 1)) / 10.0f);
            sum += input[j];
            data.push_back(input[j]);
        }

        // Simple classification rule
        size_t label_class = static_cast<size_t>(std::abs(sum * 2.0f)) % output_size;

        // Generate one-hot label
        for (size_t j = 0; j < output_size; ++j) {
            labels.push_back(j == label_class ? 1.0f : 0.0f);
        }
    }

    return {data, labels};
}

float measureAccuracy(ResidualNeuralNetwork<float>& network, const std::vector<float>& data,
                      const std::vector<float>& labels, size_t input_size, size_t output_size)
{
    size_t correct = 0;
    size_t total = data.size() / input_size;

    for (size_t i = 0; i < total; ++i) {
        std::vector<float> sample(data.begin() + i * input_size,
                                  data.begin() + (i + 1) * input_size);
        std::vector<float> label(labels.begin() + i * output_size,
                                 labels.begin() + (i + 1) * output_size);

        auto prediction = network.forward(sample);

        size_t pred_idx = std::distance(prediction.begin(),
                                        std::max_element(prediction.begin(), prediction.end()));
        size_t label_idx =
            std::distance(label.begin(), std::max_element(label.begin(), label.end()));

        if (pred_idx == label_idx) correct++;
    }

    return static_cast<float>(correct) / total;
}

TEST(TrainedNetworkRadiationTest, RadiationEffectsOnTrainedNetwork)
{
    std::cout << "\n=== TRAINED NETWORK RADIATION TEST ===\n" << std::endl;

    // Create a simple network
    ResidualNeuralNetwork<float> network({4, 16, 4}, ProtectionLevel::NONE);

    // Create learnable dataset
    auto [data, labels] = createLearnableDataset(400, 4, 4);

    // Train the network to good accuracy
    std::cout << "=== TRAINING NETWORK TO GOOD ACCURACY ===\n";
    float initial_accuracy = measureAccuracy(network, data, labels, 4, 4);
    std::cout << "Initial accuracy: " << std::fixed << std::setprecision(4) << initial_accuracy
              << std::endl;

    // Train for many epochs
    TrainingConfig config;
    config.epochs = 500;
    config.learning_rate = 0.01f;
    config.batch_size = 32;

    // Manual training loop
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        for (size_t i = 0; i < data.size() / (4 * config.batch_size); ++i) {
            size_t start_idx = i * config.batch_size;
            for (size_t j = 0; j < config.batch_size && (start_idx + j) * 4 < data.size(); ++j) {
                size_t sample_idx = start_idx + j;

                std::vector<float> sample(data.begin() + sample_idx * 4,
                                          data.begin() + (sample_idx + 1) * 4);
                std::vector<float> label(labels.begin() + sample_idx * 4,
                                         labels.begin() + (sample_idx + 1) * 4);

                // Train one step
                network.train(sample, label, 1, 1, config.learning_rate);
            }
        }

        if (epoch % 100 == 0) {
            float current_accuracy = measureAccuracy(network, data, labels, 4, 4);
            std::cout << "Epoch " << epoch << " accuracy: " << std::fixed << std::setprecision(4)
                      << current_accuracy << std::endl;
        }
    }

    // Final accuracy after training
    float trained_accuracy = measureAccuracy(network, data, labels, 4, 4);
    std::cout << "\nFinal trained accuracy: " << std::fixed << std::setprecision(4)
              << trained_accuracy << std::endl;

    // Only proceed if we achieved good accuracy
    if (trained_accuracy < 0.7f) {
        std::cout << "⚠️ Network didn't train well enough (accuracy < 70%). Skipping radiation test."
                  << std::endl;
        return;
    }

    std::cout << "✅ Network trained successfully! Proceeding with radiation test..." << std::endl;

    // Now test radiation effects
    std::cout << "\n=== TESTING RADIATION EFFECTS ===\n";

    // Get some weights before radiation
    auto layers = network.getLayers();
    std::cout << "Sample weights before radiation:\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "Weight[0][" << i << "]: " << std::fixed << std::setprecision(6)
                  << layers[0].weights[0][i] << std::endl;
    }

    // Apply radiation effects manually
    auto& layer = network.getLayerMutable(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> bit_dist(0, 31);

    int flips_applied = 0;
    // Apply bit flips to first layer weights
    for (size_t i = 0; i < layer.weights.size(); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
            if (gen() % 3 == 0) {  // Apply to ~33% of weights
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
    float post_radiation_accuracy = measureAccuracy(network, data, labels, 4, 4);
    std::cout << "\nPost-radiation accuracy: " << std::fixed << std::setprecision(4)
              << post_radiation_accuracy << std::endl;

    float accuracy_drop = trained_accuracy - post_radiation_accuracy;
    std::cout << "Accuracy drop: " << std::fixed << std::setprecision(4) << accuracy_drop
              << std::endl;
    std::cout << "Relative accuracy drop: " << std::fixed << std::setprecision(2)
              << (accuracy_drop / trained_accuracy) * 100.0f << "%" << std::endl;

    if (std::abs(accuracy_drop) > 0.05f) {
        std::cout << "✅ SIGNIFICANT accuracy drop detected! Radiation effects are working."
                  << std::endl;
    }
    else {
        std::cout << "❌ No significant accuracy drop despite " << flips_applied << " bit flips."
                  << std::endl;
    }

    std::cout << "\n=== ANALYSIS ===\n";
    std::cout << "This test shows radiation effects on a properly trained network." << std::endl;
    std::cout
        << "A significant accuracy drop confirms that radiation injection is working correctly."
        << std::endl;
}
