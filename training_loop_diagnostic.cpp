/**
 * @file training_loop_diagnostic.cpp
 * @brief Diagnostic test to trace training loop and identify radiation effect loss
 *
 * This test specifically investigates the suspected timing issue where accuracy
 * measurements happen on stale/cached data rather than fresh forward passes
 * through the radiation-corrupted network.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <rad_ml/research/radiation_aware_training.hpp>
#include <rad_ml/research/residual_network.hpp>
#include <rad_ml/sim/environment.hpp>
#include <rad_ml/utils/bit_manipulation.hpp>
#include <set>
#include <sstream>
#include <vector>

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::neural;
using namespace rad_ml::utils;
using namespace rad_ml::sim;

// Helper to create deterministic test data
std::pair<std::vector<float>, std::vector<float>> createDeterministicDataset(int num_samples,
                                                                             int input_size,
                                                                             int output_size)
{
    std::vector<float> data;
    std::vector<float> labels;

    // Use deterministic patterns instead of random
    for (int i = 0; i < num_samples; ++i) {
        // Create input pattern
        for (int j = 0; j < input_size; ++j) {
            data.push_back(std::sin(i * 0.1 + j * 0.2));
        }

        // Create output pattern
        for (int j = 0; j < output_size; ++j) {
            labels.push_back(std::cos(i * 0.15 + j * 0.25));
        }
    }

    return {data, labels};
}

// Helper to compare vectors and detect changes
bool compareVectors(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-6f)
{
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// Helper to calculate vector hash for quick comparison
size_t vectorHash(const std::vector<float>& vec)
{
    size_t hash = 0;
    for (float val : vec) {
        hash ^= std::hash<float>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

TEST(TrainingLoopDiagnostic, RadiationEffectTracing)
{
    std::cout << "\n=== RADIATION EFFECT TRACING IN TRAINING LOOP ===" << std::endl;

    // Create a simple network
    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

    // Create deterministic test data
    auto [data, labels] = createDeterministicDataset(20, 4, 4);

    // Store baseline outputs before any radiation
    std::vector<float> input_sample = {data[0], data[1], data[2], data[3]};
    std::vector<float> baseline_output = network.forward(input_sample);

    std::cout << "Baseline output: ";
    for (float val : baseline_output) {
        std::cout << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << std::endl;

    // Test 1: Direct radiation application
    std::cout << "\n--- Test 1: Direct Radiation Application ---" << std::endl;

    std::vector<std::vector<float>> radiation_outputs;
    std::vector<size_t> radiation_hashes;

    for (int trial = 0; trial < 10; ++trial) {
        // Apply radiation directly
        auto radiation_output = network.forward(input_sample, 0.1f);  // 10% bit flip probability
        radiation_outputs.push_back(radiation_output);
        radiation_hashes.push_back(vectorHash(radiation_output));

        std::cout << "Trial " << trial << " output: ";
        for (float val : radiation_output) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << " (hash: " << radiation_hashes.back() << ")" << std::endl;
    }

    // Check for diversity in radiation effects
    std::set<size_t> unique_hashes(radiation_hashes.begin(), radiation_hashes.end());
    std::cout << "Unique output patterns: " << unique_hashes.size() << "/10" << std::endl;

    if (unique_hashes.size() < 3) {
        std::cout << "❌ WARNING: Low diversity in radiation effects" << std::endl;
    }
    else {
        std::cout << "✅ Good diversity in radiation effects" << std::endl;
    }
}

TEST(TrainingLoopDiagnostic, TrainingLoopCacheInvestigation)
{
    std::cout << "\n=== TRAINING LOOP CACHE INVESTIGATION ===" << std::endl;

    // Create a radiation-aware trainer
    RadiationAwareTraining trainer(0.05f, false, Environment::JUPITER);

    // Create a network
    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

    // Create test data
    auto [data, labels] = createDeterministicDataset(40, 4, 4);

    // Configure training
    neural::TrainingConfig config;
    config.epochs = 5;
    config.batch_size = 8;
    config.learning_rate = 0.01f;

    std::cout << "Starting training with detailed tracing..." << std::endl;

    // We'll need to create a custom training loop to trace what's happening
    // Since we can't directly access the internal training loop, let's simulate it

    std::vector<float> input_sample = {data[0], data[1], data[2], data[3]};
    std::vector<float> label_sample = {labels[0], labels[1], labels[2], labels[3]};

    std::cout << "\n--- Simulating Training Loop Behavior ---" << std::endl;

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "\nEpoch " << epoch << ":" << std::endl;

        // Simulate what happens in the training loop

        // 1. Forward pass for accuracy measurement (PRE-radiation)
        auto pre_radiation_output = network.forward(input_sample, 0.0f);
        std::cout << "  Pre-radiation output: ";
        for (float val : pre_radiation_output) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;

        // 2. Apply radiation effects (this is where the bug might be)
        auto radiation_output = network.forward(input_sample, 0.05f);
        std::cout << "  Post-radiation output: ";
        for (float val : radiation_output) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;

        // 3. Check if they're different
        bool outputs_different = !compareVectors(pre_radiation_output, radiation_output, 1e-6f);
        std::cout << "  Radiation effect detected: " << (outputs_different ? "YES" : "NO")
                  << std::endl;

        // 4. Calculate accuracy on both
        float pre_accuracy = 0.0f;
        float post_accuracy = 0.0f;

        for (size_t i = 0; i < pre_radiation_output.size(); ++i) {
            float pre_error = std::abs(pre_radiation_output[i] - label_sample[i]);
            float post_error = std::abs(radiation_output[i] - label_sample[i]);
            pre_accuracy += (1.0f - pre_error);
            post_accuracy += (1.0f - post_error);
        }

        pre_accuracy /= pre_radiation_output.size();
        post_accuracy /= radiation_output.size();

        std::cout << "  Pre-radiation accuracy: " << std::fixed << std::setprecision(6)
                  << pre_accuracy << std::endl;
        std::cout << "  Post-radiation accuracy: " << std::fixed << std::setprecision(6)
                  << post_accuracy << std::endl;
        std::cout << "  Accuracy drop: " << std::fixed << std::setprecision(6)
                  << (pre_accuracy - post_accuracy) << std::endl;

        // 5. This is the key question: Which accuracy is being reported?
        if (std::abs(pre_accuracy - post_accuracy) < 1e-6f) {
            std::cout << "  ❌ POTENTIAL BUG: Identical accuracies suggest cached/stale data"
                      << std::endl;
        }
        else {
            std::cout << "  ✅ Different accuracies suggest fresh calculations" << std::endl;
        }
    }
}

TEST(TrainingLoopDiagnostic, RadiationAwareTrainingDeepTrace)
{
    std::cout << "\n=== RADIATION-AWARE TRAINING DEEP TRACE ===" << std::endl;

    // Create a custom implementation that traces what's happening
    class TracingRadiationAwareTraining : public RadiationAwareTraining {
       public:
        TracingRadiationAwareTraining(float bit_flip_prob, bool critical_targeting, Environment env)
            : RadiationAwareTraining(bit_flip_prob, critical_targeting, env)
        {
        }

        // We'll need to override or instrument the training method
        // For now, let's simulate the suspected behavior

        void simulateTrainingStep(ResidualNeuralNetwork<float>& network,
                                  const std::vector<float>& input,
                                  const std::vector<float>& expected_output)
        {
            std::cout << "  Training step simulation:" << std::endl;

            // Step 1: Calculate baseline accuracy
            auto baseline_output = network.forward(input, 0.0f);
            float baseline_accuracy = calculateAccuracy(baseline_output, expected_output);
            std::cout << "    Baseline accuracy: " << std::fixed << std::setprecision(6)
                      << baseline_accuracy << std::endl;

            // Step 2: Apply radiation
            auto radiation_output = network.forward(input, 0.05f);
            float radiation_accuracy = calculateAccuracy(radiation_output, expected_output);
            std::cout << "    Radiation accuracy: " << std::fixed << std::setprecision(6)
                      << radiation_accuracy << std::endl;

            // Step 3: Check what the training loop might be doing
            std::cout << "    Output comparison:" << std::endl;
            std::cout << "      Baseline hash: " << vectorHash(baseline_output) << std::endl;
            std::cout << "      Radiation hash: " << vectorHash(radiation_output) << std::endl;

            // Step 4: This is where the bug might be - using wrong accuracy
            float reported_accuracy = baseline_accuracy;  // BUG: Should be radiation_accuracy?
            std::cout << "    Reported accuracy: " << std::fixed << std::setprecision(6)
                      << reported_accuracy << std::endl;

            if (std::abs(reported_accuracy - baseline_accuracy) < 1e-6f) {
                std::cout << "    ❌ SUSPECTED BUG: Reporting baseline accuracy instead of "
                             "radiation accuracy"
                          << std::endl;
            }
            else {
                std::cout << "    ✅ Correctly reporting radiation-affected accuracy" << std::endl;
            }
        }

       private:
        float calculateAccuracy(const std::vector<float>& output,
                                const std::vector<float>& expected)
        {
            float accuracy = 0.0f;
            for (size_t i = 0; i < output.size() && i < expected.size(); ++i) {
                accuracy += (1.0f - std::abs(output[i] - expected[i]));
            }
            return accuracy / output.size();
        }
    };

    // Test the suspected behavior
    TracingRadiationAwareTraining tracer(0.05f, false, Environment::JUPITER);
    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

    auto [data, labels] = createDeterministicDataset(20, 4, 4);
    std::vector<float> input_sample = {data[0], data[1], data[2], data[3]};
    std::vector<float> label_sample = {labels[0], labels[1], labels[2], labels[3]};

    std::cout << "Running traced training steps..." << std::endl;

    for (int step = 0; step < 3; ++step) {
        std::cout << "\nStep " << step << ":" << std::endl;
        tracer.simulateTrainingStep(network, input_sample, label_sample);
    }
}

TEST(TrainingLoopDiagnostic, TimingSequenceAnalysis)
{
    std::cout << "\n=== TIMING SEQUENCE ANALYSIS ===" << std::endl;

    // This test checks the exact sequence of operations in the training loop
    // to identify where radiation effects might be getting lost

    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);
    auto [data, labels] = createDeterministicDataset(20, 4, 4);

    std::vector<float> input_sample = {data[0], data[1], data[2], data[3]};
    std::vector<float> label_sample = {labels[0], labels[1], labels[2], labels[3]};

    std::cout << "Analyzing potential timing sequences..." << std::endl;

    // Sequence 1: Correct sequence
    std::cout << "\n--- Sequence 1: Correct (Radiation → Accuracy) ---" << std::endl;
    auto radiation_output1 = network.forward(input_sample, 0.05f);
    float accuracy1 = 0.0f;
    for (size_t i = 0; i < radiation_output1.size(); ++i) {
        accuracy1 += (1.0f - std::abs(radiation_output1[i] - label_sample[i]));
    }
    accuracy1 /= radiation_output1.size();
    std::cout << "  Radiation-affected accuracy: " << std::fixed << std::setprecision(6)
              << accuracy1 << std::endl;

    // Sequence 2: Suspected bug sequence
    std::cout << "\n--- Sequence 2: Suspected Bug (Accuracy → Radiation) ---" << std::endl;
    auto baseline_output2 = network.forward(input_sample, 0.0f);
    float accuracy2 = 0.0f;
    for (size_t i = 0; i < baseline_output2.size(); ++i) {
        accuracy2 += (1.0f - std::abs(baseline_output2[i] - label_sample[i]));
    }
    accuracy2 /= baseline_output2.size();
    // Now apply radiation (but accuracy already calculated)
    auto radiation_output2 = network.forward(input_sample, 0.05f);
    std::cout << "  Baseline accuracy (bug): " << std::fixed << std::setprecision(6) << accuracy2
              << std::endl;

    // Sequence 3: Caching issue
    std::cout << "\n--- Sequence 3: Caching Issue (Same Forward Pass) ---" << std::endl;
    auto cached_output3 = network.forward(input_sample, 0.0f);
    // Bug: Using cached output for radiation calculation
    float accuracy3 = 0.0f;
    for (size_t i = 0; i < cached_output3.size(); ++i) {
        accuracy3 += (1.0f - std::abs(cached_output3[i] - label_sample[i]));
    }
    accuracy3 /= cached_output3.size();
    std::cout << "  Cached accuracy: " << std::fixed << std::setprecision(6) << accuracy3
              << std::endl;

    // Compare results
    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "  Correct accuracy: " << std::fixed << std::setprecision(6) << accuracy1
              << std::endl;
    std::cout << "  Bug accuracy: " << std::fixed << std::setprecision(6) << accuracy2 << std::endl;
    std::cout << "  Cached accuracy: " << std::fixed << std::setprecision(6) << accuracy3
              << std::endl;

    if (std::abs(accuracy2 - accuracy3) < 1e-6f) {
        std::cout << "  ❌ CONFIRMED: Bug and cached accuracies are identical" << std::endl;
        std::cout << "  This suggests accuracy is calculated on non-radiated data" << std::endl;
    }
    else {
        std::cout << "  ✅ Different accuracies suggest no caching issue" << std::endl;
    }

    if (std::abs(accuracy1 - accuracy2) > 1e-6f) {
        std::cout << "  ✅ Radiation does affect accuracy when calculated correctly" << std::endl;
    }
    else {
        std::cout << "  ❌ WARNING: Radiation doesn't seem to affect accuracy" << std::endl;
    }
}

// Helper function to run all diagnostics
void runFullDiagnostic()
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FULL TRAINING LOOP DIAGNOSTIC REPORT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nThis diagnostic investigates the suspected timing issue where:" << std::endl;
    std::cout << "1. Accuracy measurements happen on stale/cached data" << std::endl;
    std::cout << "2. Radiation effects are applied after accuracy calculation" << std::endl;
    std::cout << "3. Forward passes are cached and not updated with radiation" << std::endl;

    std::cout << "\nDiagnostic tests will reveal:" << std::endl;
    std::cout << "- Whether radiation effects are actually applied" << std::endl;
    std::cout << "- If accuracy calculations use fresh or stale data" << std::endl;
    std::cout << "- The exact sequence of operations in the training loop" << std::endl;
    std::cout << "- Where radiation effects are getting lost" << std::endl;
}

// Note: Using gtest_main, no custom main function needed
