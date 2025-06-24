/**
 * @file radiation_aware_training.cpp
 * @brief Implementation of radiation-aware training for neural networks
 */

#include "rad_ml/research/radiation_aware_training.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <type_traits>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/neural/protected_neural_network.hpp"
#include "rad_ml/research/residual_network.hpp"
#include "rad_ml/utils/bit_manipulation.hpp"

namespace rad_ml {
namespace research {

// Template to check if a network has a saveState method
template <typename, typename = void>
struct has_save_state_method : std::false_type {};

template <typename T>
struct has_save_state_method<T, std::void_t<decltype(std::declval<T>().saveState())>>
    : std::true_type {};

RadiationAwareTraining::RadiationAwareTraining(float bit_flip_probability,
                                               bool target_critical_weights,
                                               sim::Environment environment)
    : bit_flip_probability_(bit_flip_probability),
      target_critical_weights_(target_critical_weights),
      environment_(environment),
      uniform_dist_(0.0f, 1.0f),
      bit_dist_(0, 31)  // For 32-bit floats
{
    // Use time-based seed for random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);

    core::Logger::info("RadiationAwareTraining initialized with bit flip probability: " +
                       std::to_string(bit_flip_probability_));
}

template <typename Network>
RadiationAwareTraining::TrainingResult RadiationAwareTraining::train(
    Network& network, const std::vector<float>& train_data, const std::vector<float>& train_labels,
    const ::rad_ml::neural::TrainingConfig& config)
{
    try {
        core::Logger::info("Starting radiation-aware training...");

        // Specialized training for residual networks
        if constexpr (std::is_same_v<std::decay_t<Network>, ResidualNeuralNetwork<float>>) {
            core::Logger::info("Using specialized training for residual network");
        }

        // Get input and output sizes from the network
        const size_t input_size = network.getInputSize();
        const size_t output_size = network.getOutputSize();

        // Check that data dimensions make sense
        if (train_data.size() % input_size != 0 || train_labels.size() % output_size != 0) {
            return std::string("Error: Input/output data size mismatch with network dimensions");
        }

        // Calculate number of samples
        const size_t num_samples = train_data.size() / input_size;
        if (num_samples * output_size != train_labels.size()) {
            return std::string(
                "Error: Number of input samples doesn't match number of output samples");
        }

        core::Logger::debug("Processing " + std::to_string(num_samples) +
                            " samples with input size " + std::to_string(input_size) +
                            " and output size " + std::to_string(output_size));

        // Reshape flat data into individual samples
        std::vector<std::vector<float>> samples;
        std::vector<std::vector<float>> labels;

        // Create individual samples from flat data
        for (size_t i = 0; i < num_samples; i++) {
            // Extract this sample's data
            std::vector<float> sample(train_data.begin() + i * input_size,
                                      train_data.begin() + (i + 1) * input_size);
            std::vector<float> label(train_labels.begin() + i * output_size,
                                     train_labels.begin() + (i + 1) * output_size);

            samples.push_back(std::move(sample));
            labels.push_back(std::move(label));
        }

        // If targeting critical weights, initialize criticality map
        if (target_critical_weights_ && !weight_criticality_.has_value()) {
            core::Logger::info("Initializing weight criticality map...");
            // Use the first few samples to analyze criticality
            size_t samples_for_analysis = std::min(num_samples, size_t{10});

            // Create a subset of data for criticality analysis
            std::vector<float> analysis_data;
            std::vector<float> analysis_labels;

            for (size_t i = 0; i < samples_for_analysis; i++) {
                analysis_data.insert(analysis_data.end(), samples[i].begin(), samples[i].end());
                analysis_labels.insert(analysis_labels.end(), labels[i].begin(), labels[i].end());
            }

            updateCriticalityMap(network, analysis_data, analysis_labels);
        }

        // Calculate injection frequency
        int injection_frequency = std::max(1, static_cast<int>(1.0f / bit_flip_probability_));

        // Initial baseline accuracy - measure on all samples
        float baseline_accuracy = 0.0f;
        size_t correct_predictions = 0;

        for (size_t i = 0; i < samples.size(); i++) {
            std::vector<float> prediction = network.forward(samples[i]);
            // Find highest probability in prediction and label
            size_t pred_idx = std::distance(prediction.begin(),
                                            std::max_element(prediction.begin(), prediction.end()));
            size_t label_idx = std::distance(labels[i].begin(),
                                             std::max_element(labels[i].begin(), labels[i].end()));
            if (pred_idx == label_idx) {
                correct_predictions++;
            }
        }
        baseline_accuracy = static_cast<float>(correct_predictions) / samples.size();

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            // Process in batches
            const size_t batch_size = config.batch_size;
            const size_t num_batches =
                (num_samples + batch_size - 1) / batch_size;  // Ceiling division

            for (size_t batch = 0; batch < num_batches; batch++) {
                // Calculate batch range
                size_t start_idx = batch * batch_size;
                size_t end_idx = std::min(start_idx + batch_size, num_samples);
                size_t batch_samples = end_idx - start_idx;

                // Process each sample in the batch
                for (size_t i = 0; i < batch_samples; i++) {
                    const size_t sample_idx = start_idx + i;
                    // Standard training step - handle both old and new API
                    if constexpr (std::is_same_v<Network,
                                                 ::rad_ml::neural::ProtectedNeuralNetwork<float>>) {
                        // New API with OptimizerConfig
                        typename Network::OptimizerConfig optimizer_config;
                        optimizer_config.learning_rate = config.learning_rate;
                        optimizer_config.type = Network::OptimizerType::ADAM;
                        network.train(samples[sample_idx], labels[sample_idx], 1, 1,
                                      optimizer_config);
                    }
                    else {
                        // Old API with float learning_rate
                        network.train(samples[sample_idx], labels[sample_idx], 1, 1,
                                      config.learning_rate);
                    }
                }
            }

            // Inject radiation effects periodically
            if (epoch % injection_frequency == 0) {
                // Save network state (if supported)
                // Note: This uses SFINAE to check if the network has a saveState method
                if constexpr (has_save_state_method<Network>::value) {
                    auto state = network.saveState();
                }

                // Measure accuracy before injection
                float pre_injection_accuracy = 0.0f;
                correct_predictions = 0;

                for (size_t i = 0; i < samples.size(); i++) {
                    std::vector<float> prediction = network.forward(samples[i]);
                    size_t pred_idx = std::distance(
                        prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
                    size_t label_idx = std::distance(
                        labels[i].begin(), std::max_element(labels[i].begin(), labels[i].end()));
                    if (pred_idx == label_idx) {
                        correct_predictions++;
                    }
                }
                pre_injection_accuracy = static_cast<float>(correct_predictions) / samples.size();

                // Track time for injection operation
                auto start = std::chrono::high_resolution_clock::now();

                // Inject bit flips
                injectRadiationEffects(network);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                double injection_time = elapsed.count();

                // Measure accuracy after injection
                float post_injection_accuracy = 0.0f;
                correct_predictions = 0;

                for (size_t i = 0; i < samples.size(); i++) {
                    std::vector<float> prediction = network.forward(samples[i]);
                    size_t pred_idx = std::distance(
                        prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
                    size_t label_idx = std::distance(
                        labels[i].begin(), std::max_element(labels[i].begin(), labels[i].end()));
                    if (pred_idx == label_idx) {
                        correct_predictions++;
                    }
                }
                post_injection_accuracy = static_cast<float>(correct_predictions) / samples.size();

                // Update statistics
                stats_.avg_accuracy_drop += (pre_injection_accuracy - post_injection_accuracy);

                // Continue training for recovery - process all samples once
                for (size_t i = 0; i < samples.size(); i++) {
                    if constexpr (std::is_same_v<Network,
                                                 ::rad_ml::neural::ProtectedNeuralNetwork<float>>) {
                        // New API with OptimizerConfig
                        typename Network::OptimizerConfig optimizer_config;
                        optimizer_config.learning_rate = config.learning_rate;
                        optimizer_config.type = Network::OptimizerType::ADAM;
                        network.train(samples[i], labels[i], 1, 1, optimizer_config);
                    }
                    else {
                        // Old API with float learning_rate
                        network.train(samples[i], labels[i], 1, 1, config.learning_rate);
                    }
                }

                // Measure recovery
                float recovery_accuracy = 0.0f;
                correct_predictions = 0;

                for (size_t i = 0; i < samples.size(); i++) {
                    std::vector<float> prediction = network.forward(samples[i]);
                    size_t pred_idx = std::distance(
                        prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
                    size_t label_idx = std::distance(
                        labels[i].begin(), std::max_element(labels[i].begin(), labels[i].end()));
                    if (pred_idx == label_idx) {
                        correct_predictions++;
                    }
                }
                recovery_accuracy = static_cast<float>(correct_predictions) / samples.size();

                // Avoid division by zero
                float accuracy_drop =
                    std::max(0.001f, pre_injection_accuracy - post_injection_accuracy);
                stats_.recovery_rate +=
                    (recovery_accuracy - post_injection_accuracy) / accuracy_drop;

                if (epoch % (10 * injection_frequency) == 0) {
                    core::Logger::info(
                        "Epoch " + std::to_string(epoch) +
                        ": Accuracy before injection: " + std::to_string(pre_injection_accuracy) +
                        ", after: " + std::to_string(post_injection_accuracy) +
                        ", recovery: " + std::to_string(recovery_accuracy) +
                        ", time: " + std::to_string(injection_time) + "ms");
                }
            }

            // Update criticality map periodically if targeting critical weights
            if (target_critical_weights_ && weight_criticality_.has_value() &&
                epoch % (10 * injection_frequency) == 0) {
                // Use a subset of samples for criticality update
                size_t samples_for_analysis = std::min(num_samples, size_t{10});

                // Create a subset of data for criticality analysis
                std::vector<float> analysis_data;
                std::vector<float> analysis_labels;

                for (size_t i = 0; i < samples_for_analysis; i++) {
                    analysis_data.insert(analysis_data.end(), samples[i].begin(), samples[i].end());
                    analysis_labels.insert(analysis_labels.end(), labels[i].begin(),
                                           labels[i].end());
                }

                updateCriticalityMap(network, analysis_data, analysis_labels);
            }
        }

        // Normalize statistics
        int injection_count = config.epochs / injection_frequency;
        stats_.avg_accuracy_drop /= std::max(1, injection_count);
        stats_.recovery_rate /= std::max(1, injection_count);

        core::Logger::info("Radiation-aware training completed. Average accuracy drop: " +
                           std::to_string(stats_.avg_accuracy_drop) +
                           ", Recovery rate: " + std::to_string(stats_.recovery_rate));

        // Return successful result
        return stats_;
    }
    catch (const std::exception& e) {
        // Return error message
        return std::string("Error during training: ") + e.what();
    }
}

template <typename Network>
void RadiationAwareTraining::injectRadiationEffects(Network& network)
{
    // Get environment factor for scaling bit flip probability
    float env_factor = [this]() {
        switch (environment_) {
            case sim::Environment::EARTH_ORBIT:
            case sim::Environment::ISS:
                return 0.5f;
            case sim::Environment::MARS:
                return 2.0f;
            case sim::Environment::JUPITER:
                return 5.0f;
            case sim::Environment::EXTREME:
                return 10.0f;
            default:
                return 1.0f;
        }
    }();

    int total_flips_this_round = 0;

    // Get const layers for size information
    const auto& layers = network.getLayers();

    // Process each layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        // Get a mutable reference to the layer
        auto& layer = network.getLayerMutable(layer_idx);

        // Process weights (now correctly handling the 2D structure)
        for (size_t input_idx = 0; input_idx < layer.weights.size(); ++input_idx) {
            for (size_t output_idx = 0; output_idx < layer.weights[input_idx].size();
                 ++output_idx) {
                float flip_probability = bit_flip_probability_ * env_factor;

                // Adjust probability based on criticality if enabled
                if (target_critical_weights_ && weight_criticality_.has_value()) {
                    const auto& criticality_map = *weight_criticality_;
                    if (layer_idx < criticality_map.size() &&
                        input_idx < criticality_map[layer_idx].size() &&
                        output_idx < criticality_map[layer_idx][input_idx].size()) {
                        float criticality = criticality_map[layer_idx][input_idx][output_idx];
                        // Increase probability for more critical weights
                        flip_probability *= (1.0f + 5.0f * criticality);
                    }
                }

                // Determine if this weight should be affected
                if (uniform_dist_(generator_) < flip_probability) {
                    // Save original weight for logging
                    float original = layer.weights[input_idx][output_idx];

                    // Select a random bit to flip
                    int bit = bit_dist_(generator_);

                    // Flip the bit
                    layer.weights[input_idx][output_idx] = flipBit(original, bit);

                    total_flips_this_round++;
                }
            }
        }

        // Process biases if they exist
        if (!layer.biases.empty()) {
            for (size_t bias_idx = 0; bias_idx < layer.biases.size(); ++bias_idx) {
                if (uniform_dist_(generator_) < bit_flip_probability_ * env_factor * 0.5f) {
                    // Biases are typically less critical than weights, so use lower probability
                    float original = layer.biases[bias_idx];
                    int bit = bit_dist_(generator_);
                    layer.biases[bias_idx] = flipBit(original, bit);
                    total_flips_this_round++;
                }
            }
        }
    }

    stats_.total_bit_flips += total_flips_this_round;
    core::Logger::debug("Injected " + std::to_string(total_flips_this_round) + " bit flips");
}

float RadiationAwareTraining::flipBit(float value, int bit_position) const
{
    // Use the BitManipulation utility
    return utils::BitManipulation::flipBit(value, bit_position);
}

void RadiationAwareTraining::setBitFlipProbability(float probability)
{
    bit_flip_probability_ = probability;
}

void RadiationAwareTraining::setTargetCriticalWeights(bool target)
{
    target_critical_weights_ = target;
    if (target && !weight_criticality_.has_value()) {
        core::Logger::warning(
            "Critical weight targeting enabled but criticality map not initialized");
    }
}

void RadiationAwareTraining::setEnvironment(sim::Environment env) { environment_ = env; }

std::optional<std::vector<std::vector<std::vector<float>>>>
RadiationAwareTraining::getCriticalityMap() const
{
    return weight_criticality_;
}

RadiationAwareTraining::TrainingStats RadiationAwareTraining::getStats() const { return stats_; }

template <typename Network>
void RadiationAwareTraining::updateCriticalityMap(Network& network, const std::vector<float>& data,
                                                  const std::vector<float>& labels)
{
    core::Logger::info("Updating weight criticality map...");

    // Get const layers for size information
    const auto& layers = network.getLayers();

    // Get network input/output sizes
    const size_t input_size = network.getInputSize();
    const size_t output_size = network.getOutputSize();

    // Validate data dimensions
    if (data.size() % input_size != 0 || labels.size() % output_size != 0) {
        core::Logger::error("Input size mismatch in updateCriticalityMap");
        core::Logger::error("Expected input size: " + std::to_string(input_size) +
                            ", Actual input size: " + std::to_string(data.size()));
        return;
    }

    // Use just the first sample for criticality analysis
    std::vector<float> single_sample(data.begin(), data.begin() + input_size);
    std::vector<float> single_label(labels.begin(), labels.begin() + output_size);

    // Initialize or resize criticality map if needed
    if (!weight_criticality_.has_value()) {
        std::vector<std::vector<std::vector<float>>> criticality;
        criticality.resize(layers.size());
        for (size_t l = 0; l < layers.size(); ++l) {
            criticality[l].resize(layers[l].weights.size());
            for (size_t i = 0; i < layers[l].weights.size(); ++i) {
                criticality[l][i].resize(layers[l].weights[i].size(), 0.0f);
            }
        }
        weight_criticality_ = std::move(criticality);
    }

    // Calculate baseline loss with a single sample
    float baseline_loss = network.calculateLoss(single_sample, single_label);
    core::Logger::debug("Baseline loss: " + std::to_string(baseline_loss));

    // Determine number of weights to test (sampling to reduce computational load)
    const float sampling_ratio = 0.1f;  // Test 10% of weights

    for (size_t l = 0; l < layers.size(); ++l) {
        // Use const reference for size checks
        const auto& const_layer = layers[l];

        // Test weights in each layer
        size_t total_weights = 0;
        for (const auto& row : const_layer.weights) {
            total_weights += row.size();
        }

        const size_t weights_to_test =
            std::max(size_t{1}, static_cast<size_t>(total_weights * sampling_ratio));

        core::Logger::debug("Layer " + std::to_string(l) + ": Testing " +
                            std::to_string(weights_to_test) + " out of " +
                            std::to_string(total_weights) + " weights");

        // Generate random indices to test
        std::vector<std::pair<size_t, size_t>> indices_to_test;
        indices_to_test.reserve(weights_to_test);

        // Generate random position pairs (input_idx, output_idx)
        std::uniform_int_distribution<size_t> input_dist(0, const_layer.weights.size() - 1);

        for (size_t i = 0; i < weights_to_test; ++i) {
            size_t input_idx = input_dist(generator_);
            std::uniform_int_distribution<size_t> output_dist(
                0, const_layer.weights[input_idx].size() - 1);
            size_t output_idx = output_dist(generator_);
            indices_to_test.emplace_back(input_idx, output_idx);
        }

        // Test selected weights
        for (const auto& [input_idx, output_idx] : indices_to_test) {
            // Get a mutable reference to the layer
            auto& layer = network.getLayerMutable(l);

            // Save original weight
            float original = layer.weights[input_idx][output_idx];

            // Corrupt weight with a bit flip
            int bit = bit_dist_(generator_);
            layer.weights[input_idx][output_idx] = flipBit(original, bit);

            // Measure impact on loss with a single sample
            float corrupted_loss = network.calculateLoss(single_sample, single_label);
            float impact = corrupted_loss - baseline_loss;

            // Restore original weight
            layer.weights[input_idx][output_idx] = original;

            // Update criticality (normalize to range [0,1] with sigmoid function)
            auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            (*weight_criticality_)[l][input_idx][output_idx] = sigmoid(impact);
        }
    }

    // Calculate criticality statistics
    float avg_criticality = 0.0f;
    float max_criticality = 0.0f;
    size_t total_weights = 0;

    for (const auto& layer_criticality : *weight_criticality_) {
        for (const auto& input_criticality : layer_criticality) {
            for (float c : input_criticality) {
                avg_criticality += c;
                max_criticality = std::max(max_criticality, c);
                total_weights++;
            }
        }
    }

    avg_criticality /= std::max(size_t{1}, total_weights);

    core::Logger::info(
        "Criticality map updated. Average criticality: " + std::to_string(avg_criticality) +
        ", Max criticality: " + std::to_string(max_criticality));
}

bool RadiationAwareTraining::saveResults(const std::filesystem::path& path) const
{
    try {
        // Create directories if needed
        std::filesystem::create_directories(path.parent_path());

        std::ofstream file(path);
        if (!file) {
            return false;
        }

        // Write header
        file << "Parameter,Value\n";
        file << "BitFlipProbability," << bit_flip_probability_ << "\n";
        file << "TargetCriticalWeights," << (target_critical_weights_ ? "true" : "false") << "\n";
        file << "Environment," << static_cast<int>(environment_) << "\n";
        file << "TotalBitFlips," << stats_.total_bit_flips << "\n";
        file << "AvgAccuracyDrop," << stats_.avg_accuracy_drop << "\n";
        file << "RecoveryRate," << stats_.recovery_rate << "\n";

        // Write criticality map if available
        if (weight_criticality_.has_value()) {
            file << "\nCriticalityMap\n";
            file << "Layer,WeightIndex,Criticality\n";

            for (size_t l = 0; l < weight_criticality_->size(); ++l) {
                for (size_t i = 0; i < (*weight_criticality_)[l].size(); ++i) {
                    for (size_t j = 0; j < (*weight_criticality_)[l][i].size(); ++j) {
                        file << l << "," << i << "," << (*weight_criticality_)[l][i][j] << "\n";
                    }
                }
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        core::Logger::error("Error saving results: " + std::string(e.what()));
        return false;
    }
}

// Explicit template instantiations for common network types
template RadiationAwareTraining::TrainingResult
RadiationAwareTraining::train<::rad_ml::neural::ProtectedNeuralNetwork<float>>(
    ::rad_ml::neural::ProtectedNeuralNetwork<float>&, const std::vector<float>&,
    const std::vector<float>&, const ::rad_ml::neural::TrainingConfig&);

template RadiationAwareTraining::TrainingResult
RadiationAwareTraining::train<ResidualNeuralNetwork<float>>(
    ResidualNeuralNetwork<float>&, const std::vector<float>&, const std::vector<float>&,
    const ::rad_ml::neural::TrainingConfig&);

template void
RadiationAwareTraining::injectRadiationEffects<::rad_ml::neural::ProtectedNeuralNetwork<float>>(
    ::rad_ml::neural::ProtectedNeuralNetwork<float>&);
template void RadiationAwareTraining::injectRadiationEffects<ResidualNeuralNetwork<float>>(
    ResidualNeuralNetwork<float>&);

template void
RadiationAwareTraining::updateCriticalityMap<::rad_ml::neural::ProtectedNeuralNetwork<float>>(
    ::rad_ml::neural::ProtectedNeuralNetwork<float>&, const std::vector<float>&,
    const std::vector<float>&);
template void RadiationAwareTraining::updateCriticalityMap<ResidualNeuralNetwork<float>>(
    ResidualNeuralNetwork<float>&, const std::vector<float>&, const std::vector<float>&);

}  // namespace research
}  // namespace rad_ml
