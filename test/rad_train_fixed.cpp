/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Simple neural network implementation for testing
class SimpleNetwork {
public:
    SimpleNetwork(const std::vector<int>& layers) : layer_sizes(layers) {
        // Initialize with random weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.5, 0.5);
        
        weights.resize(layer_sizes.size() - 1);
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            weights[i].resize(layer_sizes[i] * layer_sizes[i+1], 0.0f);
            for (auto& w : weights[i]) {
                w = dist(gen);
            }
        }
    }
    
    // Forward pass
    std::vector<float> forward(const std::vector<float>& input) const {
        std::vector<float> current = input;
        
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            std::vector<float> next(layer_sizes[layer+1], 0.0f);
            
            for (int i = 0; i < layer_sizes[layer+1]; ++i) {
                for (int j = 0; j < layer_sizes[layer]; ++j) {
                    next[i] += current[j] * weights[layer][j * layer_sizes[layer+1] + i];
                }
                
                // Apply ReLU activation for hidden layers, sigmoid for output
                if (layer < weights.size() - 1) {
                    next[i] = std::max(0.0f, next[i]); // ReLU
                } else {
                    next[i] = 1.0f / (1.0f + std::exp(-next[i])); // Sigmoid
                }
            }
            
            current = next;
        }
        
        return current;
    }
    
    // Compute gradients (simplified, not a real backprop implementation)
    std::vector<float> computeGradients(const std::vector<float>& input, const std::vector<float>& target) {
        // Forward pass
        std::vector<std::vector<float>> activations;
        activations.push_back(input);
        
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            std::vector<float> next(layer_sizes[layer+1], 0.0f);
            
            for (int i = 0; i < layer_sizes[layer+1]; ++i) {
                for (int j = 0; j < layer_sizes[layer]; ++j) {
                    next[i] += activations.back()[j] * weights[layer][j * layer_sizes[layer+1] + i];
                }
                
                // Apply activation
                if (layer < weights.size() - 1) {
                    next[i] = std::max(0.0f, next[i]); // ReLU
                } else {
                    next[i] = 1.0f / (1.0f + std::exp(-next[i])); // Sigmoid
                }
            }
            
            activations.push_back(next);
        }
        
        // Output error
        std::vector<float> errors(layer_sizes.back(), 0.0f);
        for (size_t i = 0; i < errors.size(); ++i) {
            errors[i] = activations.back()[i] - target[i];
        }
        
        // Simulate random size errors (for testing)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 10);
        if (dist(gen) == 0) {
            // Intentionally return wrong gradient size to test error handling
            std::vector<float> wrong_size(totalWeights() + 5, 0.0f);
            return wrong_size;
        }
        
        // Simplified gradient computation (not real backprop)
        std::vector<float> gradients(totalWeights(), 0.0f);
        size_t offset = 0;
        
        for (size_t layer = weights.size(); layer-- > 0; ) {
            for (int i = 0; i < layer_sizes[layer]; ++i) {
                for (int j = 0; j < layer_sizes[layer+1]; ++j) {
                    size_t idx = offset + i * layer_sizes[layer+1] + j;
                    if (idx < gradients.size()) {
                        gradients[idx] = errors[j] * activations[layer][i] * 0.01f;
                    }
                }
            }
            
            offset += layer_sizes[layer] * layer_sizes[layer+1];
            
            // Calculate errors for next layer back (simplified)
            if (layer > 0) {
                std::vector<float> new_errors(layer_sizes[layer], 0.0f);
                for (int i = 0; i < layer_sizes[layer]; ++i) {
                    for (int j = 0; j < layer_sizes[layer+1]; ++j) {
                        new_errors[i] += errors[j] * weights[layer][i * layer_sizes[layer+1] + j];
                    }
                    // ReLU derivative
                    new_errors[i] *= (activations[layer][i] > 0) ? 1.0f : 0.0f;
                }
                errors = new_errors;
            }
        }
        
        return gradients;
    }
    
    // Update weights
    void updateWeights(const std::vector<float>& gradients, float learning_rate) {
        if (gradients.size() != totalWeights()) {
            std::cerr << "Error: gradient size mismatch. Expected " << totalWeights() 
                      << " but got " << gradients.size() << std::endl;
            return;
        }
        
        size_t offset = 0;
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            for (size_t i = 0; i < weights[layer].size(); ++i) {
                weights[layer][i] -= learning_rate * gradients[offset + i];
            }
            offset += weights[layer].size();
        }
    }
    
    // Get total number of weights
    size_t totalWeights() const {
        size_t total = 0;
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            total += weights[layer].size();
        }
        return total;
    }
    
    // Introduce bit errors (to simulate radiation effects)
    void introduceErrors(float error_rate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution error_dist(error_rate);
        std::uniform_int_distribution<int> bit_dist(0, 31); // 32-bit float
        
        for (auto& layer_weights : weights) {
            for (auto& w : layer_weights) {
                if (error_dist(gen)) {
                    // Flip a random bit
                    int bit = bit_dist(gen);
                    uint32_t* bits = reinterpret_cast<uint32_t*>(&w);
                    *bits ^= (1u << bit);
                }
            }
        }
    }
    
private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<float>> weights;
};

// Simple dataset structure
struct Dataset {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
    
    void add(const std::vector<float>& input, const std::vector<float>& output) {
        inputs.push_back(input);
        outputs.push_back(output);
    }
    
    size_t size() const {
        return inputs.size();
    }
};

// Generate simple XOR dataset
Dataset generateXORDataset(int samples) {
    Dataset dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.1, 0.1);
    
    for (int i = 0; i < samples; ++i) {
        float x = (i % 2);
        float y = ((i / 2) % 2);
        float xor_result = static_cast<float>((static_cast<int>(x) ^ static_cast<int>(y)));
        
        // Add some noise for more realistic training
        std::vector<float> input = {x + noise(gen), y + noise(gen)};
        std::vector<float> output = {xor_result};
        
        dataset.add(input, output);
    }
    
    return dataset;
}

// More complex 3-output dataset (predicting X, Y and XOR)
Dataset generateComplexDataset(int samples) {
    Dataset dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.1, 0.1);
    
    for (int i = 0; i < samples; ++i) {
        float x = (i % 2);
        float y = ((i / 2) % 2);
        float xor_result = static_cast<float>((static_cast<int>(x) ^ static_cast<int>(y)));
        
        // Add some noise for more realistic training
        std::vector<float> input = {x + noise(gen), y + noise(gen)};
        std::vector<float> output = {x, y, xor_result}; // 3 outputs
        
        dataset.add(input, output);
    }
    
    return dataset;
}

// Calculate accuracy
float calculateAccuracy(const SimpleNetwork& network, const Dataset& dataset, float threshold = 0.5f) {
    if (dataset.size() == 0) return 0.0f;
    
    int correct = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto prediction = network.forward(dataset.inputs[i]);
        
        // Check if all outputs are correct
        bool all_correct = true;
        for (size_t j = 0; j < prediction.size(); ++j) {
            if ((prediction[j] > threshold && dataset.outputs[i][j] <= 0.5f) ||
                (prediction[j] <= threshold && dataset.outputs[i][j] > 0.5f)) {
                all_correct = false;
                break;
            }
        }
        
        if (all_correct) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / dataset.size();
}

int main() {
    // Create more complex dataset with 2 inputs and 3 outputs
    std::cout << "Generating dataset..." << std::endl;
    Dataset training_data = generateComplexDataset(200);
    Dataset validation_data = generateComplexDataset(50);
    
    std::cout << "Training data size: " << training_data.size() << std::endl;
    std::cout << "Validation data size: " << validation_data.size() << std::endl;
    
    // Create network with 2 inputs, hidden layer of 16 neurons, 3 outputs
    std::vector<int> layers = {2, 16, 3};
    SimpleNetwork network(layers);
    
    std::cout << "Network architecture: 2-16-3" << std::endl;
    std::cout << "Total weights: " << network.totalWeights() << std::endl;
    
    // Training parameters
    float learning_rate = 0.1f;
    int epochs = 1000;
    float dropout_rate = 0.5f;
    bool enable_radiation = true;
    float radiation_factor = 0.0f;
    
    std::cout << "Training with learning rate: " << learning_rate << std::endl;
    if (enable_radiation) {
        std::cout << "Radiation simulation enabled" << std::endl;
    }
    
    // Training loop with safe gradient handling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dropout(dropout_rate);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Increase radiation effect over time
        if (enable_radiation) {
            radiation_factor = std::min(0.01f, static_cast<float>(epoch) / (epochs * 0.25f));
        }
        
        float epoch_loss = 0.0f;
        int samples_processed = 0;
        
        // Process each training sample
        for (size_t i = 0; i < training_data.size(); ++i) {
            try {
                // Compute gradients
                std::vector<float> gradients = network.computeGradients(
                    training_data.inputs[i], 
                    training_data.outputs[i]
                );
                
                // Safety check - skip samples with wrong gradient size
                if (gradients.size() != network.totalWeights()) {
                    std::cerr << "WARNING: Gradient size mismatch: expected " 
                              << network.totalWeights() << " but got " 
                              << gradients.size() << ". Skipping sample." << std::endl;
                    continue;
                }
                
                // Apply dropout
                if (dropout_rate > 0.0f) {
                    for (auto& g : gradients) {
                        if (dropout(gen)) {
                            g = 0.0f; // Drop this gradient
                        }
                    }
                }
                
                // Update weights
                network.updateWeights(gradients, learning_rate);
                
                // Apply radiation effects if enabled
                if (enable_radiation && radiation_factor > 0.0f) {
                    network.introduceErrors(radiation_factor);
                }
                
                samples_processed++;
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing sample " << i << ": " << e.what() << std::endl;
                continue; // Skip this sample and continue with the next
            }
        }
        
        // Evaluate every 100 epochs
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            float accuracy = calculateAccuracy(network, validation_data);
            
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << ", Processed samples: " << samples_processed
                      << ", Accuracy: " << accuracy;
            
            if (enable_radiation && radiation_factor > 0.0f) {
                std::cout << ", Radiation factor: " << radiation_factor;
            }
            
            std::cout << std::endl;
        }
        
        // Decay learning rate
        learning_rate *= 0.999f;
    }
    
    std::cout << "Training completed." << std::endl;
    
    // Final evaluation
    float final_accuracy = calculateAccuracy(network, validation_data);
    std::cout << "Final accuracy: " << final_accuracy << std::endl;
    
    // Test with radiation
    if (enable_radiation) {
        // Apply strong radiation effect
        network.introduceErrors(0.05f);
        
        float radiation_accuracy = calculateAccuracy(network, validation_data);
        std::cout << "Accuracy after radiation: " << radiation_accuracy << std::endl;
        std::cout << "Preservation ratio: " << radiation_accuracy / final_accuracy << std::endl;
    }
    
    return 0;
} 
