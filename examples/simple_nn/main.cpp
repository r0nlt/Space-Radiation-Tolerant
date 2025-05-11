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
#include <rad_ml/core/memory/static_allocator.hpp>
#include <rad_ml/core/redundancy/tmr.hpp>
#include <rad_ml/math/fixed_point.hpp>
#include <iostream>
#include <array>

// Simple neural network example using radiation-tolerant techniques

// Define fixed-point type for neural network calculations
using fixed_t = rad_ml::math::Fixed8_24;

// Use Triple Modular Redundancy for weights
using tmr_fixed_t = rad_ml::core::redundancy::TMR<fixed_t>;

// Simple 2-layer neural network for demonstration
class SimpleNeuralNetwork {
public:
    // Constructor with static initialization of weights
    SimpleNeuralNetwork() {
        // Initialize input to hidden layer weights
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Set initial values (would normally be loaded from a model file)
                input_to_hidden_weights_[i][j] = tmr_fixed_t(fixed_t(i * 0.1 + j * 0.2));
            }
        }
        
        // Initialize hidden to output layer weights
        for (int i = 0; i < 3; ++i) {
            // Set initial values
            hidden_to_output_weights_[i] = tmr_fixed_t(fixed_t(i * 0.3));
        }
    }
    
    // Forward pass through the network
    fixed_t predict(const std::array<fixed_t, 2>& input) {
        // Calculate hidden layer outputs
        std::array<fixed_t, 3> hidden;
        for (int i = 0; i < 3; ++i) {
            hidden[i] = fixed_t(0);
            for (int j = 0; j < 2; ++j) {
                // Use majority voting to get the correct weight value
                hidden[i] += input[j] * input_to_hidden_weights_[j][i].get();
            }
            // Apply ReLU activation (branchless version)
            hidden[i] = max_branchless(hidden[i], fixed_t(0));
        }
        
        // Calculate output layer
        fixed_t output = fixed_t(0);
        for (int i = 0; i < 3; ++i) {
            // Use majority voting to get the correct weight value
            output += hidden[i] * hidden_to_output_weights_[i].get();
        }
        
        return output;
    }
    
    // Periodically repair any bit flips in the weights
    void repair() {
        // Repair input to hidden weights
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                input_to_hidden_weights_[i][j].repair();
            }
        }
        
        // Repair hidden to output weights
        for (int i = 0; i < 3; ++i) {
            hidden_to_output_weights_[i].repair();
        }
    }
    
private:
    // Weights with Triple Modular Redundancy
    tmr_fixed_t input_to_hidden_weights_[2][3];
    tmr_fixed_t hidden_to_output_weights_[3];
    
    // Branchless max operation to avoid branch prediction issues
    static fixed_t max_branchless(fixed_t a, fixed_t b) {
        // If a >= b, mask will be all 1s, otherwise all 0s
        auto mask = -(a >= b ? 1 : 0);
        
        // Use the mask to select either a or b
        return fixed_t((mask & a.raw_value()) | (~mask & b.raw_value()));
    }
};

int main() {
    // Create a neural network
    SimpleNeuralNetwork nn;
    
    // Create input data
    std::array<fixed_t, 2> input = {fixed_t(0.5), fixed_t(0.8)};
    
    // Perform prediction
    auto result = nn.predict(input);
    
    // Display result
    std::cout << "Neural network output: " << result.to_float() << std::endl;
    
    // Periodically repair any bit flips
    nn.repair();
    
    return 0;
} 
