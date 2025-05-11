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
#include <iomanip>
#include <string>
#include <functional>
#include "../rad_ml/tmr/physics_driven_protection.hpp"
#include "../rad_ml/sim/mission_environment.hpp"
#include "../rad_ml/core/material_database.hpp"

using namespace rad_ml;

// Convert protection level enum to string
std::string protectionLevelToString(tmr::ProtectionLevel level) {
    switch (level) {
        case tmr::ProtectionLevel::NONE: return "None";
        case tmr::ProtectionLevel::BASIC_TMR: return "Basic TMR";
        case tmr::ProtectionLevel::ENHANCED_TMR: return "Enhanced TMR";
        case tmr::ProtectionLevel::STUCK_BIT_TMR: return "Stuck-Bit TMR";
        case tmr::ProtectionLevel::HEALTH_WEIGHTED_TMR: return "Health-Weighted TMR";
        case tmr::ProtectionLevel::HYBRID_REDUNDANCY: return "Hybrid Redundancy";
        default: return "Unknown";
    }
}

// Simple matrix class for demonstration
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
    
    double& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }
    
    double operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    // Equality operator for TMR
    bool operator==(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        
        for (size_t i = 0; i < data_.size(); i++) {
            // Allow small numerical differences
            if (std::abs(data_[i] - other.data_[i]) > 1e-10) return false;
        }
        return true;
    }
    
    // Not equals operator for TMR
    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }
    
private:
    size_t rows_, cols_;
    std::vector<double> data_;
};

// Simple ML model class
class SimpleNeuralNetwork {
public:
    SimpleNeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size) 
        : weights1_(input_size, hidden_size),
          weights2_(hidden_size, output_size),
          biases1_(1, hidden_size),
          biases2_(1, output_size) {
        
        // Initialize with random weights
        initializeRandom(weights1_);
        initializeRandom(weights2_);
        initializeRandom(biases1_);
        initializeRandom(biases2_);
    }
    
    // Forward pass for inference
    std::vector<double> predict(const std::vector<double>& input) const {
        // Input validation
        if (input.size() != weights1_.cols()) {
            throw std::invalid_argument("Input size doesn't match model input dimension");
        }
        
        // Convert input to matrix
        Matrix input_matrix(1, input.size());
        for (size_t i = 0; i < input.size(); i++) {
            input_matrix(0, i) = input[i];
        }
        
        // First layer
        Matrix hidden = matmul(input_matrix, weights1_);
        addBias(hidden, biases1_);
        relu(hidden);
        
        // Second layer
        Matrix output = matmul(hidden, weights2_);
        addBias(output, biases2_);
        softmax(output);
        
        // Convert output to vector
        std::vector<double> result(output.cols());
        for (size_t i = 0; i < output.cols(); i++) {
            result[i] = output(0, i);
        }
        
        return result;
    }
    
private:
    Matrix weights1_;
    Matrix weights2_;
    Matrix biases1_;
    Matrix biases2_;
    
    // Matrix multiplication
    Matrix matmul(const Matrix& a, const Matrix& b) const {
        if (a.cols() != b.rows()) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(a.rows(), b.cols());
        
        for (size_t i = 0; i < a.rows(); i++) {
            for (size_t j = 0; j < b.cols(); j++) {
                double sum = 0.0;
                for (size_t k = 0; k < a.cols(); k++) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }
    
    // Add bias
    void addBias(Matrix& matrix, const Matrix& bias) const {
        for (size_t i = 0; i < matrix.rows(); i++) {
            for (size_t j = 0; j < matrix.cols(); j++) {
                matrix(i, j) += bias(0, j);
            }
        }
    }
    
    // ReLU activation
    void relu(Matrix& matrix) const {
        for (size_t i = 0; i < matrix.rows(); i++) {
            for (size_t j = 0; j < matrix.cols(); j++) {
                matrix(i, j) = std::max(0.0, matrix(i, j));
            }
        }
    }
    
    // Softmax activation
    void softmax(Matrix& matrix) const {
        for (size_t i = 0; i < matrix.rows(); i++) {
            double max_val = matrix(i, 0);
            for (size_t j = 1; j < matrix.cols(); j++) {
                max_val = std::max(max_val, matrix(i, j));
            }
            
            double sum = 0.0;
            for (size_t j = 0; j < matrix.cols(); j++) {
                matrix(i, j) = std::exp(matrix(i, j) - max_val);
                sum += matrix(i, j);
            }
            
            for (size_t j = 0; j < matrix.cols(); j++) {
                matrix(i, j) /= sum;
            }
        }
    }
    
    // Random initialization
    void initializeRandom(Matrix& matrix) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        
        for (size_t i = 0; i < matrix.rows(); i++) {
            for (size_t j = 0; j < matrix.cols(); j++) {
                matrix(i, j) = dis(gen);
            }
        }
    }
};

// Create a radiation environment
sim::RadiationEnvironment createEnvironment(const std::string& env_name) {
    sim::RadiationEnvironment env;
    
    if (env_name == "LEO") {
        env.trapped_proton_flux = 1.0e7;
        env.trapped_electron_flux = 5.0e6;
        env.temperature.min = 270.0;
        env.temperature.max = 290.0;
        env.solar_activity = 0.2;
    } 
    else if (env_name == "JUPITER") {
        env.trapped_proton_flux = 1.0e12;
        env.trapped_electron_flux = 5.0e11;
        env.temperature.min = 120.0;
        env.temperature.max = 400.0;
        env.solar_activity = 1.0;
    }
    else {
        // Default minimal radiation
        env.trapped_proton_flux = 1.0e5;
        env.trapped_electron_flux = 1.0e4;
        env.temperature.min = 270.0;
        env.temperature.max = 280.0;
        env.solar_activity = 0.1;
    }
    
    return env;
}

// Inject random bit flips to simulate radiation effects
void injectBitFlips(std::vector<double>& values, double error_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> bit_pos(0, 31);  // Assuming 32-bit floats
    
    for (auto& val : values) {
        if (dis(gen) < error_rate) {
            // Flip a random bit in the binary representation
            uint32_t* bits = reinterpret_cast<uint32_t*>(&val);
            uint32_t mask = 1u << bit_pos(gen);
            *bits ^= mask;
        }
    }
}

// Print prediction results
void printPrediction(const std::vector<double>& prediction, const std::string& label = "") {
    std::cout << label;
    std::cout << "[";
    for (size_t i = 0; i < prediction.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << prediction[i];
        if (i < prediction.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << " Radiation-Tolerant ML Inference Example\n";
    std::cout << "====================================================\n\n";
    
    // Initialize a simple neural network
    std::cout << "Initializing neural network model...\n";
    const size_t input_size = 4;
    const size_t hidden_size = 8;
    const size_t output_size = 3;
    SimpleNeuralNetwork model(input_size, hidden_size, output_size);
    
    // Create sample input
    std::vector<double> input = {0.1, 0.2, 0.3, 0.4};
    
    // Normal prediction without radiation effects
    std::cout << "\nPerforming normal inference (no radiation)...\n";
    auto normal_prediction = model.predict(input);
    printPrediction(normal_prediction, "Normal prediction:   ");
    
    // Initialize radiation protection
    std::cout << "\nInitializing radiation protection...\n";
    core::MaterialProperties aluminum;
    aluminum.name = "Aluminum";
    aluminum.density = 2.7;
    aluminum.radiation_tolerance = 50.0;
    tmr::PhysicsDrivenProtection protection(aluminum);
    
    // Test in multiple environments
    std::vector<std::string> environments = {"NONE", "LEO", "JUPITER"};
    
    for (const auto& env_name : environments) {
        // Configure environment
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "Testing in " << env_name << " environment\n";
        std::cout << "----------------------------------------------------\n";
        
        auto env = createEnvironment(env_name);
        protection.updateEnvironment(env);
        
        // Error rate based on environment
        double error_rate = 0.0;
        if (env_name == "LEO") error_rate = 0.1;
        else if (env_name == "JUPITER") error_rate = 0.4;
        else error_rate = 0.01;
        
        std::cout << "Simulated bit flip probability: " << error_rate * 100.0 << "%\n";
        
        // First, run without protection
        std::cout << "\nRunning without protection:\n";
        for (int i = 0; i < 3; i++) {
            auto corrupted_input = input;
            injectBitFlips(corrupted_input, error_rate);
            
            try {
                auto prediction = model.predict(corrupted_input);
                printPrediction(prediction, "Unprotected prediction " + std::to_string(i+1) + ": ");
            } catch (const std::exception& e) {
                std::cout << "Error in unprotected prediction: " << e.what() << std::endl;
            }
        }
        
        // Now run with radiation protection
        std::cout << "\nRunning with " << protectionLevelToString(protection.getCurrentGlobalProtection()) << ":\n";
        
        for (int i = 0; i < 3; i++) {
            // Define the ML operation with error injection as a lambda
            auto predict_with_errors = [&]() -> std::vector<double> {
                auto corrupted_input = input;
                injectBitFlips(corrupted_input, error_rate);
                return model.predict(corrupted_input);
            };
            
            // Execute with protection
            try {
                auto result = protection.executeProtected<std::vector<double>>(predict_with_errors);
                printPrediction(result.value, "Protected prediction " + std::to_string(i+1) + ":   ");
                if (result.error_detected) {
                    std::cout << "  → Error detected" 
                              << (result.error_corrected ? " and corrected!" : " but not corrected.") 
                              << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "Error in protected prediction: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "\n====================================================\n";
    std::cout << "Example complete\n";
    std::cout << "====================================================\n";
    
    return 0;
} 
