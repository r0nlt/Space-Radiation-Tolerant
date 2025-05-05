#ifndef RAD_ML_NEURAL_ERROR_PREDICTOR_HPP
#define RAD_ML_NEURAL_ERROR_PREDICTOR_HPP

#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <memory>

namespace rad_ml {
namespace neural {

/**
 * @brief Error Predictor class that uses a small neural network to predict
 * radiation-induced error rates for different components based on radiation environment
 * 
 * @tparam T Data type used for predictions (float or double)
 */
template <typename T>
class ErrorPredictor {
public:
    ErrorPredictor()
        : input_size_(3),
          hidden_size_(5),
          output_size_(1),
          learning_rate_(0.01) {
        // Initialize weights for a small neural network
        initializeWeights();
    }

    /**
     * @brief Predict the error rate based on the given radiation level
     * 
     * @param radiation_level Radiation level in particles per bit per day
     * @return Predicted error rate as a scaling factor (0.0-1.0)
     */
    T predictErrorRate(double radiation_level) const {
        // Convert radiation level to input features
        std::vector<T> input = {
            static_cast<T>(std::log10(radiation_level + 1e-10)), // Log scale radiation
            static_cast<T>(radiation_level * 1e7),               // Linear scale
            static_cast<T>(1.0)                                  // Bias term
        };

        // Use the neural network for prediction
        std::vector<T> prediction = forward(input);
        
        // Return predicted error rate, clamped to valid range 0-1
        return std::max(static_cast<T>(0.0), 
               std::min(static_cast<T>(1.0), prediction[0]));
    }

    /**
     * @brief Update the model with actual observed error rates
     * 
     * @param radiation_level Radiation level when the error occurred
     * @param actual_error_rate Actual observed error rate
     */
    void updateModel(double radiation_level, T actual_error_rate) {
        // Convert radiation level to input features
        std::vector<T> input = {
            static_cast<T>(std::log10(radiation_level + 1e-10)),
            static_cast<T>(radiation_level * 1e7),
            static_cast<T>(1.0)
        };

        // Target output
        std::vector<T> target = {actual_error_rate};

        // Forward pass to get prediction
        std::vector<T> hidden = calculateHiddenLayer(input);
        std::vector<T> output = calculateOutputLayer(hidden);

        // Backward pass
        // Output layer error
        std::vector<T> output_error(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            output_error[i] = target[i] - output[i];
        }

        // Hidden layer error
        std::vector<T> hidden_error(hidden_size_, 0.0);
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < output_size_; ++j) {
                hidden_error[i] += output_error[j] * weights2_[i][j];
            }
            // Apply derivative of ReLU
            hidden_error[i] *= (hidden[i] > 0) ? 1.0 : 0.0;
        }

        // Update weights and biases
        // Hidden to output weights
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < output_size_; ++j) {
                weights2_[i][j] += learning_rate_ * output_error[j] * hidden[i];
            }
        }
        // Output biases
        for (int i = 0; i < output_size_; ++i) {
            biases2_[i] += learning_rate_ * output_error[i];
        }

        // Input to hidden weights
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                weights1_[i][j] += learning_rate_ * hidden_error[j] * input[i];
            }
        }
        // Hidden biases
        for (int i = 0; i < hidden_size_; ++i) {
            biases1_[i] += learning_rate_ * hidden_error[i];
        }
    }

private:
    void initializeWeights() {
        // Random number generator for weight initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-0.5, 0.5);

        // Input to hidden weights
        weights1_.resize(input_size_);
        for (int i = 0; i < input_size_; ++i) {
            weights1_[i].resize(hidden_size_);
            for (int j = 0; j < hidden_size_; ++j) {
                weights1_[i][j] = dist(gen);
            }
        }

        // Hidden to output weights
        weights2_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            weights2_[i].resize(output_size_);
            for (int j = 0; j < output_size_; ++j) {
                weights2_[i][j] = dist(gen);
            }
        }

        // Initialize biases
        biases1_.resize(hidden_size_, 0.0);
        biases2_.resize(output_size_, 0.0);
        
        // Pre-trained weights for reasonable predictions without training
        // These values were calibrated based on typical radiation environments
        // Low radiation: ~1e-9 to 1e-8, Medium: 1e-8 to 1e-7, High: >1e-7
        weights1_[0][0] = 0.15; // Log radiation to hidden 1
        weights1_[1][0] = 0.25; // Linear radiation to hidden 1
        weights1_[0][1] = 0.30; // Log radiation to hidden 2
        weights2_[0][0] = 0.20; // Hidden 1 to output
        weights2_[1][0] = 0.35; // Hidden 2 to output
        biases2_[0] = 0.05;     // Output bias
    }

    std::vector<T> calculateHiddenLayer(const std::vector<T>& input) const {
        std::vector<T> hidden(hidden_size_, 0.0);
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                hidden[i] += input[j] * weights1_[j][i];
            }
            hidden[i] += biases1_[i];
            // ReLU activation
            hidden[i] = std::max(static_cast<T>(0.0), hidden[i]);
        }
        return hidden;
    }

    std::vector<T> calculateOutputLayer(const std::vector<T>& hidden) const {
        std::vector<T> output(output_size_, 0.0);
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                output[i] += hidden[j] * weights2_[j][i];
            }
            output[i] += biases2_[i];
            // Sigmoid activation for output (constrain to 0-1)
            output[i] = 1.0 / (1.0 + std::exp(-output[i]));
        }
        return output;
    }

    std::vector<T> forward(const std::vector<T>& input) const {
        std::vector<T> hidden = calculateHiddenLayer(input);
        return calculateOutputLayer(hidden);
    }

    int input_size_;
    int hidden_size_;
    int output_size_;
    T learning_rate_;

    std::vector<std::vector<T>> weights1_; // Input to hidden
    std::vector<std::vector<T>> weights2_; // Hidden to output
    std::vector<T> biases1_;               // Hidden layer biases
    std::vector<T> biases2_;               // Output layer biases
};

} // namespace neural
} // namespace rad_ml

template <typename T>
class ErrorPredictor {
public:
    ErrorPredictor() {}
    
    // Predict error rate based on radiation level
    double predictErrorRate(double seu_rate) const {
        // Simple model:
        // - Low radiation (1e-9 to 1e-8): 0.01-0.05 error rate
        // - Medium radiation (1e-8 to 1e-7): 0.05-0.15 error rate
        // - High radiation (1e-7 to 1e-6): 0.15-0.30 error rate
        // - Extreme radiation (>1e-6): 0.30-0.90 error rate
        
        if (seu_rate < 1e-8) {
            return 0.01 + (seu_rate - 1e-9) * 4.0 / 9e-9;
        } else if (seu_rate < 1e-7) {
            return 0.05 + (seu_rate - 1e-8) * 0.10 / 9e-8;
        } else if (seu_rate < 1e-6) {
            return 0.15 + (seu_rate - 1e-7) * 0.15 / 9e-7;
        } else {
            return 0.30 + std::min(0.60, (seu_rate - 1e-6) * 0.60 / 9e-6);
        }
    }
};

#endif // RAD_ML_NEURAL_ERROR_PREDICTOR_HPP 