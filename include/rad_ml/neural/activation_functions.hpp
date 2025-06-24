/**
 * @file activation_functions.hpp
 * @brief Activation functions and their derivatives for neural networks
 */

#ifndef RAD_ML_NEURAL_ACTIVATION_FUNCTIONS_HPP
#define RAD_ML_NEURAL_ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace rad_ml {
namespace neural {

/**
 * @brief Activation function with its derivative
 */
template <typename T>
struct ActivationFunction {
    std::function<T(T)> function;
    std::function<T(T)> derivative;
    std::string name;

    ActivationFunction(std::function<T(T)> f, std::function<T(T)> df, const std::string& n)
        : function(f), derivative(df), name(n)
    {
    }
};

/**
 * @brief Standard activation functions with analytical derivatives
 */
template <typename T>
class StandardActivations {
   public:
    static ActivationFunction<T> relu()
    {
        return ActivationFunction<T>([](T x) { return std::max(T{0}, x); },
                                     [](T x) { return x > T{0} ? T{1} : T{0}; }, "ReLU");
    }

    static ActivationFunction<T> sigmoid()
    {
        return ActivationFunction<T>([](T x) { return T{1} / (T{1} + std::exp(-x)); },
                                     [](T x) {
                                         T s = T{1} / (T{1} + std::exp(-x));
                                         return s * (T{1} - s);
                                     },
                                     "Sigmoid");
    }

    static ActivationFunction<T> tanh_activation()
    {
        return ActivationFunction<T>([](T x) { return std::tanh(x); },
                                     [](T x) {
                                         T t = std::tanh(x);
                                         return T{1} - t * t;
                                     },
                                     "Tanh");
    }

    static ActivationFunction<T> linear()
    {
        return ActivationFunction<T>([](T x) { return x; }, [](T x) { return T{1}; }, "Linear");
    }

    static ActivationFunction<T> leaky_relu(T alpha = T{0.01})
    {
        return ActivationFunction<T>([alpha](T x) { return x > T{0} ? x : alpha * x; },
                                     [alpha](T x) { return x > T{0} ? T{1} : alpha; }, "LeakyReLU");
    }

    static ActivationFunction<T> elu(T alpha = T{1})
    {
        return ActivationFunction<T>(
            [alpha](T x) { return x > T{0} ? x : alpha * (std::exp(x) - T{1}); },
            [alpha](T x) { return x > T{0} ? T{1} : alpha * std::exp(x); }, "ELU");
    }
};

/**
 * @brief Validates an activation function for neural network use
 */
template <typename T>
class ActivationValidator {
   public:
    struct ValidationResult {
        bool is_valid = false;
        std::string error_message;
        T min_output = T{0};
        T max_output = T{0};
        bool produces_nan = false;
        bool produces_inf = false;
    };

    /**
     * @brief Validate an activation function
     *
     * @param function Function to validate
     * @param min_input Minimum input value to test
     * @param max_input Maximum input value to test
     * @param step Step size for testing
     * @param output_bound_check Whether to check output bounds
     * @param min_bound Minimum allowed output bound
     * @param max_bound Maximum allowed output bound
     * @return ValidationResult with details
     */
    static ValidationResult validate(const std::function<T(T)>& function, T min_input = T{-10},
                                     T max_input = T{10}, T step = T{1},
                                     bool output_bound_check = true, T min_bound = T{-1.5},
                                     T max_bound = T{1.5})
    {
        ValidationResult result;
        result.min_output = function(min_input);
        result.max_output = function(min_input);

        for (T x = min_input; x <= max_input; x += step) {
            T y = function(x);

            if (std::isnan(y)) {
                result.produces_nan = true;
                result.error_message = "Activation function produces NaN output";
                return result;
            }

            if (std::isinf(y)) {
                result.produces_inf = true;
                result.error_message = "Activation function produces Inf output";
                return result;
            }

            if (y < result.min_output) result.min_output = y;
            if (y > result.max_output) result.max_output = y;
        }

        if (output_bound_check) {
            if (result.min_output < min_bound || result.max_output > max_bound) {
                result.error_message = "Activation function output is out of expected bounds [" +
                                       std::to_string(min_bound) + ", " +
                                       std::to_string(max_bound) + "]";
                return result;
            }
        }

        result.is_valid = true;
        return result;
    }
};

/**
 * @brief Smart activation derivative computer that tries analytical detection first
 */
template <typename T>
class ActivationDerivativeComputer {
   private:
    static constexpr T EPSILON = T{1e-6};
    static constexpr T NUMERICAL_EPSILON = T{1e-4};

   public:
    /**
     * @brief Compute activation derivative using analytical detection or numerical fallback
     */
    static T computeDerivative(const std::function<T(T)>& activation_func, T z)
    {
        // Try analytical detection for common functions
        if (auto derivative = tryAnalyticalDetection(activation_func, z); derivative.has_value()) {
            return derivative.value();
        }

        // Fall back to numerical differentiation
        return computeNumericalDerivative(activation_func, z);
    }

   private:
    static std::optional<T> tryAnalyticalDetection(const std::function<T(T)>& activation_func, T z)
    {
        // Test if this is ReLU: f(x) = max(0, x)
        if (std::abs(activation_func(T{1}) - T{1}) < EPSILON &&
            std::abs(activation_func(T{-1}) - T{0}) < EPSILON) {
            return z > T{0} ? T{1} : T{0};
        }

        // Test if this is sigmoid: f(x) = 1/(1+exp(-x))
        T sigmoid_test = activation_func(T{0});
        if (std::abs(sigmoid_test - T{0.5}) < T{1e-5}) {
            T sigmoid_z = activation_func(z);
            return sigmoid_z * (T{1} - sigmoid_z);
        }

        // Test if this is tanh: f(x) = tanh(x)
        if (std::abs(activation_func(T{0}) - T{0}) < EPSILON &&
            std::abs(activation_func(T{1}) - std::tanh(T{1})) < T{1e-5}) {
            T tanh_z = activation_func(z);
            return T{1} - tanh_z * tanh_z;
        }

        // Test if this is linear: f(x) = x
        if (std::abs(activation_func(T{1}) - T{1}) < EPSILON &&
            std::abs(activation_func(T{-1}) - T{-1}) < EPSILON) {
            return T{1};
        }

        // Test if this is Leaky ReLU
        T pos_test = activation_func(T{1});
        T neg_test = activation_func(T{-1});
        T zero_test = activation_func(T{0});
        if (std::abs(pos_test - T{1}) < EPSILON && std::abs(zero_test - T{0}) < EPSILON &&
            neg_test < T{0} && neg_test > T{-0.5}) {
            T alpha = -neg_test;
            return z > T{0} ? T{1} : alpha;
        }

        // Test if this is ELU
        T pos_test_elu = activation_func(T{1});
        T zero_test_elu = activation_func(T{0});
        T neg_test_elu = activation_func(T{-1});
        T expected_neg_elu = std::exp(T{-1}) - T{1};
        if (std::abs(pos_test_elu - T{1}) < EPSILON && std::abs(zero_test_elu - T{0}) < EPSILON &&
            std::abs(neg_test_elu - expected_neg_elu) < EPSILON) {
            return z > T{0} ? T{1} : std::exp(z);
        }

        return std::nullopt;
    }

    static T computeNumericalDerivative(const std::function<T(T)>& function, T z)
    {
        const T base_epsilon = NUMERICAL_EPSILON;
        const T adaptive_epsilon = std::max(base_epsilon, std::abs(z) * T{1e-5});
        const T epsilon = std::min(adaptive_epsilon, T{1e-3});

        const T f_plus = function(z + epsilon);
        const T f_minus = function(z - epsilon);

        T derivative = (f_plus - f_minus) / (2 * epsilon);

        // Clamp extreme values
        derivative = std::max(T{-10}, std::min(T{10}, derivative));

        return derivative;
    }
};

}  // namespace neural
}  // namespace rad_ml

#endif  // RAD_ML_NEURAL_ACTIVATION_FUNCTIONS_HPP
