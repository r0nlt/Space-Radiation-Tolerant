/**
 * @file vae_example.cpp
 * @brief Example demonstrating the Variational Autoencoder in radiation environments
 *
 * This example shows how to:
 * 1. Create and configure a radiation-tolerant VAE
 * 2. Train the VAE on synthetic data
 * 3. Generate new samples
 * 4. Perform interpolation between latent points
 * 5. Test radiation tolerance capabilities
 */

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/rad_ml/core/logger.hpp"
#include "../include/rad_ml/research/variational_autoencoder.hpp"

using namespace rad_ml;
using namespace rad_ml::research;

/**
 * @brief Generate synthetic training data for demonstration
 *
 * Creates simple 2D Gaussian mixture data for the VAE to learn
 */
std::vector<std::vector<float>> generateSyntheticData(size_t num_samples, size_t input_dim)
{
    std::vector<std::vector<float>> data;
    data.reserve(num_samples);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Create mixture of Gaussians
    std::normal_distribution<float> dist1(2.0f, 0.5f);   // First mode
    std::normal_distribution<float> dist2(-2.0f, 0.5f);  // Second mode
    std::uniform_int_distribution<int> mode_selector(0, 1);

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> sample(input_dim);

        // Choose which mode to sample from
        bool use_first_mode = mode_selector(gen) == 0;

        for (size_t j = 0; j < input_dim; ++j) {
            if (use_first_mode) {
                sample[j] = dist1(gen);
            }
            else {
                sample[j] = dist2(gen);
            }

            // Add some correlation between dimensions
            if (j > 0) {
                sample[j] += 0.3f * sample[j - 1];
            }
        }

        data.push_back(sample);
    }

    return data;
}

/**
 * @brief Print vector contents for debugging
 */
void printVector(const std::vector<float>& vec, const std::string& name)
{
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Test basic VAE functionality
 */
void testBasicVAE()
{
    std::cout << "\n=== Testing Basic VAE Functionality ===" << std::endl;

    // Configuration
    size_t input_dim = 8;
    size_t latent_dim = 3;
    std::vector<size_t> hidden_dims = {16, 8};  // Two hidden layers

    VAEConfig config;
    config.latent_dim = latent_dim;
    config.learning_rate = 0.01f;
    config.epochs = 50;
    config.batch_size = 16;
    config.beta = 1.0f;  // Standard β-VAE
    config.use_interpolation = true;

    // Create VAE
    VariationalAutoencoder<float> vae(input_dim, latent_dim, hidden_dims,
                                      neural::ProtectionLevel::ADAPTIVE_TMR, config);

    std::cout << "VAE created with:" << std::endl;
    std::cout << "- Input dim: " << vae.getInputDim() << std::endl;
    std::cout << "- Latent dim: " << vae.getLatentDim() << std::endl;
    std::cout << "- Hidden dims: [";
    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        std::cout << hidden_dims[i];
        if (i < hidden_dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Test encoding/decoding
    std::vector<float> test_input(input_dim, 1.0f);
    for (size_t i = 0; i < input_dim; ++i) {
        test_input[i] = static_cast<float>(i) * 0.1f;
    }

    printVector(test_input, "Test input");

    // Encode
    auto [mean, log_var] = vae.encode(test_input);
    printVector(mean, "Encoded mean");
    printVector(log_var, "Encoded log_var");

    // Sample
    std::vector<float> latent = vae.sample(mean, log_var, 42);
    printVector(latent, "Sampled latent");

    // Decode
    std::vector<float> reconstruction = vae.decode(latent);
    printVector(reconstruction, "Decoded output");

    // Full forward pass
    std::vector<float> full_reconstruction = vae.forward(test_input);
    printVector(full_reconstruction, "Full forward reconstruction");
}

/**
 * @brief Test VAE training
 */
void testVAETraining()
{
    std::cout << "\n=== Testing VAE Training ===" << std::endl;

    // Generate training data
    size_t num_samples = 200;
    size_t input_dim = 6;
    size_t latent_dim = 2;

    std::vector<std::vector<float>> training_data = generateSyntheticData(num_samples, input_dim);

    std::cout << "Generated " << training_data.size() << " training samples" << std::endl;

    // Create VAE
    VAEConfig config;
    config.latent_dim = latent_dim;
    config.learning_rate = 0.005f;
    config.epochs = 20;
    config.batch_size = 32;
    config.beta = 0.5f;  // Lower β for better reconstruction

    VariationalAutoencoder<float> vae(input_dim, latent_dim, {12, 6},  // Hidden dimensions
                                      neural::ProtectionLevel::SELECTIVE_TMR, config);

    // Evaluate before training
    float initial_loss = vae.evaluate(training_data);
    std::cout << "Initial reconstruction error: " << initial_loss << std::endl;

    // Train the VAE
    float final_loss = vae.train(training_data);
    std::cout << "Final training loss: " << final_loss << std::endl;

    // Evaluate after training
    float final_reconstruction_error = vae.evaluate(training_data);
    std::cout << "Final reconstruction error: " << final_reconstruction_error << std::endl;

    // Show improvement
    float improvement = (initial_loss - final_reconstruction_error) / initial_loss * 100.0f;
    std::cout << "Improvement: " << improvement << "%" << std::endl;
}

/**
 * @brief Test VAE generation capabilities
 */
void testVAEGeneration()
{
    std::cout << "\n=== Testing VAE Generation ===" << std::endl;

    size_t input_dim = 4;
    size_t latent_dim = 2;

    // Create and train a simple VAE
    VAEConfig config;
    config.epochs = 10;
    config.batch_size = 16;

    VariationalAutoencoder<float> vae(input_dim, latent_dim, {8, 4},
                                      neural::ProtectionLevel::CHECKSUM_ONLY, config);

    // Generate some training data
    auto training_data = generateSyntheticData(100, input_dim);
    vae.train(training_data);

    // Generate new samples
    std::cout << "Generating new samples from prior distribution:" << std::endl;
    auto generated_samples = vae.generate(5, 0.0, 42);

    for (size_t i = 0; i < generated_samples.size(); ++i) {
        printVector(generated_samples[i], "Generated sample " + std::to_string(i + 1));
    }
}

/**
 * @brief Test VAE interpolation capabilities
 */
void testVAEInterpolation()
{
    std::cout << "\n=== Testing VAE Interpolation ===" << std::endl;

    size_t input_dim = 4;
    size_t latent_dim = 2;

    VAEConfig config;
    config.use_interpolation = true;
    config.interpolation_weight = 0.2f;

    VariationalAutoencoder<float> vae(input_dim, latent_dim, {6}, neural::ProtectionLevel::NONE,
                                      config);

    // Create two sample points
    std::vector<float> sample1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> sample2 = {-1.0f, -2.0f, -3.0f, -4.0f};

    printVector(sample1, "Sample 1");
    printVector(sample2, "Sample 2");

    // Encode both samples
    auto [mean1, log_var1] = vae.encode(sample1);
    auto [mean2, log_var2] = vae.encode(sample2);

    // Sample latent representations
    std::vector<float> latent1 = vae.sample(mean1, log_var1, 42);
    std::vector<float> latent2 = vae.sample(mean2, log_var2, 43);

    printVector(latent1, "Latent 1");
    printVector(latent2, "Latent 2");

    // Interpolate at different points
    std::cout << "Interpolation results:" << std::endl;
    for (float alpha = 0.0f; alpha <= 1.0f; alpha += 0.25f) {
        std::vector<float> interpolated = vae.interpolate(latent1, latent2, alpha);
        printVector(interpolated, "α=" + std::to_string(alpha));
    }
}

/**
 * @brief Test radiation tolerance capabilities
 */
void testRadiationTolerance()
{
    std::cout << "\n=== Testing Radiation Tolerance ===" << std::endl;

    size_t input_dim = 6;
    size_t latent_dim = 3;

    // Create VAE with strong protection
    VariationalAutoencoder<float> vae(input_dim, latent_dim, {12, 8},
                                      neural::ProtectionLevel::FULL_TMR);

    // Generate test data
    auto test_data = generateSyntheticData(50, input_dim);

    std::cout << "Training VAE with radiation protection..." << std::endl;
    vae.train(test_data, 15, 16, 0.01f);

    // Test sample
    std::vector<float> test_sample = test_data[0];
    printVector(test_sample, "Original test sample");

    // Test at different radiation levels
    std::vector<double> radiation_levels = {0.0, 0.1, 0.3, 0.5, 0.8};

    for (double radiation_level : radiation_levels) {
        std::cout << "\n--- Radiation Level: " << radiation_level << " ---" << std::endl;

        // Apply radiation effects
        vae.applyRadiationEffects(radiation_level, 12345);

        // Forward pass under radiation
        std::vector<float> result = vae.forward(test_sample, radiation_level);
        printVector(result, "Result under radiation");

        // Get error statistics
        auto [detected, corrected] = vae.getErrorStats();
        std::cout << "Errors detected: " << detected << ", corrected: " << corrected << std::endl;

        // Calculate reconstruction error
        float error = 0.0f;
        for (size_t i = 0; i < test_sample.size() && i < result.size(); ++i) {
            float diff = test_sample[i] - result[i];
            error += diff * diff;
        }
        error = std::sqrt(error / test_sample.size());
        std::cout << "Reconstruction RMSE: " << error << std::endl;

        // Reset stats for next test
        vae.resetErrorStats();
    }
}

/**
 * @brief Test different VAE configurations
 */
void testVAEConfigurations()
{
    std::cout << "\n=== Testing Different VAE Configurations ===" << std::endl;

    size_t input_dim = 4;
    size_t latent_dim = 2;
    auto test_data = generateSyntheticData(50, input_dim);

    // Test different β values (β-VAE)
    std::vector<float> beta_values = {0.1f, 1.0f, 4.0f};

    for (float beta : beta_values) {
        std::cout << "\n--- Testing β=" << beta << " ---" << std::endl;

        VAEConfig config;
        config.beta = beta;
        config.epochs = 10;
        config.loss_type = VAELossType::BETA_VAE;

        VariationalAutoencoder<float> vae(input_dim, latent_dim, {6},
                                          neural::ProtectionLevel::ADAPTIVE_TMR, config);

        float loss = vae.train(test_data);
        std::cout << "Final training loss: " << loss << std::endl;

        // Generate a sample to see the effect
        auto generated = vae.generate(1, 0.0, 42);
        printVector(generated[0], "Generated sample");
    }

    // Test different sampling techniques
    std::cout << "\n--- Testing Sampling Techniques ---" << std::endl;

    std::vector<SamplingTechnique> sampling_methods = {SamplingTechnique::STANDARD,
                                                       SamplingTechnique::REPARAMETERIZED};

    for (auto method : sampling_methods) {
        VAEConfig config;
        config.sampling = method;
        config.epochs = 5;

        VariationalAutoencoder<float> vae(input_dim, latent_dim, {6}, neural::ProtectionLevel::NONE,
                                          config);

        std::string method_name =
            (method == SamplingTechnique::STANDARD) ? "Standard" : "Reparameterized";
        std::cout << "\nSampling method: " << method_name << std::endl;

        // Test encoding and sampling
        auto [mean, log_var] = vae.encode(test_data[0]);
        std::vector<float> sampled = vae.sample(mean, log_var, 42);
        printVector(sampled, "Sampled latent");
    }
}

int main()
{
    std::cout << "=== Radiation-Tolerant Variational Autoencoder Example ===" << std::endl;

    try {
        // Run all tests
        testBasicVAE();
        testVAETraining();
        testVAEGeneration();
        testVAEInterpolation();
        testRadiationTolerance();
        testVAEConfigurations();

        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
