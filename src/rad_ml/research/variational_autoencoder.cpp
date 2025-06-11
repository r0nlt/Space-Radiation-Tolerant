/**
 * @file variational_autoencoder.cpp
 * @brief Implementation of radiation-tolerant Variational Autoencoder
 */

#include "../../../include/rad_ml/research/variational_autoencoder.hpp"

#include <algorithm>
#include <numeric>

namespace rad_ml {
namespace research {

template <typename T>
VariationalAutoencoder<T>::VariationalAutoencoder(size_t input_dim, size_t latent_dim,
                                                  const std::vector<size_t>& hidden_dims,
                                                  neural::ProtectionLevel protection_level,
                                                  const VAEConfig& config)
    : input_dim_(input_dim),
      latent_dim_(latent_dim),
      hidden_dims_(hidden_dims),
      protection_level_(protection_level),
      config_(config),
      rng_(std::random_device{}())
{
    // Initialize the three main networks
    initializeEncoder();
    initializeDecoder();
    initializeInterpolator();

    core::Logger::info("VariationalAutoencoder initialized with input_dim=" +
                       std::to_string(input_dim) + ", latent_dim=" + std::to_string(latent_dim));
}

template <typename T>
VariationalAutoencoder<T>::VariationalAutoencoder(const VariationalAutoencoder& other)
    : input_dim_(other.input_dim_),
      latent_dim_(other.latent_dim_),
      hidden_dims_(other.hidden_dims_),
      protection_level_(other.protection_level_),
      config_(other.config_),
      error_stats_(other.error_stats_),
      rng_(other.rng_)
{
    // Deep copy the networks
    if (other.encoder_) {
        encoder_ = std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.encoder_);
    }
    if (other.decoder_) {
        decoder_ = std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.decoder_);
    }
    if (other.interpolator_) {
        interpolator_ = std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.interpolator_);
    }
}

template <typename T>
VariationalAutoencoder<T>& VariationalAutoencoder<T>::operator=(const VariationalAutoencoder& other)
{
    if (this != &other) {
        input_dim_ = other.input_dim_;
        latent_dim_ = other.latent_dim_;
        hidden_dims_ = other.hidden_dims_;
        protection_level_ = other.protection_level_;
        config_ = other.config_;
        error_stats_ = other.error_stats_;
        rng_ = other.rng_;

        // Deep copy the networks
        if (other.encoder_) {
            encoder_ = std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.encoder_);
        }
        if (other.decoder_) {
            decoder_ = std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.decoder_);
        }
        if (other.interpolator_) {
            interpolator_ =
                std::make_unique<neural::ProtectedNeuralNetwork<T>>(*other.interpolator_);
        }
    }
    return *this;
}

template <typename T>
void VariationalAutoencoder<T>::initializeEncoder()
{
    // Build encoder architecture: input -> hidden_layers -> [mean, log_var]
    std::vector<size_t> encoder_layers;
    encoder_layers.push_back(input_dim_);

    // Add hidden layers
    for (size_t hidden_dim : hidden_dims_) {
        encoder_layers.push_back(hidden_dim);
    }

    // Output layer produces both mean and log_variance (2 * latent_dim)
    encoder_layers.push_back(2 * latent_dim_);

    encoder_ =
        std::make_unique<neural::ProtectedNeuralNetwork<T>>(encoder_layers, protection_level_);

    // Set activation functions
    for (size_t i = 0; i < encoder_layers.size() - 2; ++i) {
        // ReLU for hidden layers
        encoder_->setActivationFunction(i,
                                        [](T x) { return x > 0 ? x : T(0.01) * x; });  // LeakyReLU
    }
    // Linear activation for output layer (mean and log_var)
    encoder_->setActivationFunction(encoder_layers.size() - 2, [](T x) { return x; });
}

template <typename T>
void VariationalAutoencoder<T>::initializeDecoder()
{
    // Build decoder architecture: latent -> hidden_layers -> output
    std::vector<size_t> decoder_layers;
    decoder_layers.push_back(latent_dim_);

    // Add hidden layers (reverse order)
    for (auto it = hidden_dims_.rbegin(); it != hidden_dims_.rend(); ++it) {
        decoder_layers.push_back(*it);
    }

    // Output layer
    decoder_layers.push_back(input_dim_);

    decoder_ =
        std::make_unique<neural::ProtectedNeuralNetwork<T>>(decoder_layers, protection_level_);

    // Set activation functions
    for (size_t i = 0; i < decoder_layers.size() - 2; ++i) {
        // ReLU for hidden layers
        decoder_->setActivationFunction(i,
                                        [](T x) { return x > 0 ? x : T(0.01) * x; });  // LeakyReLU
    }
    // Sigmoid activation for output layer (reconstruction)
    decoder_->setActivationFunction(decoder_layers.size() - 2,
                                    [](T x) { return T(1) / (T(1) + std::exp(-x)); });
}

template <typename T>
void VariationalAutoencoder<T>::initializeInterpolator()
{
    // Build interpolator architecture: 2*latent -> hidden -> latent
    std::vector<size_t> interpolator_layers;
    interpolator_layers.push_back(2 * latent_dim_);  // Two latent vectors concatenated

    // Add a hidden layer for interpolation processing
    if (!hidden_dims_.empty()) {
        interpolator_layers.push_back(
            hidden_dims_[hidden_dims_.size() / 2]);  // Use middle hidden dim
    }
    else {
        interpolator_layers.push_back(latent_dim_);  // Default to latent_dim
    }

    interpolator_layers.push_back(latent_dim_);  // Output interpolated latent vector

    interpolator_ =
        std::make_unique<neural::ProtectedNeuralNetwork<T>>(interpolator_layers, protection_level_);

    // Set activation functions
    for (size_t i = 0; i < interpolator_layers.size() - 1; ++i) {
        if (i == interpolator_layers.size() - 2) {
            // Linear activation for output
            interpolator_->setActivationFunction(i, [](T x) { return x; });
        }
        else {
            // Tanh for hidden layers (good for interpolation)
            interpolator_->setActivationFunction(i, [](T x) { return std::tanh(x); });
        }
    }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> VariationalAutoencoder<T>::encode(
    const std::vector<T>& input, double radiation_level)
{
    // Apply preprocessing: logarithmics -> standardization
    std::vector<T> processed_input = applyLogarithmics(input);
    processed_input = applyStandardization(processed_input);

    // Forward pass through encoder
    std::vector<T> encoder_output = encoder_->forward(processed_input, radiation_level);

    // Split output into mean and log_variance
    std::vector<T> mean(latent_dim_);
    std::vector<T> log_var(latent_dim_);

    for (size_t i = 0; i < latent_dim_; ++i) {
        mean[i] = encoder_output[i];
        log_var[i] = encoder_output[i + latent_dim_];
    }

    // Clamp log_var to prevent numerical instability
    for (auto& lv : log_var) {
        lv = std::max(T(-10), std::min(T(10), lv));
    }

    return std::make_pair(mean, log_var);
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::sample(const std::vector<T>& mean,
                                                 const std::vector<T>& log_var, uint64_t seed)
{
    std::mt19937 local_rng(seed == 0 ? rng_() : seed);
    std::normal_distribution<T> dist(0.0, 1.0);

    std::vector<T> sampled(latent_dim_);

    switch (config_.sampling) {
        case SamplingTechnique::REPARAMETERIZED:
        default:
            // Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
            for (size_t i = 0; i < latent_dim_; ++i) {
                T epsilon = dist(local_rng);
                T sigma = std::exp(T(0.5) * log_var[i]);
                sampled[i] = mean[i] + sigma * epsilon;
            }
            break;

        case SamplingTechnique::STANDARD:
            // Direct sampling from N(μ, σ²)
            for (size_t i = 0; i < latent_dim_; ++i) {
                T sigma = std::exp(T(0.5) * log_var[i]);
                std::normal_distribution<T> param_dist(mean[i], sigma);
                sampled[i] = param_dist(local_rng);
            }
            break;
    }

    return sampled;
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::decode(const std::vector<T>& latent,
                                                 double radiation_level)
{
    // Forward pass through decoder
    std::vector<T> decoder_output = decoder_->forward(latent, radiation_level);

    // Apply postprocessing: de-standardization -> de-logarithmics
    std::vector<T> processed_output = applyInverseStandardization(decoder_output);
    processed_output = applyInverseLogarithmics(processed_output);

    return processed_output;
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::forward(const std::vector<T>& input,
                                                  double radiation_level)
{
    // Encode
    auto [mean, log_var] = encode(input, radiation_level);

    // Sample
    std::vector<T> latent = sample(mean, log_var);

    // Protect latent variables against radiation
    protectLatentVariables(latent, radiation_level);

    // Decode
    return decode(latent, radiation_level);
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::interpolate(const std::vector<T>& latent1,
                                                      const std::vector<T>& latent2, T alpha,
                                                      double radiation_level)
{
    if (config_.use_interpolation && interpolator_) {
        // Use learned interpolation via interpolator network
        std::vector<T> concatenated;
        concatenated.reserve(2 * latent_dim_);
        concatenated.insert(concatenated.end(), latent1.begin(), latent1.end());
        concatenated.insert(concatenated.end(), latent2.begin(), latent2.end());

        std::vector<T> interpolated_latent = interpolator_->forward(concatenated, radiation_level);

        // Scale by alpha for blending
        for (size_t i = 0; i < interpolated_latent.size(); ++i) {
            interpolated_latent[i] = (T(1) - alpha) * latent1[i] + alpha * interpolated_latent[i];
        }

        return decode(interpolated_latent, radiation_level);
    }
    else {
        // Simple linear interpolation
        std::vector<T> interpolated_latent(latent_dim_);
        for (size_t i = 0; i < latent_dim_; ++i) {
            interpolated_latent[i] = (T(1) - alpha) * latent1[i] + alpha * latent2[i];
        }

        return decode(interpolated_latent, radiation_level);
    }
}

template <typename T>
std::vector<std::vector<T>> VariationalAutoencoder<T>::generate(size_t num_samples,
                                                                double radiation_level,
                                                                uint64_t seed)
{
    std::mt19937 local_rng(seed == 0 ? rng_() : seed);
    std::normal_distribution<T> dist(0.0, 1.0);

    std::vector<std::vector<T>> generated_samples;
    generated_samples.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        // Sample from prior N(0, I)
        std::vector<T> prior_sample(latent_dim_);
        for (size_t j = 0; j < latent_dim_; ++j) {
            prior_sample[j] = dist(local_rng);
        }

        // Decode to generate sample
        std::vector<T> generated = decode(prior_sample, radiation_level);
        generated_samples.push_back(generated);
    }

    return generated_samples;
}

template <typename T>
float VariationalAutoencoder<T>::train(const std::vector<std::vector<T>>& data, int epochs,
                                       int batch_size, float learning_rate)
{
    // Use config defaults if not specified
    int actual_epochs = (epochs == -1) ? config_.epochs : epochs;
    int actual_batch_size = (batch_size == -1) ? config_.batch_size : batch_size;
    float actual_lr = (learning_rate == -1.0f) ? config_.learning_rate : learning_rate;

    float total_loss = 0.0f;

    core::Logger::info("Starting VAE training: epochs=" + std::to_string(actual_epochs) +
                       ", batch_size=" + std::to_string(actual_batch_size) +
                       ", lr=" + std::to_string(actual_lr));

    for (int epoch = 0; epoch < actual_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        // Shuffle data
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);

        // Process batches
        for (size_t batch_start = 0; batch_start < data.size(); batch_start += actual_batch_size) {
            size_t batch_end = std::min(batch_start + actual_batch_size, data.size());
            float batch_loss = 0.0f;

            // Process each sample in batch
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];

                // Forward pass
                auto [mean, log_var] = encode(data[idx]);
                std::vector<T> latent = sample(mean, log_var);
                std::vector<T> reconstruction = decode(latent);

                // Calculate loss
                float sample_loss = calculateLoss(data[idx], reconstruction, mean, log_var);
                batch_loss += sample_loss;

                // Simple weight update (in practice, would use proper backpropagation)
                updateWeights(sample_loss, actual_lr);
            }

            epoch_loss += batch_loss / (batch_end - batch_start);
        }

        total_loss = epoch_loss / (data.size() / actual_batch_size + 1);

        if (epoch % 10 == 0) {
            core::Logger::info("Epoch " + std::to_string(epoch) +
                               ", Loss: " + std::to_string(total_loss));
        }
    }

    return total_loss;
}

template <typename T>
float VariationalAutoencoder<T>::evaluate(const std::vector<std::vector<T>>& data)
{
    float total_reconstruction_error = 0.0f;

    for (const auto& sample : data) {
        std::vector<T> reconstruction = forward(sample);
        total_reconstruction_error += computeReconstructionLoss(sample, reconstruction);
    }

    return total_reconstruction_error / data.size();
}

template <typename T>
float VariationalAutoencoder<T>::calculateLoss(const std::vector<T>& input,
                                               const std::vector<T>& reconstruction,
                                               const std::vector<T>& mean,
                                               const std::vector<T>& log_var)
{
    // Reconstruction loss (L_VAE component)
    T recon_loss = computeReconstructionLoss(input, reconstruction);

    // KL divergence loss
    T kl_loss = computeKLDivergence(mean, log_var);

    // Total ELBO loss
    T total_loss = recon_loss + config_.beta * kl_loss;

    // Add interpolation loss (L_INF component) if enabled
    if (config_.use_interpolation) {
        // Sample two random latent points for interpolation loss
        std::vector<T> latent1 = sample(mean, log_var, rng_());
        std::vector<T> latent2 = sample(mean, log_var, rng_() + 1);
        std::vector<T> interpolated = interpolate(latent1, latent2, T(0.5));

        T interp_loss = computeInterpolationLoss(latent1, latent2, interpolated);
        total_loss += config_.interpolation_weight * interp_loss;
    }

    return static_cast<float>(total_loss);
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::applyLogarithmics(const std::vector<T>& input) const
{
    std::vector<T> result;
    result.reserve(input.size());

    for (T value : input) {
        // Apply safe logarithm: log(1 + |x|) * sign(x)
        T safe_log = std::log(T(1) + std::abs(value));
        result.push_back(value >= 0 ? safe_log : -safe_log);
    }

    return result;
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::applyInverseLogarithmics(
    const std::vector<T>& input) const
{
    std::vector<T> result;
    result.reserve(input.size());

    for (T value : input) {
        // Apply inverse: (exp(|x|) - 1) * sign(x)
        T safe_exp = std::exp(std::abs(value)) - T(1);
        result.push_back(value >= 0 ? safe_exp : -safe_exp);
    }

    return result;
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::applyStandardization(const std::vector<T>& input) const
{
    if (input.empty()) return input;

    // Calculate mean and standard deviation
    T mean = std::accumulate(input.begin(), input.end(), T(0)) / input.size();

    T variance = T(0);
    for (T value : input) {
        variance += (value - mean) * (value - mean);
    }
    variance /= input.size();
    T std_dev = std::sqrt(variance + T(1e-8));  // Add small epsilon for numerical stability

    // Standardize: (x - μ) / σ
    std::vector<T> result;
    result.reserve(input.size());

    for (T value : input) {
        result.push_back((value - mean) / std_dev);
    }

    return result;
}

template <typename T>
std::vector<T> VariationalAutoencoder<T>::applyInverseStandardization(
    const std::vector<T>& input) const
{
    // For simplicity, we assume unit normal distribution for inverse transform
    // In practice, you'd store the original statistics
    return input;  // Placeholder - would need stored mean/std for proper inverse
}

template <typename T>
T VariationalAutoencoder<T>::computeKLDivergence(const std::vector<T>& mean,
                                                 const std::vector<T>& log_var) const
{
    T kl_div = T(0);

    for (size_t i = 0; i < latent_dim_; ++i) {
        // KL(q(z|x) || p(z)) = -0.5 * (1 + log_var - mean^2 - exp(log_var))
        kl_div += T(0.5) * (mean[i] * mean[i] + std::exp(log_var[i]) - log_var[i] - T(1));
    }

    return kl_div;
}

template <typename T>
T VariationalAutoencoder<T>::computeReconstructionLoss(const std::vector<T>& input,
                                                       const std::vector<T>& reconstruction) const
{
    T loss = T(0);

    // Mean squared error
    for (size_t i = 0; i < input.size(); ++i) {
        T diff = input[i] - reconstruction[i];
        loss += diff * diff;
    }

    return loss / input.size();
}

template <typename T>
T VariationalAutoencoder<T>::computeInterpolationLoss(const std::vector<T>& latent1,
                                                      const std::vector<T>& latent2,
                                                      const std::vector<T>& interpolated) const
{
    // Loss encourages smooth interpolation
    T loss = T(0);

    for (size_t i = 0; i < latent_dim_; ++i) {
        T expected = T(0.5) * (latent1[i] + latent2[i]);  // Linear interpolation expectation
        T diff = interpolated[i] - expected;
        loss += diff * diff;
    }

    return loss / latent_dim_;
}

template <typename T>
void VariationalAutoencoder<T>::updateWeights(T loss, T learning_rate)
{
    // Simplified weight update - in practice would use proper gradient computation
    // This is a placeholder for demonstration

    // Apply small random updates proportional to loss (not a real gradient)
    std::uniform_real_distribution<T> dist(-learning_rate * loss * T(0.01),
                                           learning_rate * loss * T(0.01));

    // Note: This is not proper backpropagation, just a demonstration
    // Real implementation would compute gradients and update weights accordingly
}

template <typename T>
void VariationalAutoencoder<T>::protectLatentVariables(std::vector<T>& latent,
                                                       double radiation_level) const
{
    if (radiation_level > 0.1) {  // Apply protection if radiation level is significant
        // Apply simple redundancy check
        for (size_t i = 0; i < latent.size(); ++i) {
            // Store original value
            T original = latent[i];

            // Create protected copies (TMR-like)
            T copy1 = original;
            T copy2 = original;
            T copy3 = original;

            // Simulate potential radiation effects
            std::uniform_real_distribution<T> noise(-T(0.01), T(0.01));
            if (radiation_level > 0.5) {
                copy1 += noise(rng_);
                copy2 += noise(rng_);
                copy3 += noise(rng_);
            }

            // Majority voting
            if (std::abs(copy1 - copy2) < std::abs(copy1 - copy3)) {
                latent[i] = (copy1 + copy2) / T(2);
            }
            else if (std::abs(copy1 - copy3) < std::abs(copy2 - copy3)) {
                latent[i] = (copy1 + copy3) / T(2);
            }
            else {
                latent[i] = (copy2 + copy3) / T(2);
            }

            // Track if correction was applied
            if (std::abs(latent[i] - original) > T(1e-6)) {
                error_stats_.detected_errors++;
                error_stats_.corrected_errors++;
            }
        }
    }
}

template <typename T>
void VariationalAutoencoder<T>::applyRadiationEffects(double radiation_level, uint64_t seed)
{
    // Apply radiation effects to all networks
    encoder_->applyRadiationEffects(radiation_level, seed);
    decoder_->applyRadiationEffects(radiation_level, seed + 1);
    if (interpolator_) {
        interpolator_->applyRadiationEffects(radiation_level, seed + 2);
    }
}

template <typename T>
std::pair<uint64_t, uint64_t> VariationalAutoencoder<T>::getErrorStats() const
{
    uint64_t total_detected = error_stats_.detected_errors;
    uint64_t total_corrected = error_stats_.corrected_errors;

    // Add network error stats
    auto encoder_stats = encoder_->getErrorStats();
    auto decoder_stats = decoder_->getErrorStats();

    total_detected += encoder_stats.first + decoder_stats.first;
    total_corrected += encoder_stats.second + decoder_stats.second;

    if (interpolator_) {
        auto interp_stats = interpolator_->getErrorStats();
        total_detected += interp_stats.first;
        total_corrected += interp_stats.second;
    }

    return std::make_pair(total_detected, total_corrected);
}

template <typename T>
void VariationalAutoencoder<T>::resetErrorStats()
{
    error_stats_.detected_errors = 0;
    error_stats_.corrected_errors = 0;
    error_stats_.uncorrectable_errors = 0;

    encoder_->resetErrorStats();
    decoder_->resetErrorStats();
    if (interpolator_) {
        interpolator_->resetErrorStats();
    }
}

template <typename T>
bool VariationalAutoencoder<T>::saveToFile(const std::string& filename) const
{
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Save dimensions and config
        file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(input_dim_));
        file.write(reinterpret_cast<const char*>(&latent_dim_), sizeof(latent_dim_));

        // Save hidden dimensions
        size_t hidden_count = hidden_dims_.size();
        file.write(reinterpret_cast<const char*>(&hidden_count), sizeof(hidden_count));
        file.write(reinterpret_cast<const char*>(hidden_dims_.data()),
                   hidden_count * sizeof(size_t));

        // Save config
        file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));

        // Note: Saving network weights would require implementing serialization
        // in ProtectedNeuralNetwork - placeholder for now

        return true;
    }
    catch (const std::exception& e) {
        core::Logger::error("Failed to save VAE: " + std::string(e.what()));
        return false;
    }
}

template <typename T>
bool VariationalAutoencoder<T>::loadFromFile(const std::string& filename)
{
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Load dimensions
        file.read(reinterpret_cast<char*>(&input_dim_), sizeof(input_dim_));
        file.read(reinterpret_cast<char*>(&latent_dim_), sizeof(latent_dim_));

        // Load hidden dimensions
        size_t hidden_count;
        file.read(reinterpret_cast<char*>(&hidden_count), sizeof(hidden_count));
        hidden_dims_.resize(hidden_count);
        file.read(reinterpret_cast<char*>(hidden_dims_.data()), hidden_count * sizeof(size_t));

        // Load config
        file.read(reinterpret_cast<char*>(&config_), sizeof(config_));

        // Reinitialize networks
        initializeEncoder();
        initializeDecoder();
        initializeInterpolator();

        return true;
    }
    catch (const std::exception& e) {
        core::Logger::error("Failed to load VAE: " + std::string(e.what()));
        return false;
    }
}

// Explicit template instantiation
template class VariationalAutoencoder<float>;

}  // namespace research
}  // namespace rad_ml
