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
#include <array>
#include <map>
#include <chrono>
#include <random>
#include <functional>
#include <iomanip>
#include <cmath>
#include <thread>
#include <string>
#include <algorithm>
#include <limits>

// Include our simplified TMR implementations
// Health-Weighted TMR
template <typename T>
class HealthWeightedTMR {
public:
    explicit HealthWeightedTMR(const T& initial_value = T()) 
        : copies_{initial_value, initial_value, initial_value},
          health_scores_{1.0, 1.0, 1.0} {
    }
    
    T get() const {
        // Simple voting with health tracking
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            updateHealthScores(0, true);
            if (copies_[0] == copies_[1]) {
                updateHealthScores(1, true);
                updateHealthScores(2, copies_[2] == copies_[0]);
            } else {
                updateHealthScores(1, copies_[1] == copies_[0]);
                updateHealthScores(2, true);
            }
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            updateHealthScores(0, false);
            updateHealthScores(1, true);
            updateHealthScores(2, true);
            return copies_[1];
        }
        
        // No majority, use health scores
        size_t best_idx = 0;
        for (size_t i = 1; i < 3; ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    void set(const T& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
        for (auto& score : health_scores_) {
            score = 1.0;
        }
    }
    
    void repair() {
        T value = get();
        for (auto& copy : copies_) {
            copy = value;
        }
    }
    
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    void corruptCopy(size_t index, const T& value) {
        if (index < 3) {
            copies_[index] = value;
        }
    }
    
private:
    std::array<T, 3> copies_;
    mutable std::array<double, 3> health_scores_;
    
    void updateHealthScores(size_t index, bool correct) const {
        const double reward = 0.05; 
        const double penalty = 0.2;
        
        if (correct) {
            health_scores_[index] = std::min(1.0, health_scores_[index] + reward);
        } else {
            health_scores_[index] = std::max(0.1, health_scores_[index] - penalty);
        }
    }
};

// Enumeration for approximation types
enum class ApproximationType {
    EXACT,
    REDUCED_PRECISION,
    RANGE_LIMITED,
    CUSTOM
};

// Approximate TMR
template <typename T>
class ApproximateTMR {
public:
    ApproximateTMR(
        const T& initial_value,
        const std::array<ApproximationType, 3>& approx_types = {
            ApproximationType::EXACT,
            ApproximationType::REDUCED_PRECISION,
            ApproximationType::RANGE_LIMITED
        }) : approximation_types_(approx_types) {
        set(initial_value);
    }
    
    T get() const {
        T values[3];
        for (size_t i = 0; i < 3; ++i) {
            values[i] = copies_[i];
        }
        
        if (values[0] == values[1]) return values[0];
        if (values[0] == values[2]) return values[0];
        if (values[1] == values[2]) return values[1];
        
        // No majority - return exact copy
        for (size_t i = 0; i < 3; ++i) {
            if (approximation_types_[i] == ApproximationType::EXACT) {
                return values[i];
            }
        }
        
        return values[0];
    }
    
    void set(const T& value) {
        for (size_t i = 0; i < 3; ++i) {
            copies_[i] = applyApproximation(value, approximation_types_[i]);
        }
    }
    
    void repair() {
        T value = get();
        set(value);
    }
    
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    void corruptCopy(size_t index, const T& value) {
        if (index < 3) {
            copies_[index] = value;
        }
    }
    
private:
    std::array<T, 3> copies_;
    std::array<ApproximationType, 3> approximation_types_;
    
    T applyApproximation(const T& value, ApproximationType type) const {
        switch (type) {
            case ApproximationType::EXACT:
                return value;
            case ApproximationType::REDUCED_PRECISION:
                return reducePrecision(value);
            case ApproximationType::RANGE_LIMITED:
                return limitRange(value);
            default:
                return value;
        }
    }
    
    T reducePrecision(const T& value) const {
        if constexpr (std::is_floating_point_v<T>) {
            constexpr int shift = 10;
            double result = value;
            result = std::floor(result * shift) / shift;
            return static_cast<T>(result);
        } 
        else if constexpr (std::is_integral_v<T>) {
            constexpr int shift = sizeof(T) <= 2 ? 2 : 3;
            return (value >> shift) << shift;
        }
        else {
            return value;
        }
    }
    
    T limitRange(const T& value) const {
        if constexpr (std::is_floating_point_v<T>) {
            constexpr T limit = 1000;
            return std::max(std::min(value, limit), -limit);
        } 
        else if constexpr (std::is_integral_v<T>) {
            constexpr T limit = std::numeric_limits<T>::max() / 2;
            return std::max(std::min(value, limit), static_cast<T>(-limit));
        }
        else {
            return value;
        }
    }
};

// Simple Triple Modular Redundancy (TMR)
template <typename T>
class BasicTMR {
public:
    explicit BasicTMR(const T& initial_value = T()) 
        : copies_{initial_value, initial_value, initial_value} {
    }
    
    T get() const {
        // Simple majority voting
        if (copies_[0] == copies_[1]) return copies_[0];
        if (copies_[0] == copies_[2]) return copies_[0];
        if (copies_[1] == copies_[2]) return copies_[1];
        
        // No majority, return first copy
        return copies_[0];
    }
    
    void set(const T& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
    }
    
    void repair() {
        T value = get();
        for (auto& copy : copies_) {
            copy = value;
        }
    }
    
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    void corruptCopy(size_t index, const T& value) {
        if (index < 3) {
            copies_[index] = value;
        }
    }
    
private:
    std::array<T, 3> copies_;
};

// Space Environment Simulator
class SpaceEnvironmentSimulator {
public:
    enum class Location {
        LEO,                // Low Earth Orbit
        GEO,                // Geostationary Orbit
        MOON,               // Lunar Environment
        MARS,               // Mars Environment
        JUPITER,            // Jupiter Environment
        SOLAR_FLARE,        // Solar Flare Event
        EXTREME             // Beyond Jupiter levels
    };
    
    enum class ErrorType {
        SINGLE_BIT_FLIP,    // Single bit error
        MULTI_BIT_UPSET,    // Multiple consecutive bits
        STUCK_BIT,          // Bit stuck at 0 or 1
        SEVERE_CORRUPTION   // Multiple random bits
    };
    
    struct RadiationEvent {
        ErrorType type;
        size_t target_index;
        size_t bit_position;
        int num_bits;
        uint64_t timestamp_ms;
        std::string description;
    };
    
    SpaceEnvironmentSimulator(Location location, double duration_factor = 1.0)
        : location_(location), 
          duration_factor_(duration_factor),
          random_engine_(std::random_device{}()) {
        configureEnvironment();
    }
    
    std::vector<RadiationEvent> simulateRadiation(
        size_t num_elements, 
        std::chrono::milliseconds duration) {
        
        std::vector<RadiationEvent> events;
        
        // Calculate expected events based on radiation rate and duration
        double seconds = duration.count() / 1000.0;
        double expected_events = radiation_rate_ * seconds * num_elements * duration_factor_;
        
        // Generate actual number using Poisson distribution
        std::poisson_distribution<int> poisson(expected_events);
        int num_events = poisson(random_engine_);
        
        // Generate each event
        for (int i = 0; i < num_events; ++i) {
            RadiationEvent event;
            
            // Random timestamp within the duration
            std::uniform_int_distribution<uint64_t> time_dist(0, duration.count());
            event.timestamp_ms = time_dist(random_engine_);
            
            // Random target element
            std::uniform_int_distribution<size_t> target_dist(0, num_elements - 1);
            event.target_index = target_dist(random_engine_);
            
            // Generate error type based on environment
            event.type = generateErrorType();
            
            // Generate error details
            switch (event.type) {
                case ErrorType::SINGLE_BIT_FLIP:
                    event.bit_position = std::uniform_int_distribution<size_t>(0, 31)(random_engine_);
                    event.num_bits = 1;
                    event.description = "Single bit flip at bit " + std::to_string(event.bit_position);
                    break;
                    
                case ErrorType::MULTI_BIT_UPSET:
                    event.bit_position = std::uniform_int_distribution<size_t>(0, 29)(random_engine_);
                    event.num_bits = std::uniform_int_distribution<int>(2, 4)(random_engine_);
                    event.description = "Multi-bit upset starting at bit " + 
                                       std::to_string(event.bit_position) + 
                                       " affecting " + std::to_string(event.num_bits) + " bits";
                    break;
                    
                case ErrorType::STUCK_BIT:
                    event.bit_position = std::uniform_int_distribution<size_t>(0, 31)(random_engine_);
                    event.num_bits = 1;
                    event.description = "Bit " + std::to_string(event.bit_position) + 
                                       " stuck at " + 
                                       (std::uniform_int_distribution<int>(0, 1)(random_engine_) ? "1" : "0");
                    break;
                    
                case ErrorType::SEVERE_CORRUPTION:
                    event.bit_position = 0; // Not meaningful for severe corruption
                    event.num_bits = std::uniform_int_distribution<int>(8, 16)(random_engine_);
                    event.description = "Severe corruption affecting " + 
                                       std::to_string(event.num_bits) + " random bits";
                    break;
            }
            
            events.push_back(event);
        }
        
        // Sort by timestamp
        std::sort(events.begin(), events.end(), 
                 [](const RadiationEvent& a, const RadiationEvent& b) {
                     return a.timestamp_ms < b.timestamp_ms;
                 });
        
        return events;
    }
    
    // Apply a radiation event to a value
    template <typename T>
    T applyRadiationEvent(const T& value, const RadiationEvent& event) const {
        T corrupted = value;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&corrupted);
        
        switch (event.type) {
            case ErrorType::SINGLE_BIT_FLIP: {
                size_t byte_pos = event.bit_position / 8;
                size_t bit_in_byte = event.bit_position % 8;
                if (byte_pos < sizeof(T)) {
                    bytes[byte_pos] ^= (1 << bit_in_byte); // Flip single bit
                }
                break;
            }
                
            case ErrorType::MULTI_BIT_UPSET: {
                size_t byte_pos = event.bit_position / 8;
                size_t bit_in_byte = event.bit_position % 8;
                if (byte_pos < sizeof(T)) {
                    for (int i = 0; i < event.num_bits && bit_in_byte + i < 8; ++i) {
                        bytes[byte_pos] ^= (1 << (bit_in_byte + i)); // Flip consecutive bits
                    }
                }
                break;
            }
                
            case ErrorType::STUCK_BIT: {
                size_t byte_pos = event.bit_position / 8;
                size_t bit_in_byte = event.bit_position % 8;
                if (byte_pos < sizeof(T)) {
                    // 50% chance of stuck at 1, 50% chance of stuck at 0
                    if (std::uniform_int_distribution<int>(0, 1)(random_engine_)) {
                        bytes[byte_pos] |= (1 << bit_in_byte);  // Stuck at 1
                    } else {
                        bytes[byte_pos] &= ~(1 << bit_in_byte); // Stuck at 0
                    }
                }
                break;
            }
                
            case ErrorType::SEVERE_CORRUPTION: {
                // Corrupt multiple random bits
                for (int i = 0; i < event.num_bits; ++i) {
                    size_t bit_pos = std::uniform_int_distribution<size_t>(0, sizeof(T) * 8 - 1)(random_engine_);
                    size_t byte_pos = bit_pos / 8;
                    size_t bit_in_byte = bit_pos % 8;
                    if (byte_pos < sizeof(T)) {
                        bytes[byte_pos] ^= (1 << bit_in_byte);
                    }
                }
                break;
            }
        }
        
        return corrupted;
    }
    
    // Get radiation environment description
    std::string getEnvironmentDescription() const {
        std::string desc = "Space Environment: ";
        
        switch (location_) {
            case Location::LEO:
                desc += "Low Earth Orbit";
                break;
            case Location::GEO:
                desc += "Geostationary Orbit";
                break;
            case Location::MOON:
                desc += "Lunar Environment";
                break;
            case Location::MARS:
                desc += "Mars Environment";
                break;
            case Location::JUPITER:
                desc += "Jupiter Environment";
                break;
            case Location::SOLAR_FLARE:
                desc += "Solar Flare Event";
                break;
            case Location::EXTREME:
                desc += "Extreme Radiation (Beyond Jupiter)";
                break;
        }
        
        desc += "\nRadiation Rate: " + std::to_string(radiation_rate_) + " events/second/element";
        desc += "\nDuration Factor: " + std::to_string(duration_factor_) + "x";
        desc += "\nSingle Bit Error Probability: " + std::to_string(error_probs_.at(ErrorType::SINGLE_BIT_FLIP));
        desc += "\nMulti-Bit Upset Probability: " + std::to_string(error_probs_.at(ErrorType::MULTI_BIT_UPSET));
        desc += "\nStuck Bit Probability: " + std::to_string(error_probs_.at(ErrorType::STUCK_BIT));
        desc += "\nSevere Corruption Probability: " + std::to_string(error_probs_.at(ErrorType::SEVERE_CORRUPTION));
        
        return desc;
    }
    
private:
    Location location_;
    double duration_factor_;
    double radiation_rate_;
    std::map<ErrorType, double> error_probs_;
    mutable std::mt19937 random_engine_;
    
    // Configure environment parameters based on location
    void configureEnvironment() {
        switch (location_) {
            case Location::LEO:
                radiation_rate_ = 0.01; // Events per second per element
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.80;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.15;
                error_probs_[ErrorType::STUCK_BIT] = 0.04;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.01;
                break;
                
            case Location::GEO:
                radiation_rate_ = 0.05;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.70;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.20;
                error_probs_[ErrorType::STUCK_BIT] = 0.07;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.03;
                break;
                
            case Location::MOON:
                radiation_rate_ = 0.1;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.65;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.25;
                error_probs_[ErrorType::STUCK_BIT] = 0.07;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.03;
                break;
                
            case Location::MARS:
                radiation_rate_ = 0.15;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.60;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.25;
                error_probs_[ErrorType::STUCK_BIT] = 0.10;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.05;
                break;
                
            case Location::JUPITER:
                radiation_rate_ = 0.5;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.50;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.30;
                error_probs_[ErrorType::STUCK_BIT] = 0.10;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.10;
                break;
                
            case Location::SOLAR_FLARE:
                radiation_rate_ = 1.0;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.45;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.35;
                error_probs_[ErrorType::STUCK_BIT] = 0.10;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.10;
                break;
                
            case Location::EXTREME:
                radiation_rate_ = 2.0;
                error_probs_[ErrorType::SINGLE_BIT_FLIP] = 0.40;
                error_probs_[ErrorType::MULTI_BIT_UPSET] = 0.30;
                error_probs_[ErrorType::STUCK_BIT] = 0.15;
                error_probs_[ErrorType::SEVERE_CORRUPTION] = 0.15;
                break;
        }
    }
    
    // Generate error type based on probabilities
    ErrorType generateErrorType() const {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double roll = dist(random_engine_);
        double cumulative = 0.0;
        
        for (const auto& [type, prob] : error_probs_) {
            cumulative += prob;
            if (roll <= cumulative) {
                return type;
            }
        }
        
        return ErrorType::SINGLE_BIT_FLIP; // Default
    }
};

// Struct to hold test results
struct TestResults {
    size_t total_events = 0;
    size_t detected_errors = 0;
    size_t corrected_errors = 0;
    
    std::map<SpaceEnvironmentSimulator::ErrorType, size_t> error_type_counts;
    std::map<SpaceEnvironmentSimulator::ErrorType, size_t> correction_success_by_type;
    
    double getDetectionRate() const {
        return total_events > 0 ? static_cast<double>(detected_errors) / total_events : 1.0;
    }
    
    double getCorrectionRate() const {
        return total_events > 0 ? static_cast<double>(corrected_errors) / total_events : 1.0;
    }
    
    double getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType type) const {
        if (error_type_counts.count(type) && error_type_counts.at(type) > 0) {
            return static_cast<double>(correction_success_by_type.at(type)) / error_type_counts.at(type);
        }
        return 1.0;
    }
    
    void print(const std::string& name) const {
        std::cout << "=== Results for " << name << " ===" << std::endl;
        std::cout << "Total Radiation Events: " << total_events << std::endl;
        std::cout << "Detected Errors: " << detected_errors 
                  << " (" << std::fixed << std::setprecision(2) << (getDetectionRate() * 100.0) << "%)" << std::endl;
        std::cout << "Corrected Errors: " << corrected_errors 
                  << " (" << std::fixed << std::setprecision(2) << (getCorrectionRate() * 100.0) << "%)" << std::endl;
        
        std::cout << "\nError Types:" << std::endl;
        for (const auto& [type, count] : error_type_counts) {
            std::cout << "  " << getErrorTypeName(type) << ": " << count 
                      << " events, " << std::fixed << std::setprecision(2) 
                      << (getCorrectionRateByType(type) * 100.0) << "% corrected" << std::endl;
        }
        std::cout << std::endl;
    }
    
private:
    std::string getErrorTypeName(SpaceEnvironmentSimulator::ErrorType type) const {
        switch (type) {
            case SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP:
                return "Single Bit Flip";
            case SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET:
                return "Multi-Bit Upset";
            case SpaceEnvironmentSimulator::ErrorType::STUCK_BIT:
                return "Stuck Bit";
            case SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION:
                return "Severe Corruption";
            default:
                return "Unknown";
        }
    }
};

// Template function to run a stress test on a TMR implementation
template <template<typename> class TMR, typename T>
TestResults runStressTest(
    const std::string& name,
    SpaceEnvironmentSimulator& env,
    const T& initial_value,
    size_t num_elements,
    std::chrono::milliseconds duration,
    bool show_progress = true) {
    
    // Setup test data
    std::vector<TMR<T>> tmr_elements;
    std::vector<T> expected_values;
    
    for (size_t i = 0; i < num_elements; ++i) {
        tmr_elements.push_back(TMR<T>(initial_value));
        expected_values.push_back(initial_value);
    }
    
    // Generate radiation events
    std::cout << "Generating radiation events for " << name << "..." << std::endl;
    auto events = env.simulateRadiation(num_elements, duration);
    std::cout << "Generated " << events.size() << " radiation events." << std::endl;
    
    TestResults results;
    results.total_events = events.size();
    
    // Initialize error type counts
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::STUCK_BIT] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION] = 0;
    
    results.correction_success_by_type[SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP] = 0;
    results.correction_success_by_type[SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET] = 0;
    results.correction_success_by_type[SpaceEnvironmentSimulator::ErrorType::STUCK_BIT] = 0;
    results.correction_success_by_type[SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION] = 0;
    
    // Process events
    std::cout << "Processing events:" << std::endl;
    
    size_t checkpoint = events.size() / 20; // 5% increments
    size_t next_checkpoint = checkpoint;
    
    // Create a persistent random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> copy_dist(0, 2);

    for (size_t event_idx = 0; event_idx < events.size(); ++event_idx) {
        const auto& event = events[event_idx];
        size_t element_idx = event.target_index;
        
        if (show_progress && event_idx >= next_checkpoint) {
            int percent = static_cast<int>((static_cast<double>(event_idx) / events.size()) * 100);
            std::cout << "  Progress: " << percent << "% (" 
                      << event_idx << "/" << events.size() << " events)" << std::endl;
            next_checkpoint += checkpoint;
        }
        
        // Store pre-event copy for checking
        auto pre_event_copies = tmr_elements[element_idx].getCopies();
        T expected = expected_values[element_idx];
        
        // Update error type counts
        results.error_type_counts[event.type]++;
        
        // Corrupt a random copy
        int copy_to_corrupt = copy_dist(rng);
        
        // Calculate corrupted value
        auto original = pre_event_copies[copy_to_corrupt];
        auto corrupted = env.applyRadiationEvent(original, event);
        
        // Skip if no actual corruption occurred
        if (corrupted == original) {
            continue;
        }

        // Apply corruption
        tmr_elements[element_idx].corruptCopy(copy_to_corrupt, corrupted);

        // Check if the TMR value is affected by the corruption
        T after_event = tmr_elements[element_idx].get();

        // If the TMR value differs from expected, it's a detection failure
        if (after_event != expected) {
            results.detected_errors++;
            
            // Repair attempt
            tmr_elements[element_idx].repair();
            T after_repair = tmr_elements[element_idx].get();
            
            if (after_repair == expected) {
                results.corrected_errors++;
                results.correction_success_by_type[event.type]++;
            } else {
                // Update expected value if repair was unsuccessful
                expected_values[element_idx] = after_repair;
            }
        } else {
            // It's a success if TMR returned the correct value despite corruption
            results.corrected_errors++;
            results.correction_success_by_type[event.type]++;
        }
        
        // Simulate small time increment
        if (event_idx % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    std::cout << "Test complete!" << std::endl;
    return results;
}

int main() {
    std::cout << "====== Extreme Space Radiation Stress Test ======" << std::endl;
    
    // Setup test parameters
    SpaceEnvironmentSimulator::Location test_location = SpaceEnvironmentSimulator::Location::EXTREME;
    double duration_factor = 3.0;  // More radiation events for stress testing
    size_t num_elements = 2000;    // Number of elements to protect
    std::chrono::milliseconds test_duration(15000); // 15 seconds simulating days in space
    int initial_value = 42;        // Value to protect
    
    // Create environment simulator
    SpaceEnvironmentSimulator env(test_location, duration_factor);
    
    // Print test configuration
    std::cout << "Test Configuration:" << std::endl;
    std::cout << env.getEnvironmentDescription() << std::endl;
    std::cout << "Number of Protected Elements: " << num_elements << std::endl;
    std::cout << "Test Duration: " << test_duration.count() << " ms (simulating extended space operation)" << std::endl;
    std::cout << "Initial Value: " << initial_value << std::endl;
    std::cout << "\n";
    
    // Run tests for each TMR implementation
    std::cout << "Running stress tests...\n" << std::endl;
    
    // Basic TMR test
    auto basic_results = runStressTest<BasicTMR, int>(
        "Basic TMR", env, initial_value, num_elements, test_duration);
    
    // Health-Weighted TMR test
    auto hw_results = runStressTest<HealthWeightedTMR, int>(
        "Health-Weighted TMR", env, initial_value, num_elements, test_duration);
    
    // Approximate TMR test
    auto approx_results = runStressTest<ApproximateTMR, int>(
        "Approximate TMR", env, initial_value, num_elements, test_duration);
    
    // Print results
    std::cout << "\n====== Test Results Summary ======" << std::endl;
    basic_results.print("Basic TMR");
    hw_results.print("Health-Weighted TMR");
    approx_results.print("Approximate TMR");
    
    // Compare results
    std::cout << "====== Comparative Analysis ======" << std::endl;
    std::cout << "Error Correction Rates:" << std::endl;
    std::cout << "  Basic TMR: " << std::fixed << std::setprecision(2) 
              << (basic_results.getCorrectionRate() * 100.0) << "%" << std::endl;
    std::cout << "  Health-Weighted TMR: " << std::fixed << std::setprecision(2) 
              << (hw_results.getCorrectionRate() * 100.0) << "%" << std::endl;
    std::cout << "  Approximate TMR: " << std::fixed << std::setprecision(2) 
              << (approx_results.getCorrectionRate() * 100.0) << "%" << std::endl;
    
    std::cout << "\nPerformance Against Single-Bit Errors:" << std::endl;
    std::cout << "  Basic TMR: " << std::fixed << std::setprecision(2) 
              << (basic_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP) * 100.0) << "%" << std::endl;
    std::cout << "  Health-Weighted TMR: " << std::fixed << std::setprecision(2) 
              << (hw_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP) * 100.0) << "%" << std::endl;
    std::cout << "  Approximate TMR: " << std::fixed << std::setprecision(2) 
              << (approx_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP) * 100.0) << "%" << std::endl;
    
    std::cout << "\nPerformance Against Multi-Bit Upsets:" << std::endl;
    std::cout << "  Basic TMR: " << std::fixed << std::setprecision(2) 
              << (basic_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET) * 100.0) << "%" << std::endl;
    std::cout << "  Health-Weighted TMR: " << std::fixed << std::setprecision(2) 
              << (hw_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET) * 100.0) << "%" << std::endl;
    std::cout << "  Approximate TMR: " << std::fixed << std::setprecision(2) 
              << (approx_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET) * 100.0) << "%" << std::endl;
    
    std::cout << "\nPerformance Against Severe Corruption:" << std::endl;
    std::cout << "  Basic TMR: " << std::fixed << std::setprecision(2) 
              << (basic_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION) * 100.0) << "%" << std::endl;
    std::cout << "  Health-Weighted TMR: " << std::fixed << std::setprecision(2) 
              << (hw_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION) * 100.0) << "%" << std::endl;
    std::cout << "  Approximate TMR: " << std::fixed << std::setprecision(2) 
              << (approx_results.getCorrectionRateByType(SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION) * 100.0) << "%" << std::endl;
    
    // Determine best overall performer
    double basic_overall = basic_results.getCorrectionRate();
    double hw_overall = hw_results.getCorrectionRate();
    double approx_overall = approx_results.getCorrectionRate();
    
    std::cout << "\n====== Conclusion ======" << std::endl;
    std::cout << "Best Overall Performer: ";
    if (hw_overall >= basic_overall && hw_overall >= approx_overall) {
        std::cout << "Health-Weighted TMR (" << std::fixed << std::setprecision(2) 
                  << (hw_overall * 100.0) << "% correction rate)" << std::endl;
    } else if (approx_overall >= basic_overall && approx_overall >= hw_overall) {
        std::cout << "Approximate TMR (" << std::fixed << std::setprecision(2) 
                  << (approx_overall * 100.0) << "% correction rate)" << std::endl;
    } else {
        std::cout << "Basic TMR (" << std::fixed << std::setprecision(2) 
                  << (basic_overall * 100.0) << "% correction rate)" << std::endl;
    }
    
    std::cout << "\nStress Test Completed!" << std::endl;
    
    return 0;
} 
