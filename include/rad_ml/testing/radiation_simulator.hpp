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
#pragma once

#include <random>
#include <chrono>
#include <functional>
#include <string>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>

namespace rad_ml {
namespace testing {

/**
 * @brief Physics-based radiation effects simulator
 * 
 * This class simulates radiation effects based on space environment parameters,
 * providing a realistic testing framework for radiation-tolerant software.
 * It models different types of Single Event Effects (SEE) including bit flips,
 * multi-bit upsets, and latchup events.
 */
class RadiationSimulator {
public:
    /// Radiation environment parameters
    struct EnvironmentParams {
        double altitude_km;            ///< Orbital altitude
        double inclination_deg;        ///< Orbital inclination
        double solar_activity;         ///< Solar activity level (0-10)
        bool inside_saa;               ///< Whether inside South Atlantic Anomaly
        double shielding_thickness_mm; ///< Spacecraft shielding
        std::string mission_name;      ///< Optional mission name for logging
    };
    
    /// Types of radiation effects to simulate
    enum class RadiationEffectType {
        SINGLE_BIT_FLIP,       ///< Single bit changes from 0 to 1 or 1 to 0
        MULTI_BIT_UPSET,       ///< Multiple adjacent bits corrupted
        SINGLE_EVENT_LATCHUP,  ///< Bit gets "stuck" at a value
        SINGLE_EVENT_TRANSIENT ///< Temporary voltage spike
    };
    
    /// Information about a simulated radiation event
    struct RadiationEvent {
        RadiationEffectType type;   ///< Type of radiation effect
        size_t memory_offset;       ///< Offset in the memory region
        uint32_t bits_affected;     ///< Number of bits affected
        std::string description;    ///< Human-readable description
    };
    
    /// Event rates for different effect types
    struct EventRates {
        double single_bit_flip_rate;       ///< Events per second
        double multi_bit_upset_rate;       ///< Events per second
        double single_event_latchup_rate;  ///< Events per second
        double single_event_transient_rate; ///< Events per second
        double total_rate;                 ///< Sum of all rates
    };
    
    /**
     * @brief Constructor with environment configuration
     * 
     * @param params Space environment parameters
     */
    explicit RadiationSimulator(const EnvironmentParams& params)
        : env_params_(params), 
          random_engine_(std::random_device{}()) {
        // Initialize based on environment parameters
        updateRates();
    }
    
    /**
     * @brief Get predefined environment parameters for common missions
     * 
     * @param mission_name Name of the mission (e.g., "LEO", "ISS", "LUNAR", "MARS")
     * @return Environment parameters for the specified mission
     */
    static EnvironmentParams getMissionEnvironment(const std::string& mission_name) {
        if (mission_name == "LEO" || mission_name == "LEO_EARTH_OBSERVATION") {
            return {
                .altitude_km = 500.0,
                .inclination_deg = 45.0,
                .solar_activity = 3.0,
                .inside_saa = false,
                .shielding_thickness_mm = 5.0,
                .mission_name = "Low Earth Orbit"
            };
        } 
        else if (mission_name == "ISS") {
            return {
                .altitude_km = 420.0,
                .inclination_deg = 51.6,
                .solar_activity = 3.0,
                .inside_saa = false,
                .shielding_thickness_mm = 10.0,
                .mission_name = "International Space Station"
            };
        }
        else if (mission_name == "GEO" || mission_name == "GEOSTATIONARY") {
            return {
                .altitude_km = 35786.0,
                .inclination_deg = 0.0,
                .solar_activity = 3.0,
                .inside_saa = false,
                .shielding_thickness_mm = 5.0,
                .mission_name = "Geostationary Orbit"
            };
        }
        else if (mission_name == "LUNAR" || mission_name == "MOON") {
            return {
                .altitude_km = 384400.0,  // Average Earth-Moon distance
                .inclination_deg = 0.0,   // N/A for lunar missions
                .solar_activity = 3.0,
                .inside_saa = false,      // N/A for lunar missions
                .shielding_thickness_mm = 3.0,
                .mission_name = "Lunar Mission"
            };
        }
        else if (mission_name == "MARS") {
            return {
                .altitude_km = 0.0,       // N/A for Mars missions
                .inclination_deg = 0.0,   // N/A for Mars missions
                .solar_activity = 3.0,
                .inside_saa = false,      // N/A for Mars missions
                .shielding_thickness_mm = 3.0,
                .mission_name = "Mars Mission"
            };
        }
        else if (mission_name == "JUPITER") {
            return {
                .altitude_km = 0.0,       // N/A for Jupiter missions
                .inclination_deg = 0.0,   // N/A for Jupiter missions
                .solar_activity = 3.0,
                .inside_saa = false,      // N/A for Jupiter missions
                .shielding_thickness_mm = 20.0,  // Much higher for Jupiter's radiation
                .mission_name = "Jupiter Mission"
            };
        }
        else {
            // Default to LEO
            return {
                .altitude_km = 500.0,
                .inclination_deg = 45.0,
                .solar_activity = 3.0,
                .inside_saa = false,
                .shielding_thickness_mm = 5.0,
                .mission_name = "Generic Space Mission"
            };
        }
    }
    
    /**
     * @brief Run simulation for specified duration
     * 
     * @param memory Pointer to the memory region to affect
     * @param memory_size Size of the memory region in bytes
     * @param duration Duration to simulate
     * @return Vector of radiation events that occurred
     */
    template <typename T>
    std::vector<RadiationEvent> simulateEffects(
        T* memory, 
        size_t memory_size, 
        std::chrono::milliseconds duration) {
        
        std::vector<RadiationEvent> events;
        
        // Calculate expected number of events based on duration and environment
        double expected_events = calculateExpectedEvents(duration);
        
        // Use Poisson distribution to determine actual number of events
        std::poisson_distribution<int> poisson(expected_events);
        int num_events = poisson(random_engine_);
        
        for (int i = 0; i < num_events; ++i) {
            RadiationEvent event = generateRandomEvent(memory, memory_size);
            events.push_back(event);
        }
        
        return events;
    }
    
    /**
     * @brief Update environment parameters
     * 
     * @param new_params New environment parameters
     */
    void updateEnvironment(const EnvironmentParams& new_params) {
        env_params_ = new_params;
        updateRates();
    }
    
    /**
     * @brief Get current event rates
     * 
     * @return Rates for different types of radiation events
     */
    const EventRates& getEventRates() const {
        return event_rates_;
    }
    
    /**
     * @brief Get the current simulation environment parameters
     * 
     * @return Current environment parameters
     */
    EnvironmentParams getSimulationEnvironment() const {
        return env_params_;
    }
    
    /**
     * @brief Get string description of current environment
     * 
     * @return Human-readable description of the radiation environment
     */
    std::string getEnvironmentDescription() const {
        std::string description = "Radiation Environment:\n";
        description += "  Mission: " + env_params_.mission_name + "\n";
        description += "  Altitude: " + std::to_string(env_params_.altitude_km) + " km\n";
        
        if (env_params_.altitude_km < 40000) {  // Only relevant for Earth orbits
            description += "  Inclination: " + std::to_string(env_params_.inclination_deg) + " degrees\n";
            description += "  Inside SAA: " + std::string(env_params_.inside_saa ? "Yes" : "No") + "\n";
        }
        
        description += "  Solar Activity: " + std::to_string(env_params_.solar_activity) + "/10\n";
        description += "  Shielding: " + std::to_string(env_params_.shielding_thickness_mm) + " mm\n\n";
        
        description += "Event Rates (per second):\n";
        description += "  Single Bit Flips: " + std::to_string(event_rates_.single_bit_flip_rate) + "\n";
        description += "  Multi-Bit Upsets: " + std::to_string(event_rates_.multi_bit_upset_rate) + "\n";
        description += "  Single Event Latchups: " + std::to_string(event_rates_.single_event_latchup_rate) + "\n";
        description += "  Single Event Transients: " + std::to_string(event_rates_.single_event_transient_rate) + "\n";
        description += "  Total Rate: " + std::to_string(event_rates_.total_rate) + "\n";
        
        return description;
    }
    
private:
    EnvironmentParams env_params_;
    EventRates event_rates_;
    std::default_random_engine random_engine_;
    
    /**
     * @brief Update event rates based on current environment parameters
     */
    void updateRates() {
        // Calculate base rates from environment parameters
        double base_rate = calculateBaseRate();
        
        event_rates_.single_bit_flip_rate = base_rate * 0.8;
        event_rates_.multi_bit_upset_rate = base_rate * 0.15;
        event_rates_.single_event_latchup_rate = base_rate * 0.03;
        event_rates_.single_event_transient_rate = base_rate * 0.02;
        
        event_rates_.total_rate = 
            event_rates_.single_bit_flip_rate +
            event_rates_.multi_bit_upset_rate +
            event_rates_.single_event_latchup_rate +
            event_rates_.single_event_transient_rate;
    }
    
    /**
     * @brief Calculate base event rate based on environment
     * 
     * @return Base event rate (events per second)
     */
    double calculateBaseRate() const {
        // Base rate for different space environments
        double rate = 0.001; // Base rate (events per second)
        
        // Different scaling for different mission types
        if (env_params_.mission_name.find("Mars") != std::string::npos) {
            // Mars has a thinner atmosphere but is farther from the Sun
            rate = 0.002;
        }
        else if (env_params_.mission_name.find("Lunar") != std::string::npos ||
                env_params_.mission_name.find("Moon") != std::string::npos) {
            // Lunar missions experience more radiation than LEO
            rate = 0.003;
        }
        else if (env_params_.mission_name.find("Jupiter") != std::string::npos) {
            // Jupiter has an intense radiation environment
            rate = 0.01;
        }
        else {
            // Earth orbit scaling factors
            
            // Altitude effects (higher altitude = more radiation)
            if (env_params_.altitude_km <= 1000) {
                // LEO
                rate *= 1.0 + (env_params_.altitude_km / 1000.0);
            } else if (env_params_.altitude_km <= 36000) {
                // MEO to GEO
                rate *= 2.0 + (env_params_.altitude_km / 10000.0);
            } else {
                // Beyond GEO
                rate *= 5.0;
            }
            
            // Inclination effects (polar orbits have higher exposure)
            rate *= 1.0 + (env_params_.inclination_deg / 90.0);
        }
        
        // Common scaling factors for all environments
        
        // Solar activity
        rate *= 1.0 + (env_params_.solar_activity / 5.0);
        
        // South Atlantic Anomaly
        if (env_params_.inside_saa) {
            rate *= 10.0;
        }
        
        // Shielding reduces rate exponentially
        rate *= std::exp(-env_params_.shielding_thickness_mm / 10.0);
        
        return rate;
    }
    
    /**
     * @brief Calculate expected number of events for a duration
     * 
     * @param duration Simulation duration
     * @return Expected number of events
     */
    double calculateExpectedEvents(std::chrono::milliseconds duration) const {
        double seconds = duration.count() / 1000.0;
        return event_rates_.total_rate * seconds;
    }
    
    /**
     * @brief Generate a random radiation event
     * 
     * @param memory Pointer to the memory region
     * @param memory_size Size of the memory region in bytes
     * @return A randomly generated radiation event
     */
    template <typename T>
    RadiationEvent generateRandomEvent(T* memory, size_t memory_size) {
        RadiationEvent event;
        
        // Determine event type based on relative rates
        double value = std::uniform_real_distribution<double>(0.0, event_rates_.total_rate)(random_engine_);
        
        if (value < event_rates_.single_bit_flip_rate) {
            event.type = RadiationEffectType::SINGLE_BIT_FLIP;
            event.description = "Single Bit Flip";
        } else if (value < event_rates_.single_bit_flip_rate + event_rates_.multi_bit_upset_rate) {
            event.type = RadiationEffectType::MULTI_BIT_UPSET;
            event.description = "Multi-Bit Upset";
        } else if (value < event_rates_.single_bit_flip_rate + 
                        event_rates_.multi_bit_upset_rate + 
                        event_rates_.single_event_latchup_rate) {
            event.type = RadiationEffectType::SINGLE_EVENT_LATCHUP;
            event.description = "Single Event Latchup";
        } else {
            event.type = RadiationEffectType::SINGLE_EVENT_TRANSIENT;
            event.description = "Single Event Transient";
        }
        
        // Select random location in memory
        std::uniform_int_distribution<size_t> loc_dist(0, memory_size - 1);
        event.memory_offset = loc_dist(random_engine_);
        
        // Apply the effect
        uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(memory) + event.memory_offset;
        
        switch (event.type) {
            case RadiationEffectType::SINGLE_BIT_FLIP: {
                // Select random bit within byte
                std::uniform_int_distribution<int> bit_dist(0, 7);
                int bit = bit_dist(random_engine_);
                
                // Flip the bit
                *byte_ptr ^= (1 << bit);
                
                event.bits_affected = 1;
                event.description += " at offset " + std::to_string(event.memory_offset) + 
                                    ", bit " + std::to_string(bit);
                break;
            }
            
            case RadiationEffectType::MULTI_BIT_UPSET: {
                // MBUs typically affect 2-3 adjacent bits
                std::uniform_int_distribution<int> num_bits_dist(2, 3);
                int num_bits = num_bits_dist(random_engine_);
                
                // Select starting bit
                std::uniform_int_distribution<int> start_bit_dist(0, 8 - num_bits);
                int start_bit = start_bit_dist(random_engine_);
                
                // Create mask for affected bits
                uint8_t mask = 0;
                for (int i = 0; i < num_bits; ++i) {
                    mask |= (1 << (start_bit + i));
                }
                
                // Flip the bits
                *byte_ptr ^= mask;
                
                event.bits_affected = num_bits;
                event.description += " at offset " + std::to_string(event.memory_offset) + 
                                    ", " + std::to_string(num_bits) + " adjacent bits starting at bit " + 
                                    std::to_string(start_bit);
                break;
            }
            
            case RadiationEffectType::SINGLE_EVENT_LATCHUP: {
                // Select random bit within byte
                std::uniform_int_distribution<int> bit_dist(0, 7);
                int bit = bit_dist(random_engine_);
                
                // Determine if bit gets stuck at 0 or 1
                std::uniform_int_distribution<int> value_dist(0, 1);
                int value = value_dist(random_engine_);
                
                if (value == 0) {
                    // Stuck at 0 - clear the bit
                    *byte_ptr &= ~(1 << bit);
                } else {
                    // Stuck at 1 - set the bit
                    *byte_ptr |= (1 << bit);
                }
                
                event.bits_affected = 1;
                event.description += " at offset " + std::to_string(event.memory_offset) + 
                                    ", bit " + std::to_string(bit) + 
                                    " stuck at " + std::to_string(value);
                break;
            }
            
            case RadiationEffectType::SINGLE_EVENT_TRANSIENT: {
                // For simulation purposes, temporarily corrupt the byte
                uint8_t original = *byte_ptr;
                
                // Corrupt with random value
                std::uniform_int_distribution<int> value_dist(0, 255);
                *byte_ptr = static_cast<uint8_t>(value_dist(random_engine_));
                
                event.bits_affected = 8; // Potentially affects all bits in byte
                event.description += " at offset " + std::to_string(event.memory_offset) + 
                                    ", byte temporarily corrupted from " + 
                                    std::to_string(original) + " to " + 
                                    std::to_string(*byte_ptr);
                
                // In a real implementation, we would schedule restoration
                // For now, we'll note that this is a transient effect
                event.description += " (transient effect - would recover on next write)";
                break;
            }
        }
        
        return event;
    }
};

} // namespace testing
} // namespace rad_ml 
