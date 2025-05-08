#include "radiation_simulator.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace rad_ml {
namespace testing {

RadiationSimulator::RadiationSimulator(const EnvironmentParams& params)
    : params_(params) {
    // Initialize random number generators
    std::random_device rd;
    rng_ = std::mt19937(rd());
    
    // Initialize distributions based on environment parameters
    initializeDistributions();
}

void RadiationSimulator::initializeDistributions() {
    // Calculate base radiation flux based on altitude and solar activity
    double base_flux = calculateBaseFlux();
    
    // Initialize LET distribution (MeV-cm²/mg)
    let_dist_ = std::lognormal_distribution<double>(
        std::log(calculateMeanLET()), 
        calculateLETSigma()
    );
    
    // Initialize flux distribution (particles/cm²/s)
    flux_dist_ = std::gamma_distribution<double>(
        calculateFluxShape(), 
        calculateFluxScale(base_flux)
    );
    
    // Initialize energy distribution (MeV)
    energy_dist_ = std::weibull_distribution<double>(
        calculateEnergyShape(),
        calculateEnergyScale()
    );
    
    // Initialize angle distribution (radians)
    angle_dist_ = std::uniform_real_distribution<double>(0, 2 * M_PI);
}

double RadiationSimulator::calculateBaseFlux() const {
    // Base flux calculation based on altitude and solar activity
    double altitude_factor = std::exp(-params_.altitude_km / 1000.0);
    double solar_factor = 1.0 + (params_.solar_activity - 1.0) * 0.2;
    
    // Adjust for South Atlantic Anomaly
    double saa_factor = params_.inside_saa ? 10.0 : 1.0;
    
    // Base flux in particles/cm²/s
    return 1.0e4 * altitude_factor * solar_factor * saa_factor;
}

double RadiationSimulator::calculateMeanLET() const {
    // Mean LET calculation based on environment
    double base_let = 10.0; // Base LET in MeV-cm²/mg
    
    // Adjust for altitude (higher altitude = higher LET)
    double altitude_factor = 1.0 + (params_.altitude_km / 1000.0) * 0.1;
    
    // Adjust for shielding
    double shielding_factor = std::exp(-params_.shielding_thickness_mm / 10.0);
    
    return base_let * altitude_factor * shielding_factor;
}

double RadiationSimulator::calculateLETSigma() const {
    // LET distribution width
    return 0.5; // Log-normal distribution sigma
}

double RadiationSimulator::calculateFluxShape() const {
    // Shape parameter for gamma distribution
    return 2.0;
}

double RadiationSimulator::calculateFluxScale(double base_flux) const {
    // Scale parameter for gamma distribution
    return base_flux / calculateFluxShape();
}

double RadiationSimulator::calculateEnergyShape() const {
    // Shape parameter for Weibull distribution
    return 1.5;
}

double RadiationSimulator::calculateEnergyScale() const {
    // Scale parameter for Weibull distribution
    return 50.0; // MeV
}

std::vector<RadiationEvent> RadiationSimulator::simulateEffects(
    const uint8_t* memory,
    size_t size,
    std::chrono::milliseconds duration) {
    
    std::vector<RadiationEvent> events;
    
    // Calculate number of particles based on flux and duration
    double flux = flux_dist_(rng_);
    double area = 1.0; // cm²
    double time = duration.count() / 1000.0; // seconds
    int num_particles = static_cast<int>(flux * area * time);
    
    // Generate radiation events
    for (int i = 0; i < num_particles; ++i) {
        RadiationEvent event;
        
        // Generate particle properties
        event.let = let_dist_(rng_);
        event.energy = energy_dist_(rng_);
        event.angle = angle_dist_(rng_);
        
        // Calculate impact location
        event.location = calculateImpactLocation(size);
        
        // Calculate error probability based on LET and energy
        event.error_probability = calculateErrorProbability(event.let, event.energy);
        
        // Determine if error occurs
        if (std::uniform_real_distribution<double>(0, 1)(rng_) < event.error_probability) {
            event.error_type = determineErrorType(event.let);
            event.error_magnitude = calculateErrorMagnitude(event.let, event.energy);
            events.push_back(event);
        }
    }
    
    return events;
}

size_t RadiationSimulator::calculateImpactLocation(size_t memory_size) const {
    // Calculate impact location in memory
    std::uniform_int_distribution<size_t> loc_dist(0, memory_size - 1);
    return loc_dist(rng_);
}

double RadiationSimulator::calculateErrorProbability(double let, double energy) const {
    // Error probability calculation based on LET and energy
    double let_factor = std::exp(-let / 50.0);
    double energy_factor = std::exp(-energy / 100.0);
    
    // Base probability
    double base_prob = 0.1;
    
    // Adjust for shielding
    double shielding_factor = std::exp(-params_.shielding_thickness_mm / 10.0);
    
    return base_prob * let_factor * energy_factor * shielding_factor;
}

ErrorType RadiationSimulator::determineErrorType(double let) const {
    // Determine error type based on LET
    if (let < 20.0) {
        return ErrorType::SINGLE_BIT;
    } else if (let < 50.0) {
        return ErrorType::MULTI_BIT;
    } else {
        return ErrorType::BLOCK;
    }
}

double RadiationSimulator::calculateErrorMagnitude(double let, double energy) const {
    // Calculate error magnitude based on LET and energy
    double base_magnitude = 1.0;
    
    // Adjust for LET
    double let_factor = std::log10(let) / 2.0;
    
    // Adjust for energy
    double energy_factor = energy / 100.0;
    
    return base_magnitude * let_factor * energy_factor;
}

} // namespace testing
} // namespace rad_ml 