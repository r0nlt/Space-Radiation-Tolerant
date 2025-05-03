#pragma once

#include <random>
#include <chrono>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <string>
#include <map>

namespace rad_ml {
namespace testing {

/**
 * @brief Physics-based Space Radiation Simulator
 * 
 * An enhanced radiation simulator that models space radiation effects
 * with realistic physics-based models including particle energy spectra,
 * linear energy transfer (LET), and mission-specific environments.
 */
class PhysicsRadiationSimulator {
public:
    /**
     * @brief Particle types in space radiation
     */
    enum class ParticleType {
        PROTON,
        ELECTRON,
        HEAVY_ION,
        ALPHA,
        NEUTRON
    };
    
    /**
     * @brief Radiation effect types
     */
    enum class RadiationEffectType {
        SINGLE_BIT_FLIP,       // Single Event Upset (SEU)
        MULTI_BIT_UPSET,       // Multiple Bit Upset (MBU)
        SINGLE_EVENT_LATCHUP,  // Single Event Latchup (SEL)
        SINGLE_EVENT_TRANSIENT // Single Event Transient (SET)
    };
    
    /**
     * @brief Radiation environment parameters with physics-based modeling
     */
    struct EnvironmentParams {
        // Basic parameters
        std::string mission_name;        ///< Name of the mission
        double altitude_km;              ///< Orbital altitude in kilometers
        double inclination_deg;          ///< Orbital inclination in degrees
        double solar_activity;           ///< Solar activity level (0-10)
        double shielding_thickness_mm;   ///< Spacecraft shielding in mm
        bool inside_saa;                 ///< Whether inside South Atlantic Anomaly
        
        // Enhanced physics-based parameters
        double solar_cycle_phase;        ///< Phase of solar cycle (0-1)
        double gcr_modulation;           ///< Galactic cosmic ray modulation potential
        std::map<ParticleType, double> 
            particle_flux_ratios;        ///< Relative fluxes of different particles
        double magnetosphere_strength;   ///< Earth's magnetosphere strength
        double solar_flare_probability;  ///< Probability of a solar flare event
    };
    
    /**
     * @brief Detailed radiation event information
     */
    struct RadiationEvent {
        ParticleType particle_type;      ///< Type of particle
        RadiationEffectType type;        ///< Type of radiation effect
        double energy_mev;               ///< Particle energy in MeV
        double let_mev_cm2_g;            ///< Linear Energy Transfer
        size_t memory_offset;            ///< Offset in memory
        size_t bits_affected;            ///< Number of bits affected
        std::string description;         ///< Human-readable description
    };
    
    /**
     * @brief Event rates for different radiation effects
     */
    struct EventRates {
        double single_bit_flip_rate;    ///< SEU rate (events/second)
        double multi_bit_upset_rate;    ///< MBU rate (events/second)
        double single_event_latchup_rate; ///< SEL rate (events/second)
        double single_event_transient_rate; ///< SET rate (events/second)
        double total_rate;              ///< Total event rate (events/second)
    };
    
    /**
     * @brief Particle energy spectrum model
     */
    struct EnergySpectrum {
        std::function<double(double)> distribution;   ///< Energy distribution function
        double min_energy_mev;                        ///< Minimum energy in MeV
        double max_energy_mev;                        ///< Maximum energy in MeV
        double peak_energy_mev;                       ///< Peak energy in MeV
    };
    
    /**
     * @brief Constructor with environment parameters
     * 
     * @param params Space environment parameters
     */
    explicit PhysicsRadiationSimulator(const EnvironmentParams& params)
        : env_params_(params), random_engine_(std::random_device{}()) {
        initialize();
    }
    
    /**
     * @brief Get predefined environment parameters for common missions
     * 
     * @param mission_name Mission name (e.g., "LEO", "GEO", "MARS")
     * @return Environment parameters for the specified mission
     */
    static EnvironmentParams getMissionEnvironment(const std::string& mission_name) {
        EnvironmentParams params;
        params.mission_name = mission_name;
        
        // Set common baseline parameters
        params.solar_activity = 3.0;
        params.inside_saa = false;
        params.solar_cycle_phase = 0.5;
        params.solar_flare_probability = 0.01;
        
        // Initialize particle flux ratios
        params.particle_flux_ratios[ParticleType::PROTON] = 0.85;
        params.particle_flux_ratios[ParticleType::ELECTRON] = 0.10;
        params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.02;
        params.particle_flux_ratios[ParticleType::ALPHA] = 0.025;
        params.particle_flux_ratios[ParticleType::NEUTRON] = 0.005;
        
        // Set mission-specific parameters
        if (mission_name == "LEO" || mission_name == "LEO_EARTH_OBSERVATION") {
            params.altitude_km = 500.0;
            params.inclination_deg = 45.0;
            params.shielding_thickness_mm = 5.0;
            params.magnetosphere_strength = 1.0;
            params.gcr_modulation = 0.3;
        } 
        else if (mission_name == "ISS") {
            params.altitude_km = 420.0;
            params.inclination_deg = 51.6;
            params.shielding_thickness_mm = 10.0;
            params.magnetosphere_strength = 1.0;
            params.gcr_modulation = 0.3;
        }
        else if (mission_name == "GEO" || mission_name == "GEOSTATIONARY") {
            params.altitude_km = 35786.0;
            params.inclination_deg = 0.0;
            params.shielding_thickness_mm = 5.0;
            params.magnetosphere_strength = 0.4;
            params.gcr_modulation = 0.5;
            params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.03;
        }
        else if (mission_name == "LUNAR" || mission_name == "MOON") {
            params.altitude_km = 384400.0;
            params.inclination_deg = 0.0;
            params.shielding_thickness_mm = 3.0;
            params.magnetosphere_strength = 0.01;
            params.gcr_modulation = 0.8;
            params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.05;
            params.particle_flux_ratios[ParticleType::NEUTRON] = 0.01;
        }
        else if (mission_name == "MARS") {
            params.altitude_km = 0.0;
            params.inclination_deg = 0.0;
            params.shielding_thickness_mm = 3.0;
            params.magnetosphere_strength = 0.0;
            params.gcr_modulation = 0.9;
            params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.07;
            params.particle_flux_ratios[ParticleType::NEUTRON] = 0.02;
        }
        else if (mission_name == "JUPITER") {
            params.altitude_km = 0.0;
            params.inclination_deg = 0.0;
            params.shielding_thickness_mm = 20.0;
            params.magnetosphere_strength = 0.0;
            params.gcr_modulation = 1.0;
            params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.1;
            params.particle_flux_ratios[ParticleType::ELECTRON] = 0.2;
        }
        else {
            // Default to generic space environment
            params.altitude_km = 1000.0;
            params.inclination_deg = 30.0;
            params.shielding_thickness_mm = 5.0;
            params.magnetosphere_strength = 0.5;
            params.gcr_modulation = 0.5;
        }
        
        return params;
    }
    
    /**
     * @brief Get the current simulation environment
     * 
     * @return Current environment parameters
     */
    EnvironmentParams getSimulationEnvironment() const {
        return env_params_;
    }
    
    /**
     * @brief Simulate radiation effects on memory
     * 
     * @param memory Pointer to memory region
     * @param size Size of memory region in bytes
     * @param duration Simulation duration
     * @return Vector of radiation events that occurred
     */
    template <typename T>
    std::vector<RadiationEvent> simulateEffects(
        T* memory, 
        size_t size, 
        std::chrono::milliseconds duration) {
        
        std::vector<RadiationEvent> events;
        
        // Calculate expected events based on duration and environment
        double seconds = duration.count() / 1000.0;
        double expected_events = calculateTotalEventRate() * seconds;
        
        // Generate actual number of events using Poisson distribution
        std::poisson_distribution<int> poisson(expected_events);
        int num_events = poisson(random_engine_);
        
        // Generate each event
        for (int i = 0; i < num_events; ++i) {
            events.push_back(generateRadiationEvent(memory, size));
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
        initialize(); // Reinitialize with new parameters
    }
    
    /**
     * @brief Get calculated event rates for current environment
     * 
     * @return Event rates structure
     */
    EventRates getEventRates() const {
        EventRates rates;
        rates.single_bit_flip_rate = calculateSeuRate();
        rates.multi_bit_upset_rate = calculateMbuRate();
        rates.single_event_latchup_rate = calculateSelRate();
        rates.single_event_transient_rate = calculateSetRate();
        rates.total_rate = rates.single_bit_flip_rate + 
                           rates.multi_bit_upset_rate + 
                           rates.single_event_latchup_rate + 
                           rates.single_event_transient_rate;
        return rates;
    }
    
    /**
     * @brief Get human-readable description of current environment
     * 
     * @return Formatted description string
     */
    std::string getEnvironmentDescription() const {
        std::string description = "Physics-Based Radiation Environment:\n";
        description += "  Mission: " + env_params_.mission_name + "\n";
        description += "  Altitude: " + std::to_string(env_params_.altitude_km) + " km\n";
        
        if (env_params_.altitude_km < 40000) {
            description += "  Inclination: " + std::to_string(env_params_.inclination_deg) + " degrees\n";
            description += "  Inside SAA: " + std::string(env_params_.inside_saa ? "Yes" : "No") + "\n";
        }
        
        description += "  Solar Activity: " + std::to_string(env_params_.solar_activity) + "/10\n";
        description += "  Shielding: " + std::to_string(env_params_.shielding_thickness_mm) + " mm\n";
        description += "  Solar Cycle Phase: " + std::to_string(env_params_.solar_cycle_phase) + "\n";
        description += "  GCR Modulation: " + std::to_string(env_params_.gcr_modulation) + "\n";
        
        description += "\nParticle Flux Ratios:\n";
        for (const auto& [type, ratio] : env_params_.particle_flux_ratios) {
            description += "  " + getParticleTypeName(type) + ": " + 
                          std::to_string(ratio * 100.0) + "%\n";
        }
        
        const auto rates = getEventRates();
        description += "\nEvent Rates (per second):\n";
        description += "  Total: " + std::to_string(rates.total_rate) + "\n";
        description += "  SEU: " + std::to_string(rates.single_bit_flip_rate) + "\n";
        description += "  MBU: " + std::to_string(rates.multi_bit_upset_rate) + "\n";
        description += "  SEL: " + std::to_string(rates.single_event_latchup_rate) + "\n";
        description += "  SET: " + std::to_string(rates.single_event_transient_rate) + "\n";
        
        return description;
    }
    
private:
    EnvironmentParams env_params_;
    std::mt19937 random_engine_;
    
    // Energy spectra for different particle types
    std::map<ParticleType, EnergySpectrum> energy_spectra_;
    
    // LET distributions for different particle types
    std::map<ParticleType, std::function<double(double)>> let_distributions_;
    
    /**
     * @brief Initialize internal state based on environment parameters
     */
    void initialize() {
        // Initialize energy spectra for different particle types
        initializeEnergySpectra();
        
        // Initialize LET distributions
        initializeLetDistributions();
    }
    
    /**
     * @brief Initialize energy spectra for different particle types
     */
    void initializeEnergySpectra() {
        // Proton energy spectrum
        energy_spectra_[ParticleType::PROTON] = {
            // Log-normal distribution with parameters dependent on environment
            [this](double e) -> double {
                double mu = 1.5 + 0.5 * env_params_.solar_activity / 10.0;
                double sigma = 0.8;
                if (e <= 0) return 0.0;
                return (1.0 / (e * sigma * std::sqrt(2 * M_PI))) * 
                       std::exp(-std::pow(std::log(e) - mu, 2) / (2 * sigma * sigma));
            },
            0.1,   // Min energy (MeV)
            1000.0, // Max energy (MeV)
            5.0     // Peak energy (MeV)
        };
        
        // Electron energy spectrum
        energy_spectra_[ParticleType::ELECTRON] = {
            // Power law distribution
            [this](double e) -> double {
                double index = 1.5 + 0.5 * env_params_.solar_activity / 10.0;
                if (e <= 0) return 0.0;
                return std::pow(e, -index);
            },
            0.01,   // Min energy (MeV)
            10.0,   // Max energy (MeV)
            0.1     // Peak energy (MeV)
        };
        
        // Heavy ion energy spectrum
        energy_spectra_[ParticleType::HEAVY_ION] = {
            // Power law with exponential cutoff
            [this](double e) -> double {
                double index = 1.0 + 0.3 * env_params_.gcr_modulation;
                double cutoff = 100.0 + 900.0 * env_params_.gcr_modulation;
                if (e <= 0) return 0.0;
                return std::pow(e, -index) * std::exp(-e / cutoff);
            },
            10.0,    // Min energy (MeV)
            10000.0, // Max energy (MeV)
            100.0    // Peak energy (MeV)
        };
        
        // Alpha particle energy spectrum
        energy_spectra_[ParticleType::ALPHA] = {
            // Similar to heavy ions but different parameters
            [this](double e) -> double {
                double index = 1.2 + 0.4 * env_params_.gcr_modulation;
                double cutoff = 200.0 + 800.0 * env_params_.gcr_modulation;
                if (e <= 0) return 0.0;
                return std::pow(e, -index) * std::exp(-e / cutoff);
            },
            5.0,     // Min energy (MeV)
            2000.0,  // Max energy (MeV)
            50.0     // Peak energy (MeV)
        };
        
        // Neutron energy spectrum
        energy_spectra_[ParticleType::NEUTRON] = {
            // Watt spectrum
            [this](double e) -> double {
                double a = 0.5 + 0.3 * env_params_.solar_activity / 10.0;
                double b = 2.0;
                if (e <= 0) return 0.0;
                return std::sqrt(e) * std::exp(-e / a) * std::sinh(std::sqrt(b * e));
            },
            0.1,    // Min energy (MeV)
            500.0,  // Max energy (MeV)
            1.0     // Peak energy (MeV)
        };
    }
    
    /**
     * @brief Initialize Linear Energy Transfer (LET) distributions
     */
    void initializeLetDistributions() {
        // Proton LET distribution
        let_distributions_[ParticleType::PROTON] = [](double energy) -> double {
            // Simplified Bethe-Bloch formula for protons
            if (energy <= 0) return 0.0;
            double beta2 = 1.0 - 1.0 / std::pow(1.0 + energy / 938.0, 2);
            return 0.1 * (1.0 / beta2) * std::log(100 * energy);
        };
        
        // Electron LET distribution
        let_distributions_[ParticleType::ELECTRON] = [](double energy) -> double {
            // Simplified model for electrons (very low LET)
            if (energy <= 0) return 0.0;
            return 0.001 * std::log(1.0 + energy);
        };
        
        // Heavy ion LET distribution
        let_distributions_[ParticleType::HEAVY_ION] = [](double energy) -> double {
            // Simplified model for heavy ions (high LET)
            if (energy <= 0) return 0.0;
            double z_effective = 20.0; // Effective charge
            double beta2 = 1.0 - 1.0 / std::pow(1.0 + energy / 10000.0, 2);
            return 10.0 * std::pow(z_effective, 2) / beta2;
        };
        
        // Alpha particle LET distribution
        let_distributions_[ParticleType::ALPHA] = [](double energy) -> double {
            // Simplified model for alpha particles
            if (energy <= 0) return 0.0;
            double beta2 = 1.0 - 1.0 / std::pow(1.0 + energy / 3727.0, 2);
            return 0.8 * (1.0 / beta2) * std::log(10 * energy);
        };
        
        // Neutron LET distribution
        let_distributions_[ParticleType::NEUTRON] = [](double energy) -> double {
            // Neutrons don't directly ionize but produce secondary particles
            // This is a very simplified model
            if (energy <= 0) return 0.0;
            return 0.05 * std::sqrt(energy);
        };
    }
    
    /**
     * @brief Calculate total event rate based on environment
     * 
     * @return Event rate in events per second
     */
    double calculateTotalEventRate() const {
        // Base rate depends on environment
        double base_rate = calculateBaseRate();
        
        // Distribute to different event types
        double seu_rate = calculateSeuRate();
        double mbu_rate = calculateMbuRate();
        double sel_rate = calculateSelRate();
        double set_rate = calculateSetRate();
        
        return seu_rate + mbu_rate + sel_rate + set_rate;
    }
    
    /**
     * @brief Calculate Single Event Upset (SEU) rate
     * 
     * @return SEU rate in events per second
     */
    double calculateSeuRate() const {
        double base_rate = calculateBaseRate();
        return base_rate * 0.80; // 80% of events are SEUs
    }
    
    /**
     * @brief Calculate Multiple Bit Upset (MBU) rate
     * 
     * @return MBU rate in events per second
     */
    double calculateMbuRate() const {
        double base_rate = calculateBaseRate();
        return base_rate * 0.15; // 15% of events are MBUs
    }
    
    /**
     * @brief Calculate Single Event Latchup (SEL) rate
     * 
     * @return SEL rate in events per second
     */
    double calculateSelRate() const {
        double base_rate = calculateBaseRate();
        return base_rate * 0.03; // 3% of events are SELs
    }
    
    /**
     * @brief Calculate Single Event Transient (SET) rate
     * 
     * @return SET rate in events per second
     */
    double calculateSetRate() const {
        double base_rate = calculateBaseRate();
        return base_rate * 0.02; // 2% of events are SETs
    }
    
    /**
     * @brief Calculate base event rate based on environment
     * 
     * @return Base event rate in events per second
     */
    double calculateBaseRate() const {
        // Start with a base rate (events per second)
        double rate = 0.001;
        
        // Apply mission-specific scaling
        if (env_params_.mission_name.find("MARS") != std::string::npos) {
            rate = 0.002;
        }
        else if (env_params_.mission_name.find("LUNAR") != std::string::npos ||
                 env_params_.mission_name.find("MOON") != std::string::npos) {
            rate = 0.003;
        }
        else if (env_params_.mission_name.find("JUPITER") != std::string::npos) {
            rate = 0.01;
        }
        else {
            // Earth orbit scaling based on altitude
            if (env_params_.altitude_km <= 1000) {
                // LEO
                rate *= 1.0 + (env_params_.altitude_km / 1000.0);
            } 
            else if (env_params_.altitude_km <= 36000) {
                // MEO to GEO
                rate *= 2.0 + (env_params_.altitude_km / 10000.0);
            } 
            else {
                // Beyond GEO
                rate *= 5.0;
            }
            
            // Inclination effects (polar orbits have higher exposure)
            rate *= 1.0 + (env_params_.inclination_deg / 90.0);
            
            // Magnetosphere protection
            rate *= (1.0 - 0.9 * env_params_.magnetosphere_strength);
        }
        
        // Common scaling factors for all environments
        
        // Solar activity
        rate *= 1.0 + (env_params_.solar_activity / 5.0);
        
        // Solar cycle phase (solar maximum increases radiation)
        rate *= 1.0 + std::sin(2 * M_PI * env_params_.solar_cycle_phase) * 0.3;
        
        // GCR modulation (higher values mean more cosmic rays)
        rate *= 1.0 + env_params_.gcr_modulation;
        
        // South Atlantic Anomaly
        if (env_params_.inside_saa) {
            rate *= 10.0;
        }
        
        // Solar flare probability
        if (std::uniform_real_distribution<double>(0.0, 1.0)(random_engine_) < 
            env_params_.solar_flare_probability) {
            rate *= 100.0; // Solar flare dramatically increases radiation
        }
        
        // Shielding reduces rate exponentially
        rate *= std::exp(-env_params_.shielding_thickness_mm / 10.0);
        
        return rate;
    }
    
    /**
     * @brief Generate a single radiation event
     * 
     * @param memory Pointer to memory region
     * @param size Size of memory region in bytes
     * @return Details of the generated radiation event
     */
    template <typename T>
    RadiationEvent generateRadiationEvent(T* memory, size_t size) {
        RadiationEvent event;
        
        // Choose particle type based on flux ratios
        event.particle_type = selectParticleType();
        
        // Sample particle energy from appropriate distribution
        event.energy_mev = sampleParticleEnergy(event.particle_type);
        
        // Calculate LET from particle type and energy
        event.let_mev_cm2_g = calculateLET(event.particle_type, event.energy_mev);
        
        // Choose random location in memory
        std::uniform_int_distribution<size_t> loc_dist(0, size - 1);
        event.memory_offset = loc_dist(random_engine_);
        
        // Determine effect type based on particle properties and LET
        double seu_threshold = 0.1;  // LET threshold for SEU
        double mbu_threshold = 5.0;  // LET threshold for MBU
        double sel_threshold = 15.0; // LET threshold for SEL
        
        uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(memory) + event.memory_offset;
        
        // Effect depends on particle LET and random chance
        std::uniform_real_distribution<double> effect_dist(0.0, 1.0);
        double effect_rnd = effect_dist(random_engine_);
        
        if (event.let_mev_cm2_g >= sel_threshold && effect_rnd < 0.15) {
            // Single Event Latchup (SEL)
            event.type = RadiationEffectType::SINGLE_EVENT_LATCHUP;
            
            // Select random bit to get stuck
            std::uniform_int_distribution<int> bit_dist(0, 7);
            int bit = bit_dist(random_engine_);
            
            // Determine if stuck at 0 or 1
            bool stuck_at_1 = effect_dist(random_engine_) < 0.5;
            
            if (stuck_at_1) {
                *byte_ptr |= (1 << bit);  // Set bit to 1
            } else {
                *byte_ptr &= ~(1 << bit); // Clear bit to 0
            }
            
            event.bits_affected = 1;
            event.description = "Single Event Latchup at offset " + 
                               std::to_string(event.memory_offset) +
                               ", bit " + std::to_string(bit) + 
                               " stuck at " + (stuck_at_1 ? "1" : "0");
        }
        else if (event.let_mev_cm2_g >= mbu_threshold && effect_rnd < 0.6) {
            // Multiple Bit Upset (MBU)
            event.type = RadiationEffectType::MULTI_BIT_UPSET;
            
            // Number of bits depends on LET
            int num_bits = 2 + static_cast<int>(event.let_mev_cm2_g / 10.0);
            num_bits = std::min(num_bits, 5); // Cap at 5 bits
            
            // Choose starting bit
            std::uniform_int_distribution<int> start_bit_dist(0, 8 - num_bits);
            int start_bit = start_bit_dist(random_engine_);
            
            // Flip multiple adjacent bits
            uint8_t mask = 0;
            for (int i = 0; i < num_bits; ++i) {
                mask |= (1 << (start_bit + i));
            }
            *byte_ptr ^= mask;
            
            event.bits_affected = num_bits;
            event.description = "Multiple Bit Upset at offset " + 
                               std::to_string(event.memory_offset) +
                               ", " + std::to_string(num_bits) + 
                               " adjacent bits starting at bit " + 
                               std::to_string(start_bit);
        }
        else if (event.let_mev_cm2_g >= seu_threshold || 
                (event.particle_type == ParticleType::ELECTRON && effect_rnd < 0.1)) {
            // Single Bit Flip (SEU)
            event.type = RadiationEffectType::SINGLE_BIT_FLIP;
            
            std::uniform_int_distribution<int> bit_dist(0, 7);
            int bit = bit_dist(random_engine_);
            
            // Flip the bit
            *byte_ptr ^= (1 << bit);
            
            event.bits_affected = 1;
            event.description = "Single Bit Flip at offset " + 
                               std::to_string(event.memory_offset) +
                               ", bit " + std::to_string(bit);
        }
        else {
            // Single Event Transient (SET) - temporary disturbance
            event.type = RadiationEffectType::SINGLE_EVENT_TRANSIENT;
            
            uint8_t original = *byte_ptr;
            
            // Replace with random value temporarily
            std::uniform_int_distribution<int> value_dist(0, 255);
            *byte_ptr = static_cast<uint8_t>(value_dist(random_engine_));
            
            event.bits_affected = 8; // Potentially all bits in byte
            event.description = "Single Event Transient at offset " + 
                               std::to_string(event.memory_offset) +
                               ", byte temporarily changed from " + 
                               std::to_string(original) + " to " + 
                               std::to_string(*byte_ptr);
                               
            // For a real SET, the value would recover automatically
            // For simulation, we note this is transient
            event.description += " (transient effect - would recover automatically)";
        }
        
        return event;
    }
    
    /**
     * @brief Select a particle type based on flux ratios
     * 
     * @return Selected particle type
     */
    ParticleType selectParticleType() const {
        // Create cumulative distribution from flux ratios
        std::vector<double> cumulative_dist;
        std::vector<ParticleType> particle_types;
        
        double total = 0.0;
        for (const auto& [type, ratio] : env_params_.particle_flux_ratios) {
            total += ratio;
            cumulative_dist.push_back(total);
            particle_types.push_back(type);
        }
        
        // Normalize
        for (auto& val : cumulative_dist) {
            val /= total;
        }
        
        // Sample from distribution
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double rnd = dist(random_engine_);
        
        for (size_t i = 0; i < cumulative_dist.size(); ++i) {
            if (rnd <= cumulative_dist[i]) {
                return particle_types[i];
            }
        }
        
        // Default to protons if something goes wrong
        return ParticleType::PROTON;
    }
    
    /**
     * @brief Sample a particle energy from the appropriate distribution
     * 
     * @param type Particle type
     * @return Sampled energy in MeV
     */
    double sampleParticleEnergy(ParticleType type) const {
        auto it = energy_spectra_.find(type);
        if (it == energy_spectra_.end()) {
            return 10.0; // Default energy if spectrum not found
        }
        
        const auto& spectrum = it->second;
        
        // Rejection sampling from the distribution
        std::uniform_real_distribution<double> e_dist(
            spectrum.min_energy_mev, spectrum.max_energy_mev);
        std::uniform_real_distribution<double> y_dist(0.0, 1.0);
        
        // Find approximate max value of distribution for scaling
        double max_val = 0.0;
        for (double e = spectrum.min_energy_mev; 
             e <= spectrum.max_energy_mev; 
             e += (spectrum.max_energy_mev - spectrum.min_energy_mev) / 100.0) {
            max_val = std::max(max_val, spectrum.distribution(e));
        }
        
        // Add 10% margin to avoid undersampling
        max_val *= 1.1;
        
        // Rejection sampling
        while (true) {
            double e = e_dist(random_engine_);
            double y = y_dist(random_engine_) * max_val;
            
            if (y <= spectrum.distribution(e)) {
                return e;
            }
        }
    }
    
    /**
     * @brief Calculate Linear Energy Transfer for a particle
     * 
     * @param type Particle type
     * @param energy Energy in MeV
     * @return LET in MeV·cm²/g
     */
    double calculateLET(ParticleType type, double energy) const {
        auto it = let_distributions_.find(type);
        if (it == let_distributions_.end()) {
            return 0.1; // Default LET if distribution not found
        }
        
        return it->second(energy);
    }
    
    /**
     * @brief Get human-readable name for a particle type
     * 
     * @param type Particle type
     * @return Name string
     */
    std::string getParticleTypeName(ParticleType type) const {
        switch (type) {
            case ParticleType::PROTON:
                return "Proton";
            case ParticleType::ELECTRON:
                return "Electron";
            case ParticleType::HEAVY_ION:
                return "Heavy Ion";
            case ParticleType::ALPHA:
                return "Alpha Particle";
            case ParticleType::NEUTRON:
                return "Neutron";
            default:
                return "Unknown";
        }
    }
};

} // namespace testing
} // namespace rad_ml 