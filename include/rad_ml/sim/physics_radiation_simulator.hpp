#pragma once

#include <random>
#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <cmath>

namespace rad_ml {
namespace sim {

/**
 * @brief Space radiation environment model
 * 
 * Based on NASA's AE9/AP9 model and ESA's SPENVIS for radiation modeling
 */
enum class RadiationEnvironment {
    LEO,                // Low Earth Orbit (400-600km)
    MEO,                // Medium Earth Orbit (like GPS satellites)
    GEO,                // Geosynchronous Earth Orbit
    LUNAR,              // Lunar vicinity
    MARS_ORBIT,         // Mars orbit
    MARS_SURFACE,       // Mars surface (with atmosphere shielding)
    JUPITER,            // Jupiter radiation belts
    EUROPA,             // Near Europa (extreme radiation environment)
    INTERPLANETARY,     // Deep space, interplanetary transit
    SOLAR_MINIMUM,      // Solar minimum conditions (higher GCR)
    SOLAR_MAXIMUM,      // Solar maximum conditions (higher SPE probability)
    SOLAR_STORM         // Active solar storm (extreme conditions)
};

/**
 * @brief Types of radiation sources with different damage mechanisms
 * 
 * Based on experimental data showing different charge collection efficiencies
 * for various radiation sources in semiconductor devices
 */
enum class RadiationSourceType {
    CO60_GAMMA,           // Co-60 gamma rays (high charge collection efficiency)
    ELECTRONS_12MEV,      // 12-MeV electrons (high efficiency, similar to Co-60)
    ELECTRONS_5KEV,       // 5-keV electrons (medium efficiency)
    XRAYS_10KEV,          // 10-keV X-rays (medium efficiency)
    PROTONS_700KEV,       // 700-keV protons (poor charge collection)
    ALPHA_2MEV,           // 2-MeV alpha particles (very poor collection)
    SPACE_MIXED_FIELD     // Mixed space environment (weighted average)
};

/**
 * @brief Types of radiation effects in space
 */
enum class RadiationEffectType {
    SEU,                // Single Event Upset
    MBU,                // Multiple Bit Upset
    SEL,                // Single Event Latchup
    SET,                // Single Event Transient
    SEFI,               // Single Event Functional Interrupt
    TID_STUCK_BIT,      // Total Ionizing Dose induced stuck bit
    TID_THRESHOLD_SHIFT // TID induced threshold voltage shift
};

/**
 * @brief Configuration for a radiation event
 */
struct RadiationEffect {
    RadiationEffectType type;
    double probability;     // Base probability per bit per day
    size_t min_bits;        // Minimum bits affected
    size_t max_bits;        // Maximum bits affected
    bool is_persistent;     // Whether effects persist after reboot/repair
    double recovery_prob;   // Probability of spontaneous recovery (per day)
    
    // Added default constructor to fix map initialization error
    RadiationEffect() : type(RadiationEffectType::SEU),
                     probability(0.0),
                     min_bits(0),
                     max_bits(0),
                     is_persistent(false),
                     recovery_prob(0.0) {}
    
    // Constructor with typical values derived from space radiation studies
    RadiationEffect(RadiationEffectType t) : type(t) {
        // Set defaults based on radiation effect type
        switch (type) {
            case RadiationEffectType::SEU:
                probability = 1e-7;       // ~1 per 10M bits per day (typical LEO)
                min_bits = 1;
                max_bits = 1;
                is_persistent = false;
                recovery_prob = 1.0;      // Recovers immediately with power cycle
                break;
                
            case RadiationEffectType::MBU:
                probability = 2e-8;       // ~20% of SEUs are MBUs
                min_bits = 2;
                max_bits = 8;             // Typical for modern memory
                is_persistent = false;
                recovery_prob = 1.0;      // Recovers immediately with power cycle
                break;
                
            case RadiationEffectType::SEL:
                probability = 5e-9;       // Based on ESA JUICE radiation specs
                min_bits = 1;
                max_bits = 1024;          // Can affect entire regions
                is_persistent = true;     // Requires power cycle
                recovery_prob = 0.0;      // Only recovers with intervention
                break;
                
            case RadiationEffectType::SET:
                probability = 2e-7;       // More common in logic than memory
                min_bits = 1;
                max_bits = 1;
                is_persistent = false;
                recovery_prob = 1.0;      // Transient by definition
                break;
                
            case RadiationEffectType::SEFI:
                probability = 1e-9;       // Based on NASA testing data
                min_bits = 1;
                max_bits = 1024 * 1024;   // Can affect entire systems
                is_persistent = true;     // Often requires power cycle
                recovery_prob = 0.0;      // Only recovers with intervention
                break;
                
            case RadiationEffectType::TID_STUCK_BIT:
                probability = 5e-10;      // Accumulates over mission lifetime
                min_bits = 1;
                max_bits = 1;
                is_persistent = true;     // Permanent damage
                recovery_prob = 0.0;      // No spontaneous recovery
                break;
                
            case RadiationEffectType::TID_THRESHOLD_SHIFT:
                probability = 1e-9;       // Based on MESSENGER data
                min_bits = 1;
                max_bits = 1024;          // Affects regions
                is_persistent = true;     // Permanent damage
                recovery_prob = 0.0;      // No spontaneous recovery
                break;
        }
    }
};

/**
 * @brief Charge collection parameters for different radiation sources
 * 
 * Models the fractional yield (charge collection efficiency) as a function
 * of electric field strength for different radiation sources, based on
 * experimental data from semiconductor device testing
 */
struct ChargeCollectionParams {
    RadiationSourceType source_type;
    double electric_field_mv_per_cm;    // Electric field strength (MV/cm)
    double fractional_yield;            // Charge collection efficiency (0.0 to 1.0)
    double let_kev_cm2_per_mg;         // Linear Energy Transfer (keV·cm²/mg)
    
    /**
     * @brief Calculate fractional yield based on electric field and source type
     * 
     * Implements curves from experimental data showing how charge collection
     * efficiency varies with electric field for different radiation sources
     * 
     * @param field_mv_per_cm Electric field strength in MV/cm
     * @return Fractional yield (charge collection efficiency)
     */
    double calculate_fractional_yield(double field_mv_per_cm) const {
        switch (source_type) {
            case RadiationSourceType::CO60_GAMMA:
            case RadiationSourceType::ELECTRONS_12MEV:
                // High-energy sources: efficient charge collection
                // Saturates near 1.0 at moderate fields
                return std::min(1.0, 0.3 + 0.7 * (1.0 - std::exp(-field_mv_per_cm / 0.5)));
                
            case RadiationSourceType::ELECTRONS_5KEV:
                // Medium efficiency, more field-dependent
                return std::min(1.0, 0.1 + 0.8 * (1.0 - std::exp(-field_mv_per_cm / 1.0)));
                
            case RadiationSourceType::XRAYS_10KEV:
                // Similar to 5-keV electrons but slightly better collection
                return std::min(1.0, 0.15 + 0.75 * (1.0 - std::exp(-field_mv_per_cm / 0.8)));
                
            case RadiationSourceType::PROTONS_700KEV:
                // Poor charge collection, saturates at low efficiency (~0.35)
                return std::min(0.35, 0.05 + 0.3 * (1.0 - std::exp(-field_mv_per_cm / 2.0)));
                
            case RadiationSourceType::ALPHA_2MEV:
                // Very poor charge collection, saturates at ~0.15
                return std::min(0.15, 0.02 + 0.13 * (1.0 - std::exp(-field_mv_per_cm / 3.0)));
                
            case RadiationSourceType::SPACE_MIXED_FIELD:
                // Weighted average based on typical space radiation spectrum
                return std::min(0.8, 0.2 + 0.6 * (1.0 - std::exp(-field_mv_per_cm / 1.2)));
                
            default:
                return 0.5; // Conservative default
        }
    }
    
    /**
     * @brief Get typical LET value for the radiation source
     * 
     * @return LET in keV·cm²/mg
     */
    double get_typical_let() const {
        switch (source_type) {
            case RadiationSourceType::CO60_GAMMA:      return 0.2;   // Very low LET
            case RadiationSourceType::ELECTRONS_12MEV: return 0.25;  // Low LET
            case RadiationSourceType::ELECTRONS_5KEV:  return 2.0;   // Medium LET
            case RadiationSourceType::XRAYS_10KEV:     return 1.5;   // Medium LET
            case RadiationSourceType::PROTONS_700KEV:  return 15.0;  // High LET
            case RadiationSourceType::ALPHA_2MEV:      return 80.0;  // Very high LET
            case RadiationSourceType::SPACE_MIXED_FIELD: return 5.0; // Mixed spectrum average
            default: return 1.0;
        }
    }
};

/**
 * @brief Model of spacecraft orbit or trajectory
 */
struct SpacecraftTrajectory {
    std::vector<RadiationEnvironment> environments;
    std::vector<double> durations_days;   // Time spent in each environment
    
    // Common trajectory configurations based on NASA mission profiles
    static SpacecraftTrajectory Earth_LEO() {
        return {{RadiationEnvironment::LEO}, {365.0}}; // One year mission
    }
    
    static SpacecraftTrajectory Mars_Mission() {
        return {
            {RadiationEnvironment::LEO, 
             RadiationEnvironment::INTERPLANETARY,
             RadiationEnvironment::MARS_ORBIT,
             RadiationEnvironment::MARS_SURFACE,
             RadiationEnvironment::INTERPLANETARY,
             RadiationEnvironment::LEO},
            {10.0, 180.0, 30.0, 365.0, 180.0, 10.0}
        };
    }
    
    static SpacecraftTrajectory Europa_Mission() {
        return {
            {RadiationEnvironment::LEO,
             RadiationEnvironment::INTERPLANETARY,
             RadiationEnvironment::JUPITER,
             RadiationEnvironment::EUROPA,
             RadiationEnvironment::JUPITER,
             RadiationEnvironment::INTERPLANETARY,
             RadiationEnvironment::LEO},
            {10.0, 730.0, 60.0, 30.0, 60.0, 730.0, 10.0}
        };
    }
};

/**
 * @brief Physics-based space radiation simulator
 * 
 * Models radiation effects based on spacecraft trajectory,
 * shielding, and solar conditions. Based on NASA OLTARIS,
 * ESA SPENVIS, and AE9/AP9 radiation environment models.
 */
class PhysicsRadiationSimulator {
public:
    /**
     * @brief Create a new physics-based radiation simulator
     * 
     * @param memory_bits Total memory bits to simulate
     * @param word_size Word size in bits (typical: 32, 64)
     * @param shielding_thickness_mm Aluminum equivalent shielding in mm
     * @param trajectory Spacecraft trajectory
     */
    PhysicsRadiationSimulator(
        size_t memory_bits = 8 * 1024 * 1024,  // 1 MB default
        size_t word_size = 32,                 // 32-bit words
        double shielding_thickness_mm = 2.0,   // 2mm Al equivalent
        SpacecraftTrajectory trajectory = SpacecraftTrajectory::Earth_LEO()
    ) : memory_bits_(memory_bits),
        word_size_(word_size),
        shielding_thickness_mm_(shielding_thickness_mm),
        trajectory_(trajectory),
        current_environment_(trajectory.environments[0]),
        solar_activity_(0.5),  // Medium solar activity
        random_engine_(std::chrono::system_clock::now().time_since_epoch().count()),
        current_source_type_(RadiationSourceType::SPACE_MIXED_FIELD),
        device_electric_field_mv_per_cm_(0.0),
        environment_source_mapping_() {
        
        // Initialize radiation effects
        initialize_radiation_effects();
        
        // Calculate environment rate modifiers
        calculate_environment_modifiers();
        
        // Initialize charge collection physics
        initialize_charge_collection_mapping();
        
        // Set reasonable default electric field for typical semiconductor devices
        device_electric_field_mv_per_cm_ = 1.0; // 1 MV/cm typical for modern devices
    }
    
    /**
     * @brief Set the current radiation environment
     * 
     * @param environment The radiation environment
     */
    void set_environment(RadiationEnvironment environment) {
        current_environment_ = environment;
        // Update current source type based on new environment
        current_source_type_ = environment_source_mapping_[environment];
    }
    
    /**
     * @brief Set solar activity level
     * 
     * @param activity Activity level from 0.0 (minimum) to 1.0 (maximum)
     */
    void set_solar_activity(double activity) {
        solar_activity_ = std::max(0.0, std::min(1.0, activity));
    }
    
    /**
     * @brief Set spacecraft shielding
     * 
     * @param thickness_mm Aluminum equivalent thickness in mm
     */
    void set_shielding(double thickness_mm) {
        shielding_thickness_mm_ = thickness_mm;
        // Recalculate environment modifiers with new shielding
        calculate_environment_modifiers();
    }
    
    /**
     * @brief Set a custom spacecraft trajectory
     * 
     * @param trajectory The new trajectory
     */
    void set_trajectory(const SpacecraftTrajectory& trajectory) {
        trajectory_ = trajectory;
        current_environment_ = trajectory.environments[0];
    }
    
    /**
     * @brief Set device electric field strength
     * 
     * This affects charge collection efficiency for different radiation sources.
     * Typical values: 0.5-2.0 MV/cm for modern semiconductor devices
     * 
     * @param field_mv_per_cm Electric field strength in MV/cm
     */
    void set_device_electric_field(double field_mv_per_cm) {
        device_electric_field_mv_per_cm_ = std::max(0.0, field_mv_per_cm);
    }
    
    /**
     * @brief Get current charge collection efficiency
     * 
     * @return Fractional yield (0.0 to 1.0) for current conditions
     */
    double get_charge_collection_efficiency() const {
        ChargeCollectionParams charge_params;
        charge_params.source_type = current_source_type_;
        return charge_params.calculate_fractional_yield(device_electric_field_mv_per_cm_);
    }
    
    /**
     * @brief Simulate radiation for a specific time period
     * 
     * @param days Days to simulate
     * @return Vector of radiation effect events
     */
    std::vector<std::pair<RadiationEffectType, size_t>> simulate_period(double days) {
        std::vector<std::pair<RadiationEffectType, size_t>> effects;
        
        // Get environment rate modification
        double env_modifier = environment_modifiers_.at(current_environment_);
        
        // Solar activity modifier
        double solar_modifier = calculate_solar_modifier();
        
        // Calculate shielding effectiveness
        double shielding_factor = calculate_shielding_factor();
        
        // Combined rate modifier
        double rate_modifier = env_modifier * solar_modifier * shielding_factor;
        
        // For each radiation effect type
        for (const auto& [type, effect] : radiation_effects_) {
            // Base probability adjusted for environment and duration
            double event_probability = effect.probability * rate_modifier * days;
            
            // Expected number of events using Poisson distribution
            double expected_events = event_probability * memory_bits_;
            
            // Generate actual number using Poisson distribution
            std::poisson_distribution<size_t> poisson(expected_events);
            size_t num_events = poisson(random_engine_);
            
            // For each event, determine the number of bits affected
            for (size_t i = 0; i < num_events; ++i) {
                std::uniform_int_distribution<size_t> bit_dist(effect.min_bits, effect.max_bits);
                size_t bits_affected = bit_dist(random_engine_);
                
                effects.push_back({type, bits_affected});
            }
        }
        
        return effects;
    }
    
    /**
     * @brief Simulate effects of total ionizing dose for a mission duration
     * 
     * @param days Total mission days
     * @return Map of radiation types to number of occurrences
     */
    std::map<RadiationEffectType, size_t> simulate_mission_tid(double days) {
        std::map<RadiationEffectType, size_t> tid_effects;
        
        // Initialize TID effect types
        tid_effects[RadiationEffectType::TID_STUCK_BIT] = 0;
        tid_effects[RadiationEffectType::TID_THRESHOLD_SHIFT] = 0;
        
        // For each segment of the mission
        double days_simulated = 0.0;
        size_t current_segment = 0;
        
        while (days_simulated < days && current_segment < trajectory_.environments.size()) {
            // Calculate time spent in this segment
            double segment_days = std::min(
                trajectory_.durations_days[current_segment],
                days - days_simulated
            );
            
            // Set environment for this segment
            set_environment(trajectory_.environments[current_segment]);
            
            // Simulate TID effects for this segment
            const auto& effects = radiation_effects_;
            double env_modifier = environment_modifiers_.at(current_environment_);
            double shielding_factor = calculate_shielding_factor();
            
            // TID accumulates more linearly than SEE, simulate directly
            for (const auto& effect_entry : effects) {
                const auto& effect = effect_entry.second;
                
                // Only process TID effects
                if (effect.type == RadiationEffectType::TID_STUCK_BIT ||
                    effect.type == RadiationEffectType::TID_THRESHOLD_SHIFT) {
                    
                    // TID accumulates roughly linearly with time
                    double event_probability = effect.probability * env_modifier * 
                                             shielding_factor * segment_days;
                    
                    double expected_events = event_probability * memory_bits_;
                    
                    // Generate actual number using Poisson distribution
                    std::poisson_distribution<size_t> poisson(expected_events);
                    size_t new_events = poisson(random_engine_);
                    
                    tid_effects[effect.type] += new_events;
                }
            }
            
            days_simulated += segment_days;
            current_segment++;
        }
        
        return tid_effects;
    }
    
    /**
     * @brief Simulate radiation for a full mission
     * 
     * @return Time series of radiation events along mission
     */
    std::vector<std::map<RadiationEffectType, size_t>> simulate_mission() {
        std::vector<std::map<RadiationEffectType, size_t>> timeline;
        
        // For each segment of the mission
        for (size_t i = 0; i < trajectory_.environments.size(); ++i) {
            // Set environment
            set_environment(trajectory_.environments[i]);
            
            // Generate events for this segment
            std::map<RadiationEffectType, size_t> segment_events;
            
            // Initialize event counts
            for (const auto& [type, _] : radiation_effects_) {
                segment_events[type] = 0;
            }
            
            // Get events for this segment
            auto events = simulate_period(trajectory_.durations_days[i]);
            
            // Count events by type
            for (const auto& [type, bits] : events) {
                segment_events[type]++;
            }
            
            timeline.push_back(segment_events);
        }
        
        return timeline;
    }
    
    /**
     * @brief Get the expected error rates for the current environment
     * 
     * @return Map of error types to daily rates per Mbit
     */
    std::map<RadiationEffectType, double> get_error_rates() const {
        std::map<RadiationEffectType, double> rates;
        
        // Get environment rate modification
        double env_modifier = environment_modifiers_.at(current_environment_);
        
        // Solar activity modifier
        double solar_modifier = calculate_solar_modifier();
        
        // Calculate shielding effectiveness
        double shielding_factor = calculate_shielding_factor();
        
        // Calculate rates for each effect type
        for (const auto& [type, effect] : radiation_effects_) {
            // Errors per bit per day
            double rate = effect.probability * env_modifier * solar_modifier * shielding_factor;
            
            // Convert to errors per Mbit per day for easier reading
            rates[type] = rate * 1e6;
        }
        
        return rates;
    }
    
    /**
     * @brief Get a human-readable report of radiation environment
     * 
     * @return Environment description string
     */
    std::string get_environment_report() const {
        std::string report = "Space Radiation Environment Report\n";
        report += "================================\n";
        
        // Environment information
        report += "Current environment: " + get_environment_name(current_environment_) + "\n";
        report += "Relative radiation level: " + 
                 std::to_string(environment_modifiers_.at(current_environment_)) + "x baseline\n";
        report += "Solar activity level: " + std::to_string(solar_activity_) + 
                 " (" + (solar_activity_ < 0.3 ? "Low" : 
                         solar_activity_ > 0.7 ? "High" : "Medium") + ")\n";
        report += "Spacecraft shielding: " + std::to_string(shielding_thickness_mm_) + 
                 " mm Al-eq (reduction factor: " + 
                 std::to_string(calculate_shielding_factor()) + ")\n";
        
        // Charge collection physics
        report += "\nCharge Collection Physics:\n";
        report += "Device electric field: " + std::to_string(device_electric_field_mv_per_cm_) + " MV/cm\n";
        report += "Dominant radiation source: " + get_source_name(current_source_type_) + "\n";
        report += "Charge collection efficiency: " + 
                 std::to_string(get_charge_collection_efficiency() * 100.0) + "%\n";
        
        ChargeCollectionParams charge_params;
        charge_params.source_type = current_source_type_;
        report += "Typical LET: " + std::to_string(charge_params.get_typical_let()) + " keV⋅cm²/mg\n";
        
        // Error rates
        report += "\nExpected error rates (per Mbit per day):\n";
        auto rates = get_error_rates();
        
        for (const auto& [type, rate] : rates) {
            report += "  " + get_effect_name(type) + ": " + 
                     std::to_string(rate) + "\n";
        }
        
        return report;
    }
    
private:
    // Configuration parameters
    size_t memory_bits_;
    size_t word_size_;
    double shielding_thickness_mm_;
    SpacecraftTrajectory trajectory_;
    RadiationEnvironment current_environment_;
    double solar_activity_;
    
    // Radiation effect models
    std::map<RadiationEffectType, RadiationEffect> radiation_effects_;
    
    // Environment rate modifiers relative to baseline (LEO)
    std::map<RadiationEnvironment, double> environment_modifiers_;
    
    // Random number generation
    std::default_random_engine random_engine_;
    
    // Charge collection physics parameters
    RadiationSourceType current_source_type_;
    double device_electric_field_mv_per_cm_;
    std::map<RadiationEnvironment, RadiationSourceType> environment_source_mapping_;
    
    /**
     * @brief Initialize radiation effect models
     */
    void initialize_radiation_effects() {
        // Create radiation effects with scientifically accurate rates
        radiation_effects_[RadiationEffectType::SEU] = 
            RadiationEffect(RadiationEffectType::SEU);
            
        radiation_effects_[RadiationEffectType::MBU] = 
            RadiationEffect(RadiationEffectType::MBU);
            
        radiation_effects_[RadiationEffectType::SEL] = 
            RadiationEffect(RadiationEffectType::SEL);
            
        radiation_effects_[RadiationEffectType::SET] = 
            RadiationEffect(RadiationEffectType::SET);
            
        radiation_effects_[RadiationEffectType::SEFI] = 
            RadiationEffect(RadiationEffectType::SEFI);
            
        radiation_effects_[RadiationEffectType::TID_STUCK_BIT] = 
            RadiationEffect(RadiationEffectType::TID_STUCK_BIT);
            
        radiation_effects_[RadiationEffectType::TID_THRESHOLD_SHIFT] = 
            RadiationEffect(RadiationEffectType::TID_THRESHOLD_SHIFT);
    }
    
    /**
     * @brief Calculate modifiers for each radiation environment
     * 
     * Modifiers based on NASA AE9/AP9 and ESA SPENVIS models
     */
    void calculate_environment_modifiers() {
        // Rates relative to LEO (based on scientific space radiation models)
        environment_modifiers_[RadiationEnvironment::LEO] = 1.0;           // Baseline
        environment_modifiers_[RadiationEnvironment::MEO] = 10.0;          // South Atlantic Anomaly
        environment_modifiers_[RadiationEnvironment::GEO] = 5.0;           // Outside magnetosphere
        environment_modifiers_[RadiationEnvironment::LUNAR] = 4.0;         // No magnetic protection
        environment_modifiers_[RadiationEnvironment::MARS_ORBIT] = 3.0;    // No strong field
        environment_modifiers_[RadiationEnvironment::MARS_SURFACE] = 0.5;  // Atmosphere shields
        environment_modifiers_[RadiationEnvironment::JUPITER] = 1000.0;    // Extreme environment
        environment_modifiers_[RadiationEnvironment::EUROPA] = 2000.0;     // Europa mission estimates
        environment_modifiers_[RadiationEnvironment::INTERPLANETARY] = 3.0; // Deep space
        environment_modifiers_[RadiationEnvironment::SOLAR_MINIMUM] = 2.0; // Higher GCR
        environment_modifiers_[RadiationEnvironment::SOLAR_MAXIMUM] = 0.8; // Lower GCR
        environment_modifiers_[RadiationEnvironment::SOLAR_STORM] = 100.0; // Extreme conditions
    }
    
    /**
     * @brief Calculate modifier based on solar activity
     * 
     * @return Solar activity multiplier
     */
    double calculate_solar_modifier() const {
        // Solar storm probability increases with activity
        if (current_environment_ == RadiationEnvironment::SOLAR_STORM) {
            return 1.0; // Already factored into environment
        }
        
        // GCR rates are anti-correlated with solar activity
        // SPE rates are correlated with solar activity
        double gcr_component = 1.0 - 0.5 * solar_activity_; // 1.0 at min, 0.5 at max
        double spe_component = solar_activity_ * solar_activity_ * 5.0; // 0 at min, 5.0 at max
        
        // Combined effect depends on environment
        if (current_environment_ == RadiationEnvironment::LEO ||
            current_environment_ == RadiationEnvironment::MEO ||
            current_environment_ == RadiationEnvironment::GEO) {
            // Earth environments - more SPE protection
            return 0.7 * gcr_component + 0.3 * spe_component;
        } else {
            // Deep space - more exposure to both
            return 0.5 * gcr_component + 0.5 * spe_component;
        }
    }
    
    /**
     * @brief Calculate shielding effectiveness including charge collection physics
     * 
     * Based on aluminum equivalent shielding models from SPENVIS plus
     * charge collection efficiency for different radiation sources
     * 
     * @return Shielding attenuation factor including charge collection effects
     */
    double calculate_shielding_factor() const {
        // Parameters derived from SPENVIS aluminum shielding model
        const double reference_thickness = 2.0; // 2mm Al reference
        
        if (shielding_thickness_mm_ <= 0.0) {
            return 1.0; // No shielding
        }
        
        // Traditional shielding model
        double base_reduction = std::exp(-shielding_thickness_mm_ / reference_thickness);
        
        // TID typically follows closer to exponential attenuation
        double tid_reduction = std::pow(base_reduction, 1.2);
        
        // SEE typically requires higher energy particles, less shield-sensitive
        double see_reduction = std::pow(base_reduction, 0.7);
        
        // Combined traditional shielding effect
        double traditional_shielding = 0.3 * tid_reduction + 0.7 * see_reduction;
        
        // Add charge collection physics
        ChargeCollectionParams charge_params;
        charge_params.source_type = current_source_type_;
        
        // Calculate charge collection efficiency for current conditions
        double charge_collection_efficiency = charge_params.calculate_fractional_yield(device_electric_field_mv_per_cm_);
        
        // Different radiation sources have different penetration through shielding
        // AND different charge collection efficiencies
        double source_specific_factor = 1.0;
        switch (current_source_type_) {
            case RadiationSourceType::CO60_GAMMA:
            case RadiationSourceType::ELECTRONS_12MEV:
                // High-energy sources: good penetration, high collection efficiency
                source_specific_factor = 0.9 * traditional_shielding + 0.1 * charge_collection_efficiency;
                break;
                
            case RadiationSourceType::ELECTRONS_5KEV:
            case RadiationSourceType::XRAYS_10KEV:
                // Medium-energy sources: moderate penetration and collection
                source_specific_factor = 0.7 * traditional_shielding + 0.3 * charge_collection_efficiency;
                break;
                
            case RadiationSourceType::PROTONS_700KEV:
                // Protons: good penetration but poor charge collection
                source_specific_factor = 0.8 * traditional_shielding + 0.2 * charge_collection_efficiency;
                break;
                
            case RadiationSourceType::ALPHA_2MEV:
                // Alpha particles: poor penetration and very poor collection
                source_specific_factor = 0.5 * traditional_shielding + 0.5 * charge_collection_efficiency;
                break;
                
            case RadiationSourceType::SPACE_MIXED_FIELD:
                // Mixed field: weighted average behavior
                source_specific_factor = 0.75 * traditional_shielding + 0.25 * charge_collection_efficiency;
                break;
                
            default:
                source_specific_factor = traditional_shielding;
                break;
        }
        
        // Final shielding factor ranges from about 0.01 to 1.0
        return std::max(0.01, std::min(1.0, source_specific_factor));
    }
    
    /**
     * @brief Get string representation of environment
     * 
     * @param env Environment
     * @return String name
     */
    std::string get_environment_name(RadiationEnvironment env) const {
        switch (env) {
            case RadiationEnvironment::LEO: return "Low Earth Orbit";
            case RadiationEnvironment::MEO: return "Medium Earth Orbit";
            case RadiationEnvironment::GEO: return "Geosynchronous Earth Orbit";
            case RadiationEnvironment::LUNAR: return "Lunar Vicinity";
            case RadiationEnvironment::MARS_ORBIT: return "Mars Orbit";
            case RadiationEnvironment::MARS_SURFACE: return "Mars Surface";
            case RadiationEnvironment::JUPITER: return "Jupiter Radiation Belts";
            case RadiationEnvironment::EUROPA: return "Europa Vicinity";
            case RadiationEnvironment::INTERPLANETARY: return "Interplanetary Space";
            case RadiationEnvironment::SOLAR_MINIMUM: return "Solar Minimum";
            case RadiationEnvironment::SOLAR_MAXIMUM: return "Solar Maximum";
            case RadiationEnvironment::SOLAR_STORM: return "Solar Storm";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Get string representation of radiation effect
     * 
     * @param effect Effect type
     * @return String name
     */
    std::string get_effect_name(RadiationEffectType effect) const {
        switch (effect) {
            case RadiationEffectType::SEU: return "Single Event Upset";
            case RadiationEffectType::MBU: return "Multiple Bit Upset";
            case RadiationEffectType::SEL: return "Single Event Latchup";
            case RadiationEffectType::SET: return "Single Event Transient";
            case RadiationEffectType::SEFI: return "Single Event Functional Interrupt";
            case RadiationEffectType::TID_STUCK_BIT: return "TID Stuck Bit";
            case RadiationEffectType::TID_THRESHOLD_SHIFT: return "TID Threshold Shift";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Get string representation of radiation source type
     * 
     * @param source Source type
     * @return String name
     */
    std::string get_source_name(RadiationSourceType source) const {
        switch (source) {
            case RadiationSourceType::CO60_GAMMA: return "Co-60 Gamma Rays";
            case RadiationSourceType::ELECTRONS_12MEV: return "12-MeV Electrons";
            case RadiationSourceType::ELECTRONS_5KEV: return "5-keV Electrons";
            case RadiationSourceType::XRAYS_10KEV: return "10-keV X-rays";
            case RadiationSourceType::PROTONS_700KEV: return "700-keV Protons";
            case RadiationSourceType::ALPHA_2MEV: return "2-MeV Alpha Particles";
            case RadiationSourceType::SPACE_MIXED_FIELD: return "Mixed Space Radiation";
            default: return "Unknown Source";
        }
    }
    
    /**
     * @brief Initialize charge collection physics
     */
    void initialize_charge_collection_mapping() {
        // Map radiation environments to their dominant source types
        // Based on typical space radiation spectra for each environment
        
        environment_source_mapping_[RadiationEnvironment::LEO] = RadiationSourceType::SPACE_MIXED_FIELD;
        environment_source_mapping_[RadiationEnvironment::MEO] = RadiationSourceType::PROTONS_700KEV;  // SAA protons
        environment_source_mapping_[RadiationEnvironment::GEO] = RadiationSourceType::ELECTRONS_12MEV; // Outer belt electrons
        environment_source_mapping_[RadiationEnvironment::LUNAR] = RadiationSourceType::SPACE_MIXED_FIELD;
        environment_source_mapping_[RadiationEnvironment::MARS_ORBIT] = RadiationSourceType::SPACE_MIXED_FIELD;
        environment_source_mapping_[RadiationEnvironment::MARS_SURFACE] = RadiationSourceType::CO60_GAMMA; // Mostly GCR
        environment_source_mapping_[RadiationEnvironment::JUPITER] = RadiationSourceType::ELECTRONS_12MEV; // High-energy electrons
        environment_source_mapping_[RadiationEnvironment::EUROPA] = RadiationSourceType::ELECTRONS_12MEV; // Extreme electrons
        environment_source_mapping_[RadiationEnvironment::INTERPLANETARY] = RadiationSourceType::SPACE_MIXED_FIELD;
        environment_source_mapping_[RadiationEnvironment::SOLAR_MINIMUM] = RadiationSourceType::CO60_GAMMA; // GCR dominated
        environment_source_mapping_[RadiationEnvironment::SOLAR_MAXIMUM] = RadiationSourceType::PROTONS_700KEV; // SPE protons
        environment_source_mapping_[RadiationEnvironment::SOLAR_STORM] = RadiationSourceType::PROTONS_700KEV; // High-energy protons
        
        // Set current source type based on current environment
        current_source_type_ = environment_source_mapping_[current_environment_];
    }
};

} // namespace sim
} // namespace rad_ml 