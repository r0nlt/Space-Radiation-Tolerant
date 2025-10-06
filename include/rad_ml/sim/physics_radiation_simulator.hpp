#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rad_ml {
namespace sim {

/**
 * @brief Space radiation environment model
 *
 * Based on NASA's AE9/AP9 model and ESA's SPENVIS for radiation modeling
 */
enum class SpaceEnvironment {
    LEO,             // Low Earth Orbit (400-600km)
    EARTH_ORBIT,     // Earth orbit (general)
    MEO,             // Medium Earth Orbit (like GPS satellites)
    GEO,             // Geosynchronous Earth Orbit
    LUNAR,           // Lunar vicinity
    MARS,            // Mars (general)
    MARS_ORBIT,      // Mars orbit
    MARS_SURFACE,    // Mars surface (with atmosphere shielding)
    JUPITER,         // Jupiter radiation belts
    EUROPA,          // Near Europa (extreme radiation environment)
    INTERPLANETARY,  // Deep space, interplanetary transit
    SOLAR_PROBE,     // Solar probe (extreme solar radiation)
    SOLAR_MINIMUM,   // Solar minimum conditions (higher GCR)
    SOLAR_MAXIMUM,   // Solar maximum conditions (higher SPE probability)
    SOLAR_STORM      // Active solar storm (extreme conditions)
};

/**
 * @brief Types of radiation effects in space
 */
enum class RadiationEffectType {
    SEU,                 // Single Event Upset
    MBU,                 // Multiple Bit Upset
    SEL,                 // Single Event Latchup
    SET,                 // Single Event Transient
    SEFI,                // Single Event Functional Interrupt
    TID_STUCK_BIT,       // Total Ionizing Dose induced stuck bit
    TID_THRESHOLD_SHIFT  // TID induced threshold voltage shift
};

/**
 * @brief Configuration for a radiation event
 */
struct RadiationEffect {
    RadiationEffectType type;
    double probability;    // Base probability per bit per day
    size_t min_bits;       // Minimum bits affected
    size_t max_bits;       // Maximum bits affected
    bool is_persistent;    // Whether effects persist after reboot/repair
    double recovery_prob;  // Probability of spontaneous recovery (per day)

    // Added default constructor to fix map initialization error
    RadiationEffect()
        : type(RadiationEffectType::SEU),
          probability(0.0),
          min_bits(0),
          max_bits(0),
          is_persistent(false),
          recovery_prob(0.0)
    {
    }

    // Constructor with typical values derived from space radiation studies
    RadiationEffect(RadiationEffectType t) : type(t)
    {
        // Set defaults based on radiation effect type
        switch (type) {
            case RadiationEffectType::SEU:
                probability = 1e-7;  // ~1 per 10M bits per day (typical LEO)
                min_bits = 1;
                max_bits = 1;
                is_persistent = false;
                recovery_prob = 1.0;  // Recovers immediately with power cycle
                break;

            case RadiationEffectType::MBU:
                probability = 2e-8;  // ~20% of SEUs are MBUs
                min_bits = 2;
                max_bits = 8;  // Typical for modern memory
                is_persistent = false;
                recovery_prob = 1.0;  // Recovers immediately with power cycle
                break;

            case RadiationEffectType::SEL:
                probability = 5e-9;  // Based on ESA JUICE radiation specs
                min_bits = 1;
                max_bits = 1024;       // Can affect entire regions
                is_persistent = true;  // Requires power cycle
                recovery_prob = 0.0;   // Only recovers with intervention
                break;

            case RadiationEffectType::SET:
                probability = 2e-7;  // More common in logic than memory
                min_bits = 1;
                max_bits = 1;
                is_persistent = false;
                recovery_prob = 1.0;  // Transient by definition
                break;

            case RadiationEffectType::SEFI:
                probability = 1e-9;  // Based on NASA testing data
                min_bits = 1;
                max_bits = 1024 * 1024;  // Can affect entire systems
                is_persistent = true;    // Often requires power cycle
                recovery_prob = 0.0;     // Only recovers with intervention
                break;

            case RadiationEffectType::TID_STUCK_BIT:
                probability = 5e-10;  // Accumulates over mission lifetime
                min_bits = 1;
                max_bits = 1;
                is_persistent = true;  // Permanent damage
                recovery_prob = 0.0;   // No spontaneous recovery
                break;

            case RadiationEffectType::TID_THRESHOLD_SHIFT:
                probability = 1e-9;  // Based on MESSENGER data
                min_bits = 1;
                max_bits = 1024;       // Affects regions
                is_persistent = true;  // Permanent damage
                recovery_prob = 0.0;   // No spontaneous recovery
                break;
        }
    }
};

/**
 * @brief Model of spacecraft orbit or trajectory
 */
struct SpacecraftTrajectory {
    std::vector<SpaceEnvironment> environments;
    std::vector<double> durations_days;  // Time spent in each environment

    // Common trajectory configurations based on NASA mission profiles
    static SpacecraftTrajectory Earth_LEO()
    {
        return {{SpaceEnvironment::LEO}, {365.0}};  // One year mission
    }

    static SpacecraftTrajectory Mars_Mission()
    {
        return {{SpaceEnvironment::LEO, SpaceEnvironment::INTERPLANETARY,
                 SpaceEnvironment::MARS_ORBIT, SpaceEnvironment::MARS_SURFACE,
                 SpaceEnvironment::INTERPLANETARY, SpaceEnvironment::LEO},
                {10.0, 180.0, 30.0, 365.0, 180.0, 10.0}};
    }

    static SpacecraftTrajectory Europa_Mission()
    {
        return {{SpaceEnvironment::LEO, SpaceEnvironment::INTERPLANETARY, SpaceEnvironment::JUPITER,
                 SpaceEnvironment::EUROPA, SpaceEnvironment::JUPITER,
                 SpaceEnvironment::INTERPLANETARY, SpaceEnvironment::LEO},
                {10.0, 730.0, 60.0, 30.0, 60.0, 730.0, 10.0}};
    }
};

/**
 * @brief Environment parameters for radiation simulation
 */
struct EnvironmentParams {
    SpaceEnvironment environment;
    double solar_activity;
    double shielding_thickness_mm;

    EnvironmentParams(SpaceEnvironment env = SpaceEnvironment::LEO, double solar = 0.5,
                      double shielding = 2.0)
        : environment(env), solar_activity(solar), shielding_thickness_mm(shielding)
    {
    }
};

/**
 * @brief Physics-based radiation simulator for space environments
 *
 * This simulator models radiation effects based on:
 * - Spacecraft trajectory and environment
 * - Solar activity levels
 * - Shielding thickness
 * - Memory technology characteristics
 */
class PhysicsRadiationSimulator {
   public:
    /**
     * @brief Get mission environment parameters
     *
     * @param mission_profile Mission profile name
     * @return Environment parameters for the mission
     */
    static EnvironmentParams getMissionEnvironment(const std::string& mission_profile)
    {
        if (mission_profile == "LEO") {
            return EnvironmentParams(SpaceEnvironment::LEO, 0.5, 2.0);
        }
        else if (mission_profile == "MARS") {
            return EnvironmentParams(SpaceEnvironment::MARS_ORBIT, 0.3, 5.0);
        }
        else if (mission_profile == "JUPITER") {
            return EnvironmentParams(SpaceEnvironment::JUPITER, 0.2, 10.0);
        }
        else {
            return EnvironmentParams(SpaceEnvironment::LEO, 0.5, 2.0);
        }
    }

    /**
     * @brief Constructor with environment parameters
     */
    PhysicsRadiationSimulator(const EnvironmentParams& params)
        : memory_bits_(8 * 1024 * 1024),  // 1 MB default
          word_size_(32),                 // 32-bit words
          shielding_thickness_mm_(params.shielding_thickness_mm),
          trajectory_(),
          current_environment_(params.environment),
          solar_activity_(params.solar_activity),
          random_engine_(std::chrono::system_clock::now().time_since_epoch().count())
    {
        // Initialize trajectory with the environment
        trajectory_.environments = {params.environment};
        trajectory_.durations_days = {365.0};

        initialize_radiation_effects();
        calculate_environment_modifiers();
    }

    /**
     * @brief Set the current radiation environment
     *
     * @param environment The radiation environment
     */
    void set_environment(SpaceEnvironment environment) { current_environment_ = environment; }

    /**
     * @brief Set solar activity level
     *
     * @param activity Solar activity level (0.0-1.0)
     */
    void set_solar_activity(double activity) { solar_activity_ = std::clamp(activity, 0.0, 1.0); }

    /**
     * @brief Set radiation intensity
     *
     * @param intensity Radiation intensity (0.0-1.0)
     */
    void setIntensity(double intensity) { solar_activity_ = std::clamp(intensity, 0.0, 1.0); }

    /**
     * @brief Get current radiation intensity
     *
     * @return Current radiation intensity
     */
    double getIntensity() const { return solar_activity_; }

    /**
     * @brief Set environment
     *
     * @param environment New environment
     */
    void setEnvironment(SpaceEnvironment environment) { current_environment_ = environment; }

    /**
     * @brief Get current environment
     *
     * @return Current environment
     */
    SpaceEnvironment getEnvironment() const { return current_environment_; }

    /**
     * @brief Simulate radiation effects
     *
     * @param duration Duration to simulate in days
     * @return Simulation results
     */
    std::vector<std::pair<RadiationEffectType, size_t>> simulate(double duration)
    {
        return simulate_period(duration);
    }

    /**
     * @brief Simulate radiation effects on data
     *
     * @param data Pointer to data buffer
     * @param size Size of data buffer in bytes
     * @param duration Duration to simulate
     * @return Vector of radiation events
     */
    std::vector<std::pair<RadiationEffectType, size_t>> simulateEffects(
        void* data, size_t size, std::chrono::milliseconds duration)
    {
        std::vector<std::pair<RadiationEffectType, size_t>> events;

        // Convert duration to days
        double days = duration.count() / (1000.0 * 60.0 * 60.0 * 24.0);

        // Simulate radiation effects for the period
        auto period_events = simulate_period(days);

        // Apply effects to data (simplified - just mark locations)
        for (const auto& [effect_type, bit_count] : period_events) {
            events.push_back({effect_type, bit_count});
        }

        return events;
    }

    /**
     * @brief Set spacecraft shielding
     *
     * @param thickness_mm Aluminum equivalent thickness in mm
     */
    void set_shielding(double thickness_mm)
    {
        shielding_thickness_mm_ = thickness_mm;
        // Recalculate environment modifiers with new shielding
        calculate_environment_modifiers();
    }

    /**
     * @brief Set a custom spacecraft trajectory
     *
     * @param trajectory The new trajectory
     */
    void set_trajectory(const SpacecraftTrajectory& trajectory)
    {
        trajectory_ = trajectory;
        current_environment_ = trajectory.environments[0];
    }

    /**
     * @brief Simulate radiation for a specific time period
     *
     * @param days Days to simulate
     * @return Vector of radiation effect events
     */
    std::vector<std::pair<RadiationEffectType, size_t>> simulate_period(double days)
    {
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
    std::map<RadiationEffectType, size_t> simulate_mission_tid(double days)
    {
        std::map<RadiationEffectType, size_t> tid_effects;

        // Initialize TID effect types
        tid_effects[RadiationEffectType::TID_STUCK_BIT] = 0;
        tid_effects[RadiationEffectType::TID_THRESHOLD_SHIFT] = 0;

        // For each segment of the mission
        double days_simulated = 0.0;
        size_t current_segment = 0;

        while (days_simulated < days && current_segment < trajectory_.environments.size()) {
            // Calculate time spent in this segment
            double segment_days =
                std::min(trajectory_.durations_days[current_segment], days - days_simulated);

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
                    double event_probability =
                        effect.probability * env_modifier * shielding_factor * segment_days;

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
    std::vector<std::map<RadiationEffectType, size_t>> simulate_mission()
    {
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
    std::map<RadiationEffectType, double> get_error_rates() const
    {
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
    std::string get_environment_report() const
    {
        std::string report = "Space Radiation Environment Report\n";
        report += "--------------------------------\n";

        // Environment information
        report += "Current environment: " + get_environment_name(current_environment_) + "\n";
        report += "Relative radiation level: " +
                  std::to_string(environment_modifiers_.at(current_environment_)) + "x baseline\n";
        report += "Solar activity level: " + std::to_string(solar_activity_) + " (" +
                  (solar_activity_ < 0.3   ? "Low"
                   : solar_activity_ > 0.7 ? "High"
                                           : "Medium") +
                  ")\n";
        report += "Spacecraft shielding: " + std::to_string(shielding_thickness_mm_) +
                  " mm Al-eq (reduction factor: " + std::to_string(calculate_shielding_factor()) +
                  ")\n\n";

        // Error rates
        report += "Expected error rates (per Mbit per day):\n";
        auto rates = get_error_rates();

        for (const auto& [type, rate] : rates) {
            report += "  " + get_effect_name(type) + ": " + std::to_string(rate) + "\n";
        }

        return report;
    }

   private:
    // Configuration parameters
    size_t memory_bits_;
    size_t word_size_;
    double shielding_thickness_mm_;
    SpacecraftTrajectory trajectory_;
    SpaceEnvironment current_environment_;
    double solar_activity_;

    // Radiation effect models
    std::map<RadiationEffectType, RadiationEffect> radiation_effects_;

    // Environment rate modifiers relative to baseline (LEO)
    std::map<SpaceEnvironment, double> environment_modifiers_;

    // Random number generation
    std::default_random_engine random_engine_;

    /**
     * @brief Initialize radiation effect models
     */
    void initialize_radiation_effects()
    {
        // Create radiation effects with scientifically accurate rates
        radiation_effects_[RadiationEffectType::SEU] = RadiationEffect(RadiationEffectType::SEU);

        radiation_effects_[RadiationEffectType::MBU] = RadiationEffect(RadiationEffectType::MBU);

        radiation_effects_[RadiationEffectType::SEL] = RadiationEffect(RadiationEffectType::SEL);

        radiation_effects_[RadiationEffectType::SET] = RadiationEffect(RadiationEffectType::SET);

        radiation_effects_[RadiationEffectType::SEFI] = RadiationEffect(RadiationEffectType::SEFI);

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
    void calculate_environment_modifiers()
    {
        // Rates relative to LEO (based on scientific space radiation models)
        environment_modifiers_[SpaceEnvironment::LEO] = 1.0;             // Baseline
        environment_modifiers_[SpaceEnvironment::MEO] = 10.0;            // South Atlantic Anomaly
        environment_modifiers_[SpaceEnvironment::GEO] = 5.0;             // Outside magnetosphere
        environment_modifiers_[SpaceEnvironment::LUNAR] = 4.0;           // No magnetic protection
        environment_modifiers_[SpaceEnvironment::MARS_ORBIT] = 3.0;      // No strong field
        environment_modifiers_[SpaceEnvironment::MARS_SURFACE] = 0.5;    // Atmosphere shields
        environment_modifiers_[SpaceEnvironment::JUPITER] = 1000.0;      // Extreme environment
        environment_modifiers_[SpaceEnvironment::EUROPA] = 2000.0;       // Europa mission estimates
        environment_modifiers_[SpaceEnvironment::INTERPLANETARY] = 3.0;  // Deep space
        environment_modifiers_[SpaceEnvironment::SOLAR_MINIMUM] = 2.0;   // Higher GCR
        environment_modifiers_[SpaceEnvironment::SOLAR_MAXIMUM] = 0.8;   // Lower GCR
        environment_modifiers_[SpaceEnvironment::SOLAR_STORM] = 100.0;   // Extreme conditions
    }

    /**
     * @brief Calculate modifier based on solar activity
     *
     * @return Solar activity multiplier
     */
    double calculate_solar_modifier() const
    {
        // Solar storm probability increases with activity
        if (current_environment_ == SpaceEnvironment::SOLAR_STORM) {
            return 1.0;  // Already factored into environment
        }

        // GCR rates are anti-correlated with solar activity
        // SPE rates are correlated with solar activity
        double gcr_component = 1.0 - 0.5 * solar_activity_;              // 1.0 at min, 0.5 at max
        double spe_component = solar_activity_ * solar_activity_ * 5.0;  // 0 at min, 5.0 at max

        // Combined effect depends on environment
        if (current_environment_ == SpaceEnvironment::LEO ||
            current_environment_ == SpaceEnvironment::MEO ||
            current_environment_ == SpaceEnvironment::GEO) {
            // Earth environments - more SPE protection
            return 0.7 * gcr_component + 0.3 * spe_component;
        }
        else {
            // Deep space - more exposure to both
            return 0.5 * gcr_component + 0.5 * spe_component;
        }
    }

    /**
     * @brief Calculate shielding effectiveness
     *
     * Based on aluminum equivalent shielding models from SPENVIS
     *
     * @return Shielding attenuation factor
     */
    double calculate_shielding_factor() const
    {
        // Parameters derived from SPENVIS aluminum shielding model
        const double reference_thickness = 2.0;  // 2mm Al reference

        if (shielding_thickness_mm_ <= 0.0) {
            return 1.0;  // No shielding
        }

        // Model different behaviors for different radiation effects
        double base_reduction = std::exp(-shielding_thickness_mm_ / reference_thickness);

        // TID typically follows closer to exponential attenuation
        double tid_reduction = std::pow(base_reduction, 1.2);

        // SEE typically requires higher energy particles, less shield-sensitive
        double see_reduction = std::pow(base_reduction, 0.7);

        // Combined effect ranges from about 0.01 to 1.0
        return std::max(0.01, std::min(1.0, 0.3 * tid_reduction + 0.7 * see_reduction));
    }

    /**
     * @brief Get string representation of environment
     *
     * @param env Environment
     * @return String name
     */
    std::string get_environment_name(SpaceEnvironment env) const
    {
        switch (env) {
            case SpaceEnvironment::LEO:
                return "Low Earth Orbit";
            case SpaceEnvironment::EARTH_ORBIT:
                return "Earth Orbit";
            case SpaceEnvironment::MEO:
                return "Medium Earth Orbit";
            case SpaceEnvironment::GEO:
                return "Geosynchronous Earth Orbit";
            case SpaceEnvironment::LUNAR:
                return "Lunar Vicinity";
            case SpaceEnvironment::MARS_ORBIT:
                return "Mars Orbit";
            case SpaceEnvironment::MARS_SURFACE:
                return "Mars Surface";
            case SpaceEnvironment::JUPITER:
                return "Jupiter Radiation Belts";
            case SpaceEnvironment::EUROPA:
                return "Europa Vicinity";
            case SpaceEnvironment::INTERPLANETARY:
                return "Interplanetary Space";
            case SpaceEnvironment::SOLAR_MINIMUM:
                return "Solar Minimum";
            case SpaceEnvironment::SOLAR_MAXIMUM:
                return "Solar Maximum";
            case SpaceEnvironment::SOLAR_STORM:
                return "Solar Storm";
            default:
                return "Unknown";
        }
    }

    /**
     * @brief Get string representation of radiation effect
     *
     * @param effect Effect type
     * @return String name
     */
    std::string get_effect_name(RadiationEffectType effect) const
    {
        switch (effect) {
            case RadiationEffectType::SEU:
                return "Single Event Upset";
            case RadiationEffectType::MBU:
                return "Multiple Bit Upset";
            case RadiationEffectType::SEL:
                return "Single Event Latchup";
            case RadiationEffectType::SET:
                return "Single Event Transient";
            case RadiationEffectType::SEFI:
                return "Single Event Functional Interrupt";
            case RadiationEffectType::TID_STUCK_BIT:
                return "TID Stuck Bit";
            case RadiationEffectType::TID_THRESHOLD_SHIFT:
                return "TID Threshold Shift";
            default:
                return "Unknown";
        }
    }
};

}  // namespace sim
}  // namespace rad_ml
