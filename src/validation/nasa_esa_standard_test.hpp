#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <functional>

namespace rad_ml {
namespace validation {

/**
 * @brief NASA/ESA Standard Radiation Test Framework
 * 
 * Implements testing methodology according to:
 * - NASA-HDBK-4002A: Mitigating In-Space Charging Effects
 * - ECSS-E-ST-10-12C: Space engineering - Methods for the calculation of radiation received
 * - MIL-STD-883, Method 1019: Ionizing radiation (total dose) test procedure
 * - JEDEC JESD57: Test Procedures for the Measurement of SEE in Semiconductor Devices
 * - NASA/TP-2006-214373: Single Event Effect Criticality Analysis
 * - NASA-STD-8719.14: Process for Limiting Orbital Debris
 */
class NASAESAStandardTest {
public:
    // Space environment models for testing
    enum class EnvironmentModel {
        CREME96,    // NASA standard model
        SPENVIS,    // ESA standard model
        AP8_AE8,    // Trapped proton and electron models
        EMMREM,     // Earth-Moon-Mars Radiation Environment Module
        SEPEM,      // Solar Energetic Particle Environment Modelling
        PSYCHIC,    // Prediction of Solar particle Yields for CHaracterizing Integrated Circuits
        CREME_MC    // CREME Monte Carlo
    };

    // Radiation effect metrics
    struct RadiationMetrics {
        double seu_rate;                // Single Event Upset rate (events/bit-day)
        double let_threshold;           // Linear Energy Transfer threshold (MeV-cm²/mg)
        double cross_section;           // Cross-section at 40 MeV-cm²/mg (cm²/bit)
        double mtbf;                    // Mean Time Between Failures (hours)
        double tid_tolerance;           // Total Ionizing Dose tolerance (krad)
        bool sel_observed;              // Single Event Latchup observed
        double sel_threshold;           // Single Event Latchup threshold (MeV-cm²/mg)
        double power_overhead;          // Power overhead (%)
        double processing_overhead;     // Processing overhead (%)
    };

    // Mission types
    enum class MissionType {
        LEO,            // Low Earth Orbit
        GEO,            // Geosynchronous Earth Orbit
        LUNAR,          // Lunar missions
        MARS,           // Mars missions
        JUPITER         // Jupiter/Europa missions
    };

    // Mission phase for testing
    enum class TestPhase {
        LAUNCH_ASCENT,          // Launch and ascent phase
        VAN_ALLEN_TRANSIT,      // Van Allen belt transit
        NOMINAL_OPERATIONS,     // Nominal operations at destination
        SOLAR_PARTICLE_EVENT    // During solar particle event
    };

    // Protection mechanism to test
    enum class ProtectionMethod {
        NO_PROTECTION,      // Baseline with no protection
        BASIC_TMR,          // Basic Triple Modular Redundancy
        ENHANCED_TMR,       // Enhanced TMR with CRC checksums
        STUCK_BIT_TMR,      // Stuck-Bit specialized TMR
        HEALTH_WEIGHTED_TMR, // Health-weighted TMR
        MEMORY_SCRUBBING    // Memory scrubbing technique
    };

    // Test results
    struct TestResult {
        MissionType mission;
        TestPhase phase;
        ProtectionMethod protection;
        RadiationMetrics metrics;
        bool pass;
        std::string notes;
    };

    // Mission suitability assessment
    struct MissionAssessment {
        MissionType mission;
        bool suitable;
        bool nasa_std_compliance;
        double system_lifetime_years;
        double required_shielding_mm_al;
        std::vector<std::string> recommended_enhancements;
    };

    // Constructor
    NASAESAStandardTest();

    // Configure the test environment
    void setEnvironmentModel(EnvironmentModel model);
    void setMonteCarloTrials(int trials);
    void setConfidenceInterval(double confidence_level);

    // Run tests
    std::vector<TestResult> runSingleEventUpsetTests();
    std::vector<TestResult> runLETThresholdTests();
    std::vector<TestResult> runMTBFTests();
    std::vector<TestResult> runTIDTests();
    std::vector<TestResult> runSELTests();
    std::vector<TestResult> runMissionPhaseTests();
    std::vector<TestResult> runComprehensiveTests();
    
    // Evaluate protection methods
    std::vector<TestResult> evaluateProtectionMethod(ProtectionMethod method);
    std::vector<TestResult> evaluateMemoryScrubbing(const std::vector<int>& scrubbing_intervals_hours);

    // Export results
    void exportResultsToCSV(const std::string& filename, const std::vector<TestResult>& results);
    void exportResultsToHTML(const std::string& filename, const std::vector<TestResult>& results);
    void generateComprehensiveReport(const std::string& filename);

    // Assess mission suitability
    MissionAssessment assessMissionSuitability(MissionType mission, const std::vector<TestResult>& results);

private:
    EnvironmentModel environment_model_;
    int monte_carlo_trials_;
    double confidence_level_;
    std::mt19937 rng_;

    // NASA/ESA standards pass criteria
    struct PassCriteria {
        double seu_rate_threshold;
        double let_threshold;
        double mtbf_requirement;
    };

    std::map<MissionType, PassCriteria> pass_criteria_;

    // Environment simulation methods
    double simulateSEURate(MissionType mission, ProtectionMethod protection);
    double simulateLETThreshold(MissionType mission, ProtectionMethod protection);
    double simulateCrossSection(double particle_energy, ProtectionMethod protection);
    double simulateMTBF(MissionType mission, ProtectionMethod protection);
    double simulateTID(MissionType mission, double mission_duration_years);
    bool simulateSEL(MissionType mission, ProtectionMethod protection);
    
    // Statistical validation
    bool performChiSquareTest(const std::vector<double>& observed, const std::vector<double>& expected);
    bool performKolmogorovSmirnovTest(const std::vector<double>& observed, const std::vector<double>& expected);
    
    // Helper methods
    double calculateSEUMitigationRatio(ProtectionMethod protection);
    double calculatePowerOverhead(ProtectionMethod protection, MissionType mission);
    double calculateProcessingOverhead(ProtectionMethod protection, MissionType mission);
    double calculateMemoryOverhead(ProtectionMethod protection);
    double calculateLatencyImpact(ProtectionMethod protection, MissionType mission);
};

} // namespace validation
} // namespace rad_ml 