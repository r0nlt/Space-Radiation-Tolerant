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

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <array>
#include <functional>
#include "rad_ml/testing/radiation_simulator.hpp"
#include "rad_ml/testing/protection_techniques.hpp"

namespace rad_ml {
namespace validation {

// Forward declarations
class NASAESAVerificationProtocol;

// Reference model data structure
struct ReferenceModelData {
    std::vector<double> seu_rates;
    std::vector<double> let_values;
    std::vector<double> cross_sections;
    std::vector<double> mtbf_values;
    std::vector<double> ber_values;
};

// Protection technique results structure
struct ProtectionTechniqueResults {
    double effectiveness_ratio;
    double reference_effectiveness;
    double resource_overhead;
    double power_overhead;
    double performance_overhead;
    bool passed_verification;
};

// Helper functions
double calculateCorrelation(double measured, double reference);
double calculatePercentDifference(double measured, double reference);

/**
 * @brief NASA/ESA Space Radiation Framework Verification Protocol
 * 
 * Implements comprehensive verification according to:
 * - NASA-HDBK-4002A: Mitigating In-Space Charging Effects
 * - ECSS-E-ST-10-12C: Space engineering - Methods for the calculation of radiation received
 * - JEDEC JESD57: Test Procedures for the Measurement of SEE in Semiconductor Devices
 * - NASA/TP-2006-214373: Single Event Effect Criticality Analysis
 * - MIL-STD-883, Method 1019: Ionizing radiation (total dose) test procedure
 */
class NASAESAVerificationProtocol {
public:
    // Environment models
    enum class EnvironmentModel {
        CREME96,            // NASA CREME96/CRÈME-MC
        OMERE,              // ESA OMERE
        SPENVIS,            // ESA SPENVIS
        AP8_AE8,            // Trapped proton/electron models
        SHIELDOSE,          // Dose calculation behind shielding
        CUSTOM              // Custom model
    };
    
    // Standard verification status
    enum class VerificationStatus {
        PASS,               // Meets or exceeds requirements
        FAIL,               // Does not meet requirements
        NOT_TESTED,         // Test not performed
        NOT_APPLICABLE      // Requirement not applicable
    };
    
    // Space environments for testing
    enum class TestEnvironment {
        LEO,                // Low Earth Orbit
        SAA,                // South Atlantic Anomaly
        GEO,                // Geosynchronous Earth Orbit
        VAN_ALLEN,          // Van Allen radiation belts
        LUNAR,              // Lunar environment
        MARS,               // Mars environment
        JUPITER             // Jupiter environment
    };
    
    // Helper function to convert TestEnvironment to string
    static std::string toString(TestEnvironment env) {
        switch (env) {
            case TestEnvironment::LEO: return "Low Earth Orbit";
            case TestEnvironment::SAA: return "South Atlantic Anomaly";
            case TestEnvironment::GEO: return "Geostationary Earth Orbit";
            case TestEnvironment::VAN_ALLEN: return "Van Allen Radiation Belts";
            case TestEnvironment::LUNAR: return "Lunar Surface";
            case TestEnvironment::MARS: return "Mars Surface";
            case TestEnvironment::JUPITER: return "Jupiter Flyby";
            default: return "Unknown Environment";
        }
    }
    
    // Protection techniques for evaluation
    enum class ProtectionTechnique {
        TMR,                // Triple Modular Redundancy
        EDAC,               // Error Detection and Correction
        SCRUBBING,          // Memory scrubbing
        HYBRID              // Hybrid approach
    };
    
    // Helper function to convert ProtectionTechnique to string
    static std::string toString(ProtectionTechnique tech) {
        switch (tech) {
            case ProtectionTechnique::TMR: return "Triple Modular Redundancy";
            case ProtectionTechnique::EDAC: return "Error Detection And Correction";
            case ProtectionTechnique::SCRUBBING: return "Memory Scrubbing";
            case ProtectionTechnique::HYBRID: return "Hybrid Protection";
            default: return "Unknown Technique";
        }
    }
    
    // Standard metric
    struct StandardMetric {
        std::string name;                   // Metric name
        std::string unit;                   // Metric unit
        double value;                       // Measured value
        double reference_value;             // Reference value
        double threshold;                   // Pass/fail threshold
        double confidence_interval_low;     // 95% confidence interval low
        double confidence_interval_high;    // 95% confidence interval high
        VerificationStatus status;          // Pass/fail status
    };
    
    // Standard requirement
    struct StandardRequirement {
        std::string standard;               // Standard reference (e.g., "NASA-HDBK-4002A")
        std::string requirement_id;         // Specific requirement ID
        std::string description;            // Requirement description
        std::string implementation;         // How requirement is implemented
        VerificationStatus status;          // Compliance status
    };
    
    // Environment model validation
    struct EnvironmentValidation {
        TestEnvironment environment;         // Environment being validated
        EnvironmentModel model;              // Model used
        double correlation_coefficient;      // Correlation with reference data
        double percent_difference;           // Percent difference from reference
        VerificationStatus status;           // Validation status
    };
    
    // Radiation hardening assessment
    struct RadiationHardeningResult {
        TestEnvironment environment;         // Test environment
        double seu_rate;                     // SEU rate (events/bit-day)
        double let_threshold;                // LET threshold (MeV-cm²/mg)
        double cross_section;                // Cross-section at 40 MeV-cm²/mg
        double mtbf;                         // MTBF (hours)
        double ber;                          // Bit Error Rate
        double nasa_threshold;               // NASA threshold value
        VerificationStatus status;           // Pass/fail status
    };
    
    // Protection technique evaluation
    struct ProtectionEvaluation {
        ProtectionTechnique technique;       // Protection technique
        double effectiveness_ratio;          // Effectiveness ratio
        double reference_effectiveness;      // Published reference effectiveness
        double resource_overhead;            // Resource utilization overhead (%)
        double power_overhead;               // Power consumption overhead (%)
        double performance_overhead;         // Performance overhead (%)
        VerificationStatus status;           // Evaluation status
    };
    
    // Mission suitability determination
    struct MissionSuitability {
        TestEnvironment environment;         // Mission environment
        bool suitable;                       // Go/no-go determination
        std::vector<std::string> required_modifications; // Required modifications
        double required_shielding_mm_al;     // Required shielding (mm Al)
        std::string rationale;               // Suitability rationale
    };
    
    // Comprehensive verification report
    struct VerificationReport {
        std::string verification_date;
        std::string framework_version;
        std::vector<StandardRequirement> compliance_matrix;
        std::vector<EnvironmentValidation> environment_validations;
        std::vector<RadiationHardeningResult> radiation_assessments;
        std::vector<ProtectionEvaluation> protection_evaluations;
        std::vector<MissionSuitability> mission_suitabilities;
        bool overall_compliant;
        std::string verification_statement;
    };
    
    // Constructor with options
    NASAESAVerificationProtocol(
        const std::vector<TestEnvironment>& environments = {
            TestEnvironment::LEO,
            TestEnvironment::SAA,
            TestEnvironment::GEO,
            TestEnvironment::VAN_ALLEN,
            TestEnvironment::LUNAR,
            TestEnvironment::MARS,
            TestEnvironment::JUPITER
        },
        const std::vector<ProtectionTechnique>& techniques = {
            ProtectionTechnique::TMR,
            ProtectionTechnique::EDAC,
            ProtectionTechnique::SCRUBBING
        },
        int monte_carlo_trials = 10000,
        double confidence_level = 0.95
    );
    
    // Core verification functions
    void verifyEnvironmentIntegration();
    void verifyStandardMetrics();
    void evaluateProtectionTechniques();
    void performTestingMethodology();
    
    // Run comprehensive verification
    VerificationReport runComprehensiveVerification();
    
    // Export functions
    void exportComplianceMatrix(const std::string& filename);
    void exportEnvironmentValidation(const std::string& filename);
    void exportRadiationHardeningAssessment(const std::string& filename);
    void exportProtectionEvaluation(const std::string& filename);
    void exportMissionSuitability(const std::string& filename);
    void exportComprehensiveReport(const std::string& filename);
    
    // NASA reference thresholds
    static std::map<TestEnvironment, double> getNASASEUThresholds() {
        return {
            {TestEnvironment::LEO, 1e-7},        // <1×10⁻⁷ errors/bit-day
            {TestEnvironment::GEO, 5e-8},        // <5×10⁻⁸ errors/bit-day
            {TestEnvironment::LUNAR, 3e-8},      // <3×10⁻⁸ errors/bit-day
            {TestEnvironment::MARS, 1e-8},       // <1×10⁻⁸ errors/bit-day
            {TestEnvironment::JUPITER, 5e-9}     // <5×10⁻⁹ errors/bit-day
        };
    }
    
    // NASA LET threshold requirements
    static std::map<TestEnvironment, double> getNASALETThresholds() {
        return {
            {TestEnvironment::LEO, 40.0},        // >40 MeV-cm²/mg
            {TestEnvironment::GEO, 60.0},        // >60 MeV-cm²/mg
            {TestEnvironment::LUNAR, 70.0},      // >70 MeV-cm²/mg
            {TestEnvironment::MARS, 80.0},       // >80 MeV-cm²/mg
            {TestEnvironment::JUPITER, 100.0}    // >100 MeV-cm²/mg
        };
    }
    
    // NASA MTBF requirements
    static std::map<TestEnvironment, double> getNASAMTBFRequirements() {
        return {
            {TestEnvironment::LEO, 10000.0},     // >10,000 hours
            {TestEnvironment::GEO, 20000.0},     // >20,000 hours
            {TestEnvironment::LUNAR, 30000.0},   // >30,000 hours
            {TestEnvironment::MARS, 50000.0},    // >50,000 hours
            {TestEnvironment::JUPITER, 100000.0} // >100,000 hours
        };
    }
    
    // Helper function to convert to testing::ProtectionTechnique
    static testing::ProtectionTechnique toTestingProtectionTechnique(ProtectionTechnique tech) {
        switch (tech) {
            case ProtectionTechnique::TMR:
                return testing::ProtectionTechnique::TMR;
            case ProtectionTechnique::EDAC:
                return testing::ProtectionTechnique::EDAC;
            case ProtectionTechnique::SCRUBBING:
                return testing::ProtectionTechnique::SCRUBBING;
            default:
                return testing::ProtectionTechnique::NONE;
        }
    }

private:
    std::vector<TestEnvironment> environments_;
    std::vector<ProtectionTechnique> techniques_;
    int monte_carlo_trials_;
    double confidence_level_;
    EnvironmentModel primary_model_;
    
    // Statistical validation functions
    bool performChiSquareTest(const std::vector<double>& observed, const std::vector<double>& expected);
    bool performKolmogorovSmirnovTest(const std::vector<double>& observed, const std::vector<double>& expected);
    std::pair<double, double> calculateConfidenceInterval(const std::vector<double>& data, double confidence);
    
    // Reference data comparison
    double calculateCorrelationCoefficient(const std::vector<double>& measured, const std::vector<double>& reference);
    
    // Standards compliance verification
    std::vector<StandardRequirement> verifyNASAHDBK4002A();
    std::vector<StandardRequirement> verifyECSSEST1012C();
    std::vector<StandardRequirement> verifyJEDECJESD57();
    std::vector<StandardRequirement> verifyMILSTD883Method1019();
    
    // Generate verification statement
    std::string generateVerificationStatement(const VerificationReport& report);
    
    // Helper functions
    testing::RadiationSimulator::EnvironmentParams getEnvironmentParams(TestEnvironment env);
    ReferenceModelData getReferenceModelData(TestEnvironment env);
    ProtectionTechniqueResults evaluateProtectionTechnique(ProtectionTechnique tech);
    void calculateRequiredModifications(MissionSuitability& suitability, const RadiationHardeningResult& result);
    double calculateRequiredShielding(const RadiationHardeningResult& result);
    void generateSuitabilityRationale(MissionSuitability& suitability, const RadiationHardeningResult& result);
    bool determineOverallCompliance(const VerificationReport& report);
};

} // namespace validation
} // namespace rad_ml 
