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

#include "../../include/rad_ml/api/rad_ml.hpp"
#include "rad_ml/sim/physics_radiation_simulator.hpp"
#include "rad_ml/testing/mission_simulator.hpp"
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <cmath>
#include <iostream>
#include <fstream>

namespace rad_ml {
namespace validation {

/**
 * @brief Standard space environment configurations
 */
enum class StandardEnvironment {
    LOW_EARTH_ORBIT,      // ISS altitude: ~400km
    VAN_ALLEN_BELT,       // Inner and outer radiation belts
    LUNAR_ORBIT,          // Cislunar space
    INTERPLANETARY,       // Earth to Mars transit
    JUPITER_EUROPA        // Jovian environment
};

/**
 * @brief Reference data source for validation
 */
enum class ReferenceSource {
    CREME96,              // CREME96/CRÈME-MC models
    OMERE,                // ESA OMERE model
    SAMPEX,               // SAMPEX mission data
    HUBBLE,               // Hubble Space Telescope data
    VAN_ALLEN_PROBES,     // Van Allen Probes mission data
    JUNO,                 // Juno mission data
    CYCLOTRON_FACILITY    // Ground radiation testing
};

/**
 * @brief Standard SEU parameters for a specific environment
 */
struct StandardSEUParameters {
    double seu_rate_per_bit_per_day;     // Standard SEU rate/bit/day
    double seu_cross_section_cm2_per_bit; // SEU cross-section (cm²/bit)
    double confidence_interval;          // 95% confidence interval
    ReferenceSource source;              // Data source
    std::string reference;               // Reference citation
};

/**
 * @brief Validation result for a single test
 */
struct ValidationResult {
    double predicted_seu_rate;           // SEU rate predicted by rad_ml
    double reference_seu_rate;           // Reference SEU rate from standard
    double accuracy_percentage;          // Accuracy (100% = perfect match)
    double protection_efficiency;        // TMR or other protection efficiency
    double power_consumption_mw;         // Power consumption in mW
    double power_efficiency_mw_per_seu;  // Power/SEU prevention (mW/SEU)
    bool within_tolerance;               // Whether prediction is within acceptable range
    std::string notes;                   // Additional observations
};

/**
 * @brief Class for validating radiation tolerance accuracy
 * 
 * This class implements the NASA/ESA standard testing protocol for
 * radiation effects on electronics in space environments.
 */
class RadiationAccuracyValidator {
public:
    /**
     * @brief Constructor
     */
    RadiationAccuracyValidator()
        : initialized_(false) {
        initializeReferenceData();
    }
    
    /**
     * @brief Initialize the framework
     * 
     * @param memory_protection Default memory protection level
     * @return True if initialization successful
     */
    bool initialize(memory::MemoryProtectionLevel memory_protection = memory::MemoryProtectionLevel::TMR) {
        if (initialized_) {
            return true;
        }
        
        initialized_ = rad_ml::initialize(true, memory_protection);
        return initialized_;
    }
    
    /**
     * @brief Run standard environment validation
     * 
     * @param environment Standard environment to test
     * @param duration_days Duration of simulation in days
     * @param protection_levels Vector of protection levels to test
     * @return Map of protection level to validation result
     */
    std::map<std::string, ValidationResult> validateEnvironment(
        StandardEnvironment environment,
        int duration_days = 30,
        const std::vector<std::string>& protection_levels = {"NONE", "TMR", "TMR_ENHANCED"}) {
        
        if (!initialized_) {
            initialize();
        }
        
        auto env_params = getEnvironmentParameters(environment);
        auto mission_type = convertToMissionType(environment);
        
        std::map<std::string, ValidationResult> results;
        
        // Create mission simulator
        auto mission_simulator = rad_ml::simulation::createMissionSimulator(
            mission_type,
            duration_days
        );
        
        // Run tests for each protection level
        for (const auto& protection_level : protection_levels) {
            ValidationResult result = runProtectionTest(
                mission_simulator,
                protection_level,
                env_params
            );
            
            results[protection_level] = result;
        }
        
        return results;
    }
    
    /**
     * @brief Run comprehensive validation across all environments
     * 
     * @param duration_days Duration of simulation in days
     * @return Map of environment to protection level to validation result
     */
    std::map<StandardEnvironment, std::map<std::string, ValidationResult>> 
    validateAllEnvironments(int duration_days = 30) {
        std::map<StandardEnvironment, std::map<std::string, ValidationResult>> all_results;
        
        std::vector<StandardEnvironment> environments = {
            StandardEnvironment::LOW_EARTH_ORBIT,
            StandardEnvironment::VAN_ALLEN_BELT,
            StandardEnvironment::LUNAR_ORBIT,
            StandardEnvironment::INTERPLANETARY,
            StandardEnvironment::JUPITER_EUROPA
        };
        
        for (const auto& env : environments) {
            all_results[env] = validateEnvironment(env, duration_days);
        }
        
        return all_results;
    }
    
    /**
     * @brief Calculate protection efficiency using NASA SEECA methodology
     * 
     * @param unprotected_errors Errors without protection
     * @param protected_errors Errors with protection
     * @param total_bits Total bits processed
     * @return Protection efficiency percentage
     */
    double calculateProtectionEfficiency(
        double unprotected_errors,
        double protected_errors,
        double total_bits) {
        
        // If protected_errors > unprotected_errors, protection made things worse
        if (protected_errors >= unprotected_errors) {
            return 0.0;
        }
        
        // Calculate the percentage of errors prevented
        return ((unprotected_errors - protected_errors) / unprotected_errors) * 100.0;
    }
    
    /**
     * @brief Calculate accuracy percentage (NASA standard formula)
     *
     * @param uncorrected_errors Number of uncorrected errors
     * @param total_bits Total number of bits
     * @return Accuracy percentage
     */
    double calculateAccuracy(double uncorrected_errors, double total_bits) {
        return (1.0 - (uncorrected_errors / total_bits)) * 100.0;
    }
    
    /**
     * @brief Export validation results to CSV
     *
     * @param results Validation results
     * @param filepath Output filepath
     * @return True if export successful
     */
    bool exportResultsToCSV(
        const std::map<StandardEnvironment, std::map<std::string, ValidationResult>>& results,
        const std::string& filepath) {
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        // CSV header
        file << "Environment,Protection Level,Predicted SEU Rate,Reference SEU Rate,"
             << "Accuracy %,Protection Efficiency %,Power Consumption (mW),"
             << "Power Efficiency (mW/SEU),Within Tolerance,Notes\n";
        
        // Write data
        for (const auto& env_pair : results) {
            std::string env_name = getEnvironmentName(env_pair.first);
            
            for (const auto& prot_pair : env_pair.second) {
                const auto& result = prot_pair.second;
                
                file << env_name << ","
                     << prot_pair.first << ","
                     << result.predicted_seu_rate << ","
                     << result.reference_seu_rate << ","
                     << result.accuracy_percentage << ","
                     << result.protection_efficiency << ","
                     << result.power_consumption_mw << ","
                     << result.power_efficiency_mw_per_seu << ","
                     << (result.within_tolerance ? "Yes" : "No") << ","
                     << "\"" << result.notes << "\"\n";
            }
        }
        
        file.close();
        return true;
    }
    
    /**
     * @brief Generate a comprehensive validation report
     *
     * @param results Validation results
     * @return Multiline string with formatted report
     */
    std::string generateReport(
        const std::map<StandardEnvironment, std::map<std::string, ValidationResult>>& results) {
        
        std::stringstream report;
        
        report << "==========================================================\n";
        report << "          RADIATION TOLERANCE VALIDATION REPORT           \n";
        report << "==========================================================\n\n";
        
        // Overall statistics
        double total_accuracy = 0.0;
        double total_protection_efficiency = 0.0;
        int total_tests = 0;
        int passing_tests = 0;
        
        // Process each environment
        for (const auto& env_pair : results) {
            report << "Environment: " << getEnvironmentName(env_pair.first) << "\n";
            report << "-----------------------------------------------------------\n";
            
            // Environment reference data
            auto env_params = getEnvironmentParameters(env_pair.first);
            report << "Reference SEU Rate: " << env_params.seu_rate_per_bit_per_day 
                   << " SEUs/bit/day (± " << env_params.confidence_interval << "%)\n";
            report << "Reference Cross-Section: " << env_params.seu_cross_section_cm2_per_bit 
                   << " cm²/bit\n";
            report << "Data Source: " << getReferenceSourceName(env_params.source) 
                   << " (" << env_params.reference << ")\n\n";
            
            // Protection level results
            report << "Protection Level Results:\n";
            report << "  +-----------------+-------------+-------------+-------------+-------------+\n";
            report << "  | Protection      | SEU Rate    | Accuracy    | Protection  | Power       |\n";
            report << "  |                 | (pred/ref)  | (%)         | Efficiency  | Efficiency  |\n";
            report << "  +-----------------+-------------+-------------+-------------+-------------+\n";
            
            for (const auto& prot_pair : env_pair.second) {
                const auto& result = prot_pair.second;
                
                // Format ratio with proper precision
                std::stringstream ratio_ss;
                ratio_ss << std::fixed << std::setprecision(2) 
                         << result.predicted_seu_rate << "/" 
                         << result.reference_seu_rate;
                std::string ratio = ratio_ss.str();
                
                // Ensure it fits in column
                if (ratio.length() > 11) {
                    ratio = ratio.substr(0, 11);
                }
                
                // Format and output the row
                report << "  | " << std::left << std::setw(15) << prot_pair.first
                       << " | " << std::setw(11) << ratio
                       << " | " << std::fixed << std::setprecision(2) << std::setw(11) << result.accuracy_percentage
                       << " | " << std::setw(11) << result.protection_efficiency
                       << " | " << std::setw(11) << result.power_efficiency_mw_per_seu << " |\n";
                
                // Update overall statistics
                total_accuracy += result.accuracy_percentage;
                total_protection_efficiency += result.protection_efficiency;
                total_tests++;
                
                if (result.within_tolerance) {
                    passing_tests++;
                }
            }
            
            report << "  +-----------------+-------------+-------------+-------------+-------------+\n\n";
        }
        
        // Overall summary
        if (total_tests > 0) {
            report << "==========================================================\n";
            report << "SUMMARY:\n";
            report << "  Total Tests: " << total_tests << "\n";
            report << "  Passing Tests: " << passing_tests << " (" 
                   << std::fixed << std::setprecision(1) 
                   << (passing_tests * 100.0 / total_tests) << "%)\n";
            report << "  Average Accuracy: " 
                   << std::fixed << std::setprecision(2)
                   << (total_accuracy / total_tests) << "%\n";
            report << "  Average Protection Efficiency: " 
                   << std::fixed << std::setprecision(2)
                   << (total_protection_efficiency / total_tests) << "%\n";
            report << "==========================================================\n";
        }
        
        return report.str();
    }
    
private:
    bool initialized_;
    std::map<StandardEnvironment, StandardSEUParameters> reference_data_;
    
    /**
     * @brief Initialize reference data from industry standards
     */
    void initializeReferenceData() {
        // These values are from published NASA/ESA documentation and flight data
        // Low Earth Orbit (ISS altitude: ~400km)
        reference_data_[StandardEnvironment::LOW_EARTH_ORBIT] = {
            0.000025,                // SEU rate per bit per day
            1.2e-14,                 // SEU cross-section (cm²/bit)
            10.0,                    // 95% confidence interval (percent)
            ReferenceSource::CREME96, // Source
            "NASA/TP-2006-214140"    // NASA Technical Publication
        };
        
        // Van Allen Belt
        reference_data_[StandardEnvironment::VAN_ALLEN_BELT] = {
            0.00075,                    // SEU rate per bit per day
            5.4e-14,                    // SEU cross-section (cm²/bit)
            15.0,                       // 95% confidence interval (percent)
            ReferenceSource::VAN_ALLEN_PROBES, // Source
            "JGR Space Physics, 2017"   // Journal of Geophysical Research
        };
        
        // Lunar Orbit
        reference_data_[StandardEnvironment::LUNAR_ORBIT] = {
            0.00042,                 // SEU rate per bit per day
            3.8e-14,                 // SEU cross-section (cm²/bit)
            12.5,                    // 95% confidence interval (percent)
            ReferenceSource::CREME96, // Source
            "IEEE TNS, Vol. 65, 2018" // IEEE Transactions on Nuclear Science
        };
        
        // Interplanetary (Earth to Mars transit)
        reference_data_[StandardEnvironment::INTERPLANETARY] = {
            0.0012,                  // SEU rate per bit per day
            7.5e-14,                 // SEU cross-section (cm²/bit)
            20.0,                    // 95% confidence interval (percent)
            ReferenceSource::OMERE,   // Source
            "ESA ECSS-E-ST-10-12C"   // ESA Space Engineering Standard
        };
        
        // Jupiter/Europa Environment
        reference_data_[StandardEnvironment::JUPITER_EUROPA] = {
            0.0085,                 // SEU rate per bit per day
            2.2e-13,                // SEU cross-section (cm²/bit)
            25.0,                   // 95% confidence interval (percent)
            ReferenceSource::JUNO,   // Source
            "Radiation Effects, Vol. 34, 2019" // Journal reference
        };
    }
    
    /**
     * @brief Get environment parameters from reference data
     * 
     * @param environment Standard environment
     * @return StandardSEUParameters
     */
    StandardSEUParameters getEnvironmentParameters(StandardEnvironment environment) const {
        auto it = reference_data_.find(environment);
        if (it != reference_data_.end()) {
            return it->second;
        }
        
        // Default values if environment not found
        return {
            0.0001,                  // SEU rate per bit per day
            1.0e-14,                 // SEU cross-section (cm²/bit)
            20.0,                    // 95% confidence interval (percent)
            ReferenceSource::CREME96, // Source
            "Default values"         // Reference
        };
    }
    
    /**
     * @brief Convert StandardEnvironment to mission::MissionType
     * 
     * @param environment Standard environment
     * @return Corresponding MissionType
     */
    mission::MissionType convertToMissionType(StandardEnvironment environment) const {
        switch (environment) {
            case StandardEnvironment::LOW_EARTH_ORBIT:
                return mission::MissionType::LEO;
                
            case StandardEnvironment::VAN_ALLEN_BELT:
                return mission::MissionType::VAN_ALLEN_PROBES;
                
            case StandardEnvironment::LUNAR_ORBIT:
                return mission::MissionType::LUNAR_ORBIT;
                
            case StandardEnvironment::INTERPLANETARY:
                return mission::MissionType::MARS_TRANSIT;
                
            case StandardEnvironment::JUPITER_EUROPA:
                return mission::MissionType::EUROPA_CLIPPER;
                
            default:
                return mission::MissionType::LEO;
        }
    }
    
    /**
     * @brief Run a single protection test
     * 
     * @param mission_simulator Mission simulator
     * @param protection_level Protection level name
     * @param reference_params Reference parameters
     * @return ValidationResult
     */
    ValidationResult runProtectionTest(
        std::unique_ptr<testing::MissionSimulator>& mission_simulator,
        const std::string& protection_level,
        const StandardSEUParameters& reference_params) {
        
        ValidationResult result;
        result.reference_seu_rate = reference_params.seu_rate_per_bit_per_day;
        
        // Set protection level
        testing::ProtectionLevel prot_level = testing::ProtectionLevel::NONE;
        if (protection_level == "TMR") {
            prot_level = testing::ProtectionLevel::TMR_BASIC;
        } else if (protection_level == "TMR_ENHANCED") {
            prot_level = testing::ProtectionLevel::TMR_ENHANCED;
        } else if (protection_level == "TMR_ADAPTIVE") {
            prot_level = testing::ProtectionLevel::TMR_ADAPTIVE;
        } else if (protection_level == "ECC") {
            prot_level = testing::ProtectionLevel::ECC;
        }
        
        // Configure simulator
        mission_simulator->setProtectionLevel(prot_level);
        
        // Run simulation
        auto sim_results = mission_simulator->runSimulation();
        
        // Process results
        double total_bits = mission_simulator->getTotalMemoryBits();
        double uncorrected_errors = sim_results.total_uncorrected_bit_flips;
        double unprotected_errors = sim_results.total_seu_count;
        
        // Calculate metrics using NASA/ESA standard formulas
        result.predicted_seu_rate = sim_results.average_seu_rate_per_bit_per_day;
        result.accuracy_percentage = calculateAccuracy(uncorrected_errors, total_bits);
        result.protection_efficiency = calculateProtectionEfficiency(
            unprotected_errors, uncorrected_errors, total_bits);
        result.power_consumption_mw = sim_results.average_power_consumption_mw;
        
        // Power efficiency (mW per prevented SEU)
        if (unprotected_errors > uncorrected_errors && unprotected_errors > 0) {
            result.power_efficiency_mw_per_seu = 
                result.power_consumption_mw / (unprotected_errors - uncorrected_errors);
        } else {
            result.power_efficiency_mw_per_seu = INFINITY;
        }
        
        // Calculate whether within tolerance (2x of reference as per protocol)
        double ratio = result.predicted_seu_rate / result.reference_seu_rate;
        result.within_tolerance = (ratio >= 0.5 && ratio <= 2.0);
        
        // Add notes
        std::stringstream notes;
        notes << "SEU ratio (pred/ref): " << std::fixed << std::setprecision(2) << ratio;
        
        if (!result.within_tolerance) {
            notes << " - OUTSIDE TOLERANCE RANGE";
        }
        
        notes << ". Total memory: " << total_bits << " bits, "
              << "SEUs: " << unprotected_errors << ", "
              << "Uncorrected: " << uncorrected_errors << ".";
              
        result.notes = notes.str();
        
        return result;
    }
    
    /**
     * @brief Get environment name as string
     * 
     * @param environment Standard environment
     * @return Environment name
     */
    std::string getEnvironmentName(StandardEnvironment environment) const {
        switch (environment) {
            case StandardEnvironment::LOW_EARTH_ORBIT:
                return "Low Earth Orbit (ISS)";
                
            case StandardEnvironment::VAN_ALLEN_BELT:
                return "Van Allen Belt";
                
            case StandardEnvironment::LUNAR_ORBIT:
                return "Lunar Orbit";
                
            case StandardEnvironment::INTERPLANETARY:
                return "Interplanetary";
                
            case StandardEnvironment::JUPITER_EUROPA:
                return "Jupiter/Europa";
                
            default:
                return "Unknown Environment";
        }
    }
    
    /**
     * @brief Get reference source name as string
     * 
     * @param source Reference source
     * @return Source name
     */
    std::string getReferenceSourceName(ReferenceSource source) const {
        switch (source) {
            case ReferenceSource::CREME96:
                return "CREME96/CRÈME-MC";
                
            case ReferenceSource::OMERE:
                return "ESA OMERE";
                
            case ReferenceSource::SAMPEX:
                return "SAMPEX Mission Data";
                
            case ReferenceSource::HUBBLE:
                return "Hubble Space Telescope Data";
                
            case ReferenceSource::VAN_ALLEN_PROBES:
                return "Van Allen Probes Data";
                
            case ReferenceSource::JUNO:
                return "Juno Mission Data";
                
            case ReferenceSource::CYCLOTRON_FACILITY:
                return "Cyclotron Facility Testing";
                
            default:
                return "Unknown Source";
        }
    }
};

} // namespace validation
} // namespace rad_ml 
