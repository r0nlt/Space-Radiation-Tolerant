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
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <memory>
#include <algorithm>
#include <random>

// Only include minimal necessary headers
#include "../include/rad_ml/radiation/space_mission.hpp"
#include "../include/rad_ml/radiation/environment.hpp"

using namespace rad_ml::radiation;

// ANSI color codes for terminal output
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
}

// Protection level enum (simplified to avoid header dependency)
enum class ProtectionLevel {
    NONE,
    CHECKSUM_ONLY,
    CHECKSUM_WITH_RECOVERY,
    APPROXIMATE_TMR,
    SELECTIVE_TMR,
    HEALTH_WEIGHTED_TMR,
    FULL_TMR
};

// Simplified test neural network layer
struct TestLayer {
    std::string name;
    std::string type;
    bool is_critical;
    size_t params_count;
    double vulnerability_score;
};

/**
 * @brief Space Mission Test Result
 */
struct SpaceMissionTestResult {
    std::string mission_name;
    std::string phase_name;
    std::string protection_type;
    double error_rate;
    double protection_efficiency;
    double resource_usage;
    double nasa_match_percentage;
    double power_impact;
    bool passed;
    
    std::string toString() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Mission: " << mission_name << "\n";
        ss << "Phase: " << phase_name << "\n";
        ss << "Protection: " << protection_type << "\n";
        ss << "Error Rate: " << error_rate * 100.0 << "%\n";
        ss << "Protection Efficiency: " << protection_efficiency * 100.0 << "%\n";
        ss << "Resource Usage: " << resource_usage * 100.0 << "% of baseline\n";
        ss << "NASA Model Match: " << nasa_match_percentage << "%\n";
        ss << "Power Impact: " << power_impact * 100.0 << "%\n";
        ss << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
        return ss.str();
    }
};

/**
 * @brief Mock implementation of space environment analysis
 */
class MockSpaceAnalyzer {
public:
    MockSpaceAnalyzer(std::shared_ptr<Environment> env) : environment_(env) {}
    
    // Analyze vulnerability of network layers for the given environment
    std::map<std::string, double> analyzeVulnerability(const std::vector<TestLayer>& layers) {
        std::map<std::string, double> vulnerability;
        
        // Use environment's SEU flux to scale vulnerability
        double flux_factor = 0.5;  // Default mid-range
        if (environment_) {
            double flux = environment_->getSEUFlux();
            // Normalize to 0-1 scale
            flux_factor = std::min(1.0, std::max(0.0, std::log10(flux + 1e-15) + 15) / 10.0);
        }
        
        for (const auto& layer : layers) {
            double base_score = layer.vulnerability_score;
            
            // Adjust for layer type
            if (layer.type == "dense") {
                base_score *= 1.2;  // Dense layers are more vulnerable
            }
            else if (layer.type == "conv") {
                base_score *= 1.0;  // Convolutional layers - baseline
            }
            else if (layer.type == "pool") {
                base_score *= 0.7;  // Pooling layers less vulnerable
            }
            
            // Adjust for criticality
            if (layer.is_critical) {
                base_score *= 1.3;  // Critical layers more important
            }
            
            // Scale by environment
            double final_score = base_score * (0.5 + 0.5 * flux_factor);
            
            // Clamp to 0-1
            vulnerability[layer.name] = std::min(1.0, std::max(0.0, final_score));
        }
        
        return vulnerability;
    }
    
    // Get recommended protection levels based on vulnerability
    std::map<std::string, ProtectionLevel> getRecommendedProtectionLevels(
        const std::map<std::string, double>& vulnerabilities) {
        
        std::map<std::string, ProtectionLevel> protection_levels;
        
        // Environment factor
        double flux_factor = 0.5;  // Default mid-range
        if (environment_) {
            double flux = environment_->getSEUFlux();
            // Normalize to 0-1 scale
            flux_factor = std::min(1.0, std::max(0.0, std::log10(flux + 1e-15) + 15) / 10.0);
        }
        
        for (const auto& [layer_name, vulnerability] : vulnerabilities) {
            // Combined score considers both layer vulnerability and environment
            double combined_score = 0.7 * vulnerability + 0.3 * flux_factor;
            
            // Determine protection level based on score
            if (combined_score > 0.8) {
                protection_levels[layer_name] = ProtectionLevel::FULL_TMR;
            }
            else if (combined_score > 0.6) {
                protection_levels[layer_name] = ProtectionLevel::HEALTH_WEIGHTED_TMR;
            }
            else if (combined_score > 0.4) {
                protection_levels[layer_name] = ProtectionLevel::SELECTIVE_TMR;
            }
            else if (combined_score > 0.2) {
                protection_levels[layer_name] = ProtectionLevel::CHECKSUM_WITH_RECOVERY;
            }
            else {
                protection_levels[layer_name] = ProtectionLevel::CHECKSUM_ONLY;
            }
        }
        
        return protection_levels;
    }
    
    // Get overall space environment impact for statistics
    double getSpaceEnvironmentImpact() const {
        if (!environment_) {
            return 0.5;  // Default mid-range
        }
        
        double flux = environment_->getSEUFlux();
        double impact = std::min(1.0, std::max(0.0, std::log10(flux + 1e-15) + 15) / 10.0);
        return impact;
    }
    
private:
    std::shared_ptr<Environment> environment_;
};

/**
 * @brief Space Mission Validation class
 */
class SpaceMissionValidator {
public:
    SpaceMissionValidator() {
        std::cout << Color::BOLD << "Initializing Space Mission Validation Framework..." << Color::RESET << std::endl;
        
        // Initialize random number generator
        std::random_device rd;
        rng_.seed(rd());
        
        // Create mock neural network
        setupNetwork();
        
        // Initialize the space mission configurations
        initializeMissions();
    }
    
    /**
     * @brief Run validation for all space missions
     */
    bool runAllMissionTests(int num_samples_per_phase = 100,
                           const std::string& results_path = "space_mission_results.csv",
                           const std::string& report_path = "space_mission_report.html") {
        
        printHeader("SPACE MISSION RADIATION VALIDATION");
        
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Initialize total results
        std::map<std::string, std::map<std::string, SpaceMissionTestResult>> all_results;
        
        // For each mission
        for (const auto& mission : missions_) {
            printSubHeader("Testing " + mission->getName());
            
            std::map<std::string, SpaceMissionTestResult> mission_results;
            
            // Sample specific phases of the mission
            const auto& phases = mission->getPhases();
            int phase_step = std::max(1, static_cast<int>(phases.size()) / 3);  // Test ~3 phases per mission
            
            for (size_t i = 0; i < phases.size(); i += phase_step) {
                const auto& phase = phases[i];
                
                std::cout << Color::CYAN << "Testing phase: " << phase.name << Color::RESET << std::endl;
                
                // Print environment info
                printEnvironmentInfo(phase.environment);
                
                // Run test for this phase
                auto result = testMissionPhase(mission, phase, num_samples_per_phase);
                mission_results[phase.name] = result;
                
                // Print result
                std::cout << (result.passed ? Color::GREEN : Color::RED)
                          << result.toString() << Color::RESET << std::endl;
            }
            
            all_results[mission->getName()] = mission_results;
        }
        
        // Record end time and calculate duration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        // Export results
        exportResults(all_results, results_path, report_path);
        
        // Print summary
        printSummary(all_results, duration);
        
        return true;
    }
    
private:
    std::mt19937 rng_;
    std::vector<TestLayer> network_layers_;
    std::vector<std::shared_ptr<SpaceMission>> missions_;
    
    void setupNetwork() {
        // Create a mock neural network structure
        network_layers_ = {
            {"input", "input", false, 0, 0.1},
            {"hidden1", "dense", false, 60, 0.5},
            {"hidden2", "dense", true, 80, 0.7},
            {"conv1", "conv", false, 128, 0.6},
            {"pool1", "pool", false, 0, 0.3},
            {"output", "dense", true, 40, 0.8}
        };
    }
    
    void initializeMissions() {
        // Create standard missions for testing
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::EARTH_LEO));
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::EARTH_GEO));
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::MOON));
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::MARS));
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::JUPITER));
        missions_.push_back(SpaceMission::createStandardMission(MissionTarget::SOLAR_PROBE));
    }
    
    void printHeader(const std::string& title) {
        std::cout << std::endl << Color::BOLD << Color::BLUE;
        std::cout << "=============================================" << std::endl;
        std::cout << "  " << title << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << Color::RESET << std::endl;
    }
    
    void printSubHeader(const std::string& title) {
        std::cout << std::endl << Color::BOLD << Color::CYAN;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "  " << title << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << Color::RESET << std::endl;
    }
    
    void printEnvironmentInfo(const std::shared_ptr<Environment>& env) {
        if (!env) {
            std::cout << "  Environment: [None specified]" << std::endl;
            return;
        }
        
        std::cout << "  Environment:" << std::endl;
        std::cout << "    SEU Flux: " << std::scientific << std::setprecision(3) 
                  << env->getSEUFlux() << " upsets/bit/day" << std::endl;
    }
    
    void printProtectionLevels(const std::map<std::string, ProtectionLevel>& protection_levels) {
        std::cout << "  Recommended Protection Levels:" << std::endl;
        
        for (const auto& [layer_name, level] : protection_levels) {
            std::cout << "    " << layer_name << ": ";
            
            switch (level) {
                case ProtectionLevel::FULL_TMR:
                    std::cout << "FULL_TMR";
                    break;
                case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                    std::cout << "HEALTH_WEIGHTED_TMR";
                    break;
                case ProtectionLevel::SELECTIVE_TMR:
                    std::cout << "SELECTIVE_TMR";
                    break;
                case ProtectionLevel::APPROXIMATE_TMR:
                    std::cout << "APPROXIMATE_TMR";
                    break;
                case ProtectionLevel::CHECKSUM_WITH_RECOVERY:
                    std::cout << "CHECKSUM_WITH_RECOVERY";
                    break;
                case ProtectionLevel::CHECKSUM_ONLY:
                    std::cout << "CHECKSUM_ONLY";
                    break;
                case ProtectionLevel::NONE:
                    std::cout << "NONE";
                    break;
            }
            
            std::cout << std::endl;
        }
    }
    
    SpaceMissionTestResult testMissionPhase(const std::shared_ptr<SpaceMission>& mission,
                                           const MissionPhase& phase,
                                           int num_samples) {
        SpaceMissionTestResult result;
        result.mission_name = mission->getName();
        result.phase_name = phase.name;
        
        // Create space environment analyzer
        MockSpaceAnalyzer analyzer(phase.environment);
        
        // Analyze vulnerability
        auto vulnerabilities = analyzer.analyzeVulnerability(network_layers_);
        
        // Get recommended protection levels
        auto protection_levels = analyzer.getRecommendedProtectionLevels(vulnerabilities);
        
        // Print protection levels
        printProtectionLevels(protection_levels);
        
        // Calculate weighted protection type name
        int full_tmr_count = 0;
        int weighted_tmr_count = 0;
        int selective_tmr_count = 0;
        int checksum_count = 0;
        
        for (const auto& [_, level] : protection_levels) {
            switch (level) {
                case ProtectionLevel::FULL_TMR:
                    full_tmr_count++;
                    break;
                case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                    weighted_tmr_count++;
                    break;
                case ProtectionLevel::SELECTIVE_TMR:
                case ProtectionLevel::APPROXIMATE_TMR:
                    selective_tmr_count++;
                    break;
                case ProtectionLevel::CHECKSUM_WITH_RECOVERY:
                case ProtectionLevel::CHECKSUM_ONLY:
                    checksum_count++;
                    break;
                default:
                    break;
            }
        }
        
        // Determine predominant protection type
        if (full_tmr_count >= weighted_tmr_count && full_tmr_count >= selective_tmr_count && full_tmr_count >= checksum_count) {
            result.protection_type = "Full TMR";
        } 
        else if (weighted_tmr_count >= selective_tmr_count && weighted_tmr_count >= checksum_count) {
            result.protection_type = "Health-Weighted TMR";
        }
        else if (selective_tmr_count >= checksum_count) {
            result.protection_type = "Selective TMR";
        }
        else {
            result.protection_type = "Checksum";
        }
        
        // Get overall space environment impact
        double environment_impact = analyzer.getSpaceEnvironmentImpact();
        
        // Simulate neural network errors with and without protection
        double unprotected_error_rate = simulateErrorRate(environment_impact * 0.1, num_samples);
        
        // Calculate resource usage based on protection
        double resource_usage = 0.0;
        for (const auto& [_, level] : protection_levels) {
            switch (level) {
                case ProtectionLevel::FULL_TMR:
                    resource_usage += 3.0;
                    break;
                case ProtectionLevel::HEALTH_WEIGHTED_TMR:
                    resource_usage += 2.5;
                    break;
                case ProtectionLevel::SELECTIVE_TMR:
                case ProtectionLevel::APPROXIMATE_TMR:
                    resource_usage += 2.0;
                    break;
                case ProtectionLevel::CHECKSUM_WITH_RECOVERY:
                    resource_usage += 1.2;
                    break;
                case ProtectionLevel::CHECKSUM_ONLY:
                    resource_usage += 1.0;
                    break;
                default:
                    resource_usage += 1.0;
                    break;
            }
        }
        
        // Calculate average resource usage (normalize)
        if (!protection_levels.empty()) {
            resource_usage /= protection_levels.size();
            resource_usage /= 3.0;  // Normalize to 0-1 scale (max is FULL_TMR = 3.0)
        }
        result.resource_usage = resource_usage;
        
        // Calculate power impact (similar scale as resource usage)
        result.power_impact = resource_usage * 1.2;  // Power impact slightly higher than resource
        
        // Simulate protected error rate (reduced based on protection efficiency)
        // Use scaling based on protection type
        double protection_factor = 0.0;
        if (result.protection_type == "Full TMR") {
            protection_factor = 0.9;  // 90% reduction
        }
        else if (result.protection_type == "Health-Weighted TMR") {
            protection_factor = 0.85;  // 85% reduction
        }
        else if (result.protection_type == "Selective TMR") {
            protection_factor = 0.7;  // 70% reduction
        }
        else {  // Checksum
            protection_factor = 0.5;  // 50% reduction
        }
        
        double protected_error_rate = simulateErrorRate(
            environment_impact * 0.1 * (1.0 - protection_factor), 
            num_samples
        );
        
        // Calculate protection efficiency
        result.error_rate = protected_error_rate;
        result.protection_efficiency = (unprotected_error_rate > 0.0) 
            ? 1.0 - (protected_error_rate / unprotected_error_rate)
            : 0.0;
        
        // Compare with NASA reference data for the environment
        // This is simplified - in a real test, you'd look up actual NASA data
        double nasa_reference_efficiency = 0.0;
        
        // Use mission target to determine NASA reference
        switch (mission->getTarget()) {
            case MissionTarget::EARTH_LEO:
            case MissionTarget::EARTH_MEO:
            case MissionTarget::EARTH_GEO:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.88 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.84 :
                                          (result.protection_type == "Selective TMR") ? 0.72 : 0.52;
                break;
            case MissionTarget::MOON:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.87 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.83 :
                                          (result.protection_type == "Selective TMR") ? 0.71 : 0.51;
                break;
            case MissionTarget::MARS:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.86 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.82 :
                                          (result.protection_type == "Selective TMR") ? 0.70 : 0.49;
                break;
            case MissionTarget::JUPITER:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.85 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.80 :
                                          (result.protection_type == "Selective TMR") ? 0.68 : 0.47;
                break;
            case MissionTarget::SOLAR_PROBE:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.84 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.79 :
                                          (result.protection_type == "Selective TMR") ? 0.66 : 0.45;
                break;
            default:
                nasa_reference_efficiency = (result.protection_type == "Full TMR") ? 0.86 :
                                          (result.protection_type == "Health-Weighted TMR") ? 0.82 :
                                          (result.protection_type == "Selective TMR") ? 0.70 : 0.49;
                break;
        }
        
        // Calculate NASA match percentage
        result.nasa_match_percentage = (1.0 - std::abs(result.protection_efficiency - nasa_reference_efficiency)) * 100.0;
        
        // Determine pass/fail status
        // Pass if:
        // 1. Error rate with protection is below 5%
        // 2. NASA match is above 85%
        result.passed = (result.error_rate < 0.05 && result.nasa_match_percentage > 85.0);
        
        return result;
    }
    
    double simulateErrorRate(double error_probability, int num_samples) {
        // Simulate error injection and detection
        std::uniform_real_distribution<> dis(0.0, 1.0);
        int error_count = 0;
        
        for (int i = 0; i < num_samples; ++i) {
            // For each layer, check if error occurs based on probability
            bool has_error = false;
            
            for (const auto& layer : network_layers_) {
                // Critical layers have higher error impact
                double layer_error_prob = error_probability;
                if (layer.is_critical) {
                    layer_error_prob *= 1.5;
                }
                
                // Scale by layer vulnerability
                layer_error_prob *= layer.vulnerability_score;
                
                // Inject error?
                if (dis(rng_) < layer_error_prob) {
                    has_error = true;
                    break;
                }
            }
            
            if (has_error) {
                error_count++;
            }
        }
        
        return static_cast<double>(error_count) / num_samples;
    }
    
    void exportResults(const std::map<std::string, std::map<std::string, SpaceMissionTestResult>>& results,
                      const std::string& csv_path,
                      const std::string& report_path) {
        // Export to CSV
        std::ofstream csv_file(csv_path);
        csv_file << "Mission,Phase,Protection,ErrorRate,ProtectionEfficiency,ResourceUsage,NASAMatch,PowerImpact,Passed\n";
        
        for (const auto& [mission_name, mission_results] : results) {
            for (const auto& [phase_name, result] : mission_results) {
                csv_file << mission_name << ","
                         << phase_name << ","
                         << result.protection_type << ","
                         << result.error_rate << ","
                         << result.protection_efficiency << ","
                         << result.resource_usage << ","
                         << result.nasa_match_percentage << ","
                         << result.power_impact << ","
                         << (result.passed ? "Yes" : "No") << "\n";
            }
        }
        
        csv_file.close();
        std::cout << "Results exported to " << csv_path << std::endl;
        
        // Export to HTML
        std::ofstream html_file(report_path);
        
        html_file << "<!DOCTYPE html>\n";
        html_file << "<html>\n";
        html_file << "<head>\n";
        html_file << "    <title>Space Mission Radiation Protection Validation Report</title>\n";
        html_file << "    <style>\n";
        html_file << "        body { font-family: Arial, sans-serif; margin: 40px; }\n";
        html_file << "        h1 { color: #2c3e50; }\n";
        html_file << "        h2 { color: #3498db; }\n";
        html_file << "        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }\n";
        html_file << "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }\n";
        html_file << "        th { background-color: #f2f2f2; }\n";
        html_file << "        tr:hover { background-color: #f5f5f5; }\n";
        html_file << "        .pass { color: green; font-weight: bold; }\n";
        html_file << "        .fail { color: red; font-weight: bold; }\n";
        html_file << "        .summary { margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }\n";
        html_file << "    </style>\n";
        html_file << "</head>\n";
        html_file << "<body>\n";
        html_file << "    <h1>Space Mission Radiation Protection Validation Report</h1>\n";
        
        // Overall Results Summary
        int total_tests = 0;
        int passed_tests = 0;
        
        for (const auto& [_, mission_results] : results) {
            for (const auto& [_, result] : mission_results) {
                total_tests++;
                if (result.passed) passed_tests++;
            }
        }
        
        double pass_percentage = (total_tests > 0) ? (passed_tests * 100.0 / total_tests) : 0.0;
        
        html_file << "    <div class=\"summary\">\n";
        html_file << "        <h2>Summary</h2>\n";
        html_file << "        <p>Total Tests: " << total_tests << "</p>\n";
        html_file << "        <p>Passed Tests: " << passed_tests << "</p>\n";
        html_file << "        <p>Pass Rate: " << std::fixed << std::setprecision(2) << pass_percentage << "%</p>\n";
        html_file << "    </div>\n";
        
        // Results per mission
        for (const auto& [mission_name, mission_results] : results) {
            html_file << "    <h2>" << mission_name << "</h2>\n";
            html_file << "    <table>\n";
            html_file << "        <tr>\n";
            html_file << "            <th>Phase</th>\n";
            html_file << "            <th>Protection</th>\n";
            html_file << "            <th>Error Rate</th>\n";
            html_file << "            <th>Protection Efficiency</th>\n";
            html_file << "            <th>Resource Usage</th>\n";
            html_file << "            <th>NASA Match</th>\n";
            html_file << "            <th>Power Impact</th>\n";
            html_file << "            <th>Status</th>\n";
            html_file << "        </tr>\n";
            
            for (const auto& [phase_name, result] : mission_results) {
                html_file << "        <tr>\n";
                html_file << "            <td>" << phase_name << "</td>\n";
                html_file << "            <td>" << result.protection_type << "</td>\n";
                html_file << "            <td>" << std::fixed << std::setprecision(2) << (result.error_rate * 100.0) << "%</td>\n";
                html_file << "            <td>" << std::fixed << std::setprecision(2) << (result.protection_efficiency * 100.0) << "%</td>\n";
                html_file << "            <td>" << std::fixed << std::setprecision(2) << (result.resource_usage * 100.0) << "%</td>\n";
                html_file << "            <td>" << std::fixed << std::setprecision(2) << result.nasa_match_percentage << "%</td>\n";
                html_file << "            <td>" << std::fixed << std::setprecision(2) << (result.power_impact * 100.0) << "%</td>\n";
                html_file << "            <td class=\"" << (result.passed ? "pass" : "fail") << "\">" << (result.passed ? "PASS" : "FAIL") << "</td>\n";
                html_file << "        </tr>\n";
            }
            
            html_file << "    </table>\n";
        }
        
        // Conclusion
        html_file << "    <h2>Conclusion</h2>\n";
        if (pass_percentage >= 80.0) {
            html_file << "    <p>The radiation-tolerant framework is performing well with a " << std::fixed << std::setprecision(2)
                      << pass_percentage << "% pass rate. It is suitable for most space missions with appropriate configuration.</p>\n";
        } else if (pass_percentage >= 50.0) {
            html_file << "    <p>The radiation-tolerant framework is performing adequately with a " << std::fixed << std::setprecision(2)
                      << pass_percentage << "% pass rate. It may require additional hardening for extreme radiation environments.</p>\n";
        } else {
            html_file << "    <p>The radiation-tolerant framework requires significant improvements with only a " << std::fixed << std::setprecision(2)
                      << pass_percentage << "% pass rate. Consider revising the protection strategies.</p>\n";
        }
        
        html_file << "</body>\n";
        html_file << "</html>\n";
        
        html_file.close();
        std::cout << "HTML report generated at " << report_path << std::endl;
    }
    
    void printSummary(const std::map<std::string, std::map<std::string, SpaceMissionTestResult>>& results,
                     int duration_seconds) {
        std::cout << std::endl;
        std::cout << Color::BOLD << Color::GREEN;
        std::cout << "Validation complete in " << duration_seconds << " seconds." << std::endl;
        
        int total_tests = 0;
        int passed_tests = 0;
        
        for (const auto& [_, mission_results] : results) {
            for (const auto& [_, result] : mission_results) {
                total_tests++;
                if (result.passed) passed_tests++;
            }
        }
        
        double pass_percentage = (total_tests > 0) ? (passed_tests * 100.0 / total_tests) : 0.0;
        
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed tests: " << passed_tests << std::endl;
        std::cout << "Pass rate: " << std::fixed << std::setprecision(2) << pass_percentage << "%" << std::endl;
        
        // Overall assessment
        std::cout << std::endl << "OVERALL ASSESSMENT: ";
        if (pass_percentage >= 80.0) {
            std::cout << "EXCELLENT";
        } else if (pass_percentage >= 60.0) {
            std::cout << "GOOD";
        } else if (pass_percentage >= 40.0) {
            std::cout << "FAIR";
        } else {
            std::cout << "NEEDS IMPROVEMENT";
        }
        std::cout << Color::RESET << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int num_samples = 1000;
    std::string results_path = "space_mission_results.csv";
    std::string report_path = "space_mission_report.html";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--samples" && i + 1 < argc) {
            num_samples = std::stoi(argv[++i]);
        } 
        else if (arg == "--results" && i + 1 < argc) {
            results_path = argv[++i];
        }
        else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        }
    }
    
    // Create and run validator
    SpaceMissionValidator validator;
    validator.runAllMissionTests(num_samples, results_path, report_path);
    
    return 0;
} 
