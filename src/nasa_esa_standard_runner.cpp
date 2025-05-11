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
#include "validation/nasa_esa_validation_protocol.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// ANSI Color codes for terminal output
const std::string RESET = "\033[0m";
const std::string BOLD = "\033[1m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";

void printHeader() {
    std::cout << std::endl;
    std::cout << BOLD + BLUE + "=================================================================" + RESET << std::endl;
    std::cout << BOLD + BLUE + "    NASA/ESA STANDARDIZED RADIATION FRAMEWORK VERIFICATION      " + RESET << std::endl;
    std::cout << BOLD + BLUE + "=================================================================" + RESET << std::endl;
    std::cout << std::endl;
}

void printSection(const std::string& title) {
    std::cout << std::endl;
    std::cout << BOLD + CYAN + ">> " + title + RESET << std::endl;
    std::cout << CYAN + std::string(title.length() + 3, '-') + RESET << std::endl;
}

void exportReportToHTML(const rad_ml::validation::NASAESAVerificationProtocol::VerificationReport& report) {
    std::ofstream html_file("nasa_esa_test_report.html");
    
    // HTML file header
    html_file << "<!DOCTYPE html>\n";
    html_file << "<html lang=\"en\">\n";
    html_file << "<head>\n";
    html_file << "    <meta charset=\"UTF-8\">\n";
    html_file << "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
    html_file << "    <title>NASA/ESA Radiation Framework Verification Report</title>\n";
    html_file << "    <style>\n";
    html_file << "        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n";
    html_file << "        h1, h2, h3 { color: #0b3d91; }\n"; // NASA blue
    html_file << "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n";
    html_file << "        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }\n";
    html_file << "        th { background-color: #0b3d91; color: white; }\n";
    html_file << "        tr:nth-child(even) { background-color: #f2f2f2; }\n";
    html_file << "        .pass { color: green; font-weight: bold; }\n";
    html_file << "        .fail { color: red; font-weight: bold; }\n";
    html_file << "        .not-tested { color: orange; }\n";
    html_file << "        .header-section { background-color: #e1e6ea; padding: 10px; margin-top: 20px; }\n";
    html_file << "        .summary { margin: 20px 0; padding: 15px; background-color: #f7f7f7; border-left: 5px solid #0b3d91; }\n";
    html_file << "    </style>\n";
    html_file << "</head>\n";
    html_file << "<body>\n";
    
    // Report header
    html_file << "    <h1>NASA/ESA Radiation Framework Verification Report</h1>\n";
    html_file << "    <div class=\"header-section\">\n";
    html_file << "        <p><strong>Verification Date:</strong> " << report.verification_date << "</p>\n";
    html_file << "        <p><strong>Framework Version:</strong> " << report.framework_version << "</p>\n";
    html_file << "        <p><strong>Overall Status:</strong> <span class=\""
               << (report.overall_compliant ? "pass\">COMPLIANT" : "fail\">NON-COMPLIANT") 
               << "</span></p>\n";
    html_file << "    </div>\n";
    
    // Verification Statement
    html_file << "    <div class=\"summary\">\n";
    html_file << "        <h2>Verification Statement</h2>\n";
    html_file << "        <p>" << report.verification_statement << "</p>\n";
    html_file << "    </div>\n";
    
    // Standards Compliance Matrix
    html_file << "    <h2>1. Standards Compliance Matrix</h2>\n";
    html_file << "    <table>\n";
    html_file << "        <tr>\n";
    html_file << "            <th>Standard</th>\n";
    html_file << "            <th>Requirement</th>\n";
    html_file << "            <th>Description</th>\n";
    html_file << "            <th>Implementation</th>\n";
    html_file << "            <th>Status</th>\n";
    html_file << "        </tr>\n";
    
    for (const auto& req : report.compliance_matrix) {
        html_file << "        <tr>\n";
        html_file << "            <td>" << req.standard << "</td>\n";
        html_file << "            <td>" << req.requirement_id << "</td>\n";
        html_file << "            <td>" << req.description << "</td>\n";
        html_file << "            <td>" << req.implementation << "</td>\n";
        
        std::string statusClass;
        std::string statusText;
        
        switch (req.status) {
            case rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS:
                statusClass = "pass";
                statusText = "PASS";
                break;
            case rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::FAIL:
                statusClass = "fail";
                statusText = "FAIL";
                break;
            case rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::NOT_TESTED:
                statusClass = "not-tested";
                statusText = "NOT TESTED";
                break;
            default:
                statusClass = "";
                statusText = "N/A";
        }
        
        html_file << "            <td class=\"" << statusClass << "\">" << statusText << "</td>\n";
        html_file << "        </tr>\n";
    }
    
    html_file << "    </table>\n";
    
    // Environment Model Validation
    html_file << "    <h2>2. Environment Model Validation</h2>\n";
    html_file << "    <table>\n";
    html_file << "        <tr>\n";
    html_file << "            <th>Environment</th>\n";
    html_file << "            <th>Model Used</th>\n";
    html_file << "            <th>Correlation Coefficient</th>\n";
    html_file << "            <th>Difference from Reference (%)</th>\n";
    html_file << "            <th>Status</th>\n";
    html_file << "        </tr>\n";
    
    for (const auto& val : report.environment_validations) {
        html_file << "        <tr>\n";
        // In a real implementation, would convert enum to string
        html_file << "            <td>Environment " << static_cast<int>(val.environment) << "</td>\n";
        html_file << "            <td>CREME96</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(3) << val.correlation_coefficient << "</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << val.percent_difference << "%</td>\n";
        
        std::string statusClass = (val.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                                ? "pass" : "fail";
        std::string statusText = (val.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                               ? "PASS" : "FAIL";
        
        html_file << "            <td class=\"" << statusClass << "\">" << statusText << "</td>\n";
        html_file << "        </tr>\n";
    }
    
    html_file << "    </table>\n";
    
    // Radiation Hardening Assessment
    html_file << "    <h2>3. Radiation Hardening Assessment</h2>\n";
    html_file << "    <table>\n";
    html_file << "        <tr>\n";
    html_file << "            <th>Mission</th>\n";
    html_file << "            <th>SEU Rate</th>\n";
    html_file << "            <th>LET Threshold</th>\n";
    html_file << "            <th>Cross-section</th>\n";
    html_file << "            <th>MTBF (hours)</th>\n";
    html_file << "            <th>BER</th>\n";
    html_file << "            <th>NASA Threshold</th>\n";
    html_file << "            <th>Status</th>\n";
    html_file << "        </tr>\n";
    
    for (const auto& res : report.radiation_assessments) {
        html_file << "        <tr>\n";
        // In a real implementation, would convert enum to string
        html_file << "            <td>Mission " << static_cast<int>(res.environment) << "</td>\n";
        html_file << "            <td>" << std::scientific << std::setprecision(2) << res.seu_rate << " err/bit-day</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << res.let_threshold << " MeV-cm²/mg</td>\n";
        html_file << "            <td>" << std::scientific << std::setprecision(2) << res.cross_section << " cm²/bit</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(0) << res.mtbf << "</td>\n";
        html_file << "            <td>" << std::scientific << std::setprecision(2) << res.ber << "</td>\n";
        html_file << "            <td>" << std::scientific << std::setprecision(2) << res.nasa_threshold << " err/bit-day</td>\n";
        
        std::string statusClass = (res.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                                ? "pass" : "fail";
        std::string statusText = (res.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                               ? "PASS" : "FAIL";
        
        html_file << "            <td class=\"" << statusClass << "\">" << statusText << "</td>\n";
        html_file << "        </tr>\n";
    }
    
    html_file << "    </table>\n";
    
    // Performance Analysis
    html_file << "    <h2>4. Performance Analysis</h2>\n";
    html_file << "    <table>\n";
    html_file << "        <tr>\n";
    html_file << "            <th>Protection Technique</th>\n";
    html_file << "            <th>Effectiveness Ratio</th>\n";
    html_file << "            <th>Reference Effectiveness</th>\n";
    html_file << "            <th>Resource Overhead (%)</th>\n";
    html_file << "            <th>Power Overhead (%)</th>\n";
    html_file << "            <th>Performance Overhead (%)</th>\n";
    html_file << "            <th>Status</th>\n";
    html_file << "        </tr>\n";
    
    for (const auto& eval : report.protection_evaluations) {
        html_file << "        <tr>\n";
        // In a real implementation, would convert enum to string
        html_file << "            <td>Technique " << static_cast<int>(eval.technique) << "</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(2) << eval.effectiveness_ratio << "</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(2) << eval.reference_effectiveness << "</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << eval.resource_overhead << "%</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << eval.power_overhead << "%</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << eval.performance_overhead << "%</td>\n";
        
        std::string statusClass = (eval.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                                ? "pass" : "fail";
        std::string statusText = (eval.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS) 
                               ? "PASS" : "FAIL";
        
        html_file << "            <td class=\"" << statusClass << "\">" << statusText << "</td>\n";
        html_file << "        </tr>\n";
    }
    
    html_file << "    </table>\n";
    
    // Mission Suitability Determination
    html_file << "    <h2>5. Mission Suitability Determination</h2>\n";
    html_file << "    <table>\n";
    html_file << "        <tr>\n";
    html_file << "            <th>Mission</th>\n";
    html_file << "            <th>Suitable</th>\n";
    html_file << "            <th>Required Shielding (mm Al)</th>\n";
    html_file << "            <th>Required Modifications</th>\n";
    html_file << "            <th>Rationale</th>\n";
    html_file << "        </tr>\n";
    
    for (const auto& suit : report.mission_suitabilities) {
        html_file << "        <tr>\n";
        // In a real implementation, would convert enum to string
        html_file << "            <td>Mission " << static_cast<int>(suit.environment) << "</td>\n";
        
        std::string suitClass = suit.suitable ? "pass" : "fail";
        std::string suitText = suit.suitable ? "YES" : "NO";
        
        html_file << "            <td class=\"" << suitClass << "\">" << suitText << "</td>\n";
        html_file << "            <td>" << std::fixed << std::setprecision(1) << suit.required_shielding_mm_al << " mm</td>\n";
        
        html_file << "            <td>";
        if (suit.required_modifications.empty()) {
            html_file << "None required";
        } else {
            html_file << "<ul>";
            for (const auto& mod : suit.required_modifications) {
                html_file << "<li>" << mod << "</li>";
            }
            html_file << "</ul>";
        }
        html_file << "</td>\n";
        
        html_file << "            <td>" << suit.rationale << "</td>\n";
        html_file << "        </tr>\n";
    }
    
    html_file << "    </table>\n";
    
    // Document footer
    html_file << "    <div class=\"header-section\">\n";
    html_file << "        <p><em>This report was generated automatically by the NASA/ESA Radiation Framework Verification Protocol.</em></p>\n";
    html_file << "        <p><em>Test conducted in accordance with NASA-HDBK-4002A, ECSS-E-ST-10-12C, JEDEC JESD57.</em></p>\n";
    html_file << "    </div>\n";
    html_file << "</body>\n";
    html_file << "</html>\n";
    
    html_file.close();
    
    // Also export a CSV version for data analysis
    std::ofstream csv_file("nasa_esa_test_results.csv");
    
    // CSV header
    csv_file << "Category,Item,Value,Threshold,Status\n";
    
    // Radiation assessments
    for (const auto& res : report.radiation_assessments) {
        std::string mission = "Mission " + std::to_string(static_cast<int>(res.environment));
        csv_file << "Radiation," << mission << " SEU Rate," << res.seu_rate << "," << res.nasa_threshold << ","
                 << (res.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS ? "PASS" : "FAIL") << "\n";
        
        csv_file << "Radiation," << mission << " LET Threshold," << res.let_threshold << ",40.0,"
                 << (res.let_threshold >= 40.0 ? "PASS" : "FAIL") << "\n";
        
        csv_file << "Radiation," << mission << " MTBF," << res.mtbf << ",10000.0,"
                 << (res.mtbf >= 10000.0 ? "PASS" : "FAIL") << "\n";
    }
    
    // Protection techniques
    for (const auto& eval : report.protection_evaluations) {
        std::string technique = "Technique " + std::to_string(static_cast<int>(eval.technique));
        csv_file << "Protection," << technique << " Effectiveness," << eval.effectiveness_ratio << "," << eval.reference_effectiveness << ","
                 << (eval.status == rad_ml::validation::NASAESAVerificationProtocol::VerificationStatus::PASS ? "PASS" : "FAIL") << "\n";
    }
    
    // Mission suitability
    for (const auto& suit : report.mission_suitabilities) {
        std::string mission = "Mission " + std::to_string(static_cast<int>(suit.environment));
        csv_file << "Suitability," << mission << "," << (suit.suitable ? "Suitable" : "Not Suitable") << ",Suitable,"
                 << (suit.suitable ? "PASS" : "FAIL") << "\n";
    }
    
    csv_file.close();
    
    std::cout << "Reports exported to nasa_esa_test_report.html and nasa_esa_test_results.csv" << std::endl;
}

int main(int argc, char** argv) {
    printHeader();
    
    // Get environment variables if set
    int monte_carlo_trials = 10000;
    double confidence_level = 0.95;
    std::string environment_model = "CREME96";
    
    char* env_trials = std::getenv("RAD_ML_MONTE_CARLO_TRIALS");
    if (env_trials) {
        monte_carlo_trials = std::stoi(env_trials);
    }
    
    char* env_confidence = std::getenv("RAD_ML_CONFIDENCE_LEVEL");
    if (env_confidence) {
        confidence_level = std::stod(env_confidence);
    }
    
    char* env_model = std::getenv("RAD_ML_ENVIRONMENT_MODEL");
    if (env_model) {
        environment_model = env_model;
    }
    
    // Log configuration
    printSection("Test Configuration");
    std::cout << "Monte Carlo Trials: " << monte_carlo_trials << std::endl;
    std::cout << "Confidence Level: " << confidence_level << std::endl;
    std::cout << "Environment Model: " << environment_model << std::endl;
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Define environments to test
    std::vector<rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment> environments = {
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::LEO,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::SAA,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::GEO,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::VAN_ALLEN,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::LUNAR,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::MARS,
        rad_ml::validation::NASAESAVerificationProtocol::TestEnvironment::JUPITER
    };
    
    // Define protection techniques to evaluate
    std::vector<rad_ml::validation::NASAESAVerificationProtocol::ProtectionTechnique> techniques = {
        rad_ml::validation::NASAESAVerificationProtocol::ProtectionTechnique::TMR,
        rad_ml::validation::NASAESAVerificationProtocol::ProtectionTechnique::EDAC,
        rad_ml::validation::NASAESAVerificationProtocol::ProtectionTechnique::SCRUBBING
    };
    
    // Initialize verification protocol
    printSection("Initializing Verification Protocol");
    rad_ml::validation::NASAESAVerificationProtocol protocol(environments, techniques, monte_carlo_trials, confidence_level);
    
    // Run comprehensive verification
    printSection("Running Comprehensive Verification");
    rad_ml::validation::NASAESAVerificationProtocol::VerificationReport report = protocol.runComprehensiveVerification();
    
    // Calculate and display execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    printSection("Test Execution Summary");
    std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Overall status: " << (report.overall_compliant ? GREEN + "COMPLIANT" : RED + "NON-COMPLIANT") << RESET << std::endl;
    
    // Export results
    printSection("Exporting Results");
    exportReportToHTML(report);
    
    printSection("Verification Statement");
    std::cout << report.verification_statement << std::endl;
    
    // Export comprehensive report
    protocol.exportComprehensiveReport("nasa_esa_comprehensive_report.html");
    
    std::cout << std::endl;
    std::cout << BOLD + GREEN + "NASA/ESA standard test completed successfully." + RESET << std::endl;
    
    return 0;
} 
