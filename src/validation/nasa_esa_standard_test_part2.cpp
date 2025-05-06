#include "nasa_esa_standard_test.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <ctime>

namespace rad_ml {
namespace validation {

NASAESAStandardTest::MissionAssessment 
NASAESAStandardTest::assessMissionSuitability(MissionType mission, 
                                             const std::vector<TestResult>& results) {
    MissionAssessment assessment;
    assessment.mission = mission;
    assessment.suitable = false;
    assessment.nasa_std_compliance = false;
    
    // Filter results for this mission and Health-Weighted TMR (best option)
    std::vector<TestResult> mission_results;
    for (const auto& result : results) {
        if (result.mission == mission && 
            result.protection == ProtectionMethod::HEALTH_WEIGHTED_TMR) {
            mission_results.push_back(result);
        }
    }
    
    if (mission_results.empty()) {
        return assessment;
    }
    
    // Check if all tests pass
    bool all_pass = true;
    for (const auto& result : mission_results) {
        all_pass &= result.pass;
    }
    
    // System lifetime calculation based on mission and MTBF
    double mtbf = 0.0;
    for (const auto& result : mission_results) {
        if (result.metrics.mtbf > 0) {
            mtbf = result.metrics.mtbf;
            break;
        }
    }
    
    // Convert MTBF to years (with a safety factor of 0.6)
    double lifetime_years = (mtbf * 0.6) / (365.25 * 24);
    
    // Required shielding based on mission
    double shielding_mm_al = 0.0;
    switch (mission) {
        case MissionType::LEO:
            shielding_mm_al = 5.0;
            break;
        case MissionType::GEO:
            shielding_mm_al = 8.0;
            break;
        case MissionType::LUNAR:
            shielding_mm_al = 10.0;
            break;
        case MissionType::MARS:
            shielding_mm_al = 12.0;
            break;
        case MissionType::JUPITER:
            shielding_mm_al = 25.0;
            break;
    }
    
    // Recommended enhancements
    std::vector<std::string> recommended_enhancements;
    
    // Check if SEU rate needs improvement
    for (const auto& result : mission_results) {
        if (result.metrics.seu_rate > pass_criteria_[mission].seu_rate_threshold) {
            recommended_enhancements.push_back(
                "Improve SEU rate from " + 
                std::to_string(result.metrics.seu_rate) + 
                " to below " + 
                std::to_string(pass_criteria_[mission].seu_rate_threshold) + 
                " events/bit-day");
            break;
        }
    }
    
    // Check if LET threshold needs improvement
    for (const auto& result : mission_results) {
        if (result.metrics.let_threshold < pass_criteria_[mission].let_threshold) {
            recommended_enhancements.push_back(
                "Increase LET threshold from " + 
                std::to_string(result.metrics.let_threshold) + 
                " to above " + 
                std::to_string(pass_criteria_[mission].let_threshold) + 
                " MeV-cm²/mg");
            break;
        }
    }
    
    // Check if MTBF needs improvement
    for (const auto& result : mission_results) {
        if (result.metrics.mtbf < pass_criteria_[mission].mtbf_requirement) {
            recommended_enhancements.push_back(
                "Improve MTBF from " + 
                std::to_string(result.metrics.mtbf) + 
                " to above " + 
                std::to_string(pass_criteria_[mission].mtbf_requirement) + 
                " hours");
            break;
        }
    }
    
    // Add mission-specific recommendations
    switch (mission) {
        case MissionType::GEO:
            recommended_enhancements.push_back("Implement more frequent memory scrubbing");
            break;
        case MissionType::LUNAR:
            recommended_enhancements.push_back("Specialized error correction for lunar radiation profile");
            break;
        case MissionType::MARS:
            recommended_enhancements.push_back("Enhanced protection for GCRs during transit");
            break;
        case MissionType::JUPITER:
            recommended_enhancements.push_back("SEL immunity improvements");
            recommended_enhancements.push_back("3× redundant systems with independent voting");
            break;
        default:
            break;
    }
    
    // Set final assessment values
    assessment.nasa_std_compliance = all_pass;
    assessment.suitable = all_pass;
    assessment.system_lifetime_years = lifetime_years;
    assessment.required_shielding_mm_al = shielding_mm_al;
    assessment.recommended_enhancements = recommended_enhancements;
    
    return assessment;
}

void NASAESAStandardTest::exportResultsToCSV(const std::string& filename, 
                                           const std::vector<TestResult>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    // Write CSV header
    file << "Mission,Phase,Protection Method,SEU Rate (events/bit-day),LET Threshold (MeV-cm²/mg),"
         << "MTBF (hours),Cross-section (cm²/bit),TID Tolerance (krad),SEL Observed,"
         << "SEL Threshold (MeV-cm²/mg),Power Overhead (%),Processing Overhead (%),"
         << "Pass/Fail,Notes\n";
    
    // Write result rows
    for (const auto& result : results) {
        std::string mission_str;
        switch (result.mission) {
            case MissionType::LEO: mission_str = "Low Earth Orbit"; break;
            case MissionType::GEO: mission_str = "Geosynchronous Orbit"; break;
            case MissionType::LUNAR: mission_str = "Lunar Mission"; break;
            case MissionType::MARS: mission_str = "Mars Mission"; break;
            case MissionType::JUPITER: mission_str = "Jupiter/Europa Mission"; break;
            default: mission_str = "Unknown"; break;
        }
        
        std::string phase_str;
        switch (result.phase) {
            case TestPhase::LAUNCH_ASCENT: phase_str = "Launch and Ascent"; break;
            case TestPhase::VAN_ALLEN_TRANSIT: phase_str = "Van Allen Transit"; break;
            case TestPhase::NOMINAL_OPERATIONS: phase_str = "Nominal Operations"; break;
            case TestPhase::SOLAR_PARTICLE_EVENT: phase_str = "Solar Particle Event"; break;
            default: phase_str = "Unknown"; break;
        }
        
        std::string protection_str;
        switch (result.protection) {
            case ProtectionMethod::NO_PROTECTION: protection_str = "No Protection"; break;
            case ProtectionMethod::BASIC_TMR: protection_str = "Basic TMR"; break;
            case ProtectionMethod::ENHANCED_TMR: protection_str = "Enhanced TMR"; break;
            case ProtectionMethod::STUCK_BIT_TMR: protection_str = "Stuck-Bit TMR"; break;
            case ProtectionMethod::HEALTH_WEIGHTED_TMR: protection_str = "Health-Weighted TMR"; break;
            case ProtectionMethod::MEMORY_SCRUBBING: protection_str = "Memory Scrubbing"; break;
            default: protection_str = "Unknown"; break;
        }
        
        file << mission_str << ","
             << phase_str << ","
             << protection_str << ","
             << result.metrics.seu_rate << ","
             << result.metrics.let_threshold << ","
             << result.metrics.mtbf << ","
             << result.metrics.cross_section << ","
             << result.metrics.tid_tolerance << ","
             << (result.metrics.sel_observed ? "Yes" : "No") << ","
             << result.metrics.sel_threshold << ","
             << result.metrics.power_overhead << ","
             << result.metrics.processing_overhead << ","
             << (result.pass ? "PASS" : "FAIL") << ","
             << "\"" << result.notes << "\"\n";
    }
    
    file.close();
}

void NASAESAStandardTest::exportResultsToHTML(const std::string& filename, 
                                            const std::vector<TestResult>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    // Get current time
    std::time_t now = std::time(nullptr);
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    
    // Write HTML header
    file << "<!DOCTYPE html>\n"
         << "<html>\n"
         << "<head>\n"
         << "  <title>NASA/ESA Standard Radiation Test Results</title>\n"
         << "  <style>\n"
         << "    body { font-family: Arial, sans-serif; margin: 20px; }\n"
         << "    h1, h2, h3 { color: #2c3e50; }\n"
         << "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n"
         << "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n"
         << "    th { background-color: #f2f2f2; }\n"
         << "    tr:nth-child(even) { background-color: #f9f9f9; }\n"
         << "    .pass { color: green; font-weight: bold; }\n"
         << "    .fail { color: red; font-weight: bold; }\n"
         << "    .mission-section { margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }\n"
         << "  </style>\n"
         << "</head>\n"
         << "<body>\n"
         << "  <h1>NASA/ESA Standard Radiation Test Results</h1>\n"
         << "  <p>Test date: " << time_str << "</p>\n"
         << "  <p>This report presents test results following NASA and ESA standardized methodologies for radiation testing of space systems.</p>\n"
         << "\n";
    
    // Group results by mission type
    std::map<MissionType, std::vector<TestResult>> mission_results;
    for (const auto& result : results) {
        mission_results[result.mission].push_back(result);
    }
    
    // For each mission type
    for (const auto& [mission, mission_data] : mission_results) {
        // Mission header
        std::string mission_str;
        switch (mission) {
            case MissionType::LEO: mission_str = "Low Earth Orbit"; break;
            case MissionType::GEO: mission_str = "Geosynchronous Orbit"; break;
            case MissionType::LUNAR: mission_str = "Lunar Mission"; break;
            case MissionType::MARS: mission_str = "Mars Mission"; break;
            case MissionType::JUPITER: mission_str = "Jupiter/Europa Mission"; break;
            default: mission_str = "Unknown Mission"; break;
        }
        
        file << "  <div class=\"mission-section\">\n"
             << "    <h2>" << mission_str << " Test Results</h2>\n"
             << "    <table>\n"
             << "      <tr>\n"
             << "        <th>Protection Method</th>\n"
             << "        <th>SEU Rate<br>(events/bit-day)</th>\n"
             << "        <th>LET Threshold<br>(MeV-cm²/mg)</th>\n"
             << "        <th>MTBF<br>(hours)</th>\n"
             << "        <th>Status</th>\n"
             << "        <th>Notes</th>\n"
             << "      </tr>\n";
        
        // Group and sort by protection method
        std::map<ProtectionMethod, std::vector<TestResult>> protection_results;
        for (const auto& result : mission_data) {
            protection_results[result.protection].push_back(result);
        }
        
        for (const auto& [protection, protection_data] : protection_results) {
            std::string protection_str;
            switch (protection) {
                case ProtectionMethod::NO_PROTECTION: protection_str = "No Protection"; break;
                case ProtectionMethod::BASIC_TMR: protection_str = "Basic TMR"; break;
                case ProtectionMethod::ENHANCED_TMR: protection_str = "Enhanced TMR"; break;
                case ProtectionMethod::STUCK_BIT_TMR: protection_str = "Stuck-Bit TMR"; break;
                case ProtectionMethod::HEALTH_WEIGHTED_TMR: protection_str = "Health-Weighted TMR"; break;
                case ProtectionMethod::MEMORY_SCRUBBING: protection_str = "Memory Scrubbing"; break;
                default: protection_str = "Unknown"; break;
            }
            
            // Find the test result for nominal operations
            const TestResult* nominal_result = nullptr;
            for (const auto& result : protection_data) {
                if (result.phase == TestPhase::NOMINAL_OPERATIONS) {
                    nominal_result = &result;
                    break;
                }
            }
            
            if (nominal_result) {
                file << "      <tr>\n"
                     << "        <td>" << protection_str << "</td>\n"
                     << "        <td>" << std::scientific << std::setprecision(2) << nominal_result->metrics.seu_rate << "</td>\n"
                     << "        <td>" << std::fixed << std::setprecision(1) << nominal_result->metrics.let_threshold << "</td>\n"
                     << "        <td>" << std::fixed << std::setprecision(0) << nominal_result->metrics.mtbf << "</td>\n"
                     << "        <td class=\"" << (nominal_result->pass ? "pass" : "fail") << "\">"
                     << (nominal_result->pass ? "PASS" : "FAIL") << "</td>\n"
                     << "        <td>" << nominal_result->notes << "</td>\n"
                     << "      </tr>\n";
            }
        }
        
        file << "    </table>\n";
        
        // Add NASA/ESA standard criteria for this mission
        file << "    <h3>NASA/ESA Standard Requirements for " << mission_str << "</h3>\n"
             << "    <ul>\n"
             << "      <li>SEU Rate: &lt; " << std::scientific << std::setprecision(1) 
             << pass_criteria_[mission].seu_rate_threshold << " events/bit-day</li>\n"
             << "      <li>LET Threshold: &gt; " << std::fixed << std::setprecision(1) 
             << pass_criteria_[mission].let_threshold << " MeV-cm²/mg</li>\n"
             << "      <li>MTBF: &gt; " << std::fixed << std::setprecision(0) 
             << pass_criteria_[mission].mtbf_requirement << " hours</li>\n"
             << "    </ul>\n"
             << "  </div>\n\n";
    }
    
    // Add mission suitability assessment
    file << "  <h2>Mission Suitability Assessment</h2>\n"
         << "  <table>\n"
         << "    <tr>\n"
         << "      <th>Mission</th>\n"
         << "      <th>Suitability</th>\n"
         << "      <th>NASA-STD-8719.14 Compliance</th>\n"
         << "      <th>System Lifetime (years)</th>\n"
         << "      <th>Required Shielding (mm Al)</th>\n"
         << "    </tr>\n";
    
    for (int m = 0; m < 5; m++) {
        MissionType mission = static_cast<MissionType>(m);
        auto assessment = assessMissionSuitability(mission, results);
        
        std::string mission_str;
        switch (mission) {
            case MissionType::LEO: mission_str = "Low Earth Orbit"; break;
            case MissionType::GEO: mission_str = "Geosynchronous Orbit"; break;
            case MissionType::LUNAR: mission_str = "Lunar Mission"; break;
            case MissionType::MARS: mission_str = "Mars Mission"; break;
            case MissionType::JUPITER: mission_str = "Jupiter/Europa Mission"; break;
            default: mission_str = "Unknown"; break;
        }
        
        file << "    <tr>\n"
             << "      <td>" << mission_str << "</td>\n"
             << "      <td class=\"" << (assessment.suitable ? "pass" : "fail") << "\">"
             << (assessment.suitable ? "SUITABLE" : "NOT SUITABLE") << "</td>\n"
             << "      <td class=\"" << (assessment.nasa_std_compliance ? "pass" : "fail") << "\">"
             << (assessment.nasa_std_compliance ? "PASS" : "FAIL") << "</td>\n"
             << "      <td>" << std::fixed << std::setprecision(1) << assessment.system_lifetime_years << "</td>\n"
             << "      <td>" << std::fixed << std::setprecision(1) << assessment.required_shielding_mm_al << "</td>\n"
             << "    </tr>\n";
    }
    
    file << "  </table>\n";
    
    // Add footer
    file << "  <h3>Test Methodology</h3>\n"
         << "  <p>Testing was conducted according to standardized NASA and ESA protocols:</p>\n"
         << "  <ul>\n"
         << "    <li>NASA-HDBK-4002A: Mitigating In-Space Charging Effects</li>\n"
         << "    <li>ECSS-E-ST-10-12C: Space engineering - Methods for the calculation of radiation received</li>\n"
         << "    <li>MIL-STD-883, Method 1019: Ionizing radiation (total dose) test procedure</li>\n"
         << "    <li>JEDEC JESD57: Test Procedures for the Measurement of SEE in Semiconductor Devices</li>\n"
         << "    <li>NASA/TP-2006-214373: Single Event Effect Criticality Analysis</li>\n"
         << "  </ul>\n"
         << "  <p>All test results include statistical validation using Monte Carlo simulations with "
         << monte_carlo_trials_ << " trials per test case.</p>\n"
         << "</body>\n"
         << "</html>\n";
    
    file.close();
}

void NASAESAStandardTest::generateComprehensiveReport(const std::string& filename) {
    // Run all tests for comprehensive report
    auto seu_results = runSingleEventUpsetTests();
    auto let_results = runLETThresholdTests();
    auto mtbf_results = runMTBFTests();
    
    // Combine all results
    std::vector<TestResult> all_results;
    all_results.insert(all_results.end(), seu_results.begin(), seu_results.end());
    all_results.insert(all_results.end(), let_results.begin(), let_results.end());
    all_results.insert(all_results.end(), mtbf_results.begin(), mtbf_results.end());
    
    // Export to HTML with more detailed information
    exportResultsToHTML(filename, all_results);
}

} // namespace validation
} // namespace rad_ml 