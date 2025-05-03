#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "rad_ml/core/radiation/adaptive_protection.hpp"
#include "rad_ml/testing/radiation_simulator.hpp"
#include "rad_ml/core/redundancy/enhanced_tmr.hpp"
#include "rad_ml/mission/mission_profile.hpp"
#include "rad_ml/testing/mission_simulator.hpp"
#include "rad_ml/memory/memory_scrubber.hpp"

using namespace rad_ml;

// Record of radiation events and their effects
struct StressTestRecord {
    std::chrono::steady_clock::time_point timestamp;
    size_t total_events;
    size_t bit_flips;
    size_t multi_bit_upsets;
    size_t latchups;
    size_t transients;
    size_t detected_errors;
    size_t corrected_errors;
    size_t uncorrectable_errors;
    core::radiation::AdaptiveProtection::ProtectionLevel protection_level;
};

// Create an extremely harsh radiation environment
testing::RadiationSimulator::EnvironmentParams createExtremeEnvironment() {
    testing::RadiationSimulator::EnvironmentParams params;
    
    // Base on Jupiter but make it much worse
    params = testing::RadiationSimulator::getMissionEnvironment("JUPITER");
    
    // Extremely high solar activity (solar flare)
    params.solar_activity = 10.0;        // Maximum value
    
    // Almost no shielding
    params.shielding_thickness_mm = 0.05; // Minimal shielding
    
    // Inside SAA (though not applicable for Jupiter, helps increase radiation)
    params.inside_saa = true;
    
    // Custom mission name
    params.mission_name = "EXTREME RADIATION ENVIRONMENT";
    
    return params;
}

// Run a stress test with high radiation for a long duration
void runStressTest(
    std::chrono::seconds duration = std::chrono::minutes(5),   // Default: 5 minutes
    std::chrono::seconds reporting_interval = std::chrono::seconds(15)  // Report every 15 seconds
) {
    std::cout << "=========================================================" << std::endl;
    std::cout << "RADIATION-TOLERANT ML FRAMEWORK STRESS TEST" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    std::cout << "Test duration: " << duration.count() << " seconds" << std::endl;
    std::cout << "Reporting interval: " << reporting_interval.count() << " seconds" << std::endl;
    
    // Create extreme radiation environment
    auto extreme_env = createExtremeEnvironment();
    testing::RadiationSimulator simulator(extreme_env);
    
    std::cout << "\nRadiation Environment:" << std::endl;
    std::cout << simulator.getEnvironmentDescription() << std::endl;
    
    // Create adaptive protection with maximum level
    core::radiation::AdaptiveProtection protection(
        core::radiation::AdaptiveProtection::ProtectionLevel::MAXIMUM);
    
    // Create test memory with TMR-protected values (much larger array for longer test)
    const size_t memory_size = 10000;
    std::cout << "Creating " << memory_size << " TMR-protected values..." << std::endl;
    
    std::vector<core::redundancy::EnhancedTMR<float>> protected_values(memory_size, 
        core::redundancy::EnhancedTMR<float>(1.0f));
    
    // Create memory scrubber
    memory::MemoryScrubber scrubber;
    scrubber.registerMemoryRegion(protected_values.data(), 
        protected_values.size() * sizeof(core::redundancy::EnhancedTMR<float>));
    
    // Set up test record collection
    std::vector<StressTestRecord> records;
    
    // Tracking stats
    size_t total_radiation_events = 0;
    size_t total_bit_flips = 0;
    size_t total_multi_bit_upsets = 0;
    size_t total_latchups = 0;
    size_t total_transients = 0;
    size_t total_detected_errors = 0;
    size_t total_corrected_errors = 0;
    size_t total_uncorrectable_errors = 0;
    
    // Initial scrub to get baseline
    scrubber.scrubMemory();
    
    std::cout << "\nBeginning stress test...\n" << std::endl;
    
    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();
    auto last_report_time = start_time;
    
    // Run simulation in chunks matching reporting interval
    while (std::chrono::steady_clock::now() - start_time < duration) {
        // Check if it's time for a report
        auto current_time = std::chrono::steady_clock::now();
        bool report_due = (current_time - last_report_time >= reporting_interval);
        
        if (report_due) {
            // Show progress
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();
            int progress = static_cast<int>((static_cast<double>(elapsed) / duration.count()) * 100.0);
            
            std::cout << "[" << std::setw(3) << progress << "%] ";
            std::cout << "Elapsed: " << elapsed << "s | ";
            std::cout << "Events: " << total_radiation_events << " | ";
            std::cout << "Errors: " << total_detected_errors << " | ";
            std::cout << "Corrected: " << total_corrected_errors << " | ";
            std::cout << "Protection Level: " << static_cast<int>(protection.getProtectionLevel()) << std::endl;
            
            last_report_time = current_time;
            
            // Record the statistics at this point
            StressTestRecord record;
            record.timestamp = current_time;
            record.total_events = total_radiation_events;
            record.bit_flips = total_bit_flips;
            record.multi_bit_upsets = total_multi_bit_upsets;
            record.latchups = total_latchups;
            record.transients = total_transients;
            record.detected_errors = total_detected_errors;
            record.corrected_errors = total_corrected_errors;
            record.uncorrectable_errors = total_uncorrectable_errors;
            record.protection_level = protection.getProtectionLevel();
            
            records.push_back(record);
            
            // Perform memory scrubbing
            size_t errors = scrubber.scrubMemory();
            total_detected_errors += errors;
            
            // Update corrected errors
            size_t corrected_this_round = 0;
            for (auto& val : protected_values) {
                if (!val.verify()) {
                    val.repair();
                    
                    // Count as corrected if repair succeeded
                    if (val.verify()) {
                        corrected_this_round++;
                    }
                }
            }
            
            total_corrected_errors += corrected_this_round;
            total_uncorrectable_errors += (errors - corrected_this_round);
            
            // Update protection system
            protection.updateEnvironment(errors, 0);
        }
        
        // Simulate radiation effects for a shorter interval
        auto sim_interval = std::min(
            reporting_interval, 
            std::chrono::duration_cast<std::chrono::seconds>(
                duration - (current_time - start_time))
        );
        
        if (sim_interval.count() <= 0) break;
        
        // Run simulation for this interval
        auto events = simulator.simulateEffects(
            protected_values.data(), 
            protected_values.size() * sizeof(core::redundancy::EnhancedTMR<float>),
            std::chrono::milliseconds(sim_interval));
        
        // Update statistics
        total_radiation_events += events.size();
        
        for (const auto& event : events) {
            switch (event.type) {
                case testing::RadiationSimulator::RadiationEffectType::SINGLE_BIT_FLIP:
                    total_bit_flips++;
                    break;
                case testing::RadiationSimulator::RadiationEffectType::MULTI_BIT_UPSET:
                    total_multi_bit_upsets++;
                    break;
                case testing::RadiationSimulator::RadiationEffectType::SINGLE_EVENT_LATCHUP:
                    total_latchups++;
                    break;
                case testing::RadiationSimulator::RadiationEffectType::SINGLE_EVENT_TRANSIENT:
                    total_transients++;
                    break;
            }
        }
    }
    
    // Final report
    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    // Perform final scrubbing to get final error count
    size_t final_errors = scrubber.scrubMemory();
    total_detected_errors += final_errors;
    
    // Count values that are still correct after all that radiation
    int incorrect_values = 0;
    for (auto& val : protected_values) {
        if (std::abs(val.get() - 1.0f) > 0.01f) {
            incorrect_values++;
        }
    }
    
    // Calculate error rates and recovery statistics
    double events_per_second = static_cast<double>(total_radiation_events) / total_duration;
    double errors_per_second = static_cast<double>(total_detected_errors) / total_duration;
    double recovery_rate = 100.0;
    if (total_detected_errors > 0) {
        recovery_rate = 100.0 * (static_cast<double>(total_detected_errors - incorrect_values) / 
                               total_detected_errors);
    }
    
    // Print final report
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "STRESS TEST RESULTS" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    std::cout << "Test duration: " << total_duration << " seconds" << std::endl;
    std::cout << "Total radiation events: " << total_radiation_events << std::endl;
    std::cout << "  Single bit flips: " << total_bit_flips << " (" 
              << (100.0 * total_bit_flips / std::max(1UL, total_radiation_events)) << "%)" << std::endl;
    std::cout << "  Multi-bit upsets: " << total_multi_bit_upsets << " (" 
              << (100.0 * total_multi_bit_upsets / std::max(1UL, total_radiation_events)) << "%)" << std::endl;
    std::cout << "  Single event latchups: " << total_latchups << " (" 
              << (100.0 * total_latchups / std::max(1UL, total_radiation_events)) << "%)" << std::endl;
    std::cout << "  Single event transients: " << total_transients << " (" 
              << (100.0 * total_transients / std::max(1UL, total_radiation_events)) << "%)" << std::endl;
    
    std::cout << "\nError statistics:" << std::endl;
    std::cout << "  Detected errors: " << total_detected_errors << std::endl;
    std::cout << "  Corrected errors: " << total_corrected_errors << std::endl;
    std::cout << "  Uncorrectable errors: " << total_uncorrectable_errors << std::endl;
    std::cout << "  Values corrupted beyond recovery: " << incorrect_values << " (out of " 
              << memory_size << ")" << std::endl;
    
    std::cout << "\nRates:" << std::endl;
    std::cout << "  Radiation events per second: " << events_per_second << std::endl;
    std::cout << "  Errors per second: " << errors_per_second << std::endl;
    std::cout << "  Error detection rate: " << 
        (100.0 * total_detected_errors / std::max(1UL, total_radiation_events)) << "%" << std::endl;
    std::cout << "  Error correction rate: " << 
        (100.0 * total_corrected_errors / std::max(1UL, total_detected_errors)) << "%" << std::endl;
    std::cout << "  Overall data integrity preservation: " << 
        (100.0 * (memory_size - incorrect_values) / memory_size) << "%" << std::endl;
    std::cout << "  Recovery rate: " << recovery_rate << "%" << std::endl;
    
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "PROTECTION EFFECTIVENESS ANALYSIS" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    if (recovery_rate > 99.0) {
        std::cout << "EXCELLENT PROTECTION: The framework achieved >99% recovery rate even under" << std::endl;
        std::cout << "extreme radiation conditions. The combination of TMR, CRC validation," << std::endl;
        std::cout << "and memory scrubbing provided robust protection against radiation effects." << std::endl;
    }
    else if (recovery_rate > 95.0) {
        std::cout << "VERY GOOD PROTECTION: The framework achieved >95% recovery rate under" << std::endl;
        std::cout << "extreme radiation conditions. The protection mechanisms effectively" << std::endl;
        std::cout << "handled most radiation-induced errors." << std::endl;
    }
    else if (recovery_rate > 90.0) {
        std::cout << "GOOD PROTECTION: The framework achieved >90% recovery rate, which is" << std::endl;
        std::cout << "acceptable for most space missions. Some extreme radiation events" << std::endl;
        std::cout << "caused unrecoverable errors, suggesting additional protection may be" << std::endl;
        std::cout << "needed for the most critical applications." << std::endl;
    }
    else if (recovery_rate > 80.0) {
        std::cout << "MODERATE PROTECTION: The framework achieved >80% recovery rate. While" << std::endl;
        std::cout << "this provides basic protection, it may not be sufficient for critical" << std::endl;
        std::cout << "space applications. Consider enhancing the protection mechanisms." << std::endl;
    }
    else {
        std::cout << "INSUFFICIENT PROTECTION: The recovery rate was below 80%, which is" << std::endl;
        std::cout << "concerning even for the extreme radiation conditions used in this test." << std::endl;
        std::cout << "The protection mechanisms need significant improvements for space use." << std::endl;
    }
    
    // Save results to CSV for potential graphing
    std::ofstream csv_file("stress_test_results.csv");
    if (csv_file.is_open()) {
        // CSV header
        csv_file << "Timestamp,TotalEvents,BitFlips,MultiBitUpsets,Latchups,Transients,"
                 << "DetectedErrors,CorrectedErrors,UncorrectableErrors,ProtectionLevel\n";
        
        // Initial record at zero
        csv_file << "0,0,0,0,0,0,0,0,0," << static_cast<int>(records.front().protection_level) << "\n";
        
        // All records
        auto start = records.front().timestamp;
        for (const auto& record : records) {
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                record.timestamp - start).count();
                
            csv_file << seconds << ","
                     << record.total_events << ","
                     << record.bit_flips << ","
                     << record.multi_bit_upsets << ","
                     << record.latchups << ","
                     << record.transients << ","
                     << record.detected_errors << ","
                     << record.corrected_errors << ","
                     << record.uncorrectable_errors << ","
                     << static_cast<int>(record.protection_level) << "\n";
        }
        
        csv_file.close();
        std::cout << "\nResults saved to stress_test_results.csv for further analysis" << std::endl;
    }
    
    std::cout << "\nStress test completed." << std::endl;
}

int main(int argc, char* argv[]) {
    // Default duration: 5 minutes
    std::chrono::seconds duration = std::chrono::minutes(5);
    
    // Parse command line arguments for custom duration
    if (argc > 1) {
        try {
            int seconds = std::stoi(argv[1]);
            if (seconds > 0) {
                duration = std::chrono::seconds(seconds);
            }
        } catch (...) {
            std::cerr << "Invalid duration specified, using default 5 minutes." << std::endl;
        }
    }
    
    // Default reporting interval: 15 seconds
    std::chrono::seconds reporting_interval = std::chrono::seconds(15);
    
    // Run stress test
    runStressTest(duration, reporting_interval);
    
    return 0;
} 