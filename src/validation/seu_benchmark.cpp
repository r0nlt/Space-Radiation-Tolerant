#include "validation/radiation_accuracy_validator.hpp"
#include "rad_ml/api/rad_ml.hpp"
#include "rad_ml/testing/fault_injector.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>

/**
 * @brief SEU Cross-Section Benchmarking Tool
 * 
 * This tool measures SEU cross-sections (cm²/bit) for different memory
 * protection schemes under various radiation environments based on industry
 * standards from NASA and ESA.
 */

// Memory test configuration
struct MemoryTestConfig {
    size_t memory_size_mb;
    rad_ml::memory::MemoryProtectionLevel protection_level;
    double lux_per_cm2;
    std::string test_name;
    double expected_cross_section;
    std::string reference;
};

// Test result
struct SEUTestResult {
    std::string test_name;
    double cross_section_cm2_per_bit;
    double expected_cross_section_cm2_per_bit;
    double ratio;
    size_t injected_errors;
    size_t detected_errors;
    size_t corrected_errors;
    double error_rate_percent;
    double test_duration_seconds;
    std::string protection_level_name;
};

/**
 * @brief Convert memory protection level to string
 * @param level Protection level
 * @return String representation
 */
std::string protectionLevelToString(rad_ml::memory::MemoryProtectionLevel level) {
    switch (level) {
        case rad_ml::memory::MemoryProtectionLevel::NONE:
            return "None";
        case rad_ml::memory::MemoryProtectionLevel::CANARY:
            return "Canary";
        case rad_ml::memory::MemoryProtectionLevel::CRC:
            return "CRC";
        case rad_ml::memory::MemoryProtectionLevel::ECC:
            return "ECC";
        case rad_ml::memory::MemoryProtectionLevel::TMR:
            return "TMR";
        default:
            return "Unknown";
    }
}

/**
 * @brief Run a single SEU cross-section test
 * @param config Test configuration
 * @return Test result
 */
SEUTestResult runSEUTest(const MemoryTestConfig& config) {
    const size_t BYTES_PER_MB = 1024 * 1024;
    const size_t BITS_PER_BYTE = 8;
    const double CM2_PER_M2 = 10000.0;
    
    // Initialize result
    SEUTestResult result;
    result.test_name = config.test_name;
    result.expected_cross_section_cm2_per_bit = config.expected_cross_section;
    result.protection_level_name = protectionLevelToString(config.protection_level);
    
    // Calculate memory size in bytes and bits
    size_t memory_bytes = config.memory_size_mb * BYTES_PER_MB;
    size_t memory_bits = memory_bytes * BITS_PER_BYTE;
    
    // Allocate test memory
    std::cout << "Allocating " << config.memory_size_mb << " MB of memory with "
              << result.protection_level_name << " protection..." << std::endl;
    
    // Allocate memory with specified protection
    std::vector<uint8_t*> memory_blocks;
    const size_t BLOCK_SIZE = 1024 * 1024; // 1 MB per block
    
    for (size_t i = 0; i < config.memory_size_mb; i++) {
        uint8_t* block = rad_ml::memory_management::allocate<uint8_t>(
            BLOCK_SIZE,
            config.protection_level
        );
        
        // Initialize with random data
        for (size_t j = 0; j < BLOCK_SIZE; j++) {
            block[j] = static_cast<uint8_t>(rand() % 256);
        }
        
        memory_blocks.push_back(block);
    }
    
    // Create checksums of original data
    std::vector<uint32_t> original_checksums;
    original_checksums.reserve(memory_blocks.size());
    
    for (auto* block : memory_blocks) {
        // Simple CRC32-like checksum
        uint32_t checksum = 0;
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
            checksum = ((checksum << 5) + checksum) + block[i];
        }
        original_checksums.push_back(checksum);
    }
    
    // Create fault injector
    auto fault_injector = std::make_unique<rad_ml::testing::FaultInjector>();
    
    // Set fault rate based on lux and expected cross-section
    // This is a simplified model - in reality, the relationship is more complex
    double fault_rate = config.lux_per_cm2 * config.expected_cross_section * 
                         memory_bits / CM2_PER_M2;
    
    fault_injector->setFaultRate(fault_rate);
    fault_injector->setFaultType(rad_ml::testing::FaultType::BIT_FLIP);
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run test for 10 seconds
    const int TEST_DURATION_SECONDS = 10;
    size_t injected_errors = 0;
    
    std::cout << "Injecting SEUs for " << TEST_DURATION_SECONDS << " seconds..." << std::endl;
    
    for (int second = 0; second < TEST_DURATION_SECONDS; second++) {
        // Inject faults
        for (size_t i = 0; i < memory_blocks.size(); i++) {
            size_t errors = fault_injector->injectFaults(
                memory_blocks[i], 
                BLOCK_SIZE
            );
            injected_errors += errors;
        }
        
        // Simulate 1 second passing
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // Progress indicator
        std::cout << "." << std::flush;
    }
    
    std::cout << std::endl;
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    result.test_duration_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count() / 1000.0;
    
    // Check for errors
    size_t detected_errors = 0;
    
    for (size_t i = 0; i < memory_blocks.size(); i++) {
        uint32_t current_checksum = 0;
        for (size_t j = 0; j < BLOCK_SIZE; j++) {
            current_checksum = ((current_checksum << 5) + current_checksum) + memory_blocks[i][j];
        }
        
        if (current_checksum != original_checksums[i]) {
            detected_errors++;
        }
    }
    
    // Calculate SEU cross-section
    // Cross-section = (detected errors) / (flux * bits * time)
    result.injected_errors = injected_errors;
    result.detected_errors = detected_errors;
    
    // Get corrected errors from memory manager stats
    auto stats = rad_ml::memory_management::getStats();
    result.corrected_errors = stats.repaired_corruption;
    
    // Calculate error rate
    if (injected_errors > 0) {
        result.error_rate_percent = 100.0 * 
            (injected_errors - result.corrected_errors) / 
            static_cast<double>(injected_errors);
    } else {
        result.error_rate_percent = 0.0;
    }
    
    // Calculate measured cross-section (cm²/bit)
    if (config.lux_per_cm2 > 0 && result.test_duration_seconds > 0) {
        double flux = config.lux_per_cm2 * result.test_duration_seconds;
        result.cross_section_cm2_per_bit = 
            static_cast<double>(detected_errors) / (flux * memory_bits);
    } else {
        result.cross_section_cm2_per_bit = 0.0;
    }
    
    // Calculate ratio to expected
    if (config.expected_cross_section > 0) {
        result.ratio = result.cross_section_cm2_per_bit / config.expected_cross_section;
    } else {
        result.ratio = 0.0;
    }
    
    // Clean up memory
    for (auto* block : memory_blocks) {
        rad_ml::memory_management::deallocate(block);
    }
    
    return result;
}

/**
 * @brief Save results to CSV
 * @param results Test results
 * @param filename Output filename
 * @return True if successful
 */
bool saveResultsToCSV(const std::vector<SEUTestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    file << "Test Name,Protection Level,Cross-Section (cm²/bit),Expected Cross-Section (cm²/bit),"
         << "Ratio,Injected Errors,Detected Errors,Corrected Errors,Error Rate %,"
         << "Test Duration (s)\n";
    
    // Write data
    for (const auto& result : results) {
        file << result.test_name << ","
             << result.protection_level_name << ","
             << std::scientific << std::setprecision(6) << result.cross_section_cm2_per_bit << ","
             << std::scientific << std::setprecision(6) << result.expected_cross_section_cm2_per_bit << ","
             << std::fixed << std::setprecision(2) << result.ratio << ","
             << result.injected_errors << ","
             << result.detected_errors << ","
             << result.corrected_errors << ","
             << std::fixed << std::setprecision(2) << result.error_rate_percent << ","
             << std::fixed << std::setprecision(2) << result.test_duration_seconds << "\n";
    }
    
    file.close();
    return true;
}

/**
 * @brief Main program
 */
int main(int argc, char* argv[]) {
    // Parse command line args (to be implemented)
    // ...
    
    // Print header
    std::cout << "┌──────────────────────────────────────────────────────────┐\n";
    std::cout << "│        SEU CROSS-SECTION BENCHMARK UTILITY (NASA/ESA)    │\n";
    std::cout << "└──────────────────────────────────────────────────────────┘\n\n";
    
    // Initialize framework
    std::cout << "Initializing radiation framework..." << std::endl;
    if (!rad_ml::initialize(true, rad_ml::memory::MemoryProtectionLevel::NONE)) {
        std::cerr << "Error: Failed to initialize radiation framework." << std::endl;
        return 1;
    }
    
    // Define standard test configurations
    // These values are from published radiation testing data
    std::vector<MemoryTestConfig> configurations = {
        // Low Earth Orbit (ISS) - SRAM
        {
            10,                                             // 10 MB memory
            rad_ml::memory::MemoryProtectionLevel::NONE,    // No protection
            2.5e-3,                                         // 2.5e-3 LUX/cm²
            "LEO-SRAM-NONE",                                // Test name
            1.2e-14,                                        // Expected cross-section (cm²/bit)
            "NASA/TP-2006-214140"                           // Reference
        },
        // Low Earth Orbit (ISS) - SRAM with TMR
        {
            10,                                             // 10 MB memory
            rad_ml::memory::MemoryProtectionLevel::TMR,     // TMR protection
            2.5e-3,                                         // 2.5e-3 LUX/cm²
            "LEO-SRAM-TMR",                                 // Test name
            4.0e-15,                                        // Expected cross-section (cm²/bit)
            "NASA/TP-2006-214140"                           // Reference
        },
        // Jupiter/Europa - SRAM
        {
            10,                                             // 10 MB memory
            rad_ml::memory::MemoryProtectionLevel::NONE,    // No protection
            0.85,                                           // 0.85 LUX/cm²
            "JUPITER-SRAM-NONE",                            // Test name
            2.2e-13,                                        // Expected cross-section (cm²/bit)
            "JPL Publication 19-5"                          // Reference
        },
        // Jupiter/Europa - SRAM with TMR
        {
            10,                                             // 10 MB memory
            rad_ml::memory::MemoryProtectionLevel::TMR,     // TMR protection
            0.85,                                           // 0.85 LUX/cm²
            "JUPITER-SRAM-TMR",                             // Test name
            7.3e-14,                                        // Expected cross-section (cm²/bit)
            "JPL Publication 19-5"                          // Reference
        },
        // Low Earth Orbit - DRAM
        {
            50,                                             // 50 MB memory
            rad_ml::memory::MemoryProtectionLevel::NONE,    // No protection
            2.5e-3,                                         // 2.5e-3 LUX/cm²
            "LEO-DRAM-NONE",                                // Test name
            3.8e-14,                                        // Expected cross-section (cm²/bit)
            "IEEE TNS, Vol. 65, 2018"                       // Reference
        },
        // Low Earth Orbit - DRAM with ECC
        {
            50,                                             // 50 MB memory
            rad_ml::memory::MemoryProtectionLevel::ECC,     // ECC protection
            2.5e-3,                                         // 2.5e-3 LUX/cm²
            "LEO-DRAM-ECC",                                 // Test name
            9.5e-15,                                        // Expected cross-section (cm²/bit)
            "IEEE TNS, Vol. 65, 2018"                       // Reference
        }
    };
    
    // Run benchmarks
    std::vector<SEUTestResult> results;
    
    for (const auto& config : configurations) {
        std::cout << "\nRunning test: " << config.test_name << std::endl;
        std::cout << "Parameters: " << config.memory_size_mb << " MB, "
                  << protectionLevelToString(config.protection_level) << " protection, "
                  << std::scientific << config.lux_per_cm2 << " LUX/cm²" << std::endl;
        
        SEUTestResult result = runSEUTest(config);
        results.push_back(result);
        
        std::cout << "Cross-section: " << std::scientific << std::setprecision(6) 
                  << result.cross_section_cm2_per_bit << " cm²/bit (expected: " 
                  << config.expected_cross_section << ")" << std::endl;
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) 
                  << result.ratio << std::endl;
        std::cout << "Errors: " << result.injected_errors << " injected, " 
                  << result.detected_errors << " detected, "
                  << result.corrected_errors << " corrected" << std::endl;
        std::cout << "Error rate: " << std::fixed << std::setprecision(2) 
                  << result.error_rate_percent << "%" << std::endl;
    }
    
    // Save results to CSV
    std::string output_file = "seu_benchmark_results.csv";
    if (saveResultsToCSV(results, output_file)) {
        std::cout << "\nResults saved to " << output_file << std::endl;
    } else {
        std::cerr << "\nError: Failed to save results." << std::endl;
    }
    
    // Print summary
    std::cout << "\n┌──────────────────────────────────────────────────────────┐\n";
    std::cout << "│                   BENCHMARK SUMMARY                       │\n";
    std::cout << "└──────────────────────────────────────────────────────────┘\n\n";
    
    std::cout << "Test Results:\n";
    std::cout << std::left << std::setw(20) << "Test" 
              << std::setw(12) << "Protection" 
              << std::setw(16) << "Cross-Section" 
              << std::setw(8) << "Ratio" 
              << std::setw(10) << "Error Rate" << std::endl;
    
    std::cout << std::string(66, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.test_name
                  << std::setw(12) << result.protection_level_name
                  << std::scientific << std::setprecision(2) << std::setw(16) << result.cross_section_cm2_per_bit
                  << std::fixed << std::setprecision(2) << std::setw(8) << result.ratio
                  << std::fixed << std::setprecision(2) << std::setw(10) << result.error_rate_percent << "%" << std::endl;
    }
    
    return 0;
} 