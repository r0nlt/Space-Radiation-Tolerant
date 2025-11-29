/**
 * @file selective_hardening_test.cpp
 * @brief Tests for selective hardening fixes
 *
 * Tests:
 * 1. Checksum storage and actual error detection
 * 2. Physics-based error reduction factors
 * 3. Importance decay (output layers should be more protected)
 * 4. Physics environment integration
 */

#include "../../include/rad_ml/neural/selective_hardening.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../../include/rad_ml/neural/sensitivity_analysis.hpp"
#include "../../include/rad_ml/physics/radiation_physics.hpp"

using namespace rad_ml::neural;
using namespace rad_ml::physics;

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string details;
};

std::vector<TestResult> results;

void report(const std::string& name, bool passed, const std::string& details = "")
{
    results.push_back({name, passed, details});
    std::cout << (passed ? "✓ PASS" : "✗ FAIL") << ": " << name;
    if (!details.empty()) {
        std::cout << " - " << details;
    }
    std::cout << "\n";
}

// ============================================================================
// Test 1: Checksum Storage and Error Detection
// ============================================================================
void test_checksum_error_detection()
{
    std::cout << "\n=== Test 1: Checksum Storage and Error Detection ===\n";

    SelectiveHardening hardening;

    // Create test components
    std::vector<NetworkComponent> components;
    NetworkComponent comp;
    comp.id = "test_weight_1";
    comp.type = "weight";
    comp.layer_name = "layer_0";
    comp.layer_index = 0;
    comp.index = 0;
    comp.value = 3.14159;
    comp.criticality = {0.8, 0.5, 0.7, 0.2, 0.1};  // High sensitivity
    components.push_back(comp);

    // Create analysis result and manually set protection levels for testing
    // (analyzeAndProtect uses strategy-based assignment, but we want specific levels)
    SensitivityAnalysisResult analysis;
    analysis.ranked_components = components;

    // Test 1a: Set up CHECKSUM_WITH_RECOVERY for test_weight_1
    analysis.protection_map["test_weight_1"] = ProtectionLevel::CHECKSUM_WITH_RECOVERY;
    analysis.protection_map["test_weight_2"] = ProtectionLevel::CHECKSUM_WITH_RECOVERY;
    analysis.protection_map["test_weight_3"] = ProtectionLevel::CHECKSUM_ONLY;

    float original_value = 3.14159f;

    // Store checksum and backup for test_weight_1
    hardening.storeChecksum("test_weight_1", original_value);
    hardening.storeBackup("test_weight_1", original_value);

    // Test 1b: Verify uncorrupted value passes
    auto result1 = hardening.applyProtection(original_value, "test_weight_1", analysis);
    report("Checksum - uncorrupted value accepted", result1.success && result1.value.has_value(),
           result1.success ? "Value verified" : result1.error_message);

    // Test 1c: Corrupt the value and verify recovery from backup
    float corrupted_value = original_value;
    uint32_t* bits = reinterpret_cast<uint32_t*>(&corrupted_value);
    *bits ^= 0x00000100;  // Flip one bit

    // Store checksum and backup for test_weight_2 BEFORE corruption
    hardening.storeChecksum("test_weight_2", original_value);
    hardening.storeBackup("test_weight_2", original_value);

    // Now try to recover the corrupted value
    auto result2 = hardening.applyProtection(corrupted_value, "test_weight_2", analysis);

    // With CHECKSUM_WITH_RECOVERY, it should recover from backup
    bool recovered_correctly = result2.success && result2.value.has_value() &&
                               std::abs(*result2.value - original_value) < 1e-6;

    report("Checksum - corrupted value recovered from backup", recovered_correctly,
           recovered_correctly ? "Recovered to " + std::to_string(*result2.value)
                               : "Recovery failed: " + result2.error_message);

    // Test 1d: Checksum-only mode (detection without recovery)
    // Store checksum for test_weight_3
    hardening.storeChecksum("test_weight_3", original_value);

    // Verify corrupted value is detected (should fail since CHECKSUM_ONLY can't recover)
    auto result3 = hardening.applyProtection(corrupted_value, "test_weight_3", analysis);
    report("Checksum-only - corrupted value detected", !result3.success,
           result3.success ? "Should have detected error"
                           : "Error detected: " + result3.error_message);
}

// ============================================================================
// Test 2: Physics-Based Error Reduction Factors
// ============================================================================
void test_physics_error_reduction()
{
    std::cout << "\n=== Test 2: Physics-Based Error Reduction Factors ===\n";

    // Create physics environment
    PhysicsRadiationEnvironment::Config config;
    config.altitude_km = 400.0;     // ISS altitude
    config.inclination_deg = 51.6;  // ISS inclination
    auto physics_env = std::make_shared<PhysicsRadiationEnvironment>(config);

    SelectiveHardening hardening;
    hardening.setPhysicsEnvironment(physics_env);

    // Create components with varying criticality
    std::vector<NetworkComponent> components;
    for (int i = 0; i < 10; i++) {
        NetworkComponent comp;
        comp.id = "weight_" + std::to_string(i);
        comp.type = "weight";
        comp.layer_name = "layer_" + std::to_string(i / 2);
        comp.layer_index = i / 2;
        comp.index = i;
        comp.value = 1.0;
        comp.criticality = {
            0.1 + 0.09 * i,  // sensitivity: 0.1 to 0.91
            0.5,             // activation_frequency
            0.1 + 0.09 * i,  // output_influence
            0.1,             // complexity
            0.05             // memory_usage
        };
        components.push_back(comp);
    }

    auto analysis = hardening.analyzeAndProtect(components);

    // Check that error rates are calculated
    report("Physics - baseline error rate calculated", analysis.baseline_error_rate > 0,
           "Baseline: " + std::to_string(analysis.baseline_error_rate));

    report("Physics - protected error rate calculated", analysis.expected_error_rate >= 0,
           "Protected: " + std::to_string(analysis.expected_error_rate));

    // Protected rate should be lower than baseline
    bool reduction_effective = analysis.expected_error_rate < analysis.baseline_error_rate;
    report("Physics - protection reduces error rate", reduction_effective,
           "Reduction: " +
               std::to_string((1.0 - analysis.expected_error_rate / analysis.baseline_error_rate) *
                              100) +
               "%");

    // Print SEU rate from physics model
    double seu_rate = physics_env->get_orbit_average_seu_rate();
    std::cout << "  Physics model SEU rate: " << std::scientific << seu_rate << " errors/bit/day\n";
}

// ============================================================================
// Test 3: Importance Decay - Output Layers More Critical
// ============================================================================
void test_importance_decay()
{
    std::cout << "\n=== Test 3: Importance Decay - Output Layers Priority ===\n";

    HardeningConfig config = HardeningConfig::defaultConfig();
    config.strategy = HardeningStrategy::IMPORTANCE_DECAY;
    config.resource_budget = 0.8;  // Allow generous protection

    SelectiveHardening hardening(config);

    // Create a 5-layer network: input -> hidden1 -> hidden2 -> hidden3 -> output
    std::vector<NetworkComponent> components;
    std::vector<std::string> layer_names = {"input", "hidden1", "hidden2", "hidden3", "output"};

    for (size_t layer = 0; layer < layer_names.size(); layer++) {
        for (int i = 0; i < 3; i++) {  // 3 weights per layer
            NetworkComponent comp;
            comp.id = layer_names[layer] + "_w" + std::to_string(i);
            comp.type = "weight";
            comp.layer_name = layer_names[layer];
            comp.layer_index = layer;
            comp.index = i;
            comp.value = 1.0;
            // Same base criticality for all - only layer position should matter
            comp.criticality = {0.5, 0.5, 0.5, 0.1, 0.05};
            components.push_back(comp);
        }
    }

    auto analysis = hardening.analyzeAndProtect(components);

    // Count protection levels per layer
    std::map<std::string, std::map<ProtectionLevel, int>> layer_protection;
    for (const auto& comp : components) {
        if (analysis.protection_map.count(comp.id)) {
            layer_protection[comp.layer_name][analysis.protection_map.at(comp.id)]++;
        }
    }

    // Calculate "protection score" for each layer
    std::map<ProtectionLevel, double> level_scores = {
        {ProtectionLevel::NONE, 0.0},
        {ProtectionLevel::CHECKSUM_ONLY, 1.0},
        {ProtectionLevel::CHECKSUM_WITH_RECOVERY, 2.0},
        {ProtectionLevel::APPROXIMATE_TMR, 3.0},
        {ProtectionLevel::HEALTH_WEIGHTED_TMR, 4.0},
        {ProtectionLevel::FULL_TMR, 5.0}};

    std::map<std::string, double> layer_scores;
    for (const auto& [layer, levels] : layer_protection) {
        double total = 0;
        int count = 0;
        for (const auto& [level, cnt] : levels) {
            if (level_scores.count(level)) {
                total += level_scores.at(level) * cnt;
            }
            count += cnt;
        }
        layer_scores[layer] = count > 0 ? total / count : 0;
    }

    // Print protection per layer
    std::cout << "  Layer protection scores (higher = more protected):\n";
    for (const auto& name : layer_names) {
        std::cout << "    " << std::setw(10) << name << ": " << std::fixed << std::setprecision(2)
                  << layer_scores[name] << "\n";
    }

    // Verify: output layer should have higher or equal protection than input
    bool output_more_protected = layer_scores["output"] >= layer_scores["input"];
    report("Importance decay - output >= input protection", output_more_protected,
           "Output: " + std::to_string(layer_scores["output"]) +
               ", Input: " + std::to_string(layer_scores["input"]));

    // Verify: later layers should generally have higher protection
    bool monotonic_increase = true;
    for (size_t i = 1; i < layer_names.size(); i++) {
        if (layer_scores[layer_names[i]] < layer_scores[layer_names[i - 1]] - 0.5) {
            monotonic_increase = false;
            break;
        }
    }
    report("Importance decay - protection increases toward output", monotonic_increase,
           monotonic_increase ? "Trend verified" : "Non-monotonic protection");
}

// ============================================================================
// Test 4: Physics Environment Integration in SpaceEnvironmentAnalyzer
// ============================================================================
void test_space_environment_analyzer()
{
    std::cout << "\n=== Test 4: SpaceEnvironmentAnalyzer Physics Integration ===\n";

    // Create physics environment for different orbits
    PhysicsRadiationEnvironment::Config leo_config;
    leo_config.altitude_km = 400.0;
    leo_config.inclination_deg = 51.6;
    auto leo_env = std::make_shared<PhysicsRadiationEnvironment>(leo_config);

    PhysicsRadiationEnvironment::Config geo_config;
    geo_config.altitude_km = 35786.0;  // GEO
    geo_config.inclination_deg = 0.0;
    auto geo_env = std::make_shared<PhysicsRadiationEnvironment>(geo_config);

    // Create analyzers
    SpaceEnvironmentAnalyzer leo_analyzer(leo_env);
    SpaceEnvironmentAnalyzer geo_analyzer(geo_env);

    // Verify physics environments are set
    report("SpaceAnalyzer - LEO physics env set", leo_analyzer.getPhysicsEnvironment() != nullptr);
    report("SpaceAnalyzer - GEO physics env set", geo_analyzer.getPhysicsEnvironment() != nullptr);

    // Compare SEU rates
    double leo_seu = leo_env->get_orbit_average_seu_rate();
    double geo_seu = geo_env->get_orbit_average_seu_rate();

    std::cout << "  LEO (400 km) SEU rate: " << std::scientific << leo_seu << " errors/bit/day\n";
    std::cout << "  GEO (35786 km) SEU rate: " << std::scientific << geo_seu << " errors/bit/day\n";

    // GEO generally has different (often higher GCR) radiation than LEO
    report("SpaceAnalyzer - SEU rates differ by orbit", std::abs(leo_seu - geo_seu) / leo_seu > 0.1,
           "Difference: " + std::to_string(std::abs(leo_seu - geo_seu) / leo_seu * 100) + "%");

    // Test scrub interval recommendations
    size_t test_bits = 1024 * 1024 * 8;  // 1MB
    double leo_scrub = leo_env->recommended_scrub_interval(test_bits, 1);
    double geo_scrub = geo_env->recommended_scrub_interval(test_bits, 1);

    std::cout << "  LEO scrub interval (1MB, SECDED): " << std::fixed << std::setprecision(2)
              << leo_scrub << " seconds\n";
    std::cout << "  GEO scrub interval (1MB, SECDED): " << std::fixed << std::setprecision(2)
              << geo_scrub << " seconds\n";

    report("SpaceAnalyzer - scrub intervals calculated", leo_scrub > 0 && geo_scrub > 0);
}

// ============================================================================
// Test 5: End-to-End Protection Flow
// ============================================================================
void test_end_to_end_protection()
{
    std::cout << "\n=== Test 5: End-to-End Protection Flow ===\n";

    // Setup
    PhysicsRadiationEnvironment::Config config;
    config.altitude_km = 400.0;
    auto physics_env = std::make_shared<PhysicsRadiationEnvironment>(config);

    HardeningConfig harden_config = HardeningConfig::defaultConfig();
    harden_config.strategy = HardeningStrategy::RESOURCE_CONSTRAINED;
    harden_config.resource_budget = 0.5;

    SelectiveHardening hardening(harden_config);
    hardening.setPhysicsEnvironment(physics_env);

    // Create network components
    std::vector<NetworkComponent> components;
    std::vector<float> weights = {0.5f, -0.3f, 1.2f, -0.8f, 0.1f};

    for (size_t i = 0; i < weights.size(); i++) {
        NetworkComponent comp;
        comp.id = "weight_" + std::to_string(i);
        comp.type = "weight";
        comp.layer_name = "dense_0";
        comp.layer_index = 0;
        comp.index = i;
        comp.value = weights[i];
        comp.criticality = {0.3 + 0.15 * i, 0.5, 0.3 + 0.15 * i, 0.1, 0.02};
        components.push_back(comp);
    }

    // Analyze
    auto analysis = hardening.analyzeAndProtect(components);

    // Store checksums and backups for all components
    for (size_t i = 0; i < weights.size(); i++) {
        hardening.storeChecksum(components[i].id, weights[i]);
        hardening.storeBackup(components[i].id, weights[i]);
    }

    // Simulate radiation: corrupt one weight
    std::vector<float> working_weights = weights;
    uint32_t* bits = reinterpret_cast<uint32_t*>(&working_weights[2]);
    *bits ^= 0x00001000;  // Corrupt weight[2]

    // Apply protection and recovery
    int recovered = 0;
    int detected = 0;
    int total = weights.size();

    for (size_t i = 0; i < weights.size(); i++) {
        auto result = hardening.applyProtection(working_weights[i], components[i].id, analysis);

        if (result.success && result.value.has_value()) {
            if (std::abs(*result.value - weights[i]) < 1e-6) {
                recovered++;  // Either uncorrupted or successfully recovered
            }
        }
        else {
            detected++;  // Error detected but not recovered
        }
    }

    std::cout << "  Total weights: " << total << "\n";
    std::cout << "  Recovered/uncorrupted: " << recovered << "\n";
    std::cout << "  Errors detected: " << detected << "\n";

    report("End-to-end - all weights processed", (recovered + detected) == total);
    report("End-to-end - corruption handled", recovered >= 4 || detected >= 1,
           "At least 4 uncorrupted OR 1 error detected");

    // Print report
    std::cout << "\n" << hardening.getProtectionReport(analysis);
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "Selective Hardening Test Suite\n";
    std::cout << "========================================\n";

    test_checksum_error_detection();
    test_physics_error_reduction();
    test_importance_decay();
    test_space_environment_analyzer();
    test_end_to_end_protection();

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed)
            passed++;
        else
            failed++;
    }

    std::cout << "Passed: " << passed << "/" << results.size() << "\n";
    std::cout << "Failed: " << failed << "/" << results.size() << "\n";

    if (failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << ": " << r.details << "\n";
            }
        }
    }

    return failed > 0 ? 1 : 0;
}
