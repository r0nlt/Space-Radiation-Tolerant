#pragma once

#include "radiation_simulator.hpp"
#include <vector>

namespace rad_ml {
namespace testing {

enum class ProtectionTechnique {
    NONE,
    TMR,        // Triple Modular Redundancy
    EDAC,       // Error Detection and Correction
    SCRUBBING   // Memory Scrubbing
};

struct ProtectionResult {
    int corrections_successful;  // Number of successful corrections
    int total_errors;           // Total number of errors
    double seu_rate;            // Single Event Upset rate
    double let_threshold;       // LET threshold
    double cross_section;       // Cross section
    double mtbf;                // Mean Time Between Failures
    double ber;                 // Bit Error Rate
};

ProtectionResult applyProtectionTechnique(
    ProtectionTechnique technique,
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events);

void applyTMR(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result);

void applyEDAC(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result);

void applyScrubbing(
    std::vector<uint8_t>& memory,
    const std::vector<RadiationSimulator::RadiationEvent>& events,
    ProtectionResult& result);

void calculateMetrics(
    ProtectionResult& result,
    const std::vector<RadiationSimulator::RadiationEvent>& events);

} // namespace testing
} // namespace rad_ml 