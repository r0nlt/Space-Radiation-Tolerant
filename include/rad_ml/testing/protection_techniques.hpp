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
