/**
 * @file io.cpp
 * @brief Export and IO utilities for Auto Architecture Search
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <rad_ml/research/auto_arch_search.hpp>
#include <string>

namespace rad_ml {
namespace research {

void AutoArchSearch::exportResults(const std::string& filename) const
{
    std::ofstream out_file(filename);

    if (!out_file) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return;
    }

    out_file << "Architecture,Dropout,HasResidual,ProtectionLevel,Environment,"
             << "BaselineAccuracy,RadiationAccuracy,AccuracyPreservation,"
             << "ExecutionTime,ErrorsDetected,ErrorsCorrected,UncorrectableErrors,"
             << "BaselineAccuracyStdDev,RadiationAccuracyStdDev,AccuracyPreservationStdDev,"
             << "MonteCarloTrials\n";

    for (const auto& [config, result] : tested_configs_) {
        std::string arch_str;
        for (auto size : config.layer_sizes) {
            arch_str += std::to_string(size) + "-";
        }
        if (!arch_str.empty()) {
            arch_str.pop_back();
        }

        std::string protection_str;
        switch (config.protection_level) {
            case neural::ProtectionLevel::NONE:
                protection_str = "None";
                break;
            case neural::ProtectionLevel::CHECKSUM_ONLY:
                protection_str = "ChecksumOnly";
                break;
            case neural::ProtectionLevel::SELECTIVE_TMR:
                protection_str = "SelectiveTMR";
                break;
            case neural::ProtectionLevel::FULL_TMR:
                protection_str = "FullTMR";
                break;
            case neural::ProtectionLevel::ADAPTIVE_TMR:
                protection_str = "AdaptiveTMR";
                break;
            case neural::ProtectionLevel::SPACE_OPTIMIZED:
                protection_str = "SpaceOptimized";
                break;
            default:
                protection_str = "Unknown";
        }

        out_file << arch_str << "," << config.dropout_rate << ","
                 << (config.has_residual_connections ? "Yes" : "No") << "," << protection_str << ","
                 << static_cast<int>(result.environment) << "," << std::fixed
                 << std::setprecision(2) << result.baseline_accuracy << "," << std::fixed
                 << std::setprecision(2) << result.radiation_accuracy << "," << std::fixed
                 << std::setprecision(2) << result.accuracy_preservation << "," << std::fixed
                 << std::setprecision(2) << result.execution_time_ms << ","
                 << result.errors_detected << "," << result.errors_corrected << ","
                 << result.uncorrectable_errors << "," << std::fixed << std::setprecision(2)
                 << result.baseline_accuracy_stddev << "," << std::fixed << std::setprecision(2)
                 << result.radiation_accuracy_stddev << "," << std::fixed << std::setprecision(2)
                 << result.accuracy_preservation_stddev << "," << result.monte_carlo_trials << "\n";
    }

    std::cout << "Results exported to " << filename << std::endl;
}

}  // namespace research
}  // namespace rad_ml
