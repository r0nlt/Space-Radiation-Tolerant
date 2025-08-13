#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "rad_ml/neural/radiation_environment.hpp"

int main()
{
    using namespace rad_ml::neural;

    std::cout << "=== Radiation Environment Validation ===\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Configure environment: LEO Equatorial
    RadiationEnvironment env(SpaceMission::LEO_EQUATORIAL);
    env.setAverageTemperatureK(300.0);
    env.setDeviceSensitivity(1.2);  // example device sensitivity multiplier

    // Provide a precise physics calculator (per-bit per-second)
    env.setSEUCalculator([](const RadiationFlux& flux, double T, double device, bool in_saa,
                            double solar) -> double {
        // Species-weighted base (placeholder). In a full setup, replace with σ(LET)·flux.
        const double electron_factor = 1e-12;
        const double proton_factor = 1e-9;
        const double heavy_factor = 1e-6;
        const double base = flux.electron_flux * electron_factor +
                            flux.proton_flux * proton_factor + flux.heavy_ion_flux * heavy_factor;

        // Corrections consistent with paper notation
        const double Ctemp = 1.0 + std::max(0.0, (T - 273.0) / 100.0);
        const double Csolar = 1.0 + 0.5 * std::clamp(solar, 0.0, 1.0);
        const double Cregion = in_saa ? 1.5 : 1.0;
        const double Cdevice = std::max(0.0, device);
        return base * Ctemp * Csolar * Cregion * Cdevice;  // per second
    });

    env.setPrecisePhysics(true);

    // Positions to probe: nominal LEO and near SAA center
    std::vector<std::pair<std::string, OrbitalPosition>> points = {
        {"LEO-Equatorial (nominal)", {0.0, 0.0, 500.0}},
        {"LEO-SAA vicinity", {-30.0, -40.0, 500.0}},
    };

    auto run_case = [&](const std::string& label) {
        std::cout << "\n--- " << label << " ---\n";
        for (const auto& [name, pos] : points) {
            RadiationFlux flux = env.calculateRadiationFlux(pos);
            double seu_per_day_precise = env.calculateSEUProbability(pos);

            // Temporarily disable precise path to compute heuristic for comparison
            env.setPrecisePhysics(false);
            double seu_per_day_heuristic = env.calculateSEUProbability(pos);
            env.setPrecisePhysics(true);

            std::cout << std::fixed << std::setprecision(3);
            std::cout << name << ":\n";
            std::cout << "  Flux (e/p/hi) [1/cm^2/s]: " << flux.electron_flux << ", "
                      << flux.proton_flux << ", " << flux.heavy_ion_flux << "\n";
            std::cout << std::setprecision(6);
            std::cout << "  SEU per bit per day (heuristic): " << seu_per_day_heuristic << "\n";
            std::cout << "  SEU per bit per day (precise):   " << seu_per_day_precise << "\n";
        }
    };

    // Baseline (solar activity 0.5, T=300K)
    run_case("Baseline (solar=0.5, T=300K, device=1.2)");

    // Higher solar activity
    env.setSolarActivity(0.8);
    run_case("High solar activity (solar=0.8)");

    // Lower temperature
    env.setAverageTemperatureK(200.0);
    run_case("Lower temperature (T=200K)");

    // Different device sensitivity
    env.setDeviceSensitivity(0.8);
    run_case("Lower device sensitivity (device=0.8)");

    std::cout << "\nValidation complete.\n";
    return 0;
}
