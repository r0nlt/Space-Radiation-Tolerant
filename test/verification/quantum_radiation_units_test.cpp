#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "rad_ml/physics/quantum_enhanced_radiation.hpp"

namespace {

void requireNear(double actual, double expected, double relative_tolerance, const char* message)
{
    const double scale = std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > relative_tolerance * scale) {
        std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void testLetToChargeConversion()
{
    rad_ml::physics::SemiconductorProperties silicon;
    rad_ml::physics::QuantumEnhancedRadiation model(silicon);

    constexpr double elementary_charge_c = 1.602176634e-19;
    constexpr double deposited_energy_mev = 10.0 * 2329.0 * 1.0e-4;
    constexpr double expected_fc =
        deposited_energy_mev * 1.0e6 / 3.6 * elementary_charge_c * 1.0e15;

    requireNear(model.calculateClassicalChargeDeposition(100.0, 10.0), expected_fc, 1.0e-12,
                "LET-to-charge conversion is dimensionally incorrect");
}

void testSensitiveDepthScaling()
{
    rad_ml::physics::SemiconductorProperties one_micron;
    rad_ml::physics::SemiconductorProperties two_microns = one_micron;
    two_microns.sensitive_depth_um = 2.0;

    rad_ml::physics::QuantumEnhancedRadiation shallow(one_micron);
    rad_ml::physics::QuantumEnhancedRadiation deep(two_microns);
    const double shallow_charge = shallow.calculateClassicalChargeDeposition(100.0, 1.0);
    const double deep_charge = deep.calculateClassicalChargeDeposition(100.0, 1.0);

    requireNear(deep_charge, 2.0 * shallow_charge, 1.0e-12,
                "Charge deposition must scale with sensitive depth");
}

void testIncidentEnergyCap()
{
    rad_ml::physics::QuantumEnhancedRadiation model;
    constexpr double one_mev_charge_fc =
        1.0e6 / 3.6 * 1.602176634e-19 * 1.0e15;

    requireNear(model.calculateClassicalChargeDeposition(1.0, 100.0), one_mev_charge_fc, 1.0e-12,
                "Deposited energy must not exceed incident energy");
}

void testInvalidPhysicalInputs()
{
    rad_ml::physics::QuantumEnhancedRadiation model;
    require(model.calculateClassicalChargeDeposition(0.0, 10.0) == 0.0,
            "Zero particle energy must deposit zero charge");
    require(model.calculateClassicalChargeDeposition(10.0, -1.0) == 0.0,
            "Negative LET must deposit zero charge");
    require(model.calculateClassicalChargeDeposition(
                std::numeric_limits<double>::quiet_NaN(), 1.0) == 0.0,
            "Non-finite particle energy must be rejected");
    require(model.calculateDeviceSensitivity(rad_ml::physics::MemoryDeviceType::SRAM_6T, 0.0) ==
                0.0,
            "Non-positive feature size must be rejected");
}

void testConfinementEnergyUsesConsistentUnits()
{
    rad_ml::physics::QuantumEnhancedRadiation model;
    const double sensitivity =
        model.calculateDeviceSensitivity(rad_ml::physics::MemoryDeviceType::SRAM_6T, 10.0);

    // For a 10 nm silicon box with m*=0.26m_e, E1 is approximately 14.5 meV.
    const double expected = 1.0 + 0.0144623 / 1.12;
    requireNear(sensitivity, expected, 5.0e-5,
                "Quantum-confinement energy mixes eV and SI units");
}

void testCircuitCriticalChargeScaling()
{
    rad_ml::physics::CircuitProperties circuit_130;
    circuit_130.feature_size_nm = 130.0;
    rad_ml::physics::QuantumEnhancedRadiation model_130({}, circuit_130);

    rad_ml::physics::CircuitProperties circuit_180;
    circuit_180.feature_size_nm = 180.0;
    rad_ml::physics::QuantumEnhancedRadiation model_180({}, circuit_180);

    requireNear(model_130.calculateCircuitCriticalCharge(
                    rad_ml::physics::MemoryDeviceType::SRAM_6T, 300.0),
                3.30, 1.0e-12, "130 nm SRAM Qcrit must match its calibration point");
    requireNear(model_180.calculateCircuitCriticalCharge(
                    rad_ml::physics::MemoryDeviceType::SRAM_6T, 300.0),
                6.70, 1.0e-12, "180 nm SRAM Qcrit must match its calibration point");
}

void testCircuitAndParticleInputsRemainSeparated()
{
    rad_ml::physics::CircuitProperties small_circuit;
    small_circuit.feature_size_nm = 65.0;
    rad_ml::physics::CircuitProperties large_circuit;
    large_circuit.feature_size_nm = 180.0;

    rad_ml::physics::QuantumEnhancedRadiation small_model({}, small_circuit);
    rad_ml::physics::QuantumEnhancedRadiation large_model({}, large_circuit);

    const double small_deposition = small_model.calculateClassicalChargeDeposition(100.0, 2.0);
    const double large_deposition = large_model.calculateClassicalChargeDeposition(100.0, 2.0);
    requireNear(small_deposition, large_deposition, 1.0e-12,
                "Circuit feature size must not alter particle charge deposition");

    const double small_probability = small_model.calculateEnhancedBitFlipProbability(
        2.0, rad_ml::physics::MemoryDeviceType::SRAM_6T, 300.0);
    const double large_probability = large_model.calculateEnhancedBitFlipProbability(
        2.0, rad_ml::physics::MemoryDeviceType::SRAM_6T, 300.0);
    require(small_probability > large_probability,
            "Lower-Qcrit technology must be more susceptible at equal deposited charge");
}

}  // namespace

int main()
{
    testLetToChargeConversion();
    testSensitiveDepthScaling();
    testIncidentEnergyCap();
    testInvalidPhysicalInputs();
    testConfinementEnergyUsesConsistentUnits();
    testCircuitCriticalChargeScaling();
    testCircuitAndParticleInputsRemainSeparated();
    std::cout << "Quantum radiation unit tests passed\n";
    return EXIT_SUCCESS;
}
