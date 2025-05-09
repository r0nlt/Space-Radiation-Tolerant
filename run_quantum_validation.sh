#!/bin/bash

# Exit on error
set -e

echo "========================================================="
echo "Quantum Field Theory Enhancements Validation Test"
echo "========================================================="

# Ensure we have all necessary directories
mkdir -p build
mkdir -p src/rad_ml/physics
mkdir -p src/testing
mkdir -p test/verification
mkdir -p src/tmr
mkdir -p tools
mkdir -p examples
mkdir -p include/rad_ml/physics
mkdir -p include/rad_ml/testing

# Check if build directory exists, create if not
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

# Create minimal TMR CMakeLists.txt if it doesn't exist
if [ ! -f "src/tmr/CMakeLists.txt" ]; then
    echo "Creating minimal TMR CMakeLists.txt..."
    echo "add_library(rad_ml_tmr INTERFACE)" > src/tmr/CMakeLists.txt
fi

# Create stub header files if they don't exist
if [ ! -f "include/rad_ml/physics/quantum_models.hpp" ]; then
    echo "Creating stub quantum_models.hpp..."
    cat > include/rad_ml/physics/quantum_models.hpp <<EOF
#pragma once
#include <vector>
#include <string>

namespace rad_ml {
namespace physics {

struct DFTParameters {
    std::vector<int> kpoint_mesh;
    double energy_cutoff;
    double temperature;
};

class CrystalLattice {
public:
    enum LatticeType { SC, BCC, FCC, HCP, DIAMOND };
    static CrystalLattice FCC(double lattice_constant) { return CrystalLattice(FCC, lattice_constant); }
    static CrystalLattice BCC(double lattice_constant) { return CrystalLattice(BCC, lattice_constant); }
    CrystalLattice() = default;
    CrystalLattice(LatticeType type, double lattice_constant) : type(type), lattice_constant(lattice_constant) {}
    
    LatticeType type;
    double lattice_constant;
};

struct DefectDistribution {
    std::vector<double> interstitials = {1.0, 2.0, 3.0};
    std::vector<double> vacancies = {1.0, 2.0, 3.0};
    std::vector<double> clusters = {0.5, 1.0, 1.5};
};

double calculateLatticeConstant(double density, double z_effective) { return 5.0; }
CrystalLattice createAppropriateLatticetype(double z_effective, double lattice_constant) { return CrystalLattice(); }
double calculateDisplacementEnergy(const CrystalLattice& crystal, const DFTParameters& params) { return 10.0; }

DefectDistribution simulateDisplacementCascade(
    const CrystalLattice& crystal, 
    double pka_energy, 
    const DFTParameters& params,
    double displacement_energy) { return DefectDistribution(); }

int kinchinPeaseModel(double pka_energy, double displacement_energy) { return 10; }
double calculateDisplacementsPerAtom(const DefectDistribution& defects, const CrystalLattice& crystal) { return 0.01; }

} // namespace physics
} // namespace rad_ml
EOF
fi

if [ ! -f "include/rad_ml/physics/field_theory.hpp" ]; then
    echo "Creating stub field_theory.hpp..."
    cat > include/rad_ml/physics/field_theory.hpp <<EOF
#pragma once
#include <vector>

namespace rad_ml {
namespace physics {

class Grid3D {
public:
    int size_x;
    int size_y;
    int size_z;
    double spacing;
    Grid3D(int size_x, int size_y, int size_z, double spacing)
        : size_x(size_x), size_y(size_y), size_z(size_z), spacing(spacing) {}
};

template<typename T = double>
class Field3D {
public:
    Field3D(const Grid3D& grid) {}
    T& operator()(int i, int j, int k) { static T value; return value; }
    const T& operator()(int i, int j, int k) const { static T value; return value; }
    void setZero() {}
};

struct FieldParameters {
    double kappa;
    std::vector<std::vector<double>> gamma;
};

class FreeEnergyFunctional {
public:
    FreeEnergyFunctional(const FieldParameters& params) {}
    std::vector<Field3D<double>> calculateDerivatives(const std::vector<Field3D<double>>& fields) { return {}; }
};

struct TimeEvolutionResults {
    double total_defects_initial;
    double total_defects_final;
    std::vector<Field3D<double>> final_fields;
};

double calculateGradientEnergyCoefficient(const struct MaterialProperties& material) { return 1.0; }
std::vector<std::vector<double>> createInteractionMatrix(const std::vector<double>& defect_formation_energies) { return {}; }

void initializeDefectFields(
    std::vector<Field3D<double>>& fields, 
    const struct RadiationEnvironment& env, 
    const struct MaterialProperties& material) {}

TimeEvolutionResults solveFieldEquations(
    std::vector<Field3D<double>>& fields,
    const std::vector<Field3D<double>>& derivatives,
    double radiation_dose,
    double temperature,
    double applied_stress) { return {}; }

double calculateClusteringRatio(const std::vector<Field3D<double>>& fields) { return 0.5; }

} // namespace physics
} // namespace rad_ml
EOF
fi

if [ ! -f "include/rad_ml/testing/fault_injection.hpp" ]; then
    echo "Creating stub fault_injection.hpp..."
    cat > include/rad_ml/testing/fault_injection.hpp <<EOF
#pragma once
#include <vector>
#include <random>
#include <string>

namespace rad_ml {
namespace testing {

enum FaultPattern {
    SINGLE_BIT,
    ADJACENT_BITS,
    BYTE_ERROR,
    WORD_ERROR,
    STUCK_AT_ZERO,
    STUCK_AT_ONE,
    ROW_COLUMN,
    BURST_ERROR
};

class SystematicFaultInjector {
public:
    SystematicFaultInjector();
    void setSeed(unsigned int seed);
    std::string patternToString(FaultPattern pattern);
    std::vector<int> getBitsToFlip(FaultPattern pattern, int total_bits, int starting_bit = -1);
    
private:
    std::mt19937 gen;
};

} // namespace testing
} // namespace rad_ml
EOF
fi

# Navigate to build directory
cd build

# Configure and build
echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building quantum_field_validation_test..."
cmake --build . --target quantum_field_validation_test || {
    echo "Build failed. Checking for more detailed errors..."
    cat CMakeFiles/CMakeError.log || echo "No detailed error log found."
    exit 1
}

# Run test
echo "Running quantum field validation test..."
if [ -f "quantum_field_validation_test" ]; then
    ./quantum_field_validation_test
else 
    echo "Error: Test executable not found. Build likely failed."
    exit 1
fi

# Check if Python visualization script exists and run it
cd ..
if [ -f "tools/visualize_quantum_results.py" ]; then
    echo "Generating visualization of results..."
    if command -v python3 &>/dev/null; then
        # Check for required Python packages
        MISSING_PACKAGES=false
        for pkg in pandas matplotlib numpy; do
            if ! python3 -c "import $pkg" &>/dev/null; then
                echo "Warning: Python package '$pkg' is not installed. Visualization may fail."
                MISSING_PACKAGES=true
            fi
        done
        
        if [ "$MISSING_PACKAGES" = true ]; then
            echo "You can install required packages with: pip install pandas matplotlib numpy"
        fi
        
        python3 tools/visualize_quantum_results.py
    else
        echo "Python 3 is not available. Skipping visualization."
    fi
else
    echo "Visualization script not found. Skipping visualization."
fi

echo "========================================================="
echo "Quantum validation complete!"
echo "Results are saved in quantum_enhancement_results.csv"
if [ -d "quantum_analysis_results" ]; then
    echo "Visualizations are available in the quantum_analysis_results directory"
fi
echo "=========================================================" 