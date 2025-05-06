/**
 * Quantum Physics Models for Radiation Effects
 * 
 * This file contains quantum mechanical models for simulating
 * radiation-induced displacement damage in materials.
 */

#pragma once

#include <vector>
#include <string>

namespace rad_ml {
namespace physics {

/**
 * Parameters for Density Functional Theory calculations
 */
struct DFTParameters {
    std::vector<int> kpoint_mesh;
    double energy_cutoff;
    double temperature;
};

/**
 * Class representing a crystal lattice structure
 */
class CrystalLattice {
public:
    enum LatticeType { SC, BCC, FCC, HCP, DIAMOND };
    
    /**
     * Create a face-centered cubic (FCC) lattice
     * @param lattice_constant Lattice constant in Angstroms
     * @return Initialized FCC crystal lattice
     */
    static CrystalLattice FCC(double lattice_constant);
    
    /**
     * Create a body-centered cubic (BCC) lattice
     * @param lattice_constant Lattice constant in Angstroms
     * @return Initialized BCC crystal lattice
     */
    static CrystalLattice BCC(double lattice_constant);
    
    /**
     * Default constructor
     */
    CrystalLattice() = default;
    
    /**
     * Constructor with lattice type and constant
     */
    CrystalLattice(LatticeType type, double lattice_constant);
};

/**
 * Class representing a distribution of defects in a material
 */
struct DefectDistribution {
    std::vector<double> interstitials;
    std::vector<double> vacancies;
    std::vector<double> clusters;
};

/**
 * Calculate lattice constant from material density and effective atomic number
 * @param density Material density in g/cm³
 * @param z_effective Effective atomic number
 * @return Lattice constant in Angstroms
 */
double calculateLatticeConstant(double density, double z_effective);

/**
 * Create appropriate lattice type based on material properties
 * @param z_effective Effective atomic number
 * @param lattice_constant Lattice constant in Angstroms
 * @return Crystal lattice object
 */
CrystalLattice createAppropriateLatticetype(double z_effective, double lattice_constant);

/**
 * Calculate displacement energy based on crystal structure
 * @param crystal Crystal lattice
 * @param params DFT parameters
 * @return Displacement energy in eV
 */
double calculateDisplacementEnergy(const CrystalLattice& crystal, const DFTParameters& params);

/**
 * Simulate displacement cascade with quantum molecular dynamics
 * @param crystal Crystal lattice
 * @param pka_energy Primary knock-on atom energy in eV
 * @param params DFT parameters
 * @param displacement_energy Displacement energy in eV
 * @return Distribution of defects
 */
DefectDistribution simulateDisplacementCascade(
    const CrystalLattice& crystal, 
    double pka_energy, 
    const DFTParameters& params,
    double displacement_energy);

/**
 * Calculate number of defects using Kinchin-Pease model
 * @param pka_energy Primary knock-on atom energy in eV
 * @param displacement_energy Displacement energy in eV
 * @return Number of defects
 */
int kinchinPeaseModel(double pka_energy, double displacement_energy);

/**
 * Calculate displacements per atom (dpa)
 * @param defects Defect distribution
 * @param crystal Crystal lattice
 * @return Displacements per atom
 */
double calculateDisplacementsPerAtom(const DefectDistribution& defects, const CrystalLattice& crystal);

} // namespace physics
} // namespace rad_ml 