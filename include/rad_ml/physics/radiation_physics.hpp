/**
 * @file radiation_physics.hpp
 * @brief Physics-based radiation environment models
 *
 * This file provides rigorous radiation environment modeling based on:
 * - AP-8/AE-8 trapped particle models (NASA)
 * - CREME96-style Galactic Cosmic Ray models
 * - Weibull SEU cross-section models
 * - Bendel proton SEU model
 * - ISO 15390 GCR flux models
 *
 * References:
 * - Sawyer & Vette, "AP-8 Trapped Proton Environment", NSSDC 76-06, 1976
 * - Vette, "AE-8 Trapped Electron Model", NSSDC 91-24, 1991
 * - Tylka et al., "CREME96: A Revision of the Cosmic Ray Effects", IEEE TNS, 1997
 * - Petersen, "Single Event Analysis and Prediction", IEEE NSREC Short Course, 1997
 * - Bendel & Petersen, "Proton Upsets in Orbit", IEEE TNS, 1983
 * - ISO 15390:2004 "Space environment - GCR model"
 *
 * Known Limitations:
 * - Geomagnetic field uses dipole approximation (error ~10% at LEO)
 *   For production: use IGRF-13 coefficients
 * - AP-8/AE-8 models are from 1976/1991; consider AP-9/AE-9 for modern work
 * - Proton SEU uses Bendel approximation; device-specific data preferred
 *
 * @author Rishab Nuguru, Space Labs AI
 * @date 2024
 */

#ifndef RAD_ML_PHYSICS_RADIATION_PHYSICS_HPP
#define RAD_ML_PHYSICS_RADIATION_PHYSICS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rad_ml {
namespace physics {

// Physical constants
namespace constants {
constexpr double EARTH_RADIUS_KM = 6371.0;
constexpr double EARTH_MAGNETIC_MOMENT = 7.94e22;  // A·m² (dipole moment)
constexpr double PROTON_MASS_MEV = 938.272;        // MeV/c²
constexpr double ELECTRON_MASS_MEV = 0.511;        // MeV/c²
constexpr double AU_TO_KM = 1.496e8;               // km per AU
constexpr double SECONDS_PER_DAY = 86400.0;
constexpr double CM2_TO_M2 = 1e-4;
constexpr double SPEED_OF_LIGHT = 2.998e8;       // m/s
constexpr double ELEMENTARY_CHARGE = 1.602e-19;  // Coulombs
}  // namespace constants

/**
 * @brief Model uncertainty bounds for flux predictions
 *
 * AP-8/AE-8 models have known limitations. These factors represent
 * typical uncertainty ranges based on validation studies.
 */
struct FluxUncertainty {
    double factor_low = 0.5;   ///< Models can underpredict by 2x
    double factor_high = 3.0;  ///< Or overpredict by 3x (outer belt electrons)

    /// Get worst-case flux estimate
    double worst_case(double nominal_flux) const { return nominal_flux * factor_high; }

    /// Get best-case flux estimate
    double best_case(double nominal_flux) const { return nominal_flux * factor_low; }
};

/**
 * @brief Weibull cross-section parameters for SEU modeling
 *
 * The Weibull function models SEU cross-section vs. LET:
 * σ(LET) = σ_sat * (1 - exp(-(LET - LET_th)/W)^s)  for LET > LET_th
 *        = 0                                         for LET ≤ LET_th
 *
 * Parameters derived from device ground testing with heavy ions.
 * Values below are representative; use device-specific test data when available.
 */
struct WeibullParameters {
    double sigma_sat;      ///< Saturation cross-section (cm²/bit)
    double let_threshold;  ///< LET threshold (MeV·cm²/mg)
    double width;          ///< Weibull width parameter W (MeV·cm²/mg)
    double shape;          ///< Weibull shape parameter s (typically 1-4)

    // Default parameters for 65nm CMOS (typical, based on published data)
    // Ref: Dodd et al., IEEE TNS, 2010
    static WeibullParameters cmos_65nm() { return {1e-8, 0.5, 15.0, 2.0}; }

    // Parameters for 28nm CMOS (more sensitive due to smaller critical charge)
    // Ref: Massengill et al., IEEE TNS, 2012
    static WeibullParameters cmos_28nm() { return {5e-8, 0.2, 10.0, 1.5}; }

    // Parameters for radiation-hardened device (RHBD techniques)
    static WeibullParameters rad_hard() { return {1e-10, 5.0, 30.0, 3.0}; }

    // Parameters for SRAM (highly sensitive)
    static WeibullParameters sram_commercial() { return {1e-7, 0.1, 8.0, 1.2}; }
};

/**
 * @brief Bendel parameters for proton-induced SEU
 *
 * The Bendel model calculates proton SEU cross-section from nuclear reactions.
 * This is more accurate than using an effective LET for protons.
 *
 * σ_p(E) = (A/24)^14 * (1 - exp(-0.18*(sqrt(E) - sqrt(A))))^4
 *
 * Where A is the Bendel parameter (~MeV) and E is proton energy.
 * Ref: Bendel & Petersen, IEEE TNS-30, 1983
 */
struct BendelParameters {
    double A;          ///< Bendel A parameter (MeV), typically 10-50
    double sigma_inf;  ///< Limiting cross-section at high energy (cm²/bit)

    // Typical parameters for 65nm CMOS
    static BendelParameters cmos_65nm() { return {18.0, 1e-13}; }

    // Parameters for 28nm - lower threshold, higher cross-section
    static BendelParameters cmos_28nm() { return {12.0, 5e-13}; }

    // Radiation hardened
    static BendelParameters rad_hard() { return {35.0, 1e-15}; }
};

/**
 * @brief Calculate SEU cross-section using Weibull model
 *
 * @param let Linear Energy Transfer (MeV·cm²/mg)
 * @param params Weibull parameters from device characterization
 * @return Cross-section in cm²/bit
 */
inline double weibull_cross_section(double let, const WeibullParameters& params)
{
    if (let <= params.let_threshold) {
        return 0.0;
    }

    double x = (let - params.let_threshold) / params.width;
    return params.sigma_sat * (1.0 - std::exp(-std::pow(x, params.shape)));
}

/**
 * @brief Calculate proton SEU cross-section using Bendel model
 *
 * Protons cause SEUs primarily through nuclear reactions producing heavy
 * recoils, not direct ionization. The Bendel model captures this physics.
 *
 * @param energy_mev Proton kinetic energy (MeV)
 * @param params Bendel parameters from device testing
 * @return Cross-section in cm²/bit
 */
inline double bendel_proton_cross_section(double energy_mev, const BendelParameters& params)
{
    if (energy_mev <= params.A) {
        return 0.0;  // Below threshold
    }

    // Bendel two-parameter model
    double sqrt_E = std::sqrt(energy_mev);
    double sqrt_A = std::sqrt(params.A);

    double Y = 0.18 * (sqrt_E - sqrt_A);
    if (Y <= 0) return 0.0;

    double term = 1.0 - std::exp(-Y);
    double sigma = params.sigma_inf * std::pow(term, 4);

    // Apply (A/24)^14 scaling for the one-parameter Bendel form
    // For two-parameter form, this is absorbed into sigma_inf

    return sigma;
}

/**
 * @brief Temperature correction factor for SEU cross-section
 *
 * SEU cross-section has weak temperature dependence for some devices,
 * primarily through changes in critical charge and carrier mobility.
 *
 * @param temp_kelvin Device temperature in Kelvin
 * @return Multiplicative correction factor (1.0 at 300K)
 */
inline double temperature_factor(double temp_kelvin)
{
    // Approximately 1% increase per 10K for sensitive devices
    // Ref: Various device characterization studies
    constexpr double reference_temp = 300.0;  // Room temperature
    constexpr double coefficient = 0.001;     // 0.1% per K

    return 1.0 + coefficient * (temp_kelvin - reference_temp);
}

/**
 * @brief Convert geomagnetic cutoff rigidity to minimum proton energy
 *
 * Rigidity R = pc/Ze, where p is momentum, c is speed of light, Z is charge.
 * For protons (Z=1): E = sqrt((Rc)² + m²c⁴) - mc²
 *
 * @param rigidity_gv Cutoff rigidity in GV (gigavolts)
 * @return Minimum proton kinetic energy in MeV
 */
inline double proton_energy_from_rigidity(double rigidity_gv)
{
    // R (GV) = pc/e (GeV/c for protons)
    // Convert to MeV: pc = R * 1000 MeV/c
    double pc_mev = rigidity_gv * 1000.0;  // Momentum * c in MeV

    // Relativistic energy-momentum relation: E² = (pc)² + (mc²)²
    double E_total =
        std::sqrt(pc_mev * pc_mev + constants::PROTON_MASS_MEV * constants::PROTON_MASS_MEV);

    // Kinetic energy = total energy - rest mass
    return E_total - constants::PROTON_MASS_MEV;
}

/**
 * @brief Convert proton energy to approximate stopping LET in silicon
 *
 * Uses empirical fit to SRIM/PSTAR data for protons in silicon.
 *
 * @param energy_mev Proton kinetic energy (MeV)
 * @return LET in MeV·cm²/mg (approximate)
 */
inline double proton_let_in_silicon(double energy_mev)
{
    // Empirical fit to proton stopping power in silicon
    // Valid for ~1-500 MeV range
    // Ref: PSTAR database, NIST

    if (energy_mev < 0.1) return 100.0;  // Bragg peak region
    if (energy_mev > 500) return 0.01;   // Minimum ionizing

    // Power law fit: LET ≈ 30 * E^(-0.8) for E > 1 MeV
    // Plus low-energy correction
    double let = 30.0 * std::pow(energy_mev, -0.8);

    // Bragg peak enhancement at low energies
    if (energy_mev < 10.0) {
        let *= 1.0 + 5.0 * std::exp(-energy_mev / 2.0);
    }

    return let;
}

/**
 * @brief Differential LET spectrum for GCR (CREME96-style)
 *
 * Models the integral LET spectrum from galactic cosmic rays.
 * Based on Adams' "Cosmic Ray Effects on Microelectronics" model.
 *
 * Φ(>LET) ≈ A * LET^(-γ) with γ ≈ 1.9 for interplanetary space
 */
struct GCRSpectrum {
    double solar_modulation_mv;  ///< Solar modulation Φ (MV), 400-1200 typical

    explicit GCRSpectrum(double phi_mv = 650.0) : solar_modulation_mv(phi_mv) {}

    /**
     * @brief Get differential LET spectrum
     *
     * dΦ/dLET = d/dLET[A * LET^(-γ)] = -A * γ * LET^(-(γ+1))
     *
     * @param let LET in MeV·cm²/mg
     * @return Differential flux in particles/(cm²·s·sr·(MeV·cm²/mg))
     */
    double differential_flux(double let) const
    {
        if (let <= 0) return 0;

        // Adams GCR model coefficients
        // Solar minimum (Φ~400 MV): higher flux, A ≈ 0.24
        // Solar maximum (Φ~1200 MV): lower flux, A ≈ 0.08
        constexpr double A_min = 0.24;  // Solar minimum coefficient
        constexpr double A_max = 0.08;  // Solar maximum coefficient
        constexpr double gamma = 1.9;   // Power law index

        // Linear interpolation based on solar modulation
        double solar_factor = (1200.0 - solar_modulation_mv) / 800.0;
        solar_factor = std::clamp(solar_factor, 0.0, 1.0);

        double A = A_max + (A_min - A_max) * solar_factor;

        // Differential spectrum = derivative of integral spectrum
        // Units: particles/(cm²·s·sr·(MeV·cm²/mg))
        return A * gamma * std::pow(let, -(gamma + 1.0));
    }

    /**
     * @brief Get integral LET spectrum (flux above given LET)
     *
     * @param let_threshold Minimum LET to count
     * @return Integral flux in particles/(cm²·s·sr)
     */
    double integral_flux(double let_threshold) const
    {
        if (let_threshold <= 0) return 1e6;  // Cap at reasonable maximum

        double solar_factor = (1200.0 - solar_modulation_mv) / 800.0;
        solar_factor = std::clamp(solar_factor, 0.0, 1.0);

        double A = 0.08 + 0.16 * solar_factor;
        constexpr double gamma = 1.9;

        // Φ(>LET) = A * LET^(-γ)
        // Units: particles/(cm²·s·sr)
        return A * std::pow(let_threshold, -gamma);
    }
};

/**
 * @brief Trapped proton model (AP-8 simplified)
 *
 * Models the trapped proton population in Earth's inner radiation belt.
 * Based on AP-8 model with B/L coordinate transformation.
 *
 * NOTE: AP-8 dates from 1976. For modern missions, consider AP-9.
 * Uncertainty factor: typically 2-3x for flux predictions.
 *
 * Reference: Sawyer & Vette, NSSDC 76-06
 */
class TrappedProtonModel {
   public:
    /**
     * @brief Calculate differential trapped proton flux at given location
     *
     * @param L McIlwain L-shell parameter (Earth radii)
     * @param B_B0 Magnetic field ratio B/B₀ at L-shell
     * @param energy_mev Proton energy in MeV
     * @param solar_max True for solar maximum, false for solar minimum
     * @return Differential flux in protons/(cm²·s·MeV)
     */
    static double differential_flux(double L, double B_B0, double energy_mev,
                                    bool solar_max = false)
    {
        // AP-8 validity range
        if (L < 1.15 || L > 6.6) return 0.0;
        if (energy_mev < 0.1 || energy_mev > 400.0) return 0.0;

        // Simplified AP-8 fit (full model uses interpolation tables)

        // L-shell dependence: peak around L=1.5 for inner belt
        double L_factor = std::exp(-std::pow(L - 1.5, 2) / 0.5);

        // Energy spectrum: exponential rolloff with energy
        // Characteristic energy ~30-50 MeV in inner belt
        double E0 = 40.0;  // Characteristic energy (MeV)
        double E_factor = std::exp(-energy_mev / E0) / E0;

        // Pitch angle distribution: peaked near equator (B/B0 = 1)
        double B_factor = std::exp(-(B_B0 - 1.0) * 2.0);

        // Base omnidirectional flux at L=1.5, normalized
        // Calibrated to AP-8 at L=1.5: ~10^4 protons/(cm²·s) for E>10 MeV
        // Solar max has ~2x lower flux in inner belt (anti-correlation)
        double base_flux = solar_max ? 5e5 : 1e6;  // protons/(cm²·s·MeV) at peak

        return base_flux * L_factor * E_factor * B_factor;
    }

    /**
     * @brief Calculate integral flux above energy threshold
     *
     * Φ(>E) = ∫_E^Emax (dΦ/dE') dE'
     *
     * @param L L-shell parameter
     * @param B_B0 B/B₀ ratio
     * @param energy_threshold_mev Minimum energy
     * @param solar_max Solar maximum conditions
     * @return Integral flux in protons/(cm²·s)
     */
    static double integral_flux(double L, double B_B0, double energy_threshold_mev,
                                bool solar_max = false)
    {
        // Numerical integration: trapezoidal rule
        double total_flux = 0.0;
        constexpr int n_steps = 100;
        constexpr double e_max = 400.0;

        if (energy_threshold_mev >= e_max) return 0.0;

        double de = (e_max - energy_threshold_mev) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double e = energy_threshold_mev + (i + 0.5) * de;
            // Integrate differential flux over energy
            total_flux += differential_flux(L, B_B0, e, solar_max) * de;
        }

        // Result is integral flux in protons/(cm²·s)
        return total_flux;
    }

    /// Get flux uncertainty bounds
    static FluxUncertainty uncertainty()
    {
        return {0.5, 2.5};  // AP-8 typically within factor of 2-3
    }
};

/**
 * @brief Trapped electron model (AE-8 simplified)
 *
 * Models trapped electron population in Earth's radiation belts.
 * Inner belt (L<2.5): High-energy electrons from CRAND
 * Outer belt (L>3): Magnetospheric electrons, highly dynamic
 *
 * NOTE: AE-8 dates from 1991. Outer belt is highly variable and
 * model uncertainties can exceed factor of 10 during storms.
 *
 * Reference: Vette, NSSDC 91-24
 */
class TrappedElectronModel {
   public:
    /**
     * @brief Calculate differential trapped electron flux
     *
     * @param L McIlwain L-shell
     * @param B_B0 B/B₀ ratio
     * @param energy_mev Electron energy
     * @param solar_max Solar maximum conditions
     * @return Differential flux in electrons/(cm²·s·MeV)
     */
    static double differential_flux(double L, double B_B0, double energy_mev,
                                    bool solar_max = false)
    {
        if (L < 1.2 || L > 11.0) return 0.0;
        if (energy_mev < 0.04 || energy_mev > 7.0) return 0.0;

        double inner_belt = 0.0, outer_belt = 0.0;

        // Inner belt (L ~ 1.5-2.5): relatively stable, high-energy electrons
        if (L >= 1.2 && L <= 2.8) {
            double L_inner = std::exp(-std::pow(L - 1.8, 2) / 0.3);
            double E0_inner = 1.5;  // MeV
            inner_belt = 1e7 * L_inner * std::exp(-energy_mev / E0_inner) / E0_inner;
        }

        // Outer belt (L ~ 4-6): highly variable with geomagnetic activity
        if (L >= 3.0 && L <= 8.0) {
            double L_outer = std::exp(-std::pow(L - 4.5, 2) / 2.0);
            // Outer belt enhanced during solar maximum (opposite of protons!)
            double activity_factor = solar_max ? 3.0 : 1.0;
            double E0_outer = 0.8;  // MeV (softer spectrum)
            outer_belt =
                5e8 * activity_factor * L_outer * std::exp(-energy_mev / E0_outer) / E0_outer;
        }

        // Pitch angle distribution
        double B_factor = std::exp(-(B_B0 - 1.0) * 2.0);

        return (inner_belt + outer_belt) * B_factor;
    }

    /**
     * @brief Calculate integral electron flux above threshold
     *
     * @param L L-shell parameter
     * @param B_B0 B/B₀ ratio
     * @param energy_threshold_mev Minimum energy
     * @param solar_max Solar maximum conditions
     * @return Integral flux in electrons/(cm²·s)
     */
    static double integral_flux(double L, double B_B0, double energy_threshold_mev,
                                bool solar_max = false)
    {
        double total = 0.0;
        constexpr int n_steps = 100;
        constexpr double e_max = 7.0;

        if (energy_threshold_mev >= e_max) return 0.0;

        double de = (e_max - energy_threshold_mev) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double e = energy_threshold_mev + (i + 0.5) * de;
            total += differential_flux(L, B_B0, e, solar_max) * de;
        }

        return total;  // electrons/(cm²·s)
    }

    /// Get flux uncertainty bounds (outer belt is very uncertain)
    static FluxUncertainty uncertainty()
    {
        return {0.3, 10.0};  // Factor of 3-10 for outer belt
    }
};

/**
 * @brief Geomagnetic field model for coordinate calculations
 *
 * Implements centered dipole approximation for B and L calculation.
 *
 * LIMITATION: This is a dipole approximation with ~10% error at LEO.
 * For production missions, implement IGRF-13 (International Geomagnetic
 * Reference Field) with full spherical harmonic coefficients.
 */
class GeomagneticField {
   public:
    /**
     * @brief Calculate McIlwain L-shell parameter
     *
     * L = R / cos²(λ_m) where λ_m is magnetic latitude
     * For a dipole field, L represents the equatorial crossing distance
     * of the field line passing through the point.
     *
     * @param altitude_km Altitude above Earth's surface
     * @param latitude_deg Geographic latitude (≈magnetic for dipole)
     * @param longitude_deg Geographic longitude (unused in dipole model)
     * @return L-shell value in Earth radii
     */
    static double calculate_L(double altitude_km, double latitude_deg, double longitude_deg = 0.0)
    {
        (void)longitude_deg;  // Unused in dipole approximation

        // NOTE: For production, implement coordinate transform using IGRF
        // Geographic to geomagnetic coordinates differ by ~11° (dipole tilt)
        double mag_lat = latitude_deg;  // Simplified: geographic ≈ magnetic

        double r = (constants::EARTH_RADIUS_KM + altitude_km) / constants::EARTH_RADIUS_KM;
        double cos_lat = std::cos(mag_lat * M_PI / 180.0);

        // Avoid division by zero at poles
        if (std::abs(cos_lat) < 0.01) {
            return r / 0.0001;  // Very large L at poles
        }

        // L = r / cos²(λ) for dipole field
        return r / (cos_lat * cos_lat);
    }

    /**
     * @brief Calculate B/B₀ ratio
     *
     * B₀ is the minimum B along the field line (at magnetic equator).
     * B/B₀ determines the mirror point for trapped particles.
     *
     * For dipole: B/B₀ = sqrt(1 + 3sin²λ) / cos⁶λ
     *
     * @param altitude_km Altitude
     * @param latitude_deg Latitude
     * @return B/B₀ ratio (≥1, equals 1 at equator)
     */
    static double calculate_B_B0(double altitude_km, double latitude_deg)
    {
        (void)altitude_km;  // B/B0 only depends on latitude for dipole

        double mag_lat_rad = std::abs(latitude_deg) * M_PI / 180.0;
        double sin_lat = std::sin(mag_lat_rad);
        double cos_lat = std::cos(mag_lat_rad);

        // B/B₀ for dipole field
        double numerator = std::sqrt(1.0 + 3.0 * sin_lat * sin_lat);
        double denominator = std::pow(cos_lat, 6);

        if (denominator < 1e-6) return 1000.0;  // Very large at poles

        return std::max(1.0, numerator / denominator);
    }

    /**
     * @brief Calculate vertical geomagnetic cutoff rigidity
     *
     * The cutoff rigidity determines the minimum momentum/charge for
     * cosmic rays to reach a given location through Earth's magnetic field.
     *
     * Störmer formula: R_c ≈ 14.9 cos⁴(λ_m) GV for vertical incidence
     *
     * @param latitude_deg Geographic latitude
     * @return Cutoff rigidity in GV
     */
    static double cutoff_rigidity(double latitude_deg)
    {
        // Störmer vertical cutoff approximation
        // R_c = (M/4r²) * cos⁴(λ) ≈ 14.9 cos⁴(λ) GV at Earth's surface
        double mag_lat_rad = std::abs(latitude_deg) * M_PI / 180.0;
        double cos_lat = std::cos(mag_lat_rad);

        return 14.9 * std::pow(cos_lat, 4);
    }

    /**
     * @brief Calculate minimum cosmic ray energy from cutoff rigidity
     *
     * @param latitude_deg Geographic latitude
     * @return Minimum proton energy in MeV that can reach this latitude
     */
    static double minimum_proton_energy(double latitude_deg)
    {
        double Rc = cutoff_rigidity(latitude_deg);
        return proton_energy_from_rigidity(Rc);
    }
};

/**
 * @brief South Atlantic Anomaly model
 *
 * The SAA is caused by the ~500 km offset and ~11° tilt of Earth's
 * magnetic dipole from the geographic center. This creates a region
 * where the inner radiation belt dips to LEO altitudes.
 *
 * The SAA center drifts westward at ~0.3-0.5°/year.
 */
class SouthAtlanticAnomaly {
   public:
    // SAA center coordinates (2024 approximate, drifting westward)
    static constexpr double CENTER_LAT = -29.0;  // degrees
    static constexpr double CENTER_LON = -47.0;  // degrees (was -50° in 1990s)
    static constexpr double SEMI_MAJOR = 35.0;   // degrees (longitude extent)
    static constexpr double SEMI_MINOR = 25.0;   // degrees (latitude extent)

    /**
     * @brief Calculate flux enhancement factor in SAA
     *
     * @param latitude_deg Geographic latitude
     * @param longitude_deg Geographic longitude
     * @return Enhancement factor (1.0 outside SAA, up to ~100x at center)
     */
    static double enhancement_factor(double latitude_deg, double longitude_deg)
    {
        double dlat = (latitude_deg - CENTER_LAT) / SEMI_MINOR;
        double dlon = normalize_longitude_diff(longitude_deg - CENTER_LON) / SEMI_MAJOR;

        double r2 = dlat * dlat + dlon * dlon;

        if (r2 >= 1.0) return 1.0;  // Outside SAA

        // Gaussian enhancement profile within SAA
        // Peak ~100x based on ISS and other LEO measurements
        return 1.0 + 99.0 * std::exp(-r2 * 3.0);
    }

    /**
     * @brief Check if location is inside SAA boundary
     */
    static bool inside(double latitude_deg, double longitude_deg)
    {
        double dlat = (latitude_deg - CENTER_LAT) / SEMI_MINOR;
        double dlon = normalize_longitude_diff(longitude_deg - CENTER_LON) / SEMI_MAJOR;

        return (dlat * dlat + dlon * dlon) < 1.0;
    }

   private:
    static double normalize_longitude_diff(double dlon)
    {
        while (dlon > 180.0) dlon -= 360.0;
        while (dlon < -180.0) dlon += 360.0;
        return dlon;
    }
};

/**
 * @brief Solar Particle Event (SPE) model
 *
 * Models the enhancement from solar energetic particle events.
 * Based on statistical analysis of historical events from GOES data.
 *
 * Event classification follows NOAA scale:
 * - Minor (S1): >10 MeV flux > 10 pfu
 * - Moderate (S2): > 100 pfu
 * - Strong (S3): > 1000 pfu
 * - Severe (S4): > 10000 pfu
 * - Extreme (S5): > 100000 pfu
 *
 * Where pfu = proton flux unit = protons/(cm²·s·sr)
 */
class SolarParticleEvent {
   public:
    enum class Magnitude {
        MINOR,     // S1: >10 MeV flux > 10 pfu
        MODERATE,  // S2: > 100 pfu
        STRONG,    // S3: > 1000 pfu
        SEVERE,    // S4: > 10000 pfu
        EXTREME    // S5: > 100000 pfu (Carrington-class)
    };

    /**
     * @brief Get SPE differential proton spectrum
     *
     * Uses Band function fit typical of gradual SPE events.
     *
     * @param magnitude Event magnitude
     * @param energy_mev Proton kinetic energy
     * @return Differential flux in protons/(cm²·s·sr·MeV)
     */
    static double differential_spectrum(Magnitude magnitude, double energy_mev)
    {
        // J(>10 MeV) integral flux in pfu for each magnitude class
        double J10 = 0;  // pfu = protons/(cm²·s·sr)
        switch (magnitude) {
            case Magnitude::MINOR:
                J10 = 10.0;
                break;
            case Magnitude::MODERATE:
                J10 = 100.0;
                break;
            case Magnitude::STRONG:
                J10 = 1000.0;
                break;
            case Magnitude::SEVERE:
                J10 = 10000.0;
                break;
            case Magnitude::EXTREME:
                J10 = 100000.0;
                break;
        }

        // Power law spectrum with exponential rolloff
        // dJ/dE = J0 * (E/E0)^(-γ) * exp(-E/Ec)
        // Typical values: γ ~ 1.5, Ec ~ 30-100 MeV
        constexpr double E0 = 10.0;        // Reference energy (MeV)
        constexpr double gamma = 1.5;      // Power law index
        constexpr double E_cutoff = 60.0;  // Exponential cutoff (MeV)

        // Normalize so that integral above 10 MeV equals J10
        // ∫_10^∞ dJ/dE dE = J10
        // Normalization factor computed analytically
        double norm = J10 * (gamma - 1.0) / std::pow(E0, -gamma + 1.0) *
                      (1.0 + (gamma - 1.0) * E0 / E_cutoff);

        double dJdE = norm * std::pow(energy_mev / E0, -gamma) * std::exp(-energy_mev / E_cutoff);

        return dJdE;  // protons/(cm²·s·sr·MeV)
    }

    /**
     * @brief Get integral flux above energy threshold
     *
     * @param magnitude Event magnitude
     * @param energy_threshold_mev Minimum energy
     * @return Integral flux in protons/(cm²·s·sr) = pfu
     */
    static double integral_flux(Magnitude magnitude, double energy_threshold_mev)
    {
        // Numerical integration
        double total = 0.0;
        constexpr double e_max = 500.0;
        constexpr int n_steps = 100;

        double de = (e_max - energy_threshold_mev) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double e = energy_threshold_mev + (i + 0.5) * de;
            total += differential_spectrum(magnitude, e) * de;
        }

        return total;  // pfu
    }

    /**
     * @brief Get estimated event probability per year
     *
     * Based on historical solar cycle data from NOAA.
     *
     * @param magnitude Event magnitude
     * @return Expected events per year (varies with solar cycle)
     */
    static double annual_rate(Magnitude magnitude, bool solar_max = true)
    {
        // Event rates during solar maximum (factor ~10 higher than minimum)
        double rate_max = 0;
        switch (magnitude) {
            case Magnitude::MINOR:
                rate_max = 50.0;
                break;
            case Magnitude::MODERATE:
                rate_max = 10.0;
                break;
            case Magnitude::STRONG:
                rate_max = 3.0;
                break;
            case Magnitude::SEVERE:
                rate_max = 0.5;
                break;
            case Magnitude::EXTREME:
                rate_max = 0.01;
                break;  // ~once per century
        }

        return solar_max ? rate_max : rate_max * 0.1;
    }

    /**
     * @brief Get probability of at least one event during mission
     *
     * @param magnitude Event magnitude
     * @param duration_days Mission duration
     * @param solar_max Solar maximum conditions
     * @return Probability of ≥1 event (0 to 1)
     */
    static double event_probability(Magnitude magnitude, double duration_days,
                                    bool solar_max = true)
    {
        double rate = annual_rate(magnitude, solar_max);
        double duration_years = duration_days / 365.25;
        double lambda = rate * duration_years;

        // Poisson probability of ≥1 event
        return 1.0 - std::exp(-lambda);
    }
};

/**
 * @brief Complete SEU rate calculator
 *
 * Integrates flux models with cross-section to calculate SEU rate.
 * Implements: SEU_rate = ∫ Φ(E) × σ(E) dE
 */
class SEUCalculator {
   public:
    struct Environment {
        double altitude_km;
        double latitude_deg;
        double longitude_deg;
        double solar_modulation_mv;  // 400-1200 MV
        bool solar_maximum;
        bool in_spe;  // During solar particle event
        SolarParticleEvent::Magnitude spe_magnitude;
        double temperature_k = 300.0;  // Device temperature
    };

    /**
     * @brief Calculate total SEU rate from all sources
     *
     * @param env Environment parameters
     * @param heavy_ion_params Weibull parameters for heavy ion SEU
     * @param proton_params Bendel parameters for proton SEU
     * @return SEU rate in errors/bit/day
     */
    static double calculate_seu_rate(const Environment& env,
                                     const WeibullParameters& heavy_ion_params,
                                     const BendelParameters& proton_params)
    {
        double total_rate = 0.0;

        // Temperature correction
        double temp_factor = temperature_factor(env.temperature_k);

        // 1. GCR heavy ion contribution (always present)
        total_rate += gcr_heavy_ion_rate(env, heavy_ion_params) * temp_factor;

        // 2. Trapped particle contribution (Van Allen belts)
        if (env.altitude_km > 200 && env.altitude_km < 60000) {
            // Proton contribution (nuclear reactions)
            total_rate += trapped_proton_rate(env, proton_params) * temp_factor;

            // Electron contribution (significant for small geometry devices)
            if (heavy_ion_params.let_threshold < 1.0) {
                total_rate += trapped_electron_rate(env, heavy_ion_params) * temp_factor;
            }
        }

        // 3. SPE contribution (if active)
        if (env.in_spe) {
            total_rate += spe_proton_rate(env, proton_params) * temp_factor;
        }

        return total_rate;
    }

    /**
     * @brief Simplified interface using default proton parameters
     */
    static double calculate_seu_rate(const Environment& env, const WeibullParameters& device)
    {
        // Derive Bendel parameters from Weibull for consistency
        BendelParameters proton_params;
        if (device.let_threshold < 1.0) {
            proton_params = BendelParameters::cmos_28nm();
        }
        else if (device.let_threshold < 3.0) {
            proton_params = BendelParameters::cmos_65nm();
        }
        else {
            proton_params = BendelParameters::rad_hard();
        }

        return calculate_seu_rate(env, device, proton_params);
    }

    /**
     * @brief Calculate optimal scrubbing interval
     *
     * Based on: T_scrub < t_correction / (SEU_rate × N_bits)
     * Uses safety factor of 0.5 (scrub at 50% of theoretical limit)
     *
     * @param seu_rate_per_bit_per_sec SEU rate in errors/bit/second
     * @param num_bits Total protected bits
     * @param correction_capability Errors correctable before data loss
     * @return Recommended scrub interval in seconds
     */
    static double optimal_scrub_interval(double seu_rate_per_bit_per_sec, size_t num_bits,
                                         int correction_capability)
    {
        if (seu_rate_per_bit_per_sec <= 0 || num_bits == 0) {
            return 3600.0;  // Default 1 hour if no data
        }

        // Expected errors per second across all bits
        double errors_per_sec = seu_rate_per_bit_per_sec * static_cast<double>(num_bits);

        if (errors_per_sec <= 0) {
            return 3600.0;
        }

        // Time until accumulated errors exceed correction capability
        // Apply safety factor of 0.5 (scrub before reaching limit)
        double max_interval = static_cast<double>(correction_capability) / errors_per_sec;

        return max_interval * 0.5;
    }

   private:
    /**
     * @brief GCR heavy ion SEU rate
     *
     * SEU_rate = ∫ (dΦ/dLET) × σ(LET) dLET × 4π × seconds_per_day
     */
    static double gcr_heavy_ion_rate(const Environment& env, const WeibullParameters& device)
    {
        GCRSpectrum gcr(env.solar_modulation_mv);

        // Geomagnetic shielding: calculate minimum LET that penetrates
        double min_energy = GeomagneticField::minimum_proton_energy(env.latitude_deg);
        // Convert to approximate LET cutoff (simplified)
        double let_cutoff = proton_let_in_silicon(min_energy) * 0.5;  // Heavy ions have higher LET

        // Integrate flux × cross-section over LET
        // Units tracking:
        //   flux: particles/(cm²·s·sr·(MeV·cm²/mg))
        //   sigma: cm²/bit
        //   dLET: MeV·cm²/mg
        //   Result: (cm²/bit) × particles/(cm²·s·sr) = particles/(bit·s·sr)
        //   × 4π sr × 86400 s/day = errors/(bit·day)

        double rate = 0.0;
        constexpr int n_steps = 100;
        constexpr double let_min = 0.1;
        constexpr double let_max = 100.0;

        double dlet = (let_max - let_min) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double let = let_min + (i + 0.5) * dlet;

            // Skip if below geomagnetic cutoff
            if (let < let_cutoff && let < device.let_threshold) continue;

            double flux = gcr.differential_flux(let);
            double sigma = weibull_cross_section(let, device);

            rate += flux * sigma * dlet;
        }

        // Convert: multiply by 4π steradians and seconds per day
        return rate * 4.0 * M_PI * constants::SECONDS_PER_DAY;
    }

    /**
     * @brief Trapped proton SEU rate using Bendel model
     */
    static double trapped_proton_rate(const Environment& env, const BendelParameters& proton_params)
    {
        double L = GeomagneticField::calculate_L(env.altitude_km, env.latitude_deg);
        double B_B0 = GeomagneticField::calculate_B_B0(env.altitude_km, env.latitude_deg);

        // SAA enhancement
        double saa_factor =
            SouthAtlanticAnomaly::enhancement_factor(env.latitude_deg, env.longitude_deg);

        // Integrate proton flux × Bendel cross-section over energy
        // Units:
        //   flux: protons/(cm²·s·MeV)
        //   sigma: cm²/bit
        //   dE: MeV
        //   Result: errors/(bit·s) × 86400 = errors/(bit·day)

        double rate = 0.0;
        constexpr int n_steps = 50;
        constexpr double e_min = 10.0;   // MeV (Bendel threshold ~10-20 MeV)
        constexpr double e_max = 400.0;  // MeV

        double de = (e_max - e_min) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double energy = e_min + (i + 0.5) * de;

            double flux =
                TrappedProtonModel::differential_flux(L, B_B0, energy, env.solar_maximum) *
                saa_factor;
            double sigma = bendel_proton_cross_section(energy, proton_params);

            rate += flux * sigma * de;
        }

        return rate * constants::SECONDS_PER_DAY;
    }

    /**
     * @brief Trapped electron SEU rate (for small geometry devices)
     *
     * Electrons can cause SEUs in sub-45nm devices through:
     * - Direct ionization (for very low LET thresholds)
     * - Bremsstrahlung producing secondary particles
     */
    static double trapped_electron_rate(const Environment& env, const WeibullParameters& device)
    {
        double L = GeomagneticField::calculate_L(env.altitude_km, env.latitude_deg);
        double B_B0 = GeomagneticField::calculate_B_B0(env.altitude_km, env.latitude_deg);

        // SAA enhancement (less pronounced for electrons)
        double saa_factor = std::sqrt(
            SouthAtlanticAnomaly::enhancement_factor(env.latitude_deg, env.longitude_deg));

        // Electron SEU cross-section is much lower than heavy ions
        // Use effective LET approach for electrons
        constexpr double electron_effective_let = 0.3;  // MeV·cm²/mg (very low)

        // Only contributes if device has very low LET threshold
        if (device.let_threshold > electron_effective_let) {
            return 0.0;
        }

        double sigma = weibull_cross_section(electron_effective_let, device);

        // Integral electron flux above ~1 MeV
        double flux =
            TrappedElectronModel::integral_flux(L, B_B0, 1.0, env.solar_maximum) * saa_factor;

        // Electron SEU efficiency is low (~1e-3 to 1e-4 compared to heavy ions)
        constexpr double electron_efficiency = 1e-4;

        return flux * sigma * electron_efficiency * constants::SECONDS_PER_DAY;
    }

    /**
     * @brief SPE proton SEU rate contribution
     */
    static double spe_proton_rate(const Environment& env, const BendelParameters& proton_params)
    {
        // Integrate SPE proton spectrum × Bendel cross-section
        double rate = 0.0;
        constexpr int n_steps = 50;
        constexpr double e_min = 10.0;
        constexpr double e_max = 500.0;

        double de = (e_max - e_min) / n_steps;
        for (int i = 0; i < n_steps; ++i) {
            double energy = e_min + (i + 0.5) * de;

            // SPE flux is in protons/(cm²·s·sr·MeV)
            double flux = SolarParticleEvent::differential_spectrum(env.spe_magnitude, energy);
            double sigma = bendel_proton_cross_section(energy, proton_params);

            rate += flux * sigma * de;
        }

        // Convert from per steradian to omnidirectional (4π)
        return rate * 4.0 * M_PI * constants::SECONDS_PER_DAY;
    }
};

/**
 * @brief Complete radiation environment interface
 *
 * High-level interface combining all physics models for practical use.
 */
class PhysicsRadiationEnvironment {
   public:
    struct Config {
        double altitude_km;
        double inclination_deg;
        double solar_modulation_mv;
        bool solar_maximum;
        double temperature_k;
        WeibullParameters heavy_ion_device;
        BendelParameters proton_device;

        // Default constructor with ISS defaults
        Config()
            : altitude_km(400.0),
              inclination_deg(51.6),
              solar_modulation_mv(650.0),
              solar_maximum(false),
              temperature_k(300.0),
              heavy_ion_device(WeibullParameters::cmos_65nm()),
              proton_device(BendelParameters::cmos_65nm())
        {
        }
    };

    explicit PhysicsRadiationEnvironment(const Config& config = Config()) : config_(config) {}

    /**
     * @brief Get SEU rate at current position along orbit
     *
     * @param orbit_phase Orbital phase (0-1, where 0.5 is opposite side)
     * @return SEU rate in errors/bit/day
     */
    double get_seu_rate(double orbit_phase = 0.0) const
    {
        // Calculate current position based on orbit phase
        double latitude = config_.inclination_deg * std::sin(2.0 * M_PI * orbit_phase);
        double longitude = orbit_phase * 360.0 - 180.0;

        SEUCalculator::Environment env;
        env.altitude_km = config_.altitude_km;
        env.latitude_deg = latitude;
        env.longitude_deg = longitude;
        env.solar_modulation_mv = config_.solar_modulation_mv;
        env.solar_maximum = config_.solar_maximum;
        env.temperature_k = config_.temperature_k;
        env.in_spe = false;

        return SEUCalculator::calculate_seu_rate(env, config_.heavy_ion_device,
                                                 config_.proton_device);
    }

    /**
     * @brief Get orbit-averaged SEU rate
     *
     * @param samples Number of samples around orbit
     * @return Average SEU rate in errors/bit/day
     */
    double get_orbit_average_seu_rate(int samples = 100) const
    {
        double total = 0.0;
        for (int i = 0; i < samples; ++i) {
            total += get_seu_rate(static_cast<double>(i) / samples);
        }
        return total / samples;
    }

    /**
     * @brief Get worst-case SEU rate (in SAA)
     *
     * @return Peak SEU rate in errors/bit/day
     */
    double get_worst_case_seu_rate() const
    {
        SEUCalculator::Environment env;
        env.altitude_km = config_.altitude_km;
        env.latitude_deg = SouthAtlanticAnomaly::CENTER_LAT;
        env.longitude_deg = SouthAtlanticAnomaly::CENTER_LON;
        env.solar_modulation_mv = config_.solar_modulation_mv;
        env.solar_maximum = config_.solar_maximum;
        env.temperature_k = config_.temperature_k;
        env.in_spe = false;

        return SEUCalculator::calculate_seu_rate(env, config_.heavy_ion_device,
                                                 config_.proton_device);
    }

    /**
     * @brief Get SEU rate during a solar particle event
     *
     * @param magnitude SPE magnitude
     * @return SEU rate in errors/bit/day during event
     */
    double get_spe_seu_rate(SolarParticleEvent::Magnitude magnitude) const
    {
        SEUCalculator::Environment env;
        env.altitude_km = config_.altitude_km;
        env.latitude_deg = 0.0;  // Equatorial (worst case for SPE outside magnetosphere)
        env.longitude_deg = 0.0;
        env.solar_modulation_mv = config_.solar_modulation_mv;
        env.solar_maximum = true;
        env.temperature_k = config_.temperature_k;
        env.in_spe = true;
        env.spe_magnitude = magnitude;

        return SEUCalculator::calculate_seu_rate(env, config_.heavy_ion_device,
                                                 config_.proton_device);
    }

    /**
     * @brief Calculate recommended scrub interval
     *
     * @param protected_bits Number of protected bits
     * @param ecc_correction_capability Errors correctable (1 for SECDED, 16 for RS(255,223))
     * @return Recommended scrub interval in seconds
     */
    double recommended_scrub_interval(size_t protected_bits, int ecc_correction_capability) const
    {
        // Use orbit-average rate for typical operation
        double seu_per_bit_per_day = get_orbit_average_seu_rate();
        double seu_per_bit_per_sec = seu_per_bit_per_day / constants::SECONDS_PER_DAY;

        double interval = SEUCalculator::optimal_scrub_interval(seu_per_bit_per_sec, protected_bits,
                                                                ecc_correction_capability);

        // Practical bounds: 100ms to 1 hour
        return std::clamp(interval, 0.1, 3600.0);
    }

    /**
     * @brief Calculate scrub interval for worst-case (SAA passage)
     */
    double worst_case_scrub_interval(size_t protected_bits, int ecc_correction_capability) const
    {
        double seu_per_bit_per_day = get_worst_case_seu_rate();
        double seu_per_bit_per_sec = seu_per_bit_per_day / constants::SECONDS_PER_DAY;

        double interval = SEUCalculator::optimal_scrub_interval(seu_per_bit_per_sec, protected_bits,
                                                                ecc_correction_capability);

        return std::clamp(interval, 0.01, 3600.0);  // Allow faster scrubbing in SAA
    }

    /**
     * @brief Calculate orbital period using Kepler's third law
     *
     * @return Orbital period in minutes
     */
    double orbital_period_minutes() const
    {
        // T = 2π * sqrt(a³/μ)
        // μ = 3.986×10^14 m³/s² (Earth gravitational parameter)
        constexpr double MU = 3.986e14;                                          // m³/s²
        double a = (constants::EARTH_RADIUS_KM + config_.altitude_km) * 1000.0;  // meters
        double T_seconds = 2.0 * M_PI * std::sqrt(a * a * a / MU);
        return T_seconds / 60.0;
    }

    /**
     * @brief Calculate ground track position at given time
     *
     * Models the spacecraft's sub-satellite point accounting for:
     * - Orbital motion (sinusoidal latitude variation)
     * - Earth's rotation (westward drift of longitude)
     *
     * @param time_minutes Time since epoch in minutes
     * @param[out] lat_deg Latitude in degrees
     * @param[out] lon_deg Longitude in degrees
     */
    void ground_track_position(double time_minutes, double& lat_deg, double& lon_deg) const
    {
        double T = orbital_period_minutes();

        // Orbital phase (0 to 2π)
        double orbit_phase = 2.0 * M_PI * std::fmod(time_minutes, T) / T;

        // Latitude: sinusoidal with inclination as amplitude
        lat_deg = config_.inclination_deg * std::sin(orbit_phase);

        // Longitude: accounts for Earth rotation (360°/day = 0.25°/min)
        // and orbital motion
        constexpr double EARTH_ROTATION_RATE = 360.0 / (24.0 * 60.0);  // deg/min

        // Initial longitude + orbital contribution - Earth rotation
        double orbit_number = time_minutes / T;
        double lon_orbital = (orbit_phase / (2.0 * M_PI)) * 360.0;       // Where we are in orbit
        double lon_earth_rotation = EARTH_ROTATION_RATE * time_minutes;  // Earth rotated under us

        // Ascending node precession (simplified)
        lon_deg = lon_orbital - lon_earth_rotation;

        // Normalize to -180 to 180
        while (lon_deg > 180.0) lon_deg -= 360.0;
        while (lon_deg < -180.0) lon_deg += 360.0;
    }

    /**
     * @brief Get fraction of time spent in SAA over multiple orbits
     *
     * Simulates ground track over a full day (16 orbits for ISS) to get
     * realistic SAA exposure statistics.
     *
     * @param num_orbits Number of orbits to simulate (default: 16 = ~1 day for ISS)
     * @return Fraction of time in SAA (0-1), typically 10-15% for ISS
     */
    double saa_fraction(int num_orbits = 16) const
    {
        double T = orbital_period_minutes();
        double total_time = num_orbits * T;

        int in_saa = 0;
        constexpr int samples_per_orbit = 360;
        int total_samples = num_orbits * samples_per_orbit;

        for (int i = 0; i < total_samples; ++i) {
            double time_min = (static_cast<double>(i) / total_samples) * total_time;
            double lat, lon;
            ground_track_position(time_min, lat, lon);

            if (SouthAtlanticAnomaly::inside(lat, lon)) {
                ++in_saa;
            }
        }

        return static_cast<double>(in_saa) / total_samples;
    }

    /**
     * @brief Get detailed SAA crossing statistics
     *
     * @param num_orbits Number of orbits to simulate
     * @return Struct with SAA statistics
     */
    struct SAAStatistics {
        double fraction;              ///< Fraction of time in SAA
        int crossings_per_day;        ///< Number of SAA crossings per day
        double avg_crossing_minutes;  ///< Average duration of each crossing
        double peak_enhancement;      ///< Peak flux enhancement experienced
    };

    SAAStatistics get_saa_statistics(int num_orbits = 16) const
    {
        double T = orbital_period_minutes();
        double total_time = num_orbits * T;

        constexpr int samples_per_orbit = 360;
        int total_samples = num_orbits * samples_per_orbit;
        double dt = total_time / total_samples;  // Time per sample in minutes

        int in_saa_count = 0;
        int crossings = 0;
        bool was_in_saa = false;
        double total_crossing_time = 0.0;
        double peak_enhancement = 1.0;

        for (int i = 0; i < total_samples; ++i) {
            double time_min = (static_cast<double>(i) / total_samples) * total_time;
            double lat, lon;
            ground_track_position(time_min, lat, lon);

            bool is_in_saa = SouthAtlanticAnomaly::inside(lat, lon);

            if (is_in_saa) {
                ++in_saa_count;
                total_crossing_time += dt;

                double enhancement = SouthAtlanticAnomaly::enhancement_factor(lat, lon);
                peak_enhancement = std::max(peak_enhancement, enhancement);

                if (!was_in_saa) {
                    ++crossings;  // Entered SAA
                }
            }

            was_in_saa = is_in_saa;
        }

        // Scale crossings to per-day
        double orbits_per_day = 24.0 * 60.0 / T;
        double crossings_per_day = crossings * (orbits_per_day / num_orbits);

        SAAStatistics stats;
        stats.fraction = static_cast<double>(in_saa_count) / total_samples;
        stats.crossings_per_day = static_cast<int>(crossings_per_day + 0.5);
        stats.avg_crossing_minutes = (crossings > 0) ? total_crossing_time / crossings : 0.0;
        stats.peak_enhancement = peak_enhancement;

        return stats;
    }

    /**
     * @brief Get SEU rate uncertainty bounds
     *
     * @return Pair of (lower_bound, upper_bound) for orbit-average SEU rate
     */
    std::pair<double, double> seu_rate_uncertainty() const
    {
        double nominal = get_orbit_average_seu_rate();

        // Combined uncertainty from flux models
        // GCR: ~factor of 2
        // Trapped: ~factor of 2-3
        // Cross-section: ~factor of 2
        // Total: ~factor of 3-5
        constexpr double uncertainty_factor = 4.0;

        return {nominal / uncertainty_factor, nominal * uncertainty_factor};
    }

    const Config& config() const { return config_; }
    Config& config() { return config_; }

   private:
    Config config_;
};

}  // namespace physics
}  // namespace rad_ml

#endif  // RAD_ML_PHYSICS_RADIATION_PHYSICS_HPP
