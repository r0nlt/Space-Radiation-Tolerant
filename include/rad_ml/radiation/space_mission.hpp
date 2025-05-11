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

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>
#include <chrono>

#include "../common/types.hpp"
#include "environment.hpp"

namespace rad_ml {
namespace radiation {

/**
 * @brief Space mission phase type
 */
enum class MissionPhaseType {
    LAUNCH,                 ///< Launch phase
    EARTH_ORBIT,            ///< Earth orbital operations
    TRANSFER_TRAJECTORY,    ///< Trajectory between bodies
    PLANETARY_ORBIT,        ///< Orbit around another planet
    PLANETARY_SURFACE,      ///< Operations on a planetary surface
    SOLAR_ENCOUNTER,        ///< Solar proximity operations
    ASTEROID_ENCOUNTER,     ///< Asteroid/comet proximity
    RETURN_TRAJECTORY,      ///< Return trajectory to Earth
    REENTRY                 ///< Atmospheric reentry
};

/**
 * @brief Common space mission targets
 */
enum class MissionTarget {
    EARTH_LEO,              ///< Earth Low Earth Orbit
    EARTH_MEO,              ///< Earth Medium Earth Orbit
    EARTH_GEO,              ///< Earth Geostationary Orbit
    MOON,                   ///< Lunar missions
    MARS,                   ///< Mars missions
    VENUS,                  ///< Venus missions
    MERCURY,                ///< Mercury missions
    JUPITER,                ///< Jupiter missions
    SATURN,                 ///< Saturn missions
    ASTEROID_BELT,          ///< Asteroid Belt
    KUIPER_BELT,            ///< Kuiper Belt
    SOLAR_POLAR,            ///< Solar Polar
    SOLAR_PROBE             ///< Solar proximity probe
};

/**
 * @brief Mission phase with time and environment information
 */
struct MissionPhase {
    std::string name;                   ///< Phase name
    MissionPhaseType type;              ///< Phase type
    std::shared_ptr<Environment> environment; ///< Radiation environment
    std::chrono::seconds duration;      ///< Phase duration
    double distance_au;                 ///< Distance from Sun in AU
    double shielding_thickness_mm;      ///< Effective shielding in mm Al equivalent
    
    /**
     * @brief Constructor with phase parameters
     */
    MissionPhase(
        const std::string& name,
        MissionPhaseType type,
        std::shared_ptr<Environment> environment,
        std::chrono::seconds duration,
        double distance_au,
        double shielding_thickness_mm)
        : name(name)
        , type(type)
        , environment(environment)
        , duration(duration)
        , distance_au(distance_au)
        , shielding_thickness_mm(shielding_thickness_mm)
    {}
};

/**
 * @brief Space mission profile with radiation environments
 */
class SpaceMission {
public:
    /**
     * @brief Constructor with mission name and target
     * 
     * @param name Mission name
     * @param target Mission target
     */
    SpaceMission(
        const std::string& name, 
        MissionTarget target)
        : name_(name)
        , target_(target)
    {}
    
    /**
     * @brief Get mission name
     * 
     * @return Mission name
     */
    const std::string& getName() const {
        return name_;
    }
    
    /**
     * @brief Get mission target
     * 
     * @return Mission target
     */
    MissionTarget getTarget() const {
        return target_;
    }
    
    /**
     * @brief Add a mission phase
     * 
     * @param phase Mission phase to add
     * @return Reference to this mission for chaining
     */
    SpaceMission& addPhase(const MissionPhase& phase) {
        phases_.push_back(phase);
        return *this;
    }
    
    /**
     * @brief Get all mission phases
     * 
     * @return Vector of mission phases
     */
    const std::vector<MissionPhase>& getPhases() const {
        return phases_;
    }
    
    /**
     * @brief Get the current phase environment
     * 
     * @param mission_time Time since mission start
     * @return Environment for the current phase
     */
    std::shared_ptr<Environment> getEnvironmentAtTime(std::chrono::seconds mission_time) const {
        std::chrono::seconds elapsed(0);
        
        for (const auto& phase : phases_) {
            elapsed += phase.duration;
            if (mission_time < elapsed) {
                return phase.environment;
            }
        }
        
        // If past all phases, return the last phase environment
        return phases_.empty() ? nullptr : phases_.back().environment;
    }
    
    /**
     * @brief Get the worst-case environment for the mission
     * 
     * @return Environment with highest radiation levels
     */
    std::shared_ptr<Environment> getWorstCaseEnvironment() const {
        if (phases_.empty()) {
            return nullptr;
        }
        
        // Find phase with highest SEU flux
        auto worst_phase = std::max_element(phases_.begin(), phases_.end(),
            [](const MissionPhase& a, const MissionPhase& b) {
                if (!a.environment || !b.environment) {
                    return !a.environment;
                }
                return a.environment->getSEUFlux() < b.environment->getSEUFlux();
            });
        
        return worst_phase->environment;
    }
    
    /**
     * @brief Calculate total mission radiation exposure
     * 
     * @return Integrated flux-time product
     */
    double calculateTotalRadiationExposure() const {
        double total_exposure = 0.0;
        
        for (const auto& phase : phases_) {
            if (phase.environment) {
                // Exposure = flux * time
                total_exposure += phase.environment->getSEUFlux() * 
                                 phase.duration.count();
            }
        }
        
        return total_exposure;
    }
    
    /**
     * @brief Create a standard mission profile
     * 
     * @param mission_type Type of mission to create
     * @param mission_name Optional mission name (defaults to a standard name)
     * @return Shared pointer to configured mission
     */
    static std::shared_ptr<SpaceMission> createStandardMission(
        MissionTarget mission_type,
        const std::string& mission_name = "")
    {
        // Default name based on mission type
        std::string name = mission_name;
        if (name.empty()) {
            name = getDefaultMissionName(mission_type);
        }
        
        auto mission = std::make_shared<SpaceMission>(name, mission_type);
        
        // Configure phases based on mission type
        switch (mission_type) {
            case MissionTarget::EARTH_LEO:
                configureLEOMission(*mission);
                break;
                
            case MissionTarget::EARTH_GEO:
                configureGEOMission(*mission);
                break;
                
            case MissionTarget::MOON:
                configureLunarMission(*mission);
                break;
                
            case MissionTarget::MARS:
                configureMarsMission(*mission);
                break;
                
            case MissionTarget::JUPITER:
                configureJupiterMission(*mission);
                break;
                
            case MissionTarget::SOLAR_PROBE:
                configureSolarProbeMission(*mission);
                break;
                
            default:
                // Default to LEO if not implemented
                configureLEOMission(*mission);
                break;
        }
        
        return mission;
    }

private:
    std::string name_;
    MissionTarget target_;
    std::vector<MissionPhase> phases_;
    
    /**
     * @brief Get default mission name based on target
     * 
     * @param target Mission target
     * @return Default name for the mission
     */
    static std::string getDefaultMissionName(MissionTarget target) {
        switch (target) {
            case MissionTarget::EARTH_LEO: return "Low Earth Orbit Mission";
            case MissionTarget::EARTH_MEO: return "Medium Earth Orbit Mission";
            case MissionTarget::EARTH_GEO: return "Geostationary Orbit Mission";
            case MissionTarget::MOON: return "Lunar Mission";
            case MissionTarget::MARS: return "Mars Mission";
            case MissionTarget::VENUS: return "Venus Mission";
            case MissionTarget::MERCURY: return "Mercury Mission";
            case MissionTarget::JUPITER: return "Jupiter Mission";
            case MissionTarget::SATURN: return "Saturn Mission";
            case MissionTarget::ASTEROID_BELT: return "Asteroid Belt Mission";
            case MissionTarget::KUIPER_BELT: return "Kuiper Belt Mission";
            case MissionTarget::SOLAR_POLAR: return "Solar Polar Mission";
            case MissionTarget::SOLAR_PROBE: return "Solar Probe Mission";
            default: return "Space Mission";
        }
    }
    
    /**
     * @brief Configure a standard Low Earth Orbit mission
     * 
     * @param mission Mission to configure
     */
    static void configureLEOMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // LEO operations
        mission.addPhase(MissionPhase(
            "LEO Operations",
            MissionPhaseType::EARTH_ORBIT,
            Environment::createEnvironment(EnvironmentType::LOW_EARTH_ORBIT),
            std::chrono::hours(8760), // 1 year
            1.0, // Earth distance
            5.0  // Spacecraft shielding
        ));
    }
    
    /**
     * @brief Configure a standard Geostationary Orbit mission
     * 
     * @param mission Mission to configure
     */
    static void configureGEOMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // LEO parking orbit
        mission.addPhase(MissionPhase(
            "LEO Parking Orbit",
            MissionPhaseType::EARTH_ORBIT,
            Environment::createEnvironment(EnvironmentType::LOW_EARTH_ORBIT),
            std::chrono::hours(5),
            1.0, // Earth distance
            5.0  // Spacecraft shielding
        ));
        
        // GTO transfer
        auto gto_env = Environment::createEnvironment(EnvironmentType::MEDIUM_EARTH_ORBIT);
        gto_env->setSEUFlux(3e-7f); // Higher than LEO, lower than GEO
        mission.addPhase(MissionPhase(
            "GTO Transfer",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            gto_env,
            std::chrono::hours(5),
            1.0, // Earth distance
            5.0  // Spacecraft shielding
        ));
        
        // GEO operations
        mission.addPhase(MissionPhase(
            "GEO Operations",
            MissionPhaseType::EARTH_ORBIT,
            Environment::createEnvironment(EnvironmentType::GEOSTATIONARY_ORBIT),
            std::chrono::hours(8760 * 15), // 15 years
            1.0, // Earth distance
            5.0  // Spacecraft shielding
        ));
    }
    
    /**
     * @brief Configure a standard Lunar mission
     * 
     * @param mission Mission to configure
     */
    static void configureLunarMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // Earth-Moon transfer
        auto trans_env = Environment::createEnvironment(EnvironmentType::MEDIUM_EARTH_ORBIT);
        trans_env->setSEUFlux(2e-7f); // Adjusted for trans-lunar trajectory
        mission.addPhase(MissionPhase(
            "Earth-Moon Transfer",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            trans_env,
            std::chrono::hours(72),
            1.0, // Earth distance initially
            5.0  // Spacecraft shielding
        ));
        
        // Lunar orbit
        auto lunar_orbit_env = Environment::createEnvironment(EnvironmentType::LUNAR);
        lunar_orbit_env->setSEUFlux(2.5e-7f); // Higher than surface due to less shielding
        mission.addPhase(MissionPhase(
            "Lunar Orbit",
            MissionPhaseType::PLANETARY_ORBIT,
            lunar_orbit_env,
            std::chrono::hours(120),
            1.0, // Earth-Moon distance
            5.0  // Spacecraft shielding
        ));
        
        // Lunar surface (if applicable)
        mission.addPhase(MissionPhase(
            "Lunar Surface Operations",
            MissionPhaseType::PLANETARY_SURFACE,
            Environment::createEnvironment(EnvironmentType::LUNAR),
            std::chrono::hours(48),
            1.0, // Earth-Moon distance
            2.0  // Surface habitat/suit shielding
        ));
        
        // Return trajectory
        mission.addPhase(MissionPhase(
            "Moon-Earth Transfer",
            MissionPhaseType::RETURN_TRAJECTORY,
            trans_env,
            std::chrono::hours(72),
            1.0, // Returning to Earth
            5.0  // Spacecraft shielding
        ));
        
        // Reentry
        auto reentry_env = Environment::createEnvironment(EnvironmentType::AVIONIC);
        reentry_env->setSEUFlux(1e-9f); // Lower during reentry due to atmosphere
        mission.addPhase(MissionPhase(
            "Reentry",
            MissionPhaseType::REENTRY,
            reentry_env,
            std::chrono::minutes(30),
            1.0, // Earth
            15.0 // Reentry capsule shielding
        ));
    }
    
    /**
     * @brief Configure a standard Mars mission
     * 
     * @param mission Mission to configure
     */
    static void configureMarsMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // Earth-Mars transfer
        auto trans_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "Interplanetary Transfer");
        trans_env->setSEUFlux(5e-7f); // Deep space, higher radiation
        mission.addPhase(MissionPhase(
            "Earth-Mars Transfer",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            trans_env,
            std::chrono::hours(24 * 180), // ~6 months
            1.5, // Average Earth-Mars transfer distance
            10.0 // Deeper space shielding
        ));
        
        // Mars orbit
        auto mars_orbit_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "Mars Orbit");
        mars_orbit_env->setSEUFlux(2e-7f); // Less than transfer, more than surface
        mission.addPhase(MissionPhase(
            "Mars Orbit",
            MissionPhaseType::PLANETARY_ORBIT,
            mars_orbit_env,
            std::chrono::hours(24 * 60), // 60 days
            1.5, // Mars distance from Sun
            5.0  // Spacecraft shielding
        ));
        
        // Mars surface
        mission.addPhase(MissionPhase(
            "Mars Surface Operations",
            MissionPhaseType::PLANETARY_SURFACE,
            Environment::createEnvironment(EnvironmentType::MARS),
            std::chrono::hours(24 * 500), // ~500 days
            1.5, // Mars distance from Sun
            2.0  // Surface habitat shielding
        ));
        
        // Mars-Earth transfer
        mission.addPhase(MissionPhase(
            "Mars-Earth Transfer",
            MissionPhaseType::RETURN_TRAJECTORY,
            trans_env,
            std::chrono::hours(24 * 180), // ~6 months
            1.5, // Average Mars-Earth transfer distance
            10.0 // Deeper space shielding
        ));
        
        // Reentry
        auto reentry_env = Environment::createEnvironment(EnvironmentType::AVIONIC);
        reentry_env->setSEUFlux(1e-9f); // Lower during reentry due to atmosphere
        mission.addPhase(MissionPhase(
            "Reentry",
            MissionPhaseType::REENTRY,
            reentry_env,
            std::chrono::minutes(30),
            1.0, // Earth
            15.0 // Reentry capsule shielding
        ));
    }
    
    /**
     * @brief Configure a standard Jupiter mission
     * 
     * @param mission Mission to configure
     */
    static void configureJupiterMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // Earth-Jupiter transfer
        auto trans_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "Deep Space Transfer");
        trans_env->setSEUFlux(3e-7f); // Deep space
        mission.addPhase(MissionPhase(
            "Earth-Jupiter Transfer",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            trans_env,
            std::chrono::hours(24 * 365 * 2), // ~2 years
            3.0, // Average distance during transfer
            15.0 // Heavy shielding for deep space
        ));
        
        // Jupiter approach
        auto approach_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "Jupiter Approach");
        approach_env->setSEUFlux(2e-6f); // Increasing radiation
        mission.addPhase(MissionPhase(
            "Jupiter Approach",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            approach_env,
            std::chrono::hours(24 * 30), // 30 days
            5.2, // Jupiter distance from Sun
            15.0 // Heavy shielding
        ));
        
        // Jupiter orbit
        mission.addPhase(MissionPhase(
            "Jupiter Orbit",
            MissionPhaseType::PLANETARY_ORBIT,
            Environment::createEnvironment(EnvironmentType::JUPITER),
            std::chrono::hours(24 * 60), // 60 days
            5.2, // Jupiter distance from Sun
            20.0 // Maximum shielding in radiation belts
        ));
        
        // Extended mission phase (optional)
        auto extended_env = std::make_shared<Environment>(*Environment::createEnvironment(EnvironmentType::JUPITER));
        extended_env->setSEUFlux(extended_env->getSEUFlux() * 0.7f); // Lower orbit, reduced radiation
        mission.addPhase(MissionPhase(
            "Extended Mission",
            MissionPhaseType::PLANETARY_ORBIT,
            extended_env,
            std::chrono::hours(24 * 365), // 1 year
            5.2, // Jupiter distance from Sun
            20.0 // Maximum shielding
        ));
    }
    
    /**
     * @brief Configure a standard Solar Probe mission
     * 
     * @param mission Mission to configure
     */
    static void configureSolarProbeMission(SpaceMission& mission) {
        // Launch phase
        mission.addPhase(MissionPhase(
            "Launch",
            MissionPhaseType::LAUNCH,
            Environment::createEnvironment(EnvironmentType::AVIONIC),
            std::chrono::hours(1),
            1.0, // Earth distance
            10.0 // Launch vehicle shielding
        ));
        
        // Initial orbit
        auto initial_env = Environment::createEnvironment(EnvironmentType::MEDIUM_EARTH_ORBIT);
        mission.addPhase(MissionPhase(
            "Initial Orbit",
            MissionPhaseType::EARTH_ORBIT,
            initial_env,
            std::chrono::hours(48),
            1.0, // Earth distance from Sun
            10.0 // Spacecraft shielding
        ));
        
        // Venus gravity assist
        auto venus_env = std::make_shared<Environment>(EnvironmentType::CUSTOM, "Venus Flyby");
        venus_env->setSEUFlux(8e-7f); // Higher solar radiation
        mission.addPhase(MissionPhase(
            "Venus Gravity Assist",
            MissionPhaseType::PLANETARY_ORBIT,
            venus_env,
            std::chrono::hours(24),
            0.7, // Venus distance from Sun
            15.0 // Increased shielding
        ));
        
        // Solar approach
        auto approach_env = std::make_shared<Environment>(*Environment::createEnvironment(EnvironmentType::SOLAR_FLARE));
        approach_env->setSEUFlux(approach_env->getSEUFlux() * 0.5f); // Not quite solar flare level
        mission.addPhase(MissionPhase(
            "Solar Approach",
            MissionPhaseType::SOLAR_ENCOUNTER,
            approach_env,
            std::chrono::hours(24 * 7), // 1 week
            0.3, // Close solar approach
            30.0 // Maximum heat shield and radiation protection
        ));
        
        // Perihelion
        auto perihelion_env = Environment::createEnvironment(EnvironmentType::SOLAR_FLARE);
        mission.addPhase(MissionPhase(
            "Perihelion",
            MissionPhaseType::SOLAR_ENCOUNTER,
            perihelion_env,
            std::chrono::hours(24), // 1 day
            0.1, // Extremely close to Sun
            35.0 // Maximum possible shielding
        ));
        
        // Return to higher orbit
        auto return_env = std::make_shared<Environment>(*approach_env);
        mission.addPhase(MissionPhase(
            "Return to Higher Orbit",
            MissionPhaseType::TRANSFER_TRAJECTORY,
            return_env,
            std::chrono::hours(24 * 7), // 1 week
            0.3, // Moving away from Sun
            30.0 // Still high shielding
        ));
        
        // Repeat cycle
        // Additional phases would be added for a multi-cycle mission
    }
};

} // namespace radiation
} // namespace rad_ml 
