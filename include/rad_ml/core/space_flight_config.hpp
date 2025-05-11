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
/**
 * @file space_flight_config.hpp
 * @brief Space flight optimization configuration
 * 
 * This file defines configuration settings to optimize the framework
 * for space flight applications, following NASA/ESA standards.
 */

#ifndef RAD_ML_SPACE_FLIGHT_CONFIG_HPP
#define RAD_ML_SPACE_FLIGHT_CONFIG_HPP

#include <cstddef>

namespace rad_ml {
namespace core {

/**
 * Configuration parameters for space flight optimization
 */
struct SpaceFlightConfig {
    // Memory allocation parameters
    static constexpr bool DISABLE_DYNAMIC_ALLOCATION = true;
    static constexpr bool USE_FIXED_SIZE_CONTAINERS = true;
    
    // Error handling options
    static constexpr bool DISABLE_EXCEPTIONS = true;
    static constexpr bool USE_ERROR_CODES = true;
    
    // Execution parameters
    static constexpr bool ENSURE_DETERMINISTIC_EXECUTION = true;
    static constexpr bool DISABLE_RECURSION = true;
    
    // Standard library usage
    static constexpr bool MINIMIZE_STL_USAGE = true;
    
    // Pre-allocation limits for fixed containers
    static constexpr size_t MAX_ENVIRONMENT_CALLBACKS = 16;
    static constexpr size_t MAX_ERROR_HISTORY = 64;
    static constexpr size_t MAX_COMPONENTS = 128;
    static constexpr size_t MAX_LAYERS = 32;
};

} // namespace core
} // namespace rad_ml

#endif // RAD_ML_SPACE_FLIGHT_CONFIG_HPP 
