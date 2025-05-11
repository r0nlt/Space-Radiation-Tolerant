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
#include <vector>
#include <memory>

#include "../common/types.hpp"

namespace rad_ml {
namespace neural {

/**
 * @brief Base class for neural network models
 */
class NetworkModel {
public:
    virtual ~NetworkModel() = default;
    
    /**
     * @brief Get the name of the network
     * 
     * @return Network name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the number of layers in the network
     * 
     * @return Layer count
     */
    virtual size_t getLayerCount() const = 0;
    
    /**
     * @brief Get the input size of the network
     * 
     * @return Input size
     */
    virtual size_t getInputSize() const = 0;
    
    /**
     * @brief Get the output size of the network
     * 
     * @return Output size
     */
    virtual size_t getOutputSize() const = 0;
    
    /**
     * @brief Forward pass through the network
     * 
     * @param input Input tensor
     * @return Output tensor
     */
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    
    /**
     * @brief Apply protection to the network based on its criticality
     * 
     * @param criticality_threshold Threshold for protection (0-1)
     * @return True if protection was successfully applied
     */
    virtual bool applyProtection(float criticality_threshold = 0.5f) = 0;
};

} // namespace neural
} // namespace rad_ml 
