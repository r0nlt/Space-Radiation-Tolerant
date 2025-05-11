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

#include <rad_ml/core/redundancy/tmr.hpp>
#include <rad_ml/math/fixed_point.hpp>
#include <array>
#include <cassert>

namespace rad_ml {
namespace inference {
namespace layers {

using namespace rad_ml::core::redundancy;
using namespace rad_ml::math;

/**
 * @brief A radiation-hardened convolutional layer for neural networks
 * 
 * This layer applies a set of learnable filters to the input, producing
 * an activation map that shows how the filter responds at each spatial
 * position. This implementation uses Triple Modular Redundancy (TMR)
 * for the weights to protect against radiation-induced bit flips.
 * 
 * @tparam T The fixed-point type used for calculations
 * @tparam InputChannels Number of input channels
 * @tparam OutputChannels Number of output channels
 * @tparam InputHeight Height of the input image
 * @tparam InputWidth Width of the input image
 * @tparam KernelSize Size of the convolution kernel (assumed square)
 * @tparam Stride Stride of the convolution operation
 * @tparam Padding Padding size
 */
template <
    typename T, 
    size_t InputChannels, 
    size_t OutputChannels,
    size_t InputHeight,
    size_t InputWidth,
    size_t KernelSize,
    size_t Stride = 1,
    size_t Padding = 0
>
class ConvolutionalLayer {
public:
    // Calculate output dimensions
    static constexpr size_t OutputHeight = (InputHeight - KernelSize + 2 * Padding) / Stride + 1;
    static constexpr size_t OutputWidth = (InputWidth - KernelSize + 2 * Padding) / Stride + 1;
    
    /**
     * @brief Constructor
     */
    ConvolutionalLayer() {
        // Initialize weights and biases to zero
        for (size_t oc = 0; oc < OutputChannels; ++oc) {
            for (size_t ic = 0; ic < InputChannels; ++ic) {
                for (size_t kh = 0; kh < KernelSize; ++kh) {
                    for (size_t kw = 0; kw < KernelSize; ++kw) {
                        weights_[oc][ic][kh][kw] = T(0);
                    }
                }
            }
            biases_[oc] = T(0);
        }
    }
    
    /**
     * @brief Set the weights of the convolutional layer
     * 
     * @param oc Output channel index
     * @param ic Input channel index
     * @param kh Kernel height index
     * @param kw Kernel width index
     * @param value The weight value to set
     */
    void setWeight(size_t oc, size_t ic, size_t kh, size_t kw, T value) {
        assert(oc < OutputChannels);
        assert(ic < InputChannels);
        assert(kh < KernelSize);
        assert(kw < KernelSize);
        
        weights_[oc][ic][kh][kw] = value;
    }
    
    /**
     * @brief Set the bias for a specific output channel
     * 
     * @param oc Output channel index
     * @param value The bias value to set
     */
    void setBias(size_t oc, T value) {
        assert(oc < OutputChannels);
        
        biases_[oc] = value;
    }
    
    /**
     * @brief Forward pass through the convolutional layer
     * 
     * @param input Input tensor [InputChannels][InputHeight][InputWidth]
     * @param output Output tensor [OutputChannels][OutputHeight][OutputWidth]
     */
    void forward(
        const T input[InputChannels][InputHeight][InputWidth],
        T output[OutputChannels][OutputHeight][OutputWidth]
    ) {
        // Initialize output to zeros
        for (size_t oc = 0; oc < OutputChannels; ++oc) {
            for (size_t oh = 0; oh < OutputHeight; ++oh) {
                for (size_t ow = 0; ow < OutputWidth; ++ow) {
                    output[oc][oh][ow] = T(0);
                }
            }
        }
        
        // Perform convolution
        for (size_t oc = 0; oc < OutputChannels; ++oc) {
            for (size_t oh = 0; oh < OutputHeight; ++oh) {
                for (size_t ow = 0; ow < OutputWidth; ++ow) {
                    // Calculate input position based on output position
                    size_t ih_start = oh * Stride;
                    size_t iw_start = ow * Stride;
                    
                    // Accumulate weighted values
                    T sum = biases_[oc].get();
                    
                    for (size_t ic = 0; ic < InputChannels; ++ic) {
                        for (size_t kh = 0; kh < KernelSize; ++kh) {
                            for (size_t kw = 0; kw < KernelSize; ++kw) {
                                // Check padding boundaries
                                int ih = static_cast<int>(ih_start) + static_cast<int>(kh) - static_cast<int>(Padding);
                                int iw = static_cast<int>(iw_start) + static_cast<int>(kw) - static_cast<int>(Padding);
                                
                                if (ih >= 0 && ih < static_cast<int>(InputHeight) && 
                                    iw >= 0 && iw < static_cast<int>(InputWidth)) {
                                    // Get the weight and input value
                                    T weight = weights_[oc][ic][kh][kw].get();
                                    T value = input[ic][ih][iw];
                                    
                                    // Accumulate
                                    sum += weight * value;
                                }
                            }
                        }
                    }
                    
                    // Store the result
                    output[oc][oh][ow] = sum;
                }
            }
        }
    }
    
    /**
     * @brief Repair any corrupted weights
     * 
     * This should be called periodically to correct any bit flips caused by radiation
     */
    void repair() {
        for (size_t oc = 0; oc < OutputChannels; ++oc) {
            for (size_t ic = 0; ic < InputChannels; ++ic) {
                for (size_t kh = 0; kh < KernelSize; ++kh) {
                    for (size_t kw = 0; kw < KernelSize; ++kw) {
                        weights_[oc][ic][kh][kw].repair();
                    }
                }
            }
            biases_[oc].repair();
        }
    }
    
private:
    // Weights: [OutputChannels][InputChannels][KernelSize][KernelSize]
    TMR<T> weights_[OutputChannels][InputChannels][KernelSize][KernelSize];
    
    // Biases: [OutputChannels]
    TMR<T> biases_[OutputChannels];
};

} // namespace layers
} // namespace inference
} // namespace rad_ml 
