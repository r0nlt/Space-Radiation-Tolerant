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

#include <cstdint>
#include <string>
#include <vector>

namespace rad_ml {

// Common data types for the rad_ml framework

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using float32 = float;
using float64 = double;

// Forward declarations for common types
namespace memory {
    class ProtectedMemoryManager;
    class MemoryScrubber;
}

namespace tmr {
    class TMRBase;
    class ApproximateTMR;
    class HealthWeightedTMR;
    class EnhancedTMR;
}

namespace neural {
    class NetworkModel;
    class SelectiveHardening;
    class LayerProtectionPolicy;
    class TopologicalAnalyzer;
    class GradientImportanceMapper;
}

namespace radiation {
    class SEUSimulator;
    class Environment;
}

namespace error {
    class ErrorHandler;
}

} // namespace rad_ml 
