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