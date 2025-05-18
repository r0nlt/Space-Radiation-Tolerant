"""
Enums for the radiation-tolerant neural networks library.
"""

from enum import Enum, auto


class Environment(Enum):
    """Space radiation environments with different radiation profiles."""

    EARTH = 0
    EARTH_ORBIT = 1
    LEO = 2  # Low Earth Orbit
    GEO = 3  # Geostationary Earth Orbit
    LUNAR = 4  # Lunar environment
    MARS = 5  # Mars environment
    SAA = 6  # South Atlantic Anomaly
    JUPITER = 7  # Jupiter environment
    SOLAR_STORM = 8  # Solar storm conditions


class DefenseStrategy(Enum):
    """Protection strategies against radiation effects."""

    MULTI_LAYERED = 0  # Combined strategies for comprehensive protection
    REED_SOLOMON = 1  # Error correction coding using Reed-Solomon
    PHYSICS_DRIVEN = 2  # Model based on physics of radiation effects
    ADAPTIVE_PROTECTION = 3  # Adapting protection based on conditions
    HARDWARE_ACCELERATED = 4  # Hardware acceleration for protection


class ProtectionLevel(Enum):
    """Level of protection to apply to neural networks."""

    NONE = 0  # No protection
    CHECKSUM_ONLY = 1  # Only use checksums for error detection
    SELECTIVE_TMR = 2  # Selective Triple Modular Redundancy (critical parts only)
    FULL_TMR = 3  # Full Triple Modular Redundancy
    ADAPTIVE_TMR = 4  # Adaptive TMR that varies protection
    SPACE_OPTIMIZED = 5  # Space-optimized protection (memory efficient)


class ErrorType(Enum):
    """Types of radiation-induced errors."""

    SINGLE_BIT_FLIP = auto()  # Single bit upset
    MULTI_BIT_UPSET = auto()  # Multiple bit upset in same word
    BLOCK_CORRUPTION = auto()  # Corruption of memory block
    INSTRUCTION_ERROR = auto()  # Error in instruction execution
    TIMING_ERROR = auto()  # Timing error (delay or glitch)


class DetectionResult(Enum):
    """Result of error detection."""

    NO_ERROR = 0  # No error detected
    ERROR_DETECTED = 1  # Error detected
    ERROR_DETECTED_CORRECTED = 2  # Error detected and corrected
    ERROR_DETECTED_UNCORRECTABLE = 3  # Error detected but could not be corrected
