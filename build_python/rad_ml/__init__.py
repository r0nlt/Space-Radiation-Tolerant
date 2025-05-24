"""
Radiation-Tolerant Machine Learning Framework - Python Bindings

This module provides Python bindings for the C++ rad_ml framework,
enabling radiation-tolerant machine learning in Python applications.

Author: Rishab Nuguru
Copyright: © 2025 Rishab Nuguru
License: AGPL v3 license
"""

# Define version as fallback
__version__ = "2.0.0"

try:
    # Try to import core functions
    from ._core import (
        # Core functionality
        initialize,
        shutdown,
        Version,
        # Enums
        MemoryProtectionLevel,
        ProtectionLevel,
        HardeningStrategy,
        RadiationEnvironment,
        MissionType,
        ErrorSeverity,
        # Simulation
        PhysicsRadiationSimulator,
        MissionSimulator,
        FaultInjector,
        # Neural network
        SelectiveHardening,
        ErrorPredictor,
    )

    # Update version from the core if available
    __version__ = f"{Version.major}.{Version.minor}.{Version.patch}"
except ImportError as e:
    # Fallback minimal functionality
    import warnings

    warnings.warn(
        f"Some core functionality could not be imported: {e}. Using fallback implementations."
    )

    # Define minimal fallbacks for critical functions
    def initialize(enable_logging=True):
        print("Using fallback initialize function")
        return True

    def shutdown(check_for_leaks=True):
        print("Using fallback shutdown function")
        return True

    # Fallback enums if needed
    class RadiationEnvironment:
        EARTH_ORBIT = 0
        LEO = 1
        GEO = 2
        LUNAR = 3
        MARS = 4
        JUPITER = 5
        SOLAR_PROBE = 6

    class ErrorSeverity:
        INFO = 0
        WARNING = 1
        ERROR = 2
        CRITICAL = 3


# Import Pythonic TMR classes - these have built-in fallbacks
from .tmr import StandardTMR, EnhancedTMR
