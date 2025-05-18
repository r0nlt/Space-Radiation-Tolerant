"""
Core functionality for the radiation-tolerant neural networks library.
"""

import warnings

# Try to import C++ bindings
try:
    from ._cpp_binding import (
        initialize as cpp_initialize,
        shutdown as cpp_shutdown,
        apply_radiation as cpp_apply_radiation,
        detect_errors as cpp_detect_errors,
    )

    HAS_CPP = True
except ImportError as e:
    warnings.warn(f"Could not import C++ bindings: {e}. Using Python fallbacks.")
    HAS_CPP = False


def initialize():
    """Initialize the radiation-tolerant neural networks library."""
    if HAS_CPP:
        try:
            return cpp_initialize()
        except Exception as e:
            warnings.warn(f"C++ initialization failed: {e}. Using Python fallback.")

    print("Initializing rad_ml_lib with Python fallback...")
    return True


def shutdown():
    """Shut down the radiation-tolerant neural networks library."""
    if HAS_CPP:
        try:
            return cpp_shutdown()
        except Exception as e:
            warnings.warn(f"C++ shutdown failed: {e}. Using Python fallback.")

    print("Shutting down rad_ml_lib...")
    return True


# Import submodules
from .enums import Environment, DefenseStrategy, ProtectionLevel
from .config import DefenseConfig

__all__ = [
    "initialize",
    "shutdown",
    "Environment",
    "DefenseStrategy",
    "ProtectionLevel",
    "DefenseConfig",
    "HAS_CPP",
]
