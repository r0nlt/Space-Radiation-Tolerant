"""
Space Radiation-Tolerant Neural Networks
=======================================

A comprehensive library for making neural networks resilient to radiation effects in space environments.

This library provides multiple protection strategies for neural network inference and training in
radiation environments, with both software and hardware-accelerated implementations.

Basic usage:
    >>> from rad_ml_lib import protect_network, DefenseConfig, Environment
    >>> protected_model = protect_network(my_pytorch_model, DefenseConfig.for_environment(Environment.JUPITER))
    >>> outputs = protected_model(inputs)  # Run protected inference
"""

__version__ = "0.1.0"

# Import core functionality
try:
    from .core import initialize, shutdown
    from .core.enums import Environment, DefenseStrategy, ProtectionLevel
    from .core.config import DefenseConfig
    from .protection import protect_network, create_protected_network
    from .protection.tmr import TMRProtection
    from .protection.reed_solomon import ReedSolomonProtection
    from .protection.adaptive import AdaptiveProtection
    from .simulation import RadiationSimulator

    # These components form the main public API
    __all__ = [
        "initialize",
        "shutdown",
        "Environment",
        "DefenseStrategy",
        "ProtectionLevel",
        "DefenseConfig",
        "protect_network",
        "create_protected_network",
        "TMRProtection",
        "ReedSolomonProtection",
        "AdaptiveProtection",
        "RadiationSimulator",
    ]
except ImportError as e:
    import warnings

    warnings.warn(f"Error importing components: {e}")
    __all__ = []
