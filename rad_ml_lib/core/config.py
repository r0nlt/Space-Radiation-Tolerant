"""
Configuration classes for radiation protection settings.
"""

from typing import Dict, Any, Optional
from .enums import Environment, DefenseStrategy, ProtectionLevel


class DefenseConfig:
    """
    Configuration for radiation protection strategies.

    This class defines the protection strategy, level, and parameters used
    to protect neural networks against radiation effects.

    Attributes:
        strategy (DefenseStrategy): The protection strategy to use.
        protection_level (ProtectionLevel): The level of protection to apply.
        environment (Environment): The radiation environment being targeted.
        custom_params (Dict[str, Any]): Custom parameters for the protection.
        protect_gradients (bool): Whether to protect gradients during training.
    """

    def __init__(
        self,
        strategy: DefenseStrategy = DefenseStrategy.MULTI_LAYERED,
        protection_level: ProtectionLevel = ProtectionLevel.FULL_TMR,
        environment: Environment = Environment.LEO,
        custom_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a defense configuration.

        Args:
            strategy: The protection strategy to use.
            protection_level: The level of protection to apply.
            environment: The radiation environment being targeted.
            custom_params: Custom parameters for fine-tuning the protection.
        """
        self.strategy = strategy
        self.protection_level = protection_level
        self.environment = environment
        self.custom_params = custom_params or {}
        self.protect_gradients = False

    @classmethod
    def for_environment(cls, environment: Environment) -> "DefenseConfig":
        """
        Create a defense configuration optimized for a specific environment.

        Args:
            environment: The radiation environment to optimize for.

        Returns:
            A DefenseConfig instance optimized for the given environment.
        """
        if environment == Environment.EARTH:
            return cls(
                strategy=DefenseStrategy.MULTI_LAYERED,
                protection_level=ProtectionLevel.CHECKSUM_ONLY,
                environment=environment,
            )
        elif environment in [Environment.EARTH_ORBIT, Environment.LEO]:
            return cls(
                strategy=DefenseStrategy.MULTI_LAYERED,
                protection_level=ProtectionLevel.SELECTIVE_TMR,
                environment=environment,
            )
        elif environment == Environment.GEO:
            return cls(
                strategy=DefenseStrategy.MULTI_LAYERED,
                protection_level=ProtectionLevel.FULL_TMR,
                environment=environment,
            )
        elif environment in [Environment.LUNAR, Environment.MARS]:
            return cls(
                strategy=DefenseStrategy.PHYSICS_DRIVEN,
                protection_level=ProtectionLevel.FULL_TMR,
                environment=environment,
                custom_params={
                    "particle_tracking": "true",
                    "material_model": "silicon",
                },
            )
        elif environment == Environment.SAA:
            return cls(
                strategy=DefenseStrategy.REED_SOLOMON,
                protection_level=ProtectionLevel.FULL_TMR,
                environment=environment,
                custom_params={
                    "symbol_size": "8",
                    "data_symbols": "8",
                    "total_symbols": "12",
                },
            )
        elif environment == Environment.JUPITER:
            return cls(
                strategy=DefenseStrategy.MULTI_LAYERED,
                protection_level=ProtectionLevel.ADAPTIVE_TMR,
                environment=environment,
                custom_params={
                    "error_correction_threshold": "0.95",
                    "adaptive_adjustment": "true",
                    "criticality_mapping": "true",
                    "quantum_effects": "true",
                },
            )
        elif environment == Environment.SOLAR_STORM:
            return cls(
                strategy=DefenseStrategy.PHYSICS_DRIVEN,
                protection_level=ProtectionLevel.ADAPTIVE_TMR,
                environment=environment,
                custom_params={
                    "error_correction_threshold": "0.99",
                    "interleave_factor": "32",
                    "adaptive_adjustment": "true",
                    "quantum_effects": "true",
                    "transport_equation_mode": "detailed",
                },
            )
        else:
            # Default configuration for unknown environments
            return cls(environment=environment)

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"DefenseConfig(strategy={self.strategy.name}, "
            f"protection_level={self.protection_level.name}, "
            f"environment={self.environment.name}, "
            f"custom_params={self.custom_params})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            "strategy": self.strategy.value,
            "protection_level": self.protection_level.value,
            "environment": self.environment.value,
            "custom_params": self.custom_params,
            "protect_gradients": self.protect_gradients,
        }
