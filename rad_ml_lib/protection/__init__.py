"""
Protection mechanisms for neural networks against radiation effects.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any, Union, Type

from ..core import HAS_CPP
from ..core.enums import DefenseStrategy, ProtectionLevel
from ..core.config import DefenseConfig

# Import protection strategies
from .tmr import TMRProtection

# Try to import other strategies if they exist
try:
    from .reed_solomon import ReedSolomonProtection
except ImportError:
    warnings.warn("Reed-Solomon protection not available")

try:
    from .adaptive import AdaptiveProtection
except ImportError:
    warnings.warn("Adaptive protection not available")


def protect_network(
    model: nn.Module, config: Optional[DefenseConfig] = None
) -> nn.Module:
    """
    Apply radiation protection to a neural network model.

    This function wraps the model with the appropriate protection strategy based on
    the configuration.

    Args:
        model: The PyTorch neural network to protect.
        config: The protection configuration. If None, no protection is applied.

    Returns:
        A protected version of the input model.
    """
    if config is None:
        return model

    # Determine the protection strategy
    if config.strategy == DefenseStrategy.MULTI_LAYERED:
        return TMRProtection(model, config)
    elif config.strategy == DefenseStrategy.REED_SOLOMON:
        if "ReedSolomonProtection" in globals():
            return ReedSolomonProtection(model, config)
        else:
            warnings.warn("Reed-Solomon protection not available, falling back to TMR")
            return TMRProtection(model, config)
    elif config.strategy == DefenseStrategy.ADAPTIVE_PROTECTION:
        if "AdaptiveProtection" in globals():
            return AdaptiveProtection(model, config)
        else:
            warnings.warn("Adaptive protection not available, falling back to TMR")
            return TMRProtection(model, config)
    else:
        # Default to TMR for other strategies
        return TMRProtection(model, config)


def create_protected_network(
    model_class: Type[nn.Module], model_params: Dict[str, Any], config: DefenseConfig
) -> nn.Module:
    """
    Create a new protected network from a model class and parameters.

    Args:
        model_class: The PyTorch model class to instantiate.
        model_params: Parameters to pass to the model constructor.
        config: The protection configuration.

    Returns:
        A protected neural network.
    """
    # Create the base model
    base_model = model_class(**model_params)

    # Apply protection
    return protect_network(base_model, config)
