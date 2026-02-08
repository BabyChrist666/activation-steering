"""
Activation Steering - Control model behavior by modifying activations.

This module provides tools for steering language model behavior by
adding learned steering vectors to intermediate activations during inference.
"""

from activation_steering.vectors import (
    SteeringVector,
    ContrastivePair,
    extract_steering_vector,
    load_steering_vector,
    save_steering_vector,
)
from activation_steering.hooks import (
    ActivationHook,
    SteeringHook,
    CachingHook,
    AblationHook,
)
from activation_steering.steering import (
    ActivationSteerer,
    SteeringConfig,
    SteeringResult,
)
from activation_steering.analysis import (
    ActivationAnalyzer,
    ProjectionAnalysis,
    SteeringEffectiveness,
)

__version__ = "0.1.0"
__all__ = [
    # Vectors
    "SteeringVector",
    "ContrastivePair",
    "extract_steering_vector",
    "load_steering_vector",
    "save_steering_vector",
    # Hooks
    "ActivationHook",
    "SteeringHook",
    "CachingHook",
    "AblationHook",
    # Steering
    "ActivationSteerer",
    "SteeringConfig",
    "SteeringResult",
    # Analysis
    "ActivationAnalyzer",
    "ProjectionAnalysis",
    "SteeringEffectiveness",
]
