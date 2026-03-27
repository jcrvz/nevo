"""
NEVO - Neuromorphic Evolutionary Optimisation
==============================================

A neuromorphic computing framework for adaptive evolutionary optimisation.

Key Features:
- Basal ganglia-based operator selection
- Population-based parallel evaluation
- Adaptive utility learning
- Loihi-compatible neural architectures
"""

__version__ = "0.1.1"
__author__ = "Jorge Mario Cruz-Duarte"

from nevo.core.optimiser import NEVOptimiser
from nevo.core.state import StateFeatures
from nevo.operators.base import Operator

__all__ = [
    "NEVOptimiser",
    "StateFeatures",
    "Operator",
]
