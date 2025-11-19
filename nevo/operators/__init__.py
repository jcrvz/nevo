"""
Operator Registry
=================

Central registry for all available operators.
"""

from typing import Dict, Type
from nevo.operators.base import Operator
from nevo.operators.standard import (
    LevyFlight,
    DifferentialEvolution,
    ParticleSwarm,
    SpiralOptimisation,
)


# Registry of available operators
OPERATOR_REGISTRY: Dict[str, Type[Operator]] = {
    "LevyFlight": LevyFlight,
    "DifferentialEvolution": DifferentialEvolution,
    "ParticleSwarm": ParticleSwarm,
    "SpiralOptimisation": SpiralOptimisation,
}


def get_operator(name: str, **kwargs) -> Operator:
    """
    Get an operator instance by name.

    Parameters
    ----------
    name : str
        Operator name (must be in OPERATOR_REGISTRY)
    **kwargs
        Operator-specific parameters

    Returns
    -------
    operator : Operator
        Instantiated operator
    """
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator: {name}. Available: {list(OPERATOR_REGISTRY.keys())}")

    return OPERATOR_REGISTRY[name](**kwargs)


def list_operators() -> Dict[str, str]:
    """
    List all available operators with descriptions.

    Returns
    -------
    operators : Dict[str, str]
        Dictionary mapping operator names to their docstring summaries
    """
    result = {}
    for name, op_class in OPERATOR_REGISTRY.items():
        # Get first line of docstring
        doc = op_class.__doc__
        if doc:
            summary = doc.strip().split('\n')[0]
            result[name] = summary
        else:
            result[name] = "No description available"

    return result

