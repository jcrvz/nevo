"""
Base Operator Interface
=======================

This module defines the base interface for all optimisation operators in NEVO.
Each operator implements a specific search strategy (exploration or exploitation).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Operator(ABC):
    """
    Abstract base class for optimisation operators.

    Each operator must implement:
    - generate_population(): Create candidate solutions
    - get_parameters(): Return operator-specific parameters

    Operators can be either:
    - EXPLORATION: Global search, diversity promotion
    - EXPLOITATION: Local refinement, convergence
    """

    def __init__(self, name: str, operator_type: str = "exploration"):
        """
        Initialize operator.

        Parameters
        ----------
        name : str
            Unique identifier for this operator
        operator_type : str
            Either "exploration" or "exploitation"
        """
        self.name = name
        self.operator_type = operator_type
        self.usage_count = 0
        self.success_count = 0
        self.total_improvement = 0.0

    @abstractmethod
    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """
        Generate a population of candidate solutions.

        Parameters
        ----------
        centre : np.ndarray
            Current search centre (fitness-weighted centroid)
        state : Dict[str, Any]
            Current optimisation state (memory, best solution, etc.)
        population_size : int
            Number of candidates to generate

        Returns
        -------
        candidates : np.ndarray
            Population of shape (population_size, dimensions)
            All values should be in [-1, 1] (v-space)
        """
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get operator-specific parameters.

        Returns
        -------
        params : Dict[str, Any]
            Dictionary of parameter names and values
        """
        return {
            "name": self.name,
            "type": self.operator_type,
            "usage_count": self.usage_count,
            "success_rate": self.success_count / max(1, self.usage_count),
        }

    def update_statistics(self, improved: bool, improvement: float = 0.0):
        """
        Update operator usage statistics.

        Parameters
        ----------
        improved : bool
            Whether this operator improved the best solution
        improvement : float
            Magnitude of improvement (if any)
        """
        self.usage_count += 1
        if improved:
            self.success_count += 1
            self.total_improvement += improvement

    def reset_statistics(self):
        """Reset operator statistics."""
        self.usage_count = 0
        self.success_count = 0
        self.total_improvement = 0.0


class ExplorationOperator(Operator):
    """Base class for exploration operators (global search)."""

    def __init__(self, name: str):
        super().__init__(name, operator_type="exploration")


class ExploitationOperator(Operator):
    """Base class for exploitation operators (local refinement)."""

    def __init__(self, name: str):
        super().__init__(name, operator_type="exploitation")

