"""
State Feature Extraction
=========================

This module computes state features that describe the current
optimization landscape and guide operator selection.
"""

import numpy as np
from typing import Dict, Any, List


class StateFeatures:
    """
    Extracts interpretable features from optimization state.

    Features:
    - diversity: Spread of solutions in search space [0, 1]
    - improvement_rate: Fraction of recent improvements [0, 1]
    - convergence: Fitness homogeneity indicator [0, 1]
    """

    def __init__(self, history_length: int = 50):
        """
        Parameters
        ----------
        history_length : int
            Number of recent timesteps to consider for improvement rate
        """
        self.history_length = history_length
        self.improvement_history: List[float] = []

    def compute(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compute 3D feature vector from current state.

        Parameters
        ----------
        state : Dict[str, Any]
            Current optimization state containing:
            - memory_vectors: np.ndarray of shape (MU, D)
            - memory_fitness: np.ndarray of shape (MU,)
            - f_default_worst: float (sentinel value for invalid solutions)

        Returns
        -------
        features : np.ndarray
            Feature vector [diversity, improvement_rate, convergence]
        """
        features = {
            "diversity": 0.5,
            "improvement_rate": 0.5,
            "convergence": 0.0,
        }

        # Extract state components
        memory_fitness = state.get("memory_fitness", np.array([]))
        memory_vectors = state.get("memory_vectors", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        # Filter valid solutions
        valid_mask = memory_fitness < f_default_worst
        n_valid = np.sum(valid_mask)

        if n_valid < 2:
            return np.array([
                features["diversity"],
                features["improvement_rate"],
                features["convergence"]
            ])

        valid_fitness = memory_fitness[valid_mask]
        valid_vectors = memory_vectors[valid_mask]

        # 1. DIVERSITY: Spread in search space
        # Use average standard deviation across dimensions
        std_per_dim = np.std(valid_vectors, axis=0)
        avg_std = np.mean(std_per_dim)
        # In [-1,1] space, std ranges from 0 (identical) to ~0.577 (uniform)
        diversity = np.clip(avg_std / 0.6, 0.0, 1.0)
        features["diversity"] = float(diversity)

        # 2. IMPROVEMENT RATE: Fraction of recent improvements
        if len(self.improvement_history) >= 10:
            recent = self.improvement_history[-self.history_length:]
            rate = np.mean(recent)
            features["improvement_rate"] = float(rate)

        # 3. CONVERGENCE: Fitness homogeneity
        f_min = np.min(valid_fitness)
        f_max = np.max(valid_fitness)
        f_range = f_max - f_min

        # Relative convergence
        if abs(f_min) > 1e-12:
            relative_range = f_range / abs(f_min)
            # Small range = high convergence
            convergence = 1.0 / (1.0 + 10.0 * relative_range)
            features["convergence"] = float(convergence)
        else:
            # Near-zero fitness, use absolute range
            convergence = 1.0 - np.clip(f_range / 100.0, 0.0, 1.0)
            features["convergence"] = float(convergence)

        return np.array([
            features["diversity"],
            features["improvement_rate"],
            features["convergence"]
        ])

    def update_improvement_history(self, improved: bool):
        """
        Update improvement history.

        Parameters
        ----------
        improved : bool
            Whether the best solution improved this timestep
        """
        self.improvement_history.append(1.0 if improved else 0.0)
        # Keep only recent history
        if len(self.improvement_history) > self.history_length * 2:
            self.improvement_history = self.improvement_history[-self.history_length:]

    def reset(self):
        """Reset improvement history."""
        self.improvement_history = []


def compute_fitness_weighted_centre(state: Dict[str, Any]) -> np.ndarray:
    """
    Compute fitness-weighted centroid from memory.

    Uses rank-based weighting to emphasize better solutions.

    Parameters
    ----------
    state : Dict[str, Any]
        Optimization state containing memory

    Returns
    -------
    centre : np.ndarray
        Fitness-weighted centroid in v-space
    """
    memory_fitness = state.get("memory_fitness", np.array([]))
    memory_vectors = state.get("memory_vectors", np.array([]))
    f_default_worst = state.get("f_default_worst", 1e10)

    valid_mask = memory_fitness < f_default_worst

    if not np.any(valid_mask):
        # No valid solutions, return origin
        dim = len(memory_vectors[0]) if len(memory_vectors) > 0 else 10
        return np.zeros(dim)

    valid_fitness = memory_fitness[valid_mask]
    valid_vectors = memory_vectors[valid_mask]

    # Rank-based weighting (better solutions get higher weight)
    ranks = np.argsort(np.argsort(valid_fitness))
    weights = 1.0 / (ranks + 1.0)
    weights /= np.sum(weights)

    return np.average(valid_vectors, axis=0, weights=weights)

