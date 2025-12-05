"""
Standard Optimisation Operators
================================

This module implements common metaheuristic operators adapted for
neuromorphic optimisation.
"""

import numpy as np
import math
from typing import Dict, Any
from nevo.operators.base import ExplorationOperator, ExploitationOperator


class LevyFlight(ExplorationOperator):
    """
    Lévy Flight Operator

    Heavy-tailed random walk for escaping local minima.
    Uses Mantegna's algorithm to generate Lévy-distributed steps.

    Best used when: stuck in local optima, need global exploration
    """

    def __init__(self, alpha: float = 0.3, beta: float = 1.5):
        """
        Parameters
        ----------
        alpha : float
            Step size scaling factor (0.1-0.5 recommended)
        beta : float
            Lévy exponent (1.5 is standard)
        """
        super().__init__("LevyFlight")
        self.alpha = alpha
        self.beta = beta

        # Precompute sigma_u for Mantegna's algorithm
        self.sigma_u = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))
        ) ** (1 / beta)

    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """Generate candidates using Lévy flight."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for _ in range(population_size):
            # Mantegna's algorithm
            u = np.random.normal(0, self.sigma_u, dim)
            v = np.random.normal(0, 1, dim)
            step = u / (np.abs(v) ** (1 / self.beta))

            # Add direction bias towards global best
            direction = global_best - centre

            # Generate candidate
            candidate = centre + self.alpha * step + 0.1 * direction
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class DifferentialEvolution(ExplorationOperator):
    """
    Differential Evolution Operator (DE/rand/1)

    Uses memory diversity to generate directed exploration.
    Creates new solutions by combining existing memory solutions.

    Best used when: memory has diversity, exploring promising regions
    """

    def __init__(self, F: float = 0.8, CR: float = 0.9):
        """
        Parameters
        ----------
        F : float
            Mutation factor (0.5-1.0 recommended)
        CR : float
            Crossover probability (0.7-1.0 recommended)
        """
        super().__init__("DifferentialEvolution")
        self.F = F
        self.CR = CR

    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """Generate candidates using DE mutation."""
        dim = len(centre)

        # Get valid memory
        memory_fitness = state.get("memory_fitness", np.array([]))
        memory_vectors = state.get("memory_vectors", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = memory_vectors[valid_mask]

        # Fallback if insufficient memory
        if len(valid_vectors) < 3:
            noise = np.random.randn(population_size, dim)
            return np.clip(centre + 0.5 * noise, -1.0, 1.0)

        candidates = []
        for _ in range(population_size):
            # Select 3 random distinct solutions
            idx = np.random.choice(len(valid_vectors), 3, replace=False)
            a, b, c = valid_vectors[idx]

            # DE/rand/1 mutation: a + F * (b - c)
            mutant = a + self.F * (b - c)

            # Binomial crossover with centre
            trial = centre.copy()
            crossover_mask = np.random.rand(dim) < self.CR
            trial[crossover_mask] = mutant[crossover_mask]

            candidates.append(np.clip(trial, -1.0, 1.0))

        return np.array(candidates)


class ParticleSwarm(ExploitationOperator):
    """
    Particle Swarm Optimisation Operator

    Velocity-based exploitation with attraction to personal and global bests.
    Maintains velocity state for smooth convergence.

    Best used when: improving and converging, exploitation phase
    """

    def __init__(self, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Parameters
        ----------
        w : float
            Inertia weight (0.4-0.9 recommended)
        c1 : float
            Cognitive (personal best) coefficient
        c2 : float
            Social (global best) coefficient
        """
        super().__init__("ParticleSwarm")
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocities = {}  # Particle ID -> velocity

    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """Generate candidates using PSO dynamics."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for i in range(population_size):
            # Get or initialize velocity
            if i not in self.velocities:
                self.velocities[i] = np.random.randn(dim) * 0.1

            # Current position (sample around centre)
            current = centre + np.random.randn(dim) * 0.05

            # PSO velocity update
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (centre - current)
            social = self.c2 * r2 * (global_best - current)

            self.velocities[i] = (
                self.w * self.velocities[i] +
                cognitive +
                social
            )

            # Clip velocity
            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

            # Update position
            new_position = current + self.velocities[i]
            candidates.append(np.clip(new_position, -1.0, 1.0))

        return np.array(candidates)


class SpiralOptimisation(ExploitationOperator):
    """
    Spiral Optimisation Operator

    Anisotropic convergence using logarithmic spirals in 2D planes.
    Each plane has independent rotation and convergence parameters.

    Best used when: highly converged, fine-tuning phase
    """

    def __init__(self, r_base: float = 0.95):
        """
        Parameters
        ----------
        r_base : float
            Base convergence rate (0.9-0.99 recommended)
        """
        super().__init__("SpiralOptimisation")
        self.r_base = r_base
        self.min_theta = 1e-3
        self.max_theta = 2 * np.pi

    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """Generate candidates using anisotropic spiral."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        n_planes = dim // 2

        for _ in range(population_size):
            x = centre.copy()

            # Independent rotation/convergence for each 2D plane
            for plane_idx in range(n_planes):
                d = plane_idx * 2

                # Per-plane random parameters
                theta = np.random.uniform(self.min_theta, self.max_theta)
                r_variation = np.random.uniform(0.9, 1.1)
                r = np.clip(self.r_base * r_variation, 0.85, 0.999)

                # Extract 2D slice
                x_d = x[d:d+2] - global_best[d:d+2]

                # Spiral transformation
                r_theta = r ** theta
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                x_rotated = r_theta * np.array([
                    cos_t * x_d[0] - sin_t * x_d[1],
                    sin_t * x_d[0] + cos_t * x_d[1]
                ])

                x[d:d+2] = global_best[d:d+2] + x_rotated

            # Handle odd dimension
            if dim % 2 == 1:
                theta_odd = np.random.uniform(self.min_theta, self.max_theta)
                r_odd = np.clip(self.r_base * np.random.uniform(0.9, 1.1), 0.85, 0.999)
                x[-1] = global_best[-1] + r_odd ** theta_odd * (x[-1] - global_best[-1])

            candidates.append(np.clip(x, -1.0, 1.0))

        return np.array(candidates)

