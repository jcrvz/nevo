"""
Standard Optimisation Operators
================================

This module implements common metaheuristic operators adapted for
neuromorphic optimisation.
"""

import numpy as np
import nengo
import math
from typing import Dict, Any
from nevo.operators.base import ExplorationOperator, ExploitationOperator


class LevyFlight(ExplorationOperator):
    """
    Lévy Flight Operator

    Heavy-tailed random walk for escaping local minima.
    Uses Mantegna's algorithm to generate Lévy-distributed steps.

    Best used when: stuck in local optima, need global exploration.

    The update rule is:

    .. math::

        x_{\\text{new}} = x_c + \\alpha L_{\\beta} + \\gamma (x_{\\text{best}} - x_c)

    where :math:`\\alpha` is the step size, :math:`L_{\\beta}` is a Lévy-distributed
    step (Mantegna's algorithm, exponent :math:`\\beta`), and :math:`\\gamma` controls
    the direction bias towards :math:`x_{\\text{best}}`.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 1.5, gamma: float = 0.1):
        """
        Parameters
        ----------
        alpha : float
            Step size scaling factor (0.1-0.5 recommended)
        beta : float
            Lévy exponent (1.5 is standard)
        gamma : float
            Direction bias factor towards global best (0.05-0.2 recommended)
        """
        super().__init__("LevyFlight", short_name="LF", complexity=3)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Precompute sigma_u for Mantegna's algorithm
        self.sigma_u = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
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
            candidate = centre + self.alpha * step + self.gamma * direction
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class DifferentialEvolution(ExplorationOperator):
    """
    Differential Evolution Operator (DE/rand/1/bin)

    Uses memory diversity to generate directed exploration.
    Creates new solutions by combining existing memory solutions.

    Best used when: memory has diversity, exploring promising regions.

    The update rule (DE/rand/1/bin) is:

    .. math::

        x_{\\text{new}} = x_a + F (x_b - x_c)

    where :math:`x_a, x_b, x_c` are distinct memory solutions, :math:`F` is the
    mutation factor, and binomial crossover is applied with probability :math:`CR`.
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
        super().__init__("DifferentialEvolution", short_name="DE", complexity=5)
        self.F = F
        self.CR = CR

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
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

    Best used when: improving and converging, exploitation phase.

    The update equations are:

    .. math::

        v_i &= w v_i + c_1 r_1 (p_{\\text{best}} - x_i)
              + c_2 r_2 (g_{\\text{best}} - x_i) \\\\
        x_{\\text{new}} &= x_i + v_i

    where :math:`w` is the inertia weight and :math:`c_1, c_2` are the cognitive
    and social coefficients.
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
        super().__init__("ParticleSwarm", short_name="PS", complexity=6)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocities = {}  # Particle ID -> velocity

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using PSO dynamics."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for i in range(population_size):
            # Get or initialise velocity
            if i not in self.velocities:
                self.velocities[i] = np.random.randn(dim) * 0.1

            # Current position (sample around centre)
            current = centre + np.random.randn(dim) * 0.05

            # PSO velocity update
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (centre - current)
            social = self.c2 * r2 * (global_best - current)

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

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

    The update rule is:

    .. math::

        x_{new} = x_{best} + r^{\\theta}
        \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\
        \\sin\\theta & \\cos\\theta \\end{bmatrix}
        (x_i - x_{best})

    where :math:`r` is the convergence rate and :math:`\\theta` is the rotation angle.
    """

    def __init__(self, r_base: float = 0.995):
        """
        Parameters
        ----------
        r_base : float
            Base convergence rate (0.9-0.99 recommended)
        """
        super().__init__("SpiralOptimisation", short_name="SO", complexity=7)
        self.r_base = r_base
        self.min_theta = 1e-3
        self.max_theta = 2 * np.pi

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
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
                x_d = x[d : d + 2] - global_best[d : d + 2]

                # Spiral transformation
                r_theta = r**theta
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                x_rotated = r_theta * np.array(
                    [cos_t * x_d[0] - sin_t * x_d[1], sin_t * x_d[0] + cos_t * x_d[1]]
                )

                x[d : d + 2] = global_best[d : d + 2] + x_rotated

            # Handle odd dimension
            if dim % 2 == 1:
                theta_odd = np.random.uniform(self.min_theta, self.max_theta)
                r_odd = np.clip(self.r_base * np.random.uniform(0.9, 1.1), 0.85, 0.999)
                x[-1] = global_best[-1] + r_odd**theta_odd * (x[-1] - global_best[-1])

            candidates.append(np.clip(x, -1.0, 1.0))

        return np.array(candidates)


# ============================================================================
# Additional Operators (inspired by customhys framework)
# ============================================================================


class RandomSearch(ExplorationOperator):
    """
    Random Search Operator

    Simple uniform random sampling around the centre.
    Useful for initial exploration or when other operators stagnate.

    Best used when: no prior information, need baseline exploration.

    The update rule is:

    .. math::

        x_{\\text{new}} = x_c + \\delta

    where :math:`\\delta` is sampled from a uniform or Gaussian distribution
    with scale parameter :math:`s`.
    """

    def __init__(self, scale: float = 0.5, distribution: str = "uniform"):
        """
        Parameters
        ----------
        scale : float
            Perturbation scale (0.1-1.0 recommended)
        distribution : str
            Distribution type: 'uniform' or 'gaussian'
        """
        super().__init__("RandomSearch", short_name="RS", complexity=1)
        self.scale = scale
        self.distribution = distribution

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using random perturbations."""
        dim = len(centre)
        candidates = []

        for _ in range(population_size):
            if self.distribution == "gaussian":
                perturbation = np.random.randn(dim) * self.scale
            else:  # uniform
                perturbation = np.random.uniform(-self.scale, self.scale, dim)

            candidate = centre + perturbation
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class LocalRandomWalk(ExploitationOperator):
    """
    Local Random Walk Operator (from Cuckoo Search)

    Small-scale local exploration with probability-based activation.
    Generates subtle perturbations for fine-tuning solutions.

    Best used when: near optimum, need local refinement.

    The update rule is:

    .. math::

        x_{\\text{new}} = x_c + s \\, \\Delta

    where :math:`s` is the step size scale and :math:`\\Delta` is a difference
    vector from two randomly selected memory solutions.
    """

    def __init__(self, probability: float = 0.75, scale: float = 0.1):
        """
        Parameters
        ----------
        probability : float
            Probability of applying walk to each dimension (0.5-0.9 recommended)
        scale : float
            Step size scale (0.05-0.2 recommended)
        """
        super().__init__("LocalRandomWalk", short_name="LW", complexity=2)
        self.probability = probability
        self.scale = scale

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using local random walk."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        # Get two random memory solutions for difference vector
        memory_vectors = state.get("memory_vectors", np.array([]))
        memory_fitness = state.get("memory_fitness", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = (
            memory_vectors[valid_mask] if len(memory_vectors) > 0 else np.array([])
        )

        candidates = []
        for _ in range(population_size):
            candidate = centre.copy()

            # Apply walk with probability
            mask = np.random.rand(dim) < self.probability

            if len(valid_vectors) >= 2:
                # Use difference of two random solutions
                idx = np.random.choice(len(valid_vectors), 2, replace=False)
                diff = valid_vectors[idx[0]] - valid_vectors[idx[1]]
                step = self.scale * np.random.rand() * diff
            else:
                # Fallback to simple random step
                step = self.scale * np.random.randn(dim)

            candidate[mask] += step[mask]
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class GravitationalSearch(ExplorationOperator):
    """
    Gravitational Search Algorithm Operator

    Mass-based attraction dynamics where better solutions have more mass.
    Solutions are attracted towards heavier (better) solutions.

    Best used when: need directed exploration, moderate diversity.

    The update rule is:

    .. math::

        x_{\\text{new}} = x_i + a_i

    where :math:`a_i` is the acceleration from gravitational forces of all other
    solutions, weighted by their masses (derived from fitness).

    """

    def __init__(self, gravity: float = 1.0, alpha: float = 0.02):
        """
        Parameters
        ----------
        gravity : float
            Gravitational constant (0.5-2.0 recommended)
        alpha : float
            Gravity decay rate (0.01-0.05 recommended)
        """
        super().__init__("GravitationalSearch", short_name="GS", complexity=8)
        self.gravity = gravity
        self.alpha = alpha
        self.iteration = 0

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using gravitational dynamics."""
        dim = len(centre)
        self.iteration += 1

        # Get valid memory
        memory_vectors = state.get("memory_vectors", np.array([]))
        memory_fitness = state.get("memory_fitness", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = memory_vectors[valid_mask]
        valid_fitness = memory_fitness[valid_mask]

        if len(valid_vectors) < 2:
            # Fallback to random search
            noise = np.random.randn(population_size, dim) * 0.3
            return np.clip(centre + noise, -1.0, 1.0)

        # Compute masses (normalised inverse fitness)
        f_min, f_max = valid_fitness.min(), valid_fitness.max()
        if f_max - f_min < 1e-12:
            masses = np.ones(len(valid_fitness))
        else:
            masses = (f_max - valid_fitness) / (f_max - f_min + 1e-12)
        masses = masses / (masses.sum() + 1e-12)

        # Decaying gravity
        G = self.gravity * np.exp(-self.alpha * self.iteration)

        candidates = []
        for _ in range(population_size):
            # Start from centre with small noise
            pos = centre + np.random.randn(dim) * 0.05

            # Compute gravitational force
            force = np.zeros(dim)
            for j, (other, mass) in enumerate(zip(valid_vectors, masses)):
                diff = other - pos
                dist = np.linalg.norm(diff) + 1e-12
                force += G * mass * diff / dist

            # Apply force with random component
            acceleration = force * np.random.rand()
            candidate = pos + acceleration
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class FireflyAlgorithm(ExplorationOperator):
    """
    Firefly Algorithm Operator

    Light-based attraction where brighter (better) fireflies attract others.
    Combines attraction with randomisation for balanced exploration.

    Best used when: need attraction-based exploration, moderate convergence.

    The update rule is:

    .. math::

        x_{\\text{new}} = x_i + \\beta(r)(x_j - x_i) + \\alpha \\epsilon

    where :math:`\\beta(r) = \\beta_0 e^{-\\gamma r^2}` is the distance-based
    attractiveness, :math:`\\alpha` is the randomisation parameter, and
    :math:`\\epsilon` is a random perturbation.
    """

    def __init__(self, alpha: float = 0.2, beta: float = 1.0, gamma: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Randomisation parameter (0.1-0.5 recommended)
        beta : float
            Attractiveness at distance zero (0.5-2.0 recommended)
        gamma : float
            Light absorption coefficient (0.5-2.0 recommended)
        """
        super().__init__("FireflyAlgorithm", short_name="FA", complexity=7)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using firefly dynamics."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        # Get valid memory as "fireflies"
        memory_vectors = state.get("memory_vectors", np.array([]))
        memory_fitness = state.get("memory_fitness", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = memory_vectors[valid_mask]
        valid_fitness = memory_fitness[valid_mask]

        if len(valid_vectors) < 2:
            noise = np.random.randn(population_size, dim) * 0.3
            return np.clip(centre + noise, -1.0, 1.0)

        candidates = []
        for _ in range(population_size):
            # Start from centre
            pos = centre.copy()

            # Find a brighter firefly to move towards
            current_fitness = state.get("best_f", f_default_worst)
            brighter_mask = valid_fitness < current_fitness
            if brighter_mask.any():
                brighter = valid_vectors[brighter_mask]
                brighter_f = valid_fitness[brighter_mask]
                # Move towards brightest
                best_idx = np.argmin(brighter_f)
                target = brighter[best_idx]
            else:
                target = global_best

            # Distance-based attraction
            r = np.linalg.norm(target - pos)
            beta_r = self.beta * np.exp(-self.gamma * r**2)

            # Move towards target with randomisation
            candidate = (
                pos + beta_r * (target - pos) + self.alpha * np.random.randn(dim)
            )
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class CentralForce(ExplorationOperator):
    """
    Central Force Optimisation Operator

    Physics-inspired operator using gravitational attraction towards
    the global best solution with inverse-square law dynamics.

    Best used when: need strong directional bias, exploitation-exploration balance

    The mathematical expression for this operator is given by:
    x_{new} = x_{i} + F_{c}

    where:
    - \( x_{new} \) is the new candidate solution.
    - \( x_{i} \) is the current position of the solution.
    - \( F_{c} \) is the central force vector directed towards the global best solution, calculated using an inverse power law based on distance.

    """

    def __init__(self, gravity: float = 2.0, alpha: float = 2.0):
        """
        Parameters
        ----------
        gravity : float
            Gravitational constant (1.0-3.0 recommended)
        alpha : float
            Distance exponent (1.5-2.5 recommended)
        """
        super().__init__("CentralForce", short_name="CF", complexity=6)
        self.gravity = gravity
        self.alpha = alpha

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using central force dynamics."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for _ in range(population_size):
            # Start from random position around centre
            pos = centre + np.random.randn(dim) * 0.2

            # Central force towards global best
            diff = global_best - pos
            dist = np.linalg.norm(diff) + 1e-12

            # Inverse power law attraction
            force = self.gravity * diff / (dist**self.alpha)

            # Random velocity component
            velocity = np.random.rand() * force + 0.1 * np.random.randn(dim)

            candidate = pos + velocity
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class GeneticCrossover(ExplorationOperator):
    """
    Genetic Algorithm Crossover Operator

    Recombination of memory solutions using various crossover strategies.
    Creates new solutions by combining genetic material from parents.

    Best used when: memory has good diversity, want to combine good features

    The mathematical expression for this operator is given by:
    x_{new} = Crossover(x_{parent1}, x_{parent2})

    where:
    - \( x_{new} \) is the new candidate solution.
    - \( x_{parent1}, x_{parent2} \) are two parent solutions selected from memory.

    """

    def __init__(self, crossover: str = "blend", alpha: float = 0.5):
        """
        Parameters
        ----------
        crossover : str
            Crossover type: 'blend', 'single_point', 'two_point', 'uniform'
        alpha : float
            Blend crossover extension factor (0.3-0.7 recommended)
        """
        super().__init__("GeneticCrossover", short_name="GX", complexity=4)
        self.crossover = crossover
        self.alpha = alpha

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using genetic crossover."""
        dim = len(centre)

        # Get valid memory
        memory_vectors = state.get("memory_vectors", np.array([]))
        memory_fitness = state.get("memory_fitness", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = memory_vectors[valid_mask]

        if len(valid_vectors) < 2:
            noise = np.random.randn(population_size, dim) * 0.3
            return np.clip(centre + noise, -1.0, 1.0)

        # Rank-based selection probabilities
        valid_fitness = memory_fitness[valid_mask]
        ranks = np.argsort(np.argsort(valid_fitness))  # lower fitness = lower rank
        probs = (len(ranks) - ranks).astype(float)
        probs /= probs.sum()

        candidates = []
        for _ in range(population_size):
            # Select two parents (fitness-biased)
            parents_idx = np.random.choice(
                len(valid_vectors), 2, replace=False, p=probs
            )
            p1, p2 = valid_vectors[parents_idx]

            if self.crossover == "blend":
                # BLX-alpha crossover
                d = np.abs(p2 - p1)
                low = np.minimum(p1, p2) - self.alpha * d
                high = np.maximum(p1, p2) + self.alpha * d
                child = np.random.uniform(low, high)

            elif self.crossover == "single_point":
                point = np.random.randint(1, dim)
                child = np.concatenate([p1[:point], p2[point:]])

            elif self.crossover == "two_point":
                pt1, pt2 = sorted(np.random.choice(dim, 2, replace=False))
                child = p1.copy()
                child[pt1:pt2] = p2[pt1:pt2]

            else:  # uniform
                mask = np.random.rand(dim) < 0.5
                child = np.where(mask, p1, p2)

            candidates.append(np.clip(child, -1.0, 1.0))

        return np.array(candidates)


class GeneticMutation(ExploitationOperator):
    """
    Genetic Algorithm Mutation Operator

    Random perturbation of solutions with configurable mutation rate.
    Introduces diversity by modifying individual genes.

    Best used when: need local refinement, fine-tuning solutions

    The mathematical expression for this operator is given by:
    x_{new} = x_{i} + m

    where:
    - \( x_{new} \) is the new candidate solution.
    - \( x_{i} \) is the current position of the solution.
    - \( m \) is a mutation vector where each gene is altered with a certain probability according to the mutation rate and distribution.

    Note: With high mutation rates (>0.5) and large scales (>0.5), this operator
    behaves more like exploration. Default parameters favour exploitation.
    """

    def __init__(
        self,
        mutation_rate: float = 0.2,
        scale: float = 0.3,
        distribution: str = "gaussian",
    ):
        """
        Parameters
        ----------
        mutation_rate : float
            Probability of mutating each gene (0.1-0.4 recommended)
        scale : float
            Mutation magnitude (0.1-0.5 recommended)
        distribution : str
            Mutation distribution: 'gaussian' or 'uniform'
        """
        super().__init__("GeneticMutation", short_name="GM", complexity=3)
        self.mutation_rate = mutation_rate
        self.scale = scale
        self.distribution = distribution

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using genetic mutation."""
        dim = len(centre)

        # Get valid memory
        memory_vectors = state.get("memory_vectors", np.array([]))
        memory_fitness = state.get("memory_fitness", np.array([]))
        f_default_worst = state.get("f_default_worst", 1e10)

        valid_mask = memory_fitness < f_default_worst
        valid_vectors = memory_vectors[valid_mask]

        if len(valid_vectors) == 0:
            valid_vectors = np.array([centre])

        candidates = []
        for _ in range(population_size):
            # Select a random solution from memory
            idx = np.random.randint(len(valid_vectors))
            candidate = valid_vectors[idx].copy()

            # Apply mutations
            mutation_mask = np.random.rand(dim) < self.mutation_rate
            n_mutations = mutation_mask.sum()

            if n_mutations > 0:
                if self.distribution == "gaussian":
                    mutations = np.random.randn(n_mutations) * self.scale
                else:  # uniform
                    mutations = np.random.uniform(-self.scale, self.scale, n_mutations)
                candidate[mutation_mask] += mutations

            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class SimulatedAnnealing(ExploitationOperator):
    """
    Simulated Annealing Operator

    Temperature-based local search with decreasing randomness.
    Allows uphill moves early, becomes greedy over time.

    Best used when: need controlled exploitation, avoiding local minima

    The mathematical expression for this operator is given by:
    x_{new} = x_{centre} + T \cdot \delta + b

    where:
    - \( x_{new} \) is the new candidate solution.
    - \( x_{centre} \) is the current centre solution.
    - \( T \) is the current temperature controlling perturbation scale.
    - \( \delta \) is a random perturbation vector.
    - \( b \) is a bias vector towards the global best solution, increasing as temperature decreases.

    """

    def __init__(self, initial_temp: float = 1.0, cooling_rate: float = 0.995):
        """
        Parameters
        ----------
        initial_temp : float
            Initial temperature (0.5-2.0 recommended)
        cooling_rate : float
            Temperature decay per call (0.99-0.999 recommended)
        """
        super().__init__("SimulatedAnnealing", short_name="SA", complexity=4)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.temperature = initial_temp

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates using temperature-scaled perturbations."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for _ in range(population_size):
            # Temperature-scaled perturbation
            perturbation = self.temperature * np.random.randn(dim)

            # Bias towards global best (increases as temperature decreases)
            bias = (1.0 - self.temperature / self.initial_temp) * (global_best - centre)

            candidate = centre + perturbation + 0.3 * bias
            candidates.append(np.clip(candidate, -1.0, 1.0))

        # Cool down
        self.temperature = max(0.01, self.temperature * self.cooling_rate)

        return np.array(candidates)

    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.temperature = self.initial_temp


class TabuSearch(ExploitationOperator):
    """
    Tabu Search Operator

    Memory-based local search that avoids recently visited regions.
    Maintains a tabu list to prevent cycling.

    Best used when: stuck in local optima, need diversification

    The mathematical expression for this operator is given by:
    x_{new} = x_{centre} + \delta

    where:
    - \( x_{new} \) is the new candidate solution.
    - \( x_{centre} \) is the current centre solution.
    - \( \delta \) is a random perturbation vector.
    - The candidate solutions are checked against a tabu list to avoid revisiting recent solutions.

    """

    def __init__(self, tabu_tenure: int = 10, neighbourhood_size: float = 0.2):
        """
        Parameters
        ----------
        tabu_tenure : int
            Number of iterations to keep moves in tabu list (5-20 recommended)
        neighbourhood_size : float
            Size of local neighbourhood (0.1-0.3 recommended)
        """
        super().__init__("TabuSearch", short_name="TS", complexity=5)
        self.tabu_tenure = tabu_tenure
        self.neighbourhood_size = neighbourhood_size
        self.tabu_list = []

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """Generate candidates avoiding tabu regions."""
        dim = len(centre)
        global_best = state.get("best_v")
        if global_best is None:
            global_best = centre

        candidates = []
        for _ in range(population_size):
            # Generate neighbour
            perturbation = np.random.randn(dim) * self.neighbourhood_size
            candidate = centre + perturbation

            # Check if in tabu region (simplified: check distance to tabu points)
            is_tabu = False
            closest_tabu = None
            for tabu_point in self.tabu_list:
                if (
                    np.linalg.norm(candidate - tabu_point)
                    < self.neighbourhood_size * 0.5
                ):
                    is_tabu = True
                    closest_tabu = tabu_point
                    break

            if is_tabu and closest_tabu is not None:
                # Move away from tabu region
                direction = candidate - closest_tabu
                candidate = (
                    centre
                    + direction
                    / (np.linalg.norm(direction) + 1e-12)
                    * self.neighbourhood_size
                )

            candidates.append(np.clip(candidate, -1.0, 1.0))

        # Update tabu list with centre
        self.tabu_list.append(centre.copy())
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)

        return np.array(candidates)


class NeuromorphicExplorationEnsemble(ExplorationOperator):
    """
    Nengo LIF-based exploration using fast spiking neural populations.

    This operator builds Nengo neural networks inside the optimiser's model
    with LIF neurons, fast synaptic filtering, and NEF decoding.

    IMPORTANT: Unlike other operators, this creates Nengo Ensembles that
    are integrated into the main Nengo simulation loop.
    """

    def __init__(
        self,
        n_neurons: int = 150,
        tau_synapse: float = 0.005,  # 5 ms
        max_rates: tuple = (100, 200),
        intercepts: tuple = (-1.0, 1.0),
        use_numpy_fallback: bool = False,
    ):
        super().__init__(
            "NeuromorphicExplorationEnsemble", short_name="NEX", complexity=6
        )
        self.n_neurons = n_neurons
        self.tau_synapse = tau_synapse
        self.max_rates = max_rates
        self.intercepts = intercepts
        self.use_numpy_fallback = use_numpy_fallback
        self._nengo_ensemble = None
        self._nengo_model = None

    def build_network(
        self, model: "nengo.Network", state_ensemble: "nengo.Ensemble", dimension: int
    ) -> "nengo.Ensemble":
        """
        Build Nengo LIF ensemble for exploration with spike decoding.
        """
        import nengo

        with model:
            # LIF spiking ensemble
            self._nengo_ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=dimension,
                radius=1.5,
                neuron_type=nengo.LIF(),
                max_rates=nengo.dists.Uniform(self.max_rates[0], self.max_rates[1]),
                intercepts=nengo.dists.Uniform(self.intercepts[0], self.intercepts[1]),
                label="ExplorationEnsemble",
            )

            # Fast synaptic connection from state
            nengo.Connection(
                state_ensemble,
                self._nengo_ensemble,
                synapse=self.tau_synapse,
                transform=[[0.1, 0.1, 0.1]] * dimension,
            )

            # Probe to read decoded output from spikes
            self._output_probe = nengo.Probe(
                self._nengo_ensemble, synapse=self.tau_synapse
            )

            self._nengo_model = model

        return self._nengo_ensemble

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """
        Generate exploration candidates from Nengo spike decoding.
        """
        dim = len(centre)
        best_v = state.get("best_v")
        if best_v is None:
            best_v = centre

        features = state.get("state_features", np.array([0.5, 0.5, 0.0]))
        diversity, improvement, convergence = features

        # State-dependent parameters
        exploration_scale = 0.35 * (1.0 - convergence) + 0.25 * (1.0 - diversity)
        repulsion_mag = 0.3 * (1.0 - convergence)
        repulsion = np.clip(centre - best_v, -0.5, 0.5)

        candidates = []

        # Use spike decoding (default behavior)
        if not self.use_numpy_fallback:
            sim = state.get("simulator")
            if sim is None or not hasattr(self, "_output_probe"):
                raise RuntimeError(
                    "Neuromorphic operator requires Nengo simulator with probe. "
                    "Set use_numpy_fallback=True for numpy-based generation."
                )

            # Get decoded output from LIF ensemble spikes
            if len(sim.data[self._output_probe]) > 0:
                decoded_activity = sim.data[self._output_probe][-1]
            else:
                # First timestep: use small random initialization
                decoded_activity = np.random.randn(dim) * 0.1

            for i in range(population_size):
                # Use decoded neural activity as perturbation
                # Add jitter to break symmetry if neurons haven't activated yet
                if np.linalg.norm(decoded_activity) < 1e-6:
                    neural_noise = np.random.randn(dim) * exploration_scale * 0.1
                else:
                    neural_noise = (
                        decoded_activity
                        * exploration_scale
                        * (0.5 + 0.5 * np.random.rand())
                    )
                candidate = centre + repulsion_mag * repulsion + neural_noise
                candidates.append(np.clip(candidate, -1.0, 1.0))

            return np.array(candidates)

        # Numpy fallback (only if explicitly enabled)
        for _ in range(population_size):
            noise = np.random.randn(dim) * exploration_scale
            candidate = centre + repulsion_mag * repulsion + noise
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class NeuromorphicExploitationEnsemble(ExploitationOperator):
    """
    Nengo LIF-based exploitation using slow spiking neural populations.

    This operator builds Nengo neural networks inside the optimiser's model
    with LIF neurons, slow synaptic filtering for attractor dynamics.

    IMPORTANT: Unlike other operators, this creates Nengo Ensembles that
    are integrated into the main Nengo simulation loop.
    """

    def __init__(
        self,
        n_neurons: int = 200,
        tau_synapse: float = 0.020,  # 20 ms
        max_rates: tuple = (50, 100),
        intercepts: tuple = (-0.5, 0.5),
        trust_radius: float = 0.2,
        use_numpy_fallback: bool = False,
    ):
        super().__init__(
            "NeuromorphicExploitationEnsemble", short_name="NXP", complexity=6
        )
        self.n_neurons = n_neurons
        self.tau_synapse = tau_synapse
        self.max_rates = max_rates
        self.intercepts = intercepts
        self.trust_radius = trust_radius
        self.use_numpy_fallback = use_numpy_fallback
        self._nengo_ensemble = None
        self._nengo_model = None
        self._attractor = None

    def build_network(
        self, model: "nengo.Network", state_ensemble: "nengo.Ensemble", dimension: int
    ) -> "nengo.Ensemble":
        """
        Build Nengo LIF ensemble for exploitation with spike decoding.
        """
        with model:
            # LIF spiking ensemble
            self._nengo_ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=dimension,
                radius=1.5,
                neuron_type=nengo.LIF(),
                max_rates=nengo.dists.Uniform(self.max_rates[0], self.max_rates[1]),
                intercepts=nengo.dists.Uniform(self.intercepts[0], self.intercepts[1]),
                label="ExploitationEnsemble",
            )

            # Slow synaptic connection from state
            nengo.Connection(
                state_ensemble,
                self._nengo_ensemble,
                synapse=self.tau_synapse,
                transform=[[0.05, 0.05, 0.05]] * dimension,
            )

            # Probe to read decoded output from spikes
            self._output_probe = nengo.Probe(
                self._nengo_ensemble, synapse=self.tau_synapse
            )

            self._nengo_model = model

        return self._nengo_ensemble

    def generate_population(
        self, centre: np.ndarray, state: Dict[str, Any], population_size: int
    ) -> np.ndarray:
        """
        Generate exploitation candidates from Nengo spike decoding.
        """
        dim = len(centre)
        best_v = state.get("best_v")
        if best_v is None:
            best_v = centre

        features = state.get("state_features", np.array([0.5, 0.5, 0.0]))
        diversity, improvement, convergence = features

        # Attractor tracking
        blend = 0.7 * best_v + 0.3 * centre
        if self._attractor is None:
            self._attractor = blend.copy()
        self._attractor = 0.9 * self._attractor + 0.1 * blend

        # Trust region
        adaptive_radius = self.trust_radius * (0.7 + 0.5 * (1.0 - convergence))
        local_noise = 0.08 * (1.0 - convergence) + 0.05 * (1.0 - improvement)

        candidates = []

        # Use spike decoding (default behavior)
        if not self.use_numpy_fallback:
            sim = state.get("simulator")
            if sim is None or not hasattr(self, "_output_probe"):
                raise RuntimeError(
                    "Neuromorphic operator requires Nengo simulator with probe. "
                    "Set use_numpy_fallback=True for numpy-based generation."
                )

            # Get decoded output from slow LIF ensemble spikes
            if len(sim.data[self._output_probe]) > 0:
                decoded_activity = sim.data[self._output_probe][-1]
            else:
                # First timestep: use small random initialization
                decoded_activity = np.random.randn(dim) * 0.05

            for i in range(population_size):
                # Use decoded neural activity as constrained perturbation
                # Add jitter to break symmetry if neurons haven't activated yet
                if np.linalg.norm(decoded_activity) < 1e-6:
                    neural_noise = np.random.randn(dim) * local_noise * 0.1
                else:
                    neural_noise = (
                        decoded_activity * local_noise * (0.5 + 0.5 * np.random.rand())
                    )
                neural_noise = np.clip(neural_noise, -adaptive_radius, adaptive_radius)
                candidate = self._attractor + neural_noise
                candidates.append(np.clip(candidate, -1.0, 1.0))

            return np.array(candidates)

        # Numpy fallback (only if explicitly enabled)
        for _ in range(population_size):
            noise = np.random.randn(dim) * local_noise
            noise = np.clip(noise, -adaptive_radius, adaptive_radius)
            candidate = self._attractor + noise
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)
