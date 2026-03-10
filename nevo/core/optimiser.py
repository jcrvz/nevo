"""
NEVO Optimiser
==============

Main optimiser class integrating all neuromorphic components.
"""

import nengo
import numpy as np
from typing import Callable, List, Optional, Dict, Any
from nevo.operators.base import Operator
from nevo.core.td_learning import LearningRule, ValueModel
from nevo.operators.standard import (
    LevyFlight,
    DifferentialEvolution,
    ParticleSwarm,
    SpiralOptimisation,
    RandomSearch,
    LocalRandomWalk,
    GravitationalSearch,
    FireflyAlgorithm,
    CentralForce,
    GeneticCrossover,
    GeneticMutation,
    SimulatedAnnealing,
    TabuSearch,
    NeuromorphicExplorationEnsemble,
    NeuromorphicExploitationEnsemble,
)
from nevo.core.state import StateFeatures, compute_fitness_weighted_centre
from nevo.core.basal_ganglia import BasalGangliaSelector


def trs2o(v: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Transform from v-space [-1, 1] to original space [lb, ub].

    Parameters
    ----------
    v : np.ndarray
        Solution in v-space
    lb : np.ndarray
        Lower bounds in original space
    ub : np.ndarray
        Upper bounds in original space

    Returns
    -------
    x : np.ndarray
        Solution in original space
    """
    return lb + (ub - lb) * (v + 1.0) / 2.0


class NEVOptimiser:
    """
    Neuromorphic Evolutionary Optimiser.

    Uses basal ganglia circuits to adaptively select between multiple
    optimisation operators based on the current search state.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: tuple,
        dimension: int,
        population_size: int = 50,
        memory_size: int = 25,
        operators: Optional[List[Operator]] = None,
        operator_mode: str = "trad",
        neurons_per_ensemble: int = 100,
        dt: float = 0.001,
        epsilon: float = 0.1,
        learning_rate: float = 0.4,
        td_gamma: float = 0.99,
        td_lambda: float = 0.0,
        td_enabled: bool = True,
        td_learning_rule: Optional[LearningRule] = None,
        td_value_model: Optional[ValueModel] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize NEVO optimiser.

        Parameters
        ----------
        objective_function : Callable
            Function to minimize: f(x) -> float
            Input x is in original space [lb, ub]
        bounds : tuple
            (lower_bounds, upper_bounds) as numpy arrays or scalars
        dimension : int
            Problem dimension
        population_size : int
            Number of candidates per operator per timestep
        memory_size : int
            Size of solution memory (archive)
        operators : List[Operator], optional
            Custom operators (uses default if None)
        operator_mode : str
            Operator selection mode: "trad" | "traditional", "nm_dual", or "nm_softmix"
        neurons_per_ensemble : int
            Neurons per ensemble in neural networks
        dt : float
            Simulation timestep (seconds)
        epsilon : float
            Epsilon-greedy exploration rate
        learning_rate : float
            Learning rate for utility weight adaptation
        td_gamma : float
            Discount factor for TD learning
        td_lambda : float
            Lambda parameter for TD(lambda)
        td_enabled : bool
            Enable TD-based value learning in selector
        td_learning_rule : LearningRule, optional
            Custom TD learning rule instance
        td_value_model : ValueModel, optional
            Custom TD value model instance
        seed : int, optional
            Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.population_size = population_size
        self.memory_size = memory_size
        self.neurons_per_ensemble = neurons_per_ensemble
        self.dt = dt
        self.seed = seed
        self.operator_mode = operator_mode
        self.soft_mix_temperature = 0.35
        self.soft_mix_concentration = 6.0

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Parse bounds
        if np.isscalar(bounds[0]):
            self.lb = np.full(dimension, bounds[0])
            self.ub = np.full(dimension, bounds[1])
        else:
            self.lb = np.array(bounds[0])
            self.ub = np.array(bounds[1])

        # Initialize operators
        if operators is None:
            if operator_mode in ("trad", "traditional"):
                self.operators = [
                    LevyFlight(),
                    DifferentialEvolution(),
                    ParticleSwarm(),
                    SpiralOptimisation(),
                    RandomSearch(),
                    LocalRandomWalk(),
                    GravitationalSearch(),
                    FireflyAlgorithm(),
                    CentralForce(),
                    GeneticCrossover(),
                    GeneticMutation(),
                    SimulatedAnnealing(),
                    TabuSearch(),
                ]
            elif operator_mode in ("nm_dual", "nm_softmix"):
                self.operators = [
                    NeuromorphicExplorationEnsemble(),
                    NeuromorphicExploitationEnsemble(),
                ]
            else:
                raise ValueError(
                    f"Unknown operator_mode: '{operator_mode}'. "
                    "Use 'trad'/'traditional', 'nm_dual', or 'nm_softmix'."
                )
        else:
            self.operators = operators

        # Initialize state
        self.f_default_worst = 1e10
        self.state = {
            "best_v": None,
            "best_f": None,
            "memory_vectors": np.zeros((memory_size, dimension)),
            "memory_fitness": np.full(memory_size, self.f_default_worst),
            "memory_age": np.zeros(memory_size),
            "current_operator": None,
            "operator_counts": {op.name: 0 for op in self.operators},
            "total_evals": 0,
            "f_default_worst": self.f_default_worst,
            "state_features": np.array([0.5, 0.5, 0.0]),
        }

        # Initialize state features
        self.state_features = StateFeatures(history_length=50)

        # Initialize basal ganglia selector
        self.bg_selector = BasalGangliaSelector(
            operators=self.operators,
            neurons_per_ensemble=neurons_per_ensemble,
            epsilon=epsilon,
            learning_rate=learning_rate,
            gamma=td_gamma,
            lambda_coeff=td_lambda,
            learning_rule=td_learning_rule,
            value_model=td_value_model,
            td_enabled=td_enabled,
        )

        # Nengo model (will be built in run())
        self.model = None
        self.simulator = None
        self._td_episode_started = False

    def evaluate(self, v: np.ndarray) -> float:
        """
        Evaluate objective function from v-space.

        Parameters
        ----------
        v : np.ndarray
            Solution in v-space [-1, 1]

        Returns
        -------
        fitness : float
            Objective function value
        """
        x = trs2o(v, self.lb, self.ub)
        return self.objective_function(x)

    def update_memory(self, candidates: np.ndarray, fitness: np.ndarray):
        """
        Update memory with new candidates (competitive update).

        Parameters
        ----------
        candidates : np.ndarray
            Candidate solutions (population_size, dimension)
        fitness : np.ndarray
            Fitness values (population_size,)
        """
        for i in range(len(candidates)):
            worst_idx = np.argmax(self.state["memory_fitness"])
            if fitness[i] < self.state["memory_fitness"][worst_idx]:
                self.state["memory_vectors"][worst_idx] = candidates[i]
                self.state["memory_fitness"][worst_idx] = fitness[i]
                self.state["memory_age"][worst_idx] = 0.0

        # Age all memory
        self.state["memory_age"] += 1.0

    def _compute_soft_mix_weights(self, operator_selection: np.ndarray) -> np.ndarray:
        """Convert selector signal to stable softmax weights for blending."""
        signal = np.asarray(operator_selection, dtype=float)
        signal = signal - np.max(signal)
        logits = signal / max(1e-6, self.soft_mix_temperature)
        exps = np.exp(logits)
        weights = exps / (np.sum(exps) + 1e-12)

        # Prevent full collapse so both ensembles remain active.
        min_w = 0.1 if len(weights) == 2 else 0.0
        if min_w > 0.0:
            weights = np.maximum(weights, min_w)
            weights = weights / np.sum(weights)
        return weights

    def build_model(self):
        """Build Nengo model."""
        self.model = nengo.Network(label="NEVO", seed=self.seed)

        with self.model:
            # State features node
            def state_features_func(t):
                if t < 0.001:
                    features = np.array([0.5, 0.5, 0.0])
                else:
                    features = self.state_features.compute(self.state)
                self.state["state_features"] = features
                return features

            state_features_node = nengo.Node(
                state_features_func,
                label="StateFeatures"
            )

            # State ensemble (reduced neurons for speed)
            state_ensemble = nengo.Ensemble(
                n_neurons=100,  # 300
                dimensions=3,
                radius=1.5,
                label="StateEnsemble"
            )

            nengo.Connection(state_features_node, state_ensemble, synapse=None)

            # Basal ganglia operator selection
            selected_operator_ens = self.bg_selector.build_network(
                self.model,
                state_ensemble
            )

            # Build neuromorphic operator networks (if using neuromorphic modes)
            if self.operator_mode in ("nm_dual", "nm_softmix"):
                for operator in self.operators:
                    # Check if operator has build_network method (neuromorphic operators do)
                    if hasattr(operator, 'build_network'):
                        operator.build_network(
                            self.model,
                            state_ensemble,
                            self.dimension
                        )

            # Population generator node
            def population_generator_func(t, operator_selection):
                if t < 0.001:
                    self.state["total_evals"] += self.population_size
                    self.state["state_features"] = np.array([0.5, 0.5, 0.0])
                    return np.array([0.0, 0.0, 0.0])

                # Make simulator available to operators for spike decoding
                if hasattr(self, 'simulator') and self.simulator is not None:
                    self.state["simulator"] = self.simulator

                # Select operator
                current_best = self.state.get("best_f")
                if current_best is None:
                    current_best = self.f_default_worst
                operator = self.bg_selector.select_operator(
                    operator_selection,
                    current_best
                )

                # Keep latest features available for state-aware operators.
                self.state["state_features"] = self.state_features.compute(self.state)

                # Get centre
                centre = compute_fitness_weighted_centre(self.state)

                # Generate population
                if self.operator_mode == "nm_softmix" and len(self.operators) == 2:
                    explore_op, exploit_op = self.operators
                    explore_candidates = explore_op.generate_population(
                        centre,
                        self.state,
                        self.population_size
                    )
                    exploit_candidates = exploit_op.generate_population(
                        centre,
                        self.state,
                        self.population_size
                    )

                    mix_weights = self._compute_soft_mix_weights(operator_selection)
                    exploit_mean = float(mix_weights[1])
                    alpha = max(0.5, exploit_mean * self.soft_mix_concentration)
                    beta = max(0.5, (1.0 - exploit_mean) * self.soft_mix_concentration)
                    lambdas = np.random.beta(alpha, beta, size=(self.population_size, 1))

                    candidates = (1.0 - lambdas) * explore_candidates + lambdas * exploit_candidates
                    candidates = np.clip(candidates, -1.0, 1.0)
                else:
                    candidates = operator.generate_population(
                        centre,
                        self.state,
                        self.population_size
                    )

                # Evaluate - vectorised for speed
                fitness = np.array([self.evaluate(v) for v in candidates])

                # Update memory
                self.update_memory(candidates, fitness)

                # Update best
                best_idx = np.argmin(fitness)
                prev_best = self.state.get("best_f")
                if prev_best is None:
                    prev_best = self.f_default_worst

                if self.state["best_f"] is None or fitness[best_idx] < self.state["best_f"]:
                    improved = True
                    self.state["best_v"] = candidates[best_idx]
                    self.state["best_f"] = fitness[best_idx]
                else:
                    improved = False

                # Update state features
                self.state_features.update_improvement_history(improved)

                # Update operator statistics
                improvement = prev_best - fitness[best_idx] if improved else 0.0
                operator.update_statistics(improved, improvement)

                # Track
                self.state["operator_counts"][operator.name] += 1
                self.state["current_operator"] = operator.name
                self.state["total_evals"] += self.population_size

                # Return stats
                mean_f = np.mean(fitness)
                operator_idx = self.operators.index(operator)

                return np.array([self.state["best_f"], mean_f, float(operator_idx)])

            population_node = nengo.Node(
                population_generator_func,
                size_in=len(self.operators),
                size_out=3,
                label="PopulationGenerator"
            )

            nengo.Connection(selected_operator_ens, population_node, synapse=0.05)

            # Probes
            self.state_features_probe = nengo.Probe(state_features_node, synapse=None)
            self.state_probe = nengo.Probe(state_ensemble, synapse=0.01)
            self.operator_probe = nengo.Probe(selected_operator_ens, synapse=0.01)
            self.stats_probe = nengo.Probe(population_node, synapse=None)

    def run(self, time: float, verbose: bool = True, use_dl: bool = False):
        """
        Run optimisation for specified time.

        Parameters
        ----------
        time : float
            Simulation time (seconds)
        verbose : bool
            Print progress information
        use_dl : bool
            Use NengoDL backend for GPU acceleration (requires nengo-dl and tensorflow)
        """
        progress_bar = False
        if verbose:
            print(f"Starting NEVO optimisation for {time}s...")
            print(f"Problem: {self.dimension}D")
            print(f"Population size: {self.population_size}")
            print(f"Operators: {[op.name for op in self.operators]}")
            print(f"Operator mode: {self.operator_mode}")
            print(f"Expected evaluations: ~{int(time / self.dt * self.population_size):,}")
            if use_dl:
                print(f"Backend: NengoDL (GPU accelerated)")
            print()
            progress_bar = True

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Start TD episode once per optimiser lifecycle.
        if not self._td_episode_started:
            self.bg_selector.begin_episode()
            self._td_episode_started = True

        # Run simulation
        if use_dl:
            try:
                import nengo_dl
                with nengo_dl.Simulator(self.model, dt=self.dt, progress_bar=progress_bar) as sim:
                    self.simulator = sim
                    self.state["simulator"] = sim
                    sim.run(time)
            except ImportError:
                print("Warning: nengo-dl not installed, falling back to standard Nengo")
                with nengo.Simulator(self.model, dt=self.dt, progress_bar=progress_bar) as sim:
                    self.simulator = sim
                    self.state["simulator"] = sim
                    sim.run(time)
        else:
            with nengo.Simulator(self.model, dt=self.dt, progress_bar=progress_bar) as sim:
                self.simulator = sim
                self.state["simulator"] = sim
                sim.run(time)

        if verbose:
            self.print_results()

    def print_results(self):
        """Print optimisation results."""
        print("=" * 70)
        print("OPTIMISATION RESULTS")
        print("=" * 70)
        print(f"\nTotal evaluations: {self.state['total_evals']:,}")
        print(f"Best fitness: {self.state['best_f']:.6e}")

        print(f"\nOperator usage:")
        total_calls = sum(self.state["operator_counts"].values())
        td_values = self.bg_selector.get_td_values()
        td_on = self.bg_selector.td_enabled

        for i, op in enumerate(self.operators):
            count = self.state["operator_counts"][op.name]
            pct = 100 * count / max(1, total_calls)
            weight = self.bg_selector.utilities[op.name].weight
            success_rate = op.success_count / max(1, op.usage_count)
            td_str = f"  td_V={td_values[i]:.4f}" if td_on else ""
            print(f"  {op.name:25s}: {count:6d} calls ({pct:5.1f}%)  "
                  f"[weight: {weight:.3f}, success: {success_rate:.1%}{td_str}]")

        if td_on:
            stats = self.bg_selector.get_td_statistics()
            lam = self.bg_selector.td_learner.lambda_coeff
            print(f"\nTD learning  λ={lam:.2f}  "
                  f"mean_δ={stats.get('mean_td_error', 0.0):.4f}  "
                  f"std_δ={stats.get('std_td_error', 0.0):.4f}")

        print(f"\nBest solution (v-space):")
        if self.state["best_v"] is not None:
            print(f"  {self.state['best_v']}")

        print(f"\nBest solution (original space):")
        if self.state["best_v"] is not None:
            x_best = trs2o(self.state["best_v"], self.lb, self.ub)
            print(f"  {x_best}")

    def get_best_solution(self) -> tuple:
        """
        Get best solution found.

        Returns
        -------
        x_best : np.ndarray
            Best solution in original space
        f_best : float
            Best fitness value
        """
        if self.state["best_v"] is None:
            return None, None

        x_best = trs2o(self.state["best_v"], self.lb, self.ub)
        return x_best, self.state["best_f"]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimisation statistics.

        Returns
        -------
        stats : Dict[str, Any]
            Dictionary containing various statistics
        """
        return {
            "total_evaluations": self.state["total_evals"],
            "best_fitness": self.state["best_f"],
            "operator_counts": self.state["operator_counts"].copy(),
            "operator_weights": self.bg_selector.get_utility_weights(),
            "operator_success_rates": {
                op.name: op.success_count / max(1, op.usage_count)
                for op in self.operators
            },
        }

    def reset_td_episode(self):
        """Start a fresh TD episode without rebuilding the network."""
        self.bg_selector.begin_episode()
        self._td_episode_started = True

    def set_td_lambda(self, lambda_coeff: float):
        """Pass-through TD(lambda) configuration."""
        self.bg_selector.set_td_lambda(lambda_coeff)

    def set_td_learning_rule(self, learning_rule: LearningRule):
        """Pass-through learning-rule swap for basal ganglia TD."""
        self.bg_selector.set_learning_rule(learning_rule)

    def set_td_value_model(self, value_model: ValueModel):
        """Pass-through value-model swap for basal ganglia TD."""
        self.bg_selector.set_value_model(value_model)
