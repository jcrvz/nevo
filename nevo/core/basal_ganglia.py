"""
Basal Ganglia Operator Selection
=================================

This module implements neuromorphic operator selection using
basal ganglia circuits in Nengo.
"""

import nengo
import numpy as np
from typing import List, Dict, Any, Callable
from nevo.operators.base import Operator


class UtilityFunction:
    """
    Defines utility function for an operator.

    Maps state features → scalar utility value.
    Higher utility = more suitable for current state.
    """

    def __init__(
        self,
        name: str,
        function: Callable[[np.ndarray], float],
        initial_weight: float = 1.0
    ):
        """
        Parameters
        ----------
        name : str
            Operator name
        function : Callable
            Function mapping state features [diversity, improvement, convergence] → utility
        initial_weight : float
            Initial adaptive weight
        """
        self.name = name
        self.function = function
        self.weight = initial_weight

    def compute(self, features: np.ndarray) -> float:
        """
        Compute weighted utility.

        Parameters
        ----------
        features : np.ndarray
            State features [diversity, improvement_rate, convergence]

        Returns
        -------
        utility : float
            Weighted utility value
        """
        base_utility = self.function(features)
        return base_utility * self.weight

    def update_weight(self, reward: float, learning_rate: float = 0.1):
        """
        Update weight based on operator performance.

        Parameters
        ----------
        reward : float
            Performance reward (positive = good, negative = bad)
        learning_rate : float
            Learning rate for weight update
        """
        self.weight += learning_rate * reward
        self.weight = np.clip(self.weight, 0.1, 5.0)


# Standard utility functions for common operators
def utility_levy_flight(x: np.ndarray) -> float:
    """
    LevyFlight utility: high when stuck and not converged.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when not improving and not converged (need global exploration)
    return (1.0 - improvement) * 0.5 + (1.0 - convergence) * 0.3


def utility_differential_evolution(x: np.ndarray) -> float:
    """
    DifferentialEvolution utility: high when diversity exists.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when memory has diversity
    return diversity * 0.8 + (1.0 - convergence) * 0.3 + 0.2


def utility_particle_swarm(x: np.ndarray) -> float:
    """
    ParticleSwarm utility: high when improving and converging.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use during exploitation phase
    return improvement * 0.8 + convergence * 0.4 + 0.2


def utility_spiral(x: np.ndarray) -> float:
    """
    SpiralOptimisation utility: high when highly converged.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use for fine-tuning
    return convergence * 0.8 + improvement * 0.4 + 0.1


# Default utility function mapping
DEFAULT_UTILITY_FUNCTIONS = {
    "LevyFlight": utility_levy_flight,
    "DifferentialEvolution": utility_differential_evolution,
    "ParticleSwarm": utility_particle_swarm,
    "SpiralOptimisation": utility_spiral,
}


class BasalGangliaSelector:
    """
    Basal ganglia-based operator selection network.

    Implements winner-take-all selection of operators based on
    state-dependent utility functions.
    """

    def __init__(
        self,
        operators: List[Operator],
        utility_functions: Dict[str, Callable] = None,
        neurons_per_ensemble: int = 100,
        epsilon: float = 0.1,
        learning_rate: float = 0.4,
    ):
        """
        Parameters
        ----------
        operators : List[Operator]
            List of available operators
        utility_functions : Dict[str, Callable], optional
            Custom utility functions (uses defaults if None)
        neurons_per_ensemble : int
            Neurons per ensemble in basal ganglia
        epsilon : float
            Epsilon-greedy exploration rate (0.0-1.0)
        learning_rate : float
            Learning rate for utility weight adaptation
        """
        self.operators = operators
        self.n_operators = len(operators)
        self.neurons_per_ensemble = neurons_per_ensemble
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # Initialize utility functions
        if utility_functions is None:
            utility_functions = DEFAULT_UTILITY_FUNCTIONS

        self.utilities = {}
        for op in operators:
            if op.name in utility_functions:
                self.utilities[op.name] = UtilityFunction(
                    op.name,
                    utility_functions[op.name]
                )
            else:
                # Default neutral utility
                self.utilities[op.name] = UtilityFunction(
                    op.name,
                    lambda x: 0.5
                )

        # Tracking for learning
        self.last_operator = None
        self.last_best_fitness = None

    def build_network(
        self,
        model: nengo.Network,
        state_ensemble: nengo.Ensemble
    ) -> nengo.Ensemble:
        """
        Build basal ganglia selection network.

        Parameters
        ----------
        model : nengo.Network
            Parent Nengo network
        state_ensemble : nengo.Ensemble
            State feature ensemble (3D: diversity, improvement, convergence)

        Returns
        -------
        selected_operator_ens : nengo.Ensemble
            One-hot encoding of selected operator
        """
        with model:
            # Utility ensembles (one per operator)
            utility_ensembles = []

            for op in self.operators:
                utility_ens = nengo.Ensemble(
                    n_neurons=self.neurons_per_ensemble,
                    dimensions=1,
                    radius=3.0,
                    label=f"Utility_{op.name}"
                )

                # Connect state to utility via utility function
                def make_utility_func(op_name):
                    def utility_wrapper(x):
                        return self.utilities[op_name].compute(x)
                    return utility_wrapper

                nengo.Connection(
                    state_ensemble,
                    utility_ens,
                    function=make_utility_func(op.name),
                    synapse=0.01
                )

                utility_ensembles.append(utility_ens)

            # Basal ganglia (winner-take-all)
            bg = nengo.networks.BasalGanglia(
                self.n_operators,
                n_neurons_per_ensemble=self.neurons_per_ensemble
            )

            # Connect utilities to basal ganglia
            for i, utility_ens in enumerate(utility_ensembles):
                nengo.Connection(utility_ens, bg.input[i], synapse=None)

            # Thalamus (action gating)
            thalamus = nengo.networks.Thalamus(
                self.n_operators,
                n_neurons_per_ensemble=self.neurons_per_ensemble
            )
            nengo.Connection(bg.output, thalamus.input, synapse=None)

            # Selected operator ensemble (one-hot)
            selected_operator_ens = nengo.Ensemble(
                n_neurons=self.neurons_per_ensemble * self.n_operators,
                dimensions=self.n_operators,
                radius=1.5,
                label="SelectedOperator"
            )

            nengo.Connection(thalamus.output, selected_operator_ens, synapse=0.01)

            return selected_operator_ens

    def select_operator(
        self,
        operator_selection: np.ndarray,
        current_best_fitness: float
    ) -> Operator:
        """
        Select operator based on basal ganglia output with epsilon-greedy.

        Parameters
        ----------
        operator_selection : np.ndarray
            One-hot encoded operator selection from basal ganglia
        current_best_fitness : float
            Current best fitness value

        Returns
        -------
        operator : Operator
            Selected operator
        """
        # Update utility weights based on last operator's performance
        if self.last_operator is not None and self.last_best_fitness is not None:
            if current_best_fitness < self.last_best_fitness:
                # Improvement! Positive reward
                reward = (self.last_best_fitness - current_best_fitness) / (
                    abs(self.last_best_fitness) + 1e-12
                )
                self.utilities[self.last_operator].update_weight(
                    reward,
                    self.learning_rate
                )
            else:
                # No improvement, small penalty
                self.utilities[self.last_operator].update_weight(
                    -0.01,
                    self.learning_rate
                )

        # Epsilon-greedy selection
        if np.random.rand() < self.epsilon:
            # Random exploration
            operator_idx = np.random.randint(self.n_operators)
        else:
            # Use basal ganglia selection
            operator_idx = np.argmax(operator_selection)

        selected_operator = self.operators[operator_idx]

        # Track for next iteration
        self.last_operator = selected_operator.name
        self.last_best_fitness = current_best_fitness

        return selected_operator

    def get_utility_weights(self) -> Dict[str, float]:
        """
        Get current utility weights.

        Returns
        -------
        weights : Dict[str, float]
            Mapping of operator names to their current weights
        """
        return {name: util.weight for name, util in self.utilities.items()}

