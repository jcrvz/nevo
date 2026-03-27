"""
Basal Ganglia Operator Selection
=================================

This module implements neuromorphic operator selection using
basal ganglia circuits in Nengo, with pluggable Temporal Difference
learning for adaptive value estimation.
"""

import nengo
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from nevo.operators.base import Operator
from nevo.core.td_learning import (
    TemporalDifferenceLearner,
    LearningRule,
    ValueModel,
)


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
        initial_weight: float = 1.0,
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

    def update_weight(self, reward: float, lr: float = 0.1):
        """
        Update weight based on operator performance.

        Parameters
        ----------
        reward : float
            Performance reward (positive = good, negative = bad)
        lr : float
            Learning rate for weight update
        """
        self.weight += lr * reward
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


# ---
# Utility functions for additional operators (inspired by CUSTOMHyS)
# ---

def utility_random_search(x: np.ndarray) -> float:
    """
    RandomSearch utility: high when stuck, baseline exploration.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when nothing else works, low improvement
    return (1.0 - improvement) * 0.4 + (1.0 - convergence) * 0.2 + 0.1


def utility_local_random_walk(x: np.ndarray) -> float:
    """
    LocalRandomWalk utility: high when converging, need local refinement.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use for local refinement when near optimal
    return convergence * 0.6 + improvement * 0.3 + 0.1


def utility_gravitational_search(x: np.ndarray) -> float:
    """
    GravitationalSearch utility: high when diversity exists, need directed exploration.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when memory has diversity and need attraction-based search
    return diversity * 0.6 + (1.0 - convergence) * 0.3 + 0.2


def utility_firefly(x: np.ndarray) -> float:
    """
    FireflyAlgorithm utility: high when moderate convergence, need attraction.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use for attraction-based exploration with some convergence
    return diversity * 0.4 + convergence * 0.3 + improvement * 0.2 + 0.1


def utility_central_force(x: np.ndarray) -> float:
    """
    CentralForce utility: high when need strong directional bias.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when converging but need global attraction
    return convergence * 0.4 + (1.0 - diversity) * 0.3 + improvement * 0.2


def utility_genetic_crossover(x: np.ndarray) -> float:
    """
    GeneticCrossover utility: high when diversity exists, want recombination.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when memory has diversity to combine features
    return diversity * 0.7 + (1.0 - convergence) * 0.2 + 0.2


def utility_genetic_mutation(x: np.ndarray) -> float:
    """
    GeneticMutation utility: high when converging too fast, need diversity.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when converging and low diversity
    return (1.0 - diversity) * 0.5 + convergence * 0.3 + 0.1


def utility_simulated_annealing(x: np.ndarray) -> float:
    """
    SimulatedAnnealing utility: high when need controlled exploitation.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use for exploitation with adaptive randomness
    return convergence * 0.5 + improvement * 0.4 + 0.2


def utility_harmony_search(x: np.ndarray) -> float:
    """
    HarmonySearch utility: high when need memory-guided search.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use for balanced memory-guided exploitation
    return diversity * 0.3 + convergence * 0.4 + improvement * 0.3 + 0.1


def utility_tabu_search(x: np.ndarray) -> float:
    """
    TabuSearch utility: high when stuck in local optima.

    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Use when stuck and need to escape visited regions
    return (1.0 - improvement) * 0.5 + convergence * 0.3 + 0.1


def utility_neuromorphic_exploration(x: np.ndarray) -> float:
    """
    Neuromorphic exploration utility: prefer when progress is low or diversity drops.
    """
    diversity, improvement, convergence = x
    return (
        (1.0 - improvement) * 0.5 + (1.0 - diversity) * 0.3 + (1.0 - convergence) * 0.2
    )


def utility_neuromorphic_exploitation(x: np.ndarray) -> float:
    """
    Neuromorphic exploitation utility: prefer when converged and still improving.
    """
    diversity, improvement, convergence = x
    return convergence * 0.6 + improvement * 0.3 + (1.0 - diversity) * 0.1 + 0.05


# Default utility function mapping
DEFAULT_UTILITY_FUNCTIONS = {
    # Traditional candidate generators
    "LevyFlight":               utility_levy_flight,
    "DifferentialEvolution":    utility_differential_evolution,
    "ParticleSwarm":            utility_particle_swarm,
    "SpiralOptimisation":       utility_spiral,
    "RandomSearch":             utility_random_search,
    "GravitationalSearch":      utility_gravitational_search,
    "FireflyAlgorithm":         utility_firefly,
    "CentralForce":             utility_central_force,
    "GeneticCrossover":         utility_genetic_crossover,
    "GeneticMutation":          utility_genetic_mutation,
    "LocalRandomWalk":          utility_local_random_walk,
    "SimulatedAnnealing":       utility_simulated_annealing,
    "TabuSearch":               utility_tabu_search,
    # Neuromorphic candidate generators
    "NeuromorphicExplorationEnsemble":  utility_neuromorphic_exploration,
    "NeuromorphicExploitationEnsemble": utility_neuromorphic_exploitation,
}


class BasalGangliaSelector:
    """
    Basal ganglia-based operator selection network with modular TD learning.

    Implements Winner-Take-All (WTA) selection of operators based on state-dependent utility functions and learned value
    estimates via Temporal Difference learning (TD(0) or TD(λ)).
    """

    def __init__(
        self,
        operators: List[Operator],
        utility_functions: Dict[str, Callable] = None,
        neurons_per_ensemble: int = 100,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        lambda_coeff: float = 0.0,
        learning_rule: Optional[LearningRule] = None,
        value_model: Optional[ValueModel] = None,
        td_enabled: bool = True,
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
            TD learning rate α
        gamma : float
            Discount factor for upcoming rewards
        lambda_coeff : float
            λ parameter (0.0=TD(0), 1.0=Monte Carlo)
        learning_rule : LearningRule, optional
            Pluggable learning rule (default: SimpleTDRule)
        value_model : ValueModel, optional
            Pluggable value model (default: LinearValueModel)
        td_enabled : bool
            Enable TD learning (vs. basic utility weight adaptation)
        """
        self.operators = operators
        self.n_operators = len(operators)
        self.neurons_per_ensemble = neurons_per_ensemble
        self.epsilon = epsilon
        self.td_enabled = td_enabled

        # Initialise utility functions
        if utility_functions is None:
            utility_functions = DEFAULT_UTILITY_FUNCTIONS

        self.utilities = {}
        for op in operators:
            if op.name in utility_functions:
                self.utilities[op.name] = UtilityFunction(
                    op.name, utility_functions[op.name]
                )
            else:
                # Default neutral utility
                self.utilities[op.name] = UtilityFunction(op.name, lambda x: 0.5)

        # Initialise TD learner if enabled
        if td_enabled:
            self.td_learner = TemporalDifferenceLearner(
                n_operators=self.n_operators,
                learning_rate=learning_rate,
                gamma=gamma,
                lambda_coeff=lambda_coeff,
                learning_rule=learning_rule,
                value_model=value_model,
            )
        else:
            self.td_learner = None

        # Tracking for learning
        self.last_operator_idx = None
        self.last_operator = None
        self.last_best_fitness = None
        self.episode_count = 0

    def build_network(
        self, model: nengo.Network, state_ensemble: nengo.Ensemble
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
                    label=f"Utility_{op.name}",
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
                    synapse=0.01,
                )

                utility_ensembles.append(utility_ens)

            # Basal ganglia (winner-take-all)
            bg = nengo.networks.BasalGanglia(
                self.n_operators, n_neurons_per_ensemble=self.neurons_per_ensemble
            )

            # Connect utilities to basal ganglia
            for i, utility_ens in enumerate(utility_ensembles):
                nengo.Connection(utility_ens, bg.input[i], synapse=None)

            # Thalamus (action gating)
            thalamus = nengo.networks.Thalamus(
                self.n_operators, n_neurons_per_ensemble=self.neurons_per_ensemble
            )
            nengo.Connection(bg.output, thalamus.input, synapse=None)

            # Selected operator ensemble (one-hot)
            selected_operator_ens = nengo.Ensemble(
                n_neurons=self.neurons_per_ensemble * self.n_operators,
                dimensions=self.n_operators,
                radius=1.5,
                label="SelectedOperator",
            )

            nengo.Connection(thalamus.output, selected_operator_ens, synapse=0.01)

            return selected_operator_ens

    def select_operator(
        self,
        operator_selection: np.ndarray,
        current_best_fitness: float,
    ) -> Operator:
        """
        Select operator using Nengo basal ganglia output + epsilon-greedy policy.

        When td_enabled=True, TD(0)/TD(λ) value estimates bias the utility weights that feed the Nengo BG network, so
        learned knowledge flows back through the neuromorphic circuit rather than bypassing it.

        Decision flow
        -------------
        1. Compute reward from fitness improvement (after last operator executed).
        2. Update TD values for the last operator using reward + bootstrap.
        3. Add TD value bias to utility weights (scales Nengo BG input).
        4. Read Nengo thalamus output (operator_selection) as action scores.
        5. Epsilon-greedy: random with prob ε, else argmax of BG output.

        Parameters
        ----------
        operator_selection : np.ndarray
            Thalamus output from Nengo BG network (n_operators,)
        current_best_fitness : float
            Current best fitness value

        Returns
        -------
        operator : Operator
            Selected operator
        """
        #
        # Step 1 – compute reward for the operator that just ran
        # ------------------------------------------------------------------
        reward = 0.0
        if self.last_operator_idx is not None and self.last_best_fitness is not None:
            if current_best_fitness < self.last_best_fitness:
                reward = (self.last_best_fitness - current_best_fitness) / (
                    abs(self.last_best_fitness) + 1e-12
                )
            else:
                reward = -0.01

            # Step 2 – TD update for the last operator
            #   Bootstrap target: r + γ * max_j V(j)  [snapshot BEFORE update]
            # ------------------------------------------------------------------
            if self.td_enabled and self.td_learner is not None:
                v_snapshot = self.td_learner.get_values()  # V before update
                next_state_value = float(np.max(v_snapshot))
                self.td_learner.update(
                    self.last_operator_idx,
                    reward,
                    next_state_value=next_state_value,
                    is_terminal=False,
                )

            # Also update utility weight (used by Nengo BG utility functions)
            self.utilities[self.last_operator.name].update_weight(reward, lr=0.1)

        # Step 3 – let Nengo BG output drive action scores
        #   If TD is on, TD values act as an *additive bias* to the BG signal
        #   (normalised so neither dominates completely).
        # ------------------------------------------------------------------
        bg_signal = np.asarray(operator_selection, dtype=float)

        if self.td_enabled and self.td_learner is not None:
            td_values = self.td_learner.get_values()  # updated values

            # Normalise both signals to [0, 1] before mixing so scales match
            bg_norm  = bg_signal - bg_signal.min()
            bg_range = bg_norm.max() + 1e-8
            bg_norm /= bg_range

            td_norm  = td_values - td_values.min()
            td_range = td_norm.max() + 1e-8
            td_norm /= td_range

            # BG keeps primary authority; TD is a mild bias (20 %)
            combined = 0.8 * bg_norm + 0.2 * td_norm
        else:
            combined = bg_signal

        # Step 4 – epsilon-greedy selection over combined scores
        # ------------------------------------------------------------------
        if np.random.rand() < self.epsilon:
            operator_idx = np.random.randint(self.n_operators)
        else:
            if np.allclose(combined, combined[0], atol=1e-6):
                operator_idx = np.random.randint(self.n_operators)
            else:
                operator_idx = int(np.argmax(combined))

        selected_operator = self.operators[operator_idx]

        # Step 5 – store state for next call
        # ------------------------------------------------------------------
        self.last_operator_idx = operator_idx
        self.last_operator = selected_operator
        self.last_best_fitness = current_best_fitness

        return selected_operator

    def begin_episode(self):
        """
        Reset for new episode.

        Call this at the start of optimisation to initialise TD learning.
        """
        self.last_operator_idx  = None
        self.last_operator      = None
        self.last_best_fitness  = None
        self.episode_count     += 1
        if self.td_enabled and self.td_learner is not None:
            self.td_learner.begin_episode()

    def end_episode(self):
        """
        Finalise episode learning.

        Call this at the end of optimisation run.
        """
        pass

    def set_td_lambda(self, lambda_coeff: float):
        """
        Adjust TD(λ) parameter dynamically.

        Parameters
        ----------
        lambda_coeff : float
            New λ value (0.0 = TD(0), 1.0 = Monte Carlo)
        """
        if self.td_enabled and self.td_learner is not None:
            self.td_learner.set_lambda(lambda_coeff)

    def set_learning_rule(self, learning_rule: LearningRule):
        """
        Replace learning rule on the fly.

        Parameters
        ----------
        learning_rule : LearningRule
            New learning rule instance
        """
        if self.td_enabled and self.td_learner is not None:
            self.td_learner.learning_rule = learning_rule

    def set_value_model(self, value_model: ValueModel):
        """
        Replace value model on the fly.

        Parameters
        ----------
        value_model : ValueModel
            New value model instance
        """
        if self.td_enabled and self.td_learner is not None:
            self.td_learner.value_model = value_model

    def get_utility_weights(self) -> Dict[str, float]:
        """
        Get current utility weights.

        Returns
        -------
        weights : Dict[str, float]
            Mapping of operator names to their current weights
        """
        return {name: util.weight for name, util in self.utilities.items()}

    def get_td_values(self) -> np.ndarray:
        """
        Get current TD-learned values for all operators.

        Returns
        -------
        values : np.ndarray
            Value estimates (only if TD learning enabled)
        """
        if self.td_enabled and self.td_learner is not None:
            return self.td_learner.get_values()
        else:
            return np.full(self.n_operators, 0.5)

    def get_td_statistics(self) -> Dict[str, Any]:
        """
        Get TD learning statistics.

        Returns
        -------
        stats : Dict[str, Any]
            Statistics about TD learning process
        """
        if self.td_enabled and self.td_learner is not None:
            return self.td_learner.get_statistics()
        else:
            return {}

    def reset_td_learning(self):
        """Reset all TD value estimates."""
        if self.td_enabled and self.td_learner is not None:
            self.td_learner.reset_values()
