"""
Temporal Difference (TD) Learning Algorithms
=============================================

This module implements modular TD learning variants for adaptive operator selection.
Supports TD(0), TD(λ), and pluggable learning rules and value models.
"""

import numpy as np
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod


class LearningRule(ABC):
    """
    Abstract base class for TD learning rules.

    A learning rule defines how TD errors are used to update value estimates.
    """

    @abstractmethod
    def compute_update(
        self, td_error: float, learning_rate: float, current_value: float, **kwargs
    ) -> float:
        """
        Compute value update from TD error.

        Parameters
        ----------
        td_error : float
            Temporal difference error
        learning_rate : float
            Learning rate α
        current_value : float
            Current value estimate
        **kwargs : dict
            Additional parameters specific to the rule

        Returns
        -------
        update : float
            Value increment (delta_V)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state."""
        pass


class SimpleTDRule(LearningRule):
    """
    Simple TD(0) update rule: ΔV = α * δ_t

    Standard TD learning update.
    """

    def compute_update(
        self, td_error: float, learning_rate: float, current_value: float, **kwargs
    ) -> float:
        """Direct proportional update to TD error."""
        return learning_rate * td_error

    def reset(self):
        """No internal state to reset."""
        pass


class DecayingTDRule(LearningRule):
    """
    Decaying TD update rule with eligibility traces.

    Allows different weighting schemes: constant, linear, or exponential decay.
    """

    def __init__(self, decay_type: str = "exponential", decay_rate: float = 0.9):
        """
        Parameters
        ----------
        decay_type : str
            "constant" (no decay), "linear", or "exponential"
        decay_rate : float
            Decay parameter (for exponential/linear)
        """
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.trace_history = []

    def compute_update(
        self,
        td_error: float,
        learning_rate: float,
        current_value: float,
        timestep: int = 0,
        **kwargs,
    ) -> float:
        """Update with decay applied based on history depth."""
        if self.decay_type == "constant":
            decay_factor = 1.0
        elif self.decay_type == "linear":
            decay_factor = max(0.0, 1.0 - (timestep * self.decay_rate))
        elif self.decay_type == "exponential":
            decay_factor = self.decay_rate**timestep
        else:
            decay_factor = 1.0

        return learning_rate * decay_factor * td_error

    def reset(self):
        """Clear trace history."""
        self.trace_history = []


class ConservativeTDRule(LearningRule):
    """
    Conservative TD update rule with value stability.

    Includes magnitude thresholding and clipping to prevent wild swings.
    """

    def __init__(self, stability_weight: float = 0.5):
        """
        Parameters
        ----------
        stability_weight : float
            How much to dampen updates (0=full update, 1=no update)
        """
        self.stability_weight = stability_weight

    def compute_update(
        self, td_error: float, learning_rate: float, current_value: float, **kwargs
    ) -> float:
        """Damped update with stability."""
        # Dampen the TD error by stability weight
        damped_error = td_error * (1.0 - self.stability_weight)

        # Clip the update to prevent wild swings
        update = learning_rate * damped_error
        max_update = learning_rate * 0.1 * (abs(current_value) + 1.0)
        update = np.clip(update, -max_update, max_update)

        return update

    def reset(self):
        """No internal state to reset."""
        pass


class AdaptiveTDRule(LearningRule):
    """
    Adaptive TD update rule with magnitude-dependent learning rate.

    Scales learning rate based on recent TD error magnitude.
    """

    def __init__(self, window_size: int = 10):
        """
        Parameters
        ----------
        window_size : int
            Number of recent TD errors to track for adaptation
        """
        self.window_size = window_size
        self.error_history = []

    def compute_update(
        self, td_error: float, learning_rate: float, current_value: float, **kwargs
    ) -> float:
        """Update with adaptive learning rate."""
        self.error_history.append(abs(td_error))
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        # Adaptive learning rate: smaller when errors are large
        mean_abs_error = np.mean(self.error_history) if self.error_history else 1.0
        adaptation_factor = 1.0 / (1.0 + mean_abs_error)

        return learning_rate * adaptation_factor * td_error

    def reset(self):
        """Clear error history."""
        self.error_history = []


class ValueModel(ABC):
    """
    Abstract base class for value function models.

    A value model stores and updates value estimates for each operator.
    """

    @abstractmethod
    def get_value(self, operator_idx: int) -> float:
        """Get current value estimate for operator."""
        pass

    @abstractmethod
    def set_value(self, operator_idx: int, value: float):
        """Set value estimate for operator."""
        pass

    @abstractmethod
    def update(self, operator_idx: int, delta: float):
        """Increment value by delta."""
        pass

    @abstractmethod
    def reset(self):
        """Reset all values."""
        pass

    @abstractmethod
    def get_values_array(self) -> np.ndarray:
        """Get all values as array."""
        pass


class LinearValueModel(ValueModel):
    """
    Simple linear value model: V(s, a) = w_a

    One value per operator, no state dependence.
    """

    def __init__(self, n_operators: int, initial_value: float = 0.5):
        """
        Parameters
        ----------
        n_operators : int
            Number of operators
        initial_value : float
            Initial value for all operators
        """
        self.n_operators = n_operators
        self.values = np.full(n_operators, initial_value, dtype=np.float64)

    def get_value(self, operator_idx: int) -> float:
        """Get value for operator."""
        return float(self.values[operator_idx])

    def set_value(self, operator_idx: int, value: float):
        """Set value for operator."""
        self.values[operator_idx] = float(value)
        self.values[operator_idx] = np.clip(self.values[operator_idx], 0.1, 5.0)

    def update(self, operator_idx: int, delta: float):
        """Update value by delta."""
        self.values[operator_idx] += delta
        self.values[operator_idx] = np.clip(self.values[operator_idx], 0.1, 5.0)

    def reset(self):
        """Reset all values to initial."""
        self.values.fill(0.5)

    def get_values_array(self) -> np.ndarray:
        """Get copy of values array."""
        return self.values.copy()


class BoundedValueModel(ValueModel):
    """
    Value model with learnable bounds for stability.

    Maintains per-operator lower/upper bounds on values.
    """

    def __init__(
        self,
        n_operators: int,
        initial_value: float = 0.5,
        min_bound: float = 0.1,
        max_bound: float = 5.0,
        adapt_bounds: bool = True,
    ):
        """
        Parameters
        ----------
        n_operators : int
            Number of operators
        initial_value : float
            Initial value for all operators
        min_bound : float
            Minimum value bound
        max_bound : float
            Maximum value bound
        adapt_bounds : bool
            Whether bounds adapt over time
        """
        self.n_operators = n_operators
        self.values = np.full(n_operators, initial_value, dtype=np.float64)
        self.min_bounds = np.full(n_operators, min_bound, dtype=np.float64)
        self.max_bounds = np.full(n_operators, max_bound, dtype=np.float64)
        self.adapt_bounds = adapt_bounds
        self.min_history = [[] for _ in range(n_operators)]
        self.max_history = [[] for _ in range(n_operators)]

    def get_value(self, operator_idx: int) -> float:
        """Get value for operator."""
        return float(self.values[operator_idx])

    def set_value(self, operator_idx: int, value: float):
        """Set value for operator."""
        self.values[operator_idx] = np.clip(
            value, self.min_bounds[operator_idx], self.max_bounds[operator_idx]
        )

    def update(self, operator_idx: int, delta: float):
        """Update value by delta with bounds checking."""
        new_value = self.values[operator_idx] + delta
        self.values[operator_idx] = np.clip(
            new_value, self.min_bounds[operator_idx], self.max_bounds[operator_idx]
        )

    def adapt_bounds(self, operator_idx: int, window_size: int = 20):
        """Adapt bounds based on value history (called periodically)."""
        if not self.adapt_bounds or len(self.min_history[operator_idx]) < 5:
            return

        recent_values = self.values[operator_idx : operator_idx + 1]

        # Gradually widen bounds if values approach limits
        min_val = np.percentile(recent_values, 10)
        max_val = np.percentile(recent_values, 90)

        margin = 0.2 * (max_val - min_val + 0.1)
        new_min = max(0.1, min_val - margin)
        new_max = min(10.0, max_val + margin)

        # Smooth transition to new bounds
        self.min_bounds[operator_idx] = (
            0.9 * self.min_bounds[operator_idx] + 0.1 * new_min
        )
        self.max_bounds[operator_idx] = (
            0.9 * self.max_bounds[operator_idx] + 0.1 * new_max
        )

    def reset(self):
        """Reset all values to initial."""
        self.values.fill(0.5)
        self.min_history = [[] for _ in range(self.n_operators)]
        self.max_history = [[] for _ in range(self.n_operators)]

    def get_values_array(self) -> np.ndarray:
        """Get copy of values array."""
        return self.values.copy()


class EligibilityTraceManager:
    """
    Manages eligibility traces for TD(λ) learning.

    Maintains traces that decay over time, enabling multi-step credit assignment.
    """

    def __init__(
        self, n_operators: int, lambda_coeff: float = 0.9, trace_decay: float = 0.99
    ):
        """
        Parameters
        ----------
        n_operators : int
            Number of operators
        lambda_coeff : float
            λ coefficient for trace decay (0.0 = TD(0), 1.0 = MC)
        trace_decay : float
            Per-timestep decay of all traces
        """
        self.n_operators = n_operators
        self.lambda_coeff = lambda_coeff
        self.trace_decay = trace_decay
        self.traces = np.zeros(n_operators, dtype=np.float64)
        self.trace_history = []

    def update_trace(self, operator_idx: int, increment: float = 1.0):
        """
        Update trace for visited operator and decay all traces.

        Parameters
        ----------
        operator_idx : int
            Index of visited operator
        increment : float
            Increment to add to trace
        """
        # Decay all traces
        self.traces *= self.trace_decay * self.lambda_coeff

        # Increment trace for current operator
        self.traces[operator_idx] += increment

        # Track history
        self.trace_history.append(self.traces.copy())

    def get_traces(self) -> np.ndarray:
        """Get current trace vector."""
        return self.traces.copy()

    def reset(self):
        """Reset all traces."""
        self.traces.fill(0.0)
        self.trace_history = []

    def set_lambda(self, lambda_coeff: float):
        """
        Adjust λ coefficient dynamically.

        Parameters
        ----------
        lambda_coeff : float
            New λ value (0.0 to 1.0)
        """
        self.lambda_coeff = np.clip(lambda_coeff, 0.0, 1.0)


class TemporalDifferenceLearner:
    """
    Temporal Difference learner with pluggable rules and value models.

    Implements TD(0) and TD(λ) for operator value learning.
    """

    def __init__(
        self,
        n_operators: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        lambda_coeff: float = 0.0,
        learning_rule: Optional[LearningRule] = None,
        value_model: Optional[ValueModel] = None,
    ):
        """
        Parameters
        ----------
        n_operators : int
            Number of operators
        learning_rate : float
            Learning rate α
        gamma : float
            Discount factor
        lambda_coeff : float
            λ for trace decay (0.0 = TD(0), 0.9 = TD(0.9), 1.0 = Monte Carlo)
        learning_rule : LearningRule, optional
            Learning rule to use (default: SimpleTDRule)
        value_model : ValueModel, optional
            Value function model (default: LinearValueModel)
        """
        self.n_operators = n_operators
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff

        # Set default learning rule and value model if not provided
        self.learning_rule = learning_rule or SimpleTDRule()
        self.value_model = value_model or LinearValueModel(n_operators)

        # Initialize eligibility traces for TD(λ)
        self.trace_manager = EligibilityTraceManager(n_operators, lambda_coeff)

        # Tracking
        self.last_state_value = None
        self.timestep = 0
        self.td_error_history = []

    def set_learning_rate(self, learning_rate: float):
        """Update learning rate."""
        self.learning_rate = learning_rate

    def set_lambda(self, lambda_coeff: float):
        """
        Update λ coefficient (switches between TD(0) and TD(λ)).

        Parameters
        ----------
        lambda_coeff : float
            New λ value (0.0 to 1.0)
        """
        self.lambda_coeff = np.clip(lambda_coeff, 0.0, 1.0)
        self.trace_manager.set_lambda(self.lambda_coeff)

    def begin_episode(self):
        """Reset traces and timestep for a new episode. Value estimates are preserved."""
        self.trace_manager.reset()
        self.last_state_value = None
        self.timestep = 0
        self.td_error_history = []

    def update(
        self,
        operator_idx: int,
        reward: float,
        next_state_value: float = 0.0,
        is_terminal: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform TD update for visited operator.

        Parameters
        ----------
        operator_idx : int
            Index of operator to update
        reward : float
            Immediate reward signal
        next_state_value : float
            Value of next state (for bootstrapping)
        is_terminal : bool
            Whether this is terminal state

        Returns
        -------
        update_info : Dict
            Information about the update (TD error, magnitude, etc.)
        """
        # Get current value
        current_value = self.value_model.get_value(operator_idx)

        # Compute temporal difference error
        if is_terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_state_value

        td_error = td_target - current_value

        # Update eligibility traces
        self.trace_manager.update_trace(operator_idx, increment=1.0)

        # Get traces for all operators
        traces = self.trace_manager.get_traces()

        # Compute updates using learning rule
        updates = {}
        for i in range(self.n_operators):
            if traces[i] > 1e-8:  # Only update if trace is non-negligible
                delta = self.learning_rule.compute_update(
                    td_error,
                    self.learning_rate,
                    self.value_model.get_value(i),
                    timestep=self.timestep,
                    trace=traces[i],
                )
                self.value_model.update(i, delta * traces[i])
                updates[i] = delta * traces[i]

        # Track TD error
        self.td_error_history.append(td_error)
        if len(self.td_error_history) > 1000:
            self.td_error_history.pop(0)

        # Increment timestep
        self.timestep += 1

        return {
            "td_error": td_error,
            "td_target": td_target,
            "current_value": current_value,
            "updates": updates,
            "traces": traces.copy(),
            "timestep": self.timestep,
        }

    def get_values(self) -> np.ndarray:
        """Get current value estimates for all operators."""
        return self.value_model.get_values_array()

    def get_value(self, operator_idx: int) -> float:
        """Get value for specific operator."""
        return self.value_model.get_value(operator_idx)

    def reset_values(self):
        """Reset all value estimates."""
        self.value_model.reset()
        self.learning_rule.reset()
        self.trace_manager.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.td_error_history:
            return {"mean_td_error": 0.0, "std_td_error": 0.0}

        recent_errors = self.td_error_history[-100:]
        return {
            "mean_td_error": float(np.mean(recent_errors)),
            "std_td_error": float(np.std(recent_errors)),
            "max_td_error": float(np.max(recent_errors)),
            "min_td_error": float(np.min(recent_errors)),
        }
