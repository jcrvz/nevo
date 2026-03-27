# Modular Temporal Difference (TD) Learning System

## Overview

This module adds a modular, pluggable TD learning system to NEVO while preserving the Nengo neuromorphic implementation. The system supports TD(0), TD(λ), and allows dynamic switching between different learning rules and value function models.

### Key Features

- TD(0) and TD(λ) learning variants.
- Pluggable learning rules: Simple, Decaying, Conservative, Adaptive.
- Pluggable value models: Linear, Bounded.
- Eligibility traces for multi-step credit assignment.
- Nengo neuromorphic networks preserved. TD learning operates on top.
- Dynamic switching. Change rules and models at runtime.
- Backward compatible. Existing code still works.

---

## Architecture

### Components

#### 1. **Learning Rules** (`LearningRule` ABC)
Define how TD errors are converted to value updates.

**Available Rules:**
- `SimpleTDRule`: Standard TD(0) update: ΔV = α * δ_t
- `DecayingTDRule`: Decaying updates (exponential, linear, or constant)
- `ConservativeTDRule`: Dampened updates for stability
- `AdaptiveTDRule`: Magnitude-dependent learning rates

#### 2. **Value Models** (`ValueModel` ABC)
Store and manage value function estimates.

**Available Models:**
- `LinearValueModel`: Simple one-per-operator values
- `BoundedValueModel`: Values with adaptive bounds and stability

#### 3. **Eligibility Traces** (`EligibilityTraceManager`)
Manages credit assignment across multiple operators using traces that decay over time.

Key for TD(λ) implementation.

#### 4. **Temporal Difference Learner** (`TemporalDifferenceLearner`)
Core TD(0)/TD(λ) algorithm integrating rules, models, and traces.

#### 5. **Basal Ganglia Selector** (Modified)
Now uses TD learning alongside Nengo neuromorphic networks.

---

## Usage

### Basic TD(0) Learning

```python
from nevo.core.basal_ganglia import BasalGangliaSelector
from nevo.core.td_learning import SimpleTDRule, LinearValueModel
from nevo.operators.standard import LevyFlight, ParticleSwarm

# Create operators
operators = [LevyFlight(), ParticleSwarm()]

# Create selector with TD(0) learning
selector = BasalGangliaSelector(
    operators=operators,
    td_enabled=True,
    lambda_coeff=0.0,      # TD(0)
    learning_rate=0.1,
    gamma=0.99,            # Discount factor
    learning_rule=SimpleTDRule(),
    value_model=LinearValueModel(len(operators)),
)

# Start optimization episode
selector.begin_episode()

# Select operators during optimization
best_fitness = 10.0
operator_selection = np.array([1.0, 0.5])  # From basal ganglia

op = selector.select_operator(operator_selection, best_fitness)
# ... use operator, generate candidates, evaluate, get reward ...

# Next selection (TD learning updates values automatically)
better_fitness = 8.0
op = selector.select_operator(operator_selection, better_fitness)
```

### TD(λ) Learning with Eligibility Traces

```python
# TD(λ) with λ=0.9
selector = BasalGangliaSelector(
    operators=operators,
    td_enabled=True,
    lambda_coeff=0.9,      # TD(0.9) - multi-step credit
    learning_rate=0.1,
    gamma=0.99,
)

# With eligibility traces, updates propagate to previously visited operators
# Example: visiting operators [0, 1, 0, 2] assigns credit proportional to recency
```

### Dynamic Learning Rule Switching

```python
from nevo.core.td_learning import ConservativeTDRule, AdaptiveTDRule

# Start with simple rule
selector = BasalGangliaSelector(
    operators=operators,
    learning_rule=SimpleTDRule(),
)

# Switch to conservative rule for stability
selector.set_learning_rule(ConservativeTDRule(stability_weight=0.5))

# Switch to adaptive rule
selector.set_learning_rule(AdaptiveTDRule(window_size=20))
```

### Dynamic Value Model Switching

```python
from nevo.core.td_learning import BoundedValueModel

# Start with linear model
selector = BasalGangliaSelector(
    operators=operators,
    value_model=LinearValueModel(len(operators)),
)

# Switch to bounded model for stability
bounded_model = BoundedValueModel(
    n_operators=len(operators),
    min_bound=0.2,
    max_bound=3.0,
    adapt_bounds=True,
)
selector.set_value_model(bounded_model)
```

### Accessing Learning Information

```python
# Get current TD-learned values
td_values = selector.get_td_values()
print(f"Operator values: {td_values}")

# Get TD learning statistics
stats = selector.get_td_statistics()
print(f"Mean TD error: {stats['mean_td_error']:.4f}")
print(f"Std TD error: {stats['std_td_error']:.4f}")

# Get utility weights (state-dependent)
utility_weights = selector.get_utility_weights()

# Reset TD learning
selector.reset_td_learning()
```

### Adjusting TD Parameters Dynamically

```python
# Change learning rate
selector.td_learner.set_learning_rate(0.05)

# Change λ coefficient (interpolate between TD(0) and MC)
selector.set_td_lambda(0.5)  # TD(0.5)
selector.set_td_lambda(0.9)  # TD(0.9)
```

---

## Neuromorphic Integration

The Nengo-based basal ganglia network remains unchanged and neuromorphic:

```python
# build_network() creates Nengo ensembles as before
selected_operator_ens = selector.build_network(model, state_ensemble)

# TD learning operates at the decision level (post-simulation).
# It updates value estimates based on rewards from the search.
```

### How TD Learning Works with Nengo

1. The Nengo network computes utility functions for each operator.
2. The basal ganglia performs winner-take-all selection based on utilities.
3. TD learning observes the reward and updates value estimates.
4. Updated values inform the next Nengo utility computation.

This keeps the neuromorphic computation intact while adding learning capability on top.

---

## Learning Rules Comparison

| Rule | Type | Use Case |
|------|------|----------|
| SimpleTDRule | Direct | Default, fast learning |
| DecayingTDRule | Decaying | Fade old information |
| ConservativeTDRule | Stable | Prevent wild swings |
| AdaptiveTDRule | Adaptive | Scale with error magnitude |

---

## Value Models Comparison

| Model | Features | Use Case |
|-------|----------|----------|
| LinearValueModel | Simple, fast | Most cases |
| BoundedValueModel | Stable bounds, adaptive | Prevent value explosion |

---

## Implementation Details

### TD(0) Update

For each operator `i`:

```
δ_t = r_t + γ * max_j(V_j(t+1)) - V_i(t)     # TD error
V_i(t+1) = V_i(t) + α * δ_t                  # Value update
```

### TD(λ) Update

Uses eligibility traces that decay:

```
e_i(t) = λ * γ * e_i(t-1) + 1[i visited]     # Eligibility trace
δ_t = r_t + γ * max_j(V_j(t+1)) - V_i(t)    # TD error
V_i(t+1) = V_i(t) + α * δ_t * e_i(t)        # Multi-step update
```

With traces, credit propagates to previously visited operators, enabling multi-step learning.

---

## Configuration in Optimiser

The `NEVOptimiser` can be configured to use TD learning:

```python
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import ConservativeTDRule, BoundedValueModel

optimiser = NEVOptimiser(
    objective_function=f,
    bounds=bounds,
    dimension=10,
    epsilon=0.1,
    learning_rate=0.1,
    td_enabled=True,
)

# Swap learning rule before or after the first run()
optimiser.bg_selector.set_learning_rule(
    ConservativeTDRule(stability_weight=0.5)
)
optimiser.run(time=10.0)
```

---

## Testing

Run the test suite:

```bash
pytest tests/test_td_learning.py        # unit tests
pytest tests/test_td_integration.py    # integration tests (nm_dual mode)
```

Tests cover:
- All learning rules
- All value models
- Eligibility traces
- TD(0) and TD(λ) learning
- Basal ganglia integration
- Dynamic switching

---

## Advanced: Custom Learning Rules

Implement custom rules by extending `LearningRule`:

```python
from nevo.core.td_learning import LearningRule

class CustomTDRule(LearningRule):
    def compute_update(self, td_error, learning_rate, current_value, **kwargs):
        # Your custom update formula
        return learning_rate * td_error * some_factor
    
    def reset(self):
        # Reset internal state if needed
        pass

# Use it
selector.set_learning_rule(CustomTDRule())
```

---

## Advanced: Custom Value Models

Implement custom models by extending `ValueModel`:

```python
from nevo.core.td_learning import ValueModel

class CustomValueModel(ValueModel):
    def get_value(self, operator_idx):
        # Your implementation
        pass
    
    def set_value(self, operator_idx, value):
        # Your implementation
        pass
    
    def update(self, operator_idx, delta):
        # Your implementation
        pass
    
    def reset(self):
        # Reset to initial
        pass
    
    def get_values_array(self):
        # Return numpy array
        pass

# Use it
selector.set_value_model(CustomValueModel())
```

---

## References

### Temporal Difference Learning
- Sutton, R. S. (1988). "Learning to predict by the method of temporal differences."
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"

### TD(λ) and Eligibility Traces
- Sutton, R. S. (1984). "Temporal Credit Assignment in Reinforcement Learning"
- Watkins, C. (1989). "Learning with Delayed Rewards"

### Nengo Neuromorphic Computing
- Bekolay, T., Bergstra, J., Hunsberger, E., et al. (2014). "Nengo: A Python tool for building large-scale functional brain models"
- https://www.nengo.ai/

---

## Future Extensions

1. **Multi-step bootstrapping**: n-step returns instead of 1-step
2. **Importance sampling**: For off-policy learning
3. **Function approximation**: Neural network value functions
4. **Experience replay**: For sample efficiency
5. **Asynchronous TD**: Parallel multi-threaded learning

---

## Module Structure

```
nevo/core/
├── td_learning.py              # TD learning algorithms
│   ├── LearningRule             # Abstract base
│   ├── SimpleTDRule
│   ├── DecayingTDRule
│   ├── ConservativeTDRule
│   ├── AdaptiveTDRule
│   ├── ValueModel               # Abstract base
│   ├── LinearValueModel
│   ├── BoundedValueModel
│   ├── EligibilityTraceManager
│   └── TemporalDifferenceLearner
│
├── basal_ganglia.py             # Modified with TD integration
│   ├── UtilityFunction          # (unchanged)
│   └── BasalGangliaSelector     # (now with TD learning)
│
└── optimiser.py                 # Uses BasalGangliaSelector

tests/
└── test_td_learning.py          # Comprehensive tests
```

---

## Summary

This system provides a modular TD learning layer that:

1. Preserves Nengo neuromorphic networks. Operators are unchanged.
2. Adds TD learning. Both TD(0) and TD(λ) are supported.
3. Allows pluggable rules and models. Easy to extend.
4. Enables dynamic switching. Change algorithms at runtime.
5. Is backward compatible. Existing code continues to work.

