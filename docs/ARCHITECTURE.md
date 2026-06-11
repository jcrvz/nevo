# Architecture and Design Principles

## Overview

NEVO implements adaptive metaheuristic optimisation using neuromorphic computing principles. The framework uses spiking neural networks to dynamically select and coordinate optimisation operators based on the current search state.

## Key Design Principles

### 1. Modularity

**Operator Independence**: Each optimisation operator is a self-contained module that:
- Inherits from `ExplorationOperator` or `ExploitationOperator` (both subclasses of the `Operator` ABC).
- Implements `generate_population()`.
- Maintains its own statistics.
- Can be added or removed without modifying core code.

**Separation of Concerns**:
- `operators/`: Operator implementations.
- `core/`: Neuromorphic selection, state management, and TD learning.
- `utils/`: Visualisation and helper functions.

### 2. Neuromorphic Computing Integration

**Basal Ganglia Circuit**: Winner-Take-All (WTA) action selection mimicking Cortico-Basal Ganglia-Thalamic Loops:
```
State Features → Utility Functions → Basal Ganglia → Thalamus → Selected Operator
```

**Population-Based Parallelism**: Each timestep evaluates `population_size` candidates in parallel, enabling:
- Speedup on neuromorphic hardware.
- Natural parallelism across the solution space.
- Efficient use of neural ensemble dynamics.

**Adaptive Learning**: Operator selection is governed by two complementary mechanisms:
- **Utility-weight adaptation**: Each operator's utility weight is updated online via a reward signal (relative fitness improvement). Weights are clipped to `[0.1, 5.0]`.
- **Temporal Difference (TD) learning**: A `TemporalDifferenceLearner` (TD(0) or TD($\lambda$)) maintains per-operator value estimates $V(a)$. After each timestep, a TD error $\delta = r + \gamma \cdot \max_j V(j) - V(a)$ is computed and $V(a)$ is updated via the configured learning rule. The resulting TD values are mixed with the Nengo BG signal as a 20 % additive bias during action selection.
- **Epsilon-greedy exploration**: With probability $\varepsilon$, a random operator is chosen regardless of utility or TD values.

### 3. State-Aware Optimisation

**Feature Extraction**: Three-dimensional state representation:
1. **Diversity** ($\phi_d$): Average per-dimension standard deviation of valid memory vectors, normalised by 0.6 (clipped to `[0, 1]`).
2. **Improvement Rate** ($\phi_i$): Fraction of recent timesteps that improved the best solution over a sliding window of length 50. Returns the default 0.5 until at least 10 history entries exist.
3. **Convergence** ($\phi_c$): Fitness homogeneity; i.e., `1 / (1 + 10 · f_range / |f_min|)` for non-zero minima; `1 − f_range/100` otherwise.

**Utility Functions**: Each operator has a state-dependent utility:
- **LevyFlight**: High when stuck (low $\phi_i$) and not converged.
- **DifferentialEvolution**: High when diversity exists (high $\phi_d$). Requires $\ge$ 3 valid memory slots; falls back to Gaussian noise otherwise.
- **ParticleSwarm**: High when improving and converging.
- **SpiralOptimisation**: High when highly converged (high $\phi_c$).

### 4. Memory-Based Search

**Competitive Memory**: Fixed-size archive (`memory_size` solutions) maintained via:
- Competitive replacement (worst-fitness slot evicted).
- Age tracking (all slots aged by 1 each timestep).
- **Rank-weighted centroid computation** (`compute_fitness_weighted_centre()`): solutions are weighted by `1 / (rank + 1)` so better-ranked solutions contribute more; note the function name uses "fitness-weighted" historically but the implementation is rank-based.

**v-Space Normalisation**: All solutions are stored in `[-1, 1]^D`:
- Bounds-independent operators.
- Consistent step sizes.
- Transform to original space only for evaluation (`trs2o()` in `optimiser.py`).

---

## Architecture Components

### Operators (`nevo/operators/`)

Base class hierarchy:
```
Operator (ABC)
├── ExplorationOperator
│   ├── LevyFlight
│   ├── DifferentialEvolution
│   ├── RandomSearch
│   ├── GravitationalSearch
│   ├── FireflyAlgorithm
│   ├── CentralForce
│   ├── GeneticCrossover
│   └── NeuromorphicExplorationEnsemble   (150 LIF neurons, 5 ms synapse)
└── ExploitationOperator
    ├── ParticleSwarm
    ├── SpiralOptimisation
    ├── LocalRandomWalk
    ├── GeneticMutation
    ├── SimulatedAnnealing
    ├── TabuSearch
    └── NeuromorphicExploitationEnsemble  (200 LIF neurons, 20 ms synapse)
```

Each operator implements:
```python
def generate_population(
    self,
    centre: np.ndarray,
    state: Dict[str, Any],
    population_size: int
) -> np.ndarray:
    """Generate population of candidates in v-space [-1, 1]^D."""
    pass
```

### Operator Modes

| `operator_mode` | Operators loaded | Population generation |
|---|---|---|
| `"trad"` / `"traditional"` | 13 standard heuristic operators | BG winner calls `generate_population()` exclusively |
| `"nm_dual"` | 2 neuromorphic ensembles | BG winner calls `generate_population()` exclusively (hard WTA) |
| `"nm_softmix"` | 2 neuromorphic ensembles | Both `generate_population()` are called every timestep; per-candidate λ ~ Beta(α, β) blends results (`soft_mix_temperature=0.35`, `soft_mix_concentration=6.0`; minimum ensemble weight 0.1) |

**`nm_softmix` blending detail**: The BG thalamus output is passed through a softmax with `temperature=0.35` to obtain weights `[w_explore, w_exploit]`. These parameterise a Beta distribution: $\alpha = \max\{0.5, \texttt{w_exploit} \times 6.0\}$, $\beta = \max\{0.5, \texttt{w_explore} \times 6.0\}$. A per-candidate scalar λ is sampled from $\mathsf{Beta}(\alpha, \beta)$, giving the exploitation blend fraction. Final candidate is given by $ (1 − \lambda) \cdot \texttt{explore} + \lambda \cdot \texttt{exploit}$.

> **Note**: In both `nm_dual` and `nm_softmix`, `select_operator()` is still invoked each timestep for reward computation and TD/utility-weight bookkeeping; in `nm_softmix` the winning operator is tracked for learning purposes even though population generation is always blended.

### Core Components (`nevo/core/`)

**StateFeatures** (`state.py`):
- Computes the 3-D feature vector `[diversity, improvement_rate, convergence]` from raw optimisation state.
- Maintains a sliding improvement history (length 50).
- Improvement rate defaults to 0.5 until at least 10 history entries are present.

**BasalGangliaSelector** (`basal_ganglia.py`):
- Builds the Nengo neural network: one utility `Ensemble` per operator (each connected to `StateEnsemble` via its utility function), a `nengo.networks.BasalGanglia`, a `nengo.networks.Thalamus`, and a final `SelectedOperator` ensemble whose output is read by `population_generator_func`.
- `select_operator()` follows a 5-step decision flow each timestep:
  1. **Compute reward** from relative fitness improvement (`r = Δf / |f_prev|`; −0.01 penalty if no improvement).
  2. **TD update**: call `TemporalDifferenceLearner.update()` with `r + γ · max_j V(j)` as the bootstrap target.
  3. **Utility weight update**: update the winning operator's `UtilityFunction.weight` by `lr=0.1 · reward`.
  4. **Combine signals**: normalise BG output and TD values to `[0, 1]`; form `combined = 0.8 · BG + 0.2 · TD`.
  5. **ε-greedy**: select `argmax(combined)` with probability `1 − ε`; random operator otherwise.

**TemporalDifferenceLearner** (`td_learning.py`):
- Maintains per-operator value estimates V($a$) and eligibility traces for TD($\lambda$).
- Composed of a pluggable **LearningRule** and a pluggable **ValueModel**.
- **Learning rules**: `SimpleTDRule` ($\Delta V = \alpha\cdot\delta$), `DecayingTDRule` (decaying factor per rule type), `ConservativeTDRule` (damped + clipped update), `AdaptiveTDRule` (magnitude-adaptive α).
- **Value models**: `LinearValueModel` (one scalar per operator, clipped to `[0.1, 5.0]`), `BoundedValueModel` (same with per-operator adaptive bounds).
- `EligibilityTraceManager` handles trace decay: `traces(i) ← traces(i) × γ·λ` each step, then `traces(selected) += 1`.
- All TD components can be swapped at runtime without rebuilding the Nengo model.

**NEVOptimiser** (`optimiser.py`):
- Main user-facing class.
- Integrates all components: operators, `StateFeatures`, `BasalGangliaSelector`, and the Nengo simulation loop.
- Model is built lazily on the first `run()` call and reused on subsequent calls.
- TD episode is started automatically on the first `run()` call; call `reset_td_episode()` to reset eligibility traces between independent runs without rebuilding.

### Probes

After `run()`, time-series data is accessible via `optimiser.simulator.data[probe]`:

| Probe | Shape | Contents |
|---|---|---|
| `optimiser.stats_probe` | `(T, 3)` | `[best_f, mean_f, operator_idx]` per timestep |
| `optimiser.state_features_probe` | `(T, 3)` | `[diversity, improvement_rate, convergence]` per timestep |
| `optimiser.state_probe` | `(T, 3)` | Decoded `StateEnsemble` output (filtered, synapse=0.01) |
| `optimiser.operator_probe` | `(T, n_operators)` | `SelectedOperator` ensemble decoded output |

### Utilities (`nevo/utils/`)

**Visualisation** (`visualisation.py`):
- `plot_optimisation_results()`: Three-panel plot.
  - Fitness evolution with error tracking.
  - Operator selection timeline.
  - State feature trajectories.
- `plot_operator_statistics()`: Usage and performance metrics.

---

## Usage Patterns

### Basic Usage

```python
from nevo import NEVOptimiser

optimiser = NEVOptimiser(
    objective_function=my_function,
    bounds=(-5, 5),
    dimension=10,
)

optimiser.run(time=20.0)
x_best, f_best = optimiser.get_best_solution()
```

### Custom Operators

```python
from nevo.operators.base import ExplorationOperator

class MyOperator(ExplorationOperator):
    def __init__(self):
        super().__init__("MyOperator")

    def generate_population(self, centre, state, population_size):
        # Implementation here
        return candidates

optimiser = NEVOptimiser(
    objective_function=my_function,
    bounds=(-5, 5),
    dimension=10,
    operators=[MyOperator(), LevyFlight()],
)
```

### Custom Utility Functions

Utility functions can be patched on `optimiser.bg_selector.utilities` after construction,
or by supplying a `utility_functions` dict directly to `BasalGangliaSelector`:

```python
from nevo.core.basal_ganglia import BasalGangliaSelector, UtilityFunction

def my_utility(features):
    diversity, improvement, convergence = features
    return diversity * 2.0 + convergence

# Patch after construction
optimiser = NEVOptimiser(objective_function=my_function, bounds=(-5, 5), dimension=10)
optimiser.bg_selector.utilities["LevyFlight"] = UtilityFunction(
    "LevyFlight", my_utility
)
```

### Runtime TD Configuration

```python
from nevo.core.td_learning import ConservativeTDRule, BoundedValueModel

optimiser.set_td_lambda(0.9)                              # TD(λ) coefficient
optimiser.set_td_learning_rule(ConservativeTDRule(0.5))   # swap rule
optimiser.set_td_value_model(BoundedValueModel(n_ops))    # swap value model
```

### Accessing Probe Data

```python
import numpy as np

optimiser.run(time=20.0)
stats = optimiser.simulator.data[optimiser.stats_probe]   # shape (T, 3)
best_f_trace  = stats[:, 0]   # best fitness over time
mean_f_trace  = stats[:, 1]   # mean population fitness over time
op_idx_trace  = stats[:, 2]   # operator index selected each timestep
```

---

## Neuromorphic Benefits

### 1. Hardware Efficiency

**Loihi Compatibility** (design intent):
- All networks use standard Nengo constructs.
- No unsupported operations.
- Direct compilation to Loihi possible.
- Energy-efficient spiking neurons.

**Parallel Evaluation**:
- Population-based approach is natural for neuromorphic chips.
- Dimension-wise parallelism in operators.

### 2. Adaptive Behaviour

**Online Learning**:
- Utility weights and TD values adapt during the search.
- No offline training required.
- TD components (`LearningRule`, `ValueModel`) are swappable at runtime.

**State-Dependent Selection**:
- Neural dynamics encode the current search state.
- Emergent behaviour from neural interactions.

### 3. Biological Inspiration

**Brain-Like Processing**:
- Mimics action selection in the mammalian brain.
- Winner-take-all through lateral inhibition.
- Thalamic gating of selected actions.

**Continuous-Time Dynamics**:
- No discrete decision points.
- Smooth neural trajectories.
- Temporal filtering of noisy signals.

---

## Extending NEVO

### Adding New Operators

1. Create an operator class inheriting from `ExplorationOperator` or `ExploitationOperator`.
2. Implement `generate_population()`. Return values must be in `[-1, 1]`.
3. Add to `OPERATOR_REGISTRY` in `operators/__init__.py`.
4. Optionally add a `utility_<name>` function in `core/basal_ganglia.py` and register it in `DEFAULT_UTILITY_FUNCTIONS`.

### Adding New State Features

1. Extend `StateFeatures.compute()` to return more dimensions.
2. Update neural ensemble dimensions accordingly.
3. Update utility functions to use new features.

### Custom Memory Strategies

1. Override `update_memory()` in `NEVOptimiser`.
2. Implement custom replacement or ageing logic.
3. Ensure compatibility with `compute_fitness_weighted_centre()`.

### Custom TD Learning

1. Subclass `LearningRule` and implement `compute_update()`.
2. Subclass `ValueModel` and implement the abstract interface.
3. Pass instances to `NEVOptimiser` via `td_learning_rule` / `td_value_model`, or swap at runtime via `set_td_learning_rule()` / `set_td_value_model()`.

---

## Performance Considerations

**Timestep (`dt`)**:
- Smaller `dt` means more evaluations but slower wall-clock time.
- Recommended: `0.001` s — with `population_size=N`, each simulated second performs `1000 × N` objective evaluations.

**Population Size**:
- Controls candidates per timestep, not total budget.
- Recommended: 50 for moderate dimensions.

**Memory Size (`MU`)**:
- Recommended: `MU = population_size / 2`.
- `DifferentialEvolution` requires $\geq$ 3 valid (non-sentinel) slots; it falls back to Gaussian noise around the centroid if fewer are available.

**Neurons Per Ensemble**:
- Recommended: 100 for research, 50 for Loihi.
- `StateEnsemble` is fixed at 100 neurons (3D, `radius = 1.5`).
- `NeuromorphicExplorationEnsemble`: 150 LIF neurons, $\tau_\text{syn} = 5$ ms.
- `NeuromorphicExploitationEnsemble`: 200 LIF neurons, $τ_\text{syn} = 20$ ms.

---

## References

- Bekolay, T., et al. "Nengo: a Python tool for building large-scale functional brain models." Frontiers in neuroinformatics 7 (2013): 48. _[Link](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2013.00048/full)_
- Gurney, K., et al. "A computational model of action selection in the basal ganglia. I. A new functional anatomy." Biological cybernetics 84, no. 6 (2001): 401-410. _[Link](https://doi.org/10.1007/PL00007984)_
