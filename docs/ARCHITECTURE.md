# NEVO Architecture and Design Principles

## Overview

NEVO (Neuromorphic EVolutionary Optimisation) implements adaptive metaheuristic optimisation using neuromorphic computing principles. The framework uses spiking neural networks to dynamically select and coordinate optimisation operators based on the current search state.

## Key Design Principles

### 1. Modularity

**Operator Independence**: Each optimisation operator is a self-contained module that:
- Inherits from the `Operator` base class.
- Implements `generate_population()`.
- Maintains its own statistics.
- Can be added or removed without modifying core code.

**Separation of Concerns**:
- `operators/` — Operator implementations.
- `core/` — Neuromorphic selection and state management.
- `utils/` — Visualisation and helper functions.

### 2. Neuromorphic Computing Integration

**Basal Ganglia Circuit**: Winner-take-all action selection mimicking cortico-basal ganglia-thalamic loops:
```
State Features → Utility Functions → Basal Ganglia → Thalamus → Selected Operator
```

**Population-Based Parallelism**: Each timestep evaluates `population_size` candidates in parallel, enabling:
- Speedup on neuromorphic hardware.
- Natural parallelism across the solution space.
- Efficient use of neural ensemble dynamics.

**Adaptive Learning**: Utility weights are learned online through:
- Reward-based weight updates.
- Operator performance tracking.
- Epsilon-greedy exploration.

### 3. State-Aware Optimisation

**Feature Extraction**: Three-dimensional state representation:
1. **Diversity** (φ_d): Spread of solutions in search space [0, 1].
2. **Improvement Rate** (φ_i): Fraction of recent improvements [0, 1].
3. **Convergence** (φ_c): Fitness homogeneity indicator [0, 1].

**Utility Functions**: Each operator has a state-dependent utility:
- **LevyFlight**: High when stuck (low φ_i) and not converged.
- **DifferentialEvolution**: High when diversity exists (high φ_d).
- **ParticleSwarm**: High when improving and converging.
- **SpiralOptimisation**: High when highly converged (high φ_c).

### 4. Memory-Based Search

**Competitive Memory**: Fixed-size archive (`memory_size` solutions) maintained via:
- Competitive replacement (worst fitness evicted).
- Age tracking.
- Fitness-weighted centroid computation.

**v-Space Normalisation**: All solutions are stored in [-1, 1]^D:
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
│   └── NeuromorphicExplorationEnsemble
└── ExploitationOperator
    ├── ParticleSwarm
    ├── SpiralOptimisation
    ├── LocalRandomWalk
    ├── GeneticMutation
    ├── SimulatedAnnealing
    ├── TabuSearch
    └── NeuromorphicExploitationEnsemble
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

| `operator_mode` | Operators loaded |
|---|---|
| `"trad"` / `"traditional"` | 13 standard heuristic operators |
| `"nm_dual"` | 2 neuromorphic ensembles (hard WTA switching) |
| `"nm_softmix"` | 2 neuromorphic ensembles (softmax-blended) |

### Core Components (`nevo/core/`)

**StateFeatures** (`state.py`):
- Computes interpretable features from the raw optimisation state.
- Maintains improvement history.
- Provides feature vector for neural networks.

**BasalGangliaSelector** (`basal_ganglia.py`):
- Builds the Nengo neural network for operator selection.
- Implements utility ensembles and winner-take-all circuit.
- Manages adaptive utility weight learning and TD values.
- Handles epsilon-greedy exploration.

**NEVOptimiser** (`optimiser.py`):
- Main user-facing class.
- Integrates all components.
- Manages the Nengo simulation loop.
- Provides results and statistics.

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

```python
from nevo.core.basal_ganglia import BasalGangliaSelector

def my_utility(features):
    diversity, improvement, convergence = features
    return diversity * 2.0 + convergence

selector = BasalGangliaSelector(
    operators=my_operators,
    utility_functions={"MyOperator": my_utility},
)
```

---

## Neuromorphic Benefits

### 1. Hardware Efficiency

**Loihi Compatibility**:
- All networks use standard Nengo constructs.
- No unsupported operations.
- Direct compilation to Loihi possible.
- Energy-efficient spiking neurons.

**Parallel Evaluation**:
- Population-based approach is natural for neuromorphic chips.
- Dimension-wise parallelism in operators.

### 2. Adaptive Behaviour

**Online Learning**:
- Utility weights adapt during the search.
- No offline training required.

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
4. Define a utility function in `core/basal_ganglia.py`.

### Adding New State Features

1. Extend `StateFeatures.compute()` to return more dimensions.
2. Update neural ensemble dimensions accordingly.
3. Update utility functions to use new features.

### Custom Memory Strategies

1. Override `update_memory()` in `NEVOptimiser`.
2. Implement custom replacement or ageing logic.
3. Ensure compatibility with `compute_fitness_weighted_centre()`.

---

## Performance Considerations

**Timestep (`dt`)**:
- Smaller `dt` means more evaluations but slower wall-clock time.
- Recommended: `0.001` s (1000 evals/second with `population_size=1`).

**Population Size**:
- Controls candidates per timestep, not total budget.
- Recommended: 50 for moderate dimensions.

**Memory Size (`MU`)**:
- Recommended: `MU = population_size / 2`.
- Must be > 3 for `DifferentialEvolution`.

**Neurons Per Ensemble**:
- Recommended: 100 for research, 50 for Loihi.

---

## References

- Nengo: Bekolay et al. (2014), Frontiers in Neuroinformatics.
- Basal Ganglia: Gurney et al. (2001), Biological Cybernetics.
- Differential Evolution: Storn & Price (1997), Journal of Global Optimisation.
- Particle Swarm: Kennedy & Eberhart (1995), IEEE ICNN.
- IOHexperimenter: Doerr et al. (2018), arXiv:1810.05281.
