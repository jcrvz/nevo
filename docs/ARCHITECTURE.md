# NEVO Architecture and Design Principles

## Overview

NEVO (Neuromorphic EVolutionary Optimization) implements adaptive metaheuristic optimization using neuromorphic computing principles. The framework uses spiking neural networks to dynamically select and coordinate optimization operators based on the current search state.

## Key Design Principles

### 1. Modularity

**Operator Independence**: Each optimization operator is a self-contained module that:
- Inherits from the `Operator` base class
- Implements `generate_population()` method
- Maintains its own statistics
- Can be added/removed without modifying core code

**Separation of Concerns**:
- `operators/` - Operator implementations
- `core/` - Neuromorphic selection and state management
- `utils/` - Visualization and helper functions

### 2. Neuromorphic Computing Integration

**Basal Ganglia Circuit**: Winner-take-all action selection mimicking cortico-basal ganglia-thalamic loops:
```
State Features → Utility Functions → Basal Ganglia → Thalamus → Selected Operator
```

**Population-Based Parallelism**: Each timestep evaluates LAMBDA candidates in parallel, enabling:
- Massive speedup on neuromorphic hardware
- Natural parallelism across solution space
- Efficient use of neural ensemble dynamics

**Adaptive Learning**: Utility weights are learned online through:
- Reward-based weight updates
- Operator performance tracking
- Exploration-exploitation balance (epsilon-greedy)

### 3. State-Aware Optimization

**Feature Extraction**: Three-dimensional state representation:
1. **Diversity** (φ_d): Spread of solutions in search space [0, 1]
2. **Improvement Rate** (φ_i): Fraction of recent improvements [0, 1]
3. **Convergence** (φ_c): Fitness homogeneity indicator [0, 1]

**Utility Functions**: Each operator has a state-dependent utility:
- **LevyFlight**: High when stuck (low φ_i) and not converged
- **DifferentialEvolution**: High when diversity exists (high φ_d)
- **ParticleSwarm**: High when improving and converging
- **SpiralOptimisation**: High when highly converged (high φ_c)

### 4. Memory-Based Search

**Competitive Memory**: Fixed-size archive (MU solutions) maintained through:
- Competitive replacement (worst fitness evicted)
- Age tracking
- Fitness-weighted centroid computation

**v-Space Normalization**: All solutions stored in [-1, 1]^D space:
- Bounds-independent operators
- Consistent step sizes
- Transform to original space only for evaluation

## Architecture Components

### Operators (`nevo/operators/`)

Base class hierarchy:
```
Operator (ABC)
├── ExplorationOperator
│   ├── LevyFlight
│   └── DifferentialEvolution
└── ExploitationOperator
    ├── ParticleSwarm
    └── SpiralOptimisation
```

Each operator implements:
```python
def generate_population(
    self,
    centre: np.ndarray,
    state: Dict[str, Any],
    population_size: int
) -> np.ndarray:
    """Generate population of candidates."""
    pass
```

### Core Components (`nevo/core/`)

**StateFeatures** (`state.py`):
- Computes interpretable features from raw optimization state
- Maintains improvement history
- Provides feature vector for neural networks

**BasalGangliaSelector** (`basal_ganglia.py`):
- Builds Nengo neural network for operator selection
- Implements utility ensembles and winner-take-all circuit
- Manages adaptive utility weight learning
- Handles epsilon-greedy exploration

**NEVOptimizer** (`optimizer.py`):
- Main user-facing class
- Integrates all components
- Manages simulation loop
- Provides results and statistics

### Utilities (`nevo/utils/`)

**Visualization** (`visualization.py`):
- `plot_optimization_results()`: Comprehensive 3-panel plot
  - Fitness evolution with error tracking
  - Operator selection timeline
  - State feature trajectories
- `plot_operator_statistics()`: Usage and performance metrics

## Usage Patterns

### Basic Usage

```python
from nevo import NEVOptimizer

optimizer = NEVOptimizer(
    objective_function=my_function,
    bounds=(-5, 5),
    dimension=10,
)

optimizer.run(time=20.0)
x_best, f_best = optimizer.get_best_solution()
```

### Custom Operators

```python
from nevo.operators.base import ExplorationOperator

class MyOperator(ExplorationOperator):
    def __init__(self):
        super().__init__("MyOperator")
    
    def generate_population(self, centre, state, population_size):
        # Your implementation
        return candidates

optimizer = NEVOptimizer(
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

## Neuromorphic Benefits

### 1. Hardware Efficiency

**Loihi Compatibility**:
- All networks use standard Nengo constructs
- No unsupported operations
- Direct compilation to Loihi possible
- Energy-efficient spiking neurons

**Parallel Evaluation**:
- Population-based approach natural for neuromorphic chips
- Dimension-wise parallelism in operators
- Concurrent operator evaluation possible

### 2. Adaptive Behavior

**Online Learning**:
- Utility weights adapt during search
- No offline training required
- Continuous improvement through experience

**State-Dependent Selection**:
- Neural dynamics encode current search state
- Smooth transitions between operators
- Emergent behaviour from neural interactions

### 3. Biological Inspiration

**Brain-Like Processing**:
- Mimics action selection in mammalian brain
- Winner-take-all through lateral inhibition
- Thalamic gating of selected actions

**Continuous-Time Dynamics**:
- No discrete decision points
- Smooth neural trajectories
- Temporal filtering of noisy signals

## Extending NEVO

### Adding New Operators

1. Create operator class inheriting from `Operator`
2. Implement `generate_population()`
3. Add to `OPERATOR_REGISTRY` in `operators/__init__.py`
4. Define utility function in `core/basal_ganglia.py`

### Adding New State Features

1. Extend `StateFeatures.compute()` to return more dimensions
2. Update neural ensemble dimensions accordingly
3. Modify utility functions to use new features

### Custom Memory Strategies

1. Override `update_memory()` in `NEVOptimizer`
2. Implement custom replacement/aging logic
3. Ensure compatibility with `compute_fitness_weighted_centre()`

## Performance Considerations

**Timestep (dt)**:
- Smaller dt = more evaluations but slower simulation
- Recommended: 0.001s (1000 evals/second with LAMBDA=1)
- Trade-off: neural accuracy vs computational cost

**Population Size (LAMBDA)**:
- Larger = more parallelism but more evaluations
- Recommended: 50 for moderate dimensions
- Scale with problem difficulty

**Memory Size (MU)**:
- Larger = more diversity but slower centroid computation
- Recommended: MU = LAMBDA / 2
- Must be > 3 for DifferentialEvolution

**Neurons Per Ensemble**:
- More neurons = better representation but slower
- Recommended: 100 for research, 50 for Loihi
- Affects neural accuracy of decisions

## Future Directions

### Loihi Deployment

- Use `nengo_loihi` backend for direct hardware deployment
- Optimize neuron counts for chip constraints
- Benchmark energy efficiency vs CPU/GPU

### Multi-Objective Optimization

- Extend state features to include Pareto front metrics
- Multiple basal ganglia for different objectives
- Archive of non-dominated solutions

### Constraint Handling

- Add constraint violation to state features
- Penalty-based or repair operators
- Feasibility-driven selection

### Online Operator Discovery

- Meta-learning of new operators
- Genetic programming of mutation strategies
- Neural networks as operator generators

## References

Key papers and resources:
- Nengo: Bekolay et al. (2014), Frontiers in Neuroinformatics
- Basal Ganglia: Gurney et al. (2001), Biological Cybernetics
- Differential Evolution: Storn & Price (1997), Journal of Global Optimization
- Particle Swarm: Kennedy & Eberhart (1995), IEEE ICNN
- IOHexperimenter: Doerr et al. (2018), arXiv:1810.05281

