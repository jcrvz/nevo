# NEVO — Neuromorphic Evolutionary Optimisation

**NEVO** (Neuromorphic EVolutionary Optimisation) is a Python framework that bridges evolutionary computation and neuromorphic computing. It uses spiking neural networks to adaptively select and coordinate optimisation operators in real time.

## Key Features

- 🧠 **Neuromorphic operator selection** using basal ganglia circuits
- ⚡ **Population-based parallel evaluation** at each simulation timestep
- 🔄 **Adaptive TD learning** of operator utility weights
- 🎯 **State-aware optimisation** with real-time feature extraction
- 🔌 **Loihi-compatible** neural architectures
- 📊 **Built-in visualisation** tools

## Installation

### Using pip

```bash
pip install nevo
```

Or install from source:

```bash
git clone https://github.com/jcrvz/nevo.git
cd nevo
pip install -e .
```

### Using uv (recommended)

```bash
uv pip install -e .
uv pip install -e . --extra dev   # includes dev dependencies
```

NEVO uses `pyproject.toml` for dependency management and build configuration.

## Quick Start

```python
from nevo import NEVOptimiser
from ioh import get_problem

# Define optimisation problem
problem = get_problem(fid=1, dimension=10, instance=1)

# Create optimiser
optimiser = NEVOptimiser(
    objective_function=problem,
    bounds=(problem.bounds.lb, problem.bounds.ub),
    dimension=10,
    population_size=50,
    memory_size=25,
)

# Run optimisation
optimiser.run(time=20.0)

# Get results
x_best, f_best = optimiser.get_best_solution()
print(f"Best fitness: {f_best:.6e}")
```

## Architecture

NEVO uses a **basal ganglia** neural circuit to select between optimisation operators. Three operator modes are available:

| `operator_mode` | Description |
|---|---|
| `"trad"` | 13 standard heuristic operators (default) |
| `"nm_dual"` | 2 neuromorphic LIF ensembles, hard WTA switching |
| `"nm_softmix"` | 2 neuromorphic LIF ensembles, softmax-blended |

Selection is driven by three **state features**:
- **Diversity**: Spread of solutions in search space.
- **Improvement rate**: Recent success frequency.
- **Convergence**: Fitness homogeneity.

## Examples

See the `examples/` directory:

```bash
python examples/basic_example.py --time 2.0 --dimensions 5
python examples/td_learning_examples.py
```

## Neuromorphic Modes

On multimodal problems, spike-driven dynamics outperform tuned heuristics:

```python
optimiser = NEVOptimiser(
    objective_function=rastrigin,
    bounds=(-5.12, 5.12),
    dimension=10,
    operator_mode="nm_dual",
    population_size=30,
)
optimiser.run(time=10.0)
```

## Citation

If you use NEVO in your research, please cite:

```bibtex
@software{nevo2025,
  title={NEVO: Neuromorphic Evolutionary Optimisation},
  author={Cruz-Duarte, Jorge Mario and Talbi, El-Ghazali},
  year={2025},
  url={https://github.com/jcrvz/nevo}
}
```

## Licence

MIT Licence — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please submit a pull request.

## Related Projects

- [Nengo](https://www.nengo.ai/) — Neural Engineering Framework
- [IOHexperimenter](https://iohprofiler.github.io/) — Benchmarking suite
