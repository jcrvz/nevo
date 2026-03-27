# NEVO — Neuromorphic Evolutionary Optimisation

**NEVO** (Neuromorphic EVolutionary Optimisation) is a Python framework that bridges evolutionary computation and neuromorphic computing. It uses spiking neural networks to adaptively select and coordinate optimisation operators in real time.

## Key Features

- 🧠 **Neuromorphic operator selection** using basal ganglia circuits
- ⚡ **Population-based parallel evaluation** at each simulation timestep
- 🔄 **Adaptive TD learning** of operator utility weights
- 🎯 **State-aware optimisation** with real-time feature extraction
- 🔌 **Loihi-compatible** neural architectures
- 🚀 **Optional GPU acceleration** via `nengo-dl` / TensorFlow
- 📊 **Built-in visualisation** tools with LaTeX-ready output
- 🏆 **IOH / COCO benchmark** integration for reproducible experiments

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

# Access time-series probe data
data = optimiser.simulator.data[optimiser.stats_probe]  # shape (T, 3)
```

## Architecture

NEVO uses a **basal ganglia** neural circuit to select between optimisation operators. Three operator modes are available:

| `operator_mode` | Description |
|---|---|
| `"trad"` | 13 standard heuristic operators (default) |
| `"nm_dual"` | 2 neuromorphic LIF ensembles, hard WTA switching |
| `"nm_softmix"` | 2 neuromorphic LIF ensembles, softmax-blended |

### Operators (13 total)

| Type | Operator | Description                                        |
|---|---|----------------------------------------------------|
| Exploration | `LevyFlight` | Heavy-tailed random walk for escaping local minima |
| Exploration | `DifferentialEvolution` | Memory-based directed recombination                |
| Exploration | `ParticleSwarm` | Velocity-based swarm search                        |
| Exploration | `SpiralOptimisation` | Anisotropic spiral trajectories                    |
| Exploration | `RandomSearch` | Baseline uniform random sampling                   |
| Exploration | `GravitationalSearch` | Mass-attraction directed exploration               |
| Exploration | `FireflyAlgorithm` | Brightness-attraction swarm search                 |
| Exploration | `CentralForce` | Newtonian gravitational force-based movement       |
| Exploration | `GeneticCrossover` | Recombination of memory solutions                  |
| Exploitation | `GeneticMutation` | Perturbation for diversity injection               |
| Exploitation | `LocalRandomWalk` | Gaussian local refinement                          |
| Exploitation | `SimulatedAnnealing` | Controlled stochastic local search                 |
| Exploitation | `TabuSearch` | Memory-guided local search with avoidance          |

### State Features

Selection is based on a **3-D state vector** computed each timestep, which captures the current search dynamics as described below.
- **Diversity**: spread of solutions in v-space `[-1, 1]^D`
- **Improvement rate**: fraction of recent timesteps with fitness gain
- **Convergence**: fitness homogeneity in the memory archive

## Examples
See the `examples/` directory:

```bash
python examples/basic_example.py --time 2.0 --dimensions 5
```

```bash
python nevo/examples/benchmark_experiment.py --suite cocoex --problems 1-24 --dimensions 2,3,5,10
```

```bash
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
@software{nevo2026repo,
  title={NEVO: Neuromorphic EVolutionary Optimisation},
  author={Cruz-Duarte, Jorge Mario and Talbi, El-Ghazali},
  year={2026},
  url={https://github.com/jcrvz/nevo}
}
```

Also consider the related paper and dataset from Zenodo:

```bibtex
@inproceedings{nevo2026cecpaper,
    author    = {Cruz-Duarte, Jorge M. and Talbi, El-ghazali},
    title     = {{NEVO: A Neuromorphic EVolutionary Optimiser with Spike-Driven Cortico-Basal-Thalamic Coordination}},
    booktitle = {Proceedings of the 2026 IEEE Congress on Evolutionary Computation (CEC)},
    year = {2026},
    pages = {1--6},
    venue = {Maastricht, Netherlands},
    publisher = {IEEE},
    note = {Accepted for publication},
}
```

```bibtex
@dataset{nevo2026cecdataset,
  author    = {Cruz-Duarte, Jorge M. and Talbi, El-ghazali},
  title     = {NEVO: A Neuromorphic EVolutionary Optimiser with Spike-Driven Cortico-Basal-Thalamic Coordination - Codes and Results},
  month     = jan,
  year      = 2026,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.18444113},
  url       = {https://doi.org/10.5281/zenodo.18444113},
}
```

## License

BSD 3-Clause License — see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please submit a pull request.

## Related Projects

- [Nengo](https://www.nengo.ai/) — Neural Engineering Framework
- [IOHexperimenter](https://iohprofiler.github.io/) — Benchmarking suite
