# NEVO - Neuromorphic Evolutionary Optimisation

**NEVO** (Neuromorphic EVolutionary Optimisation) is a Python framework that bridges evolutionary computation and neuromorphic computing. It uses spiking neural networks to adaptively select and coordinate optimisation operators in real-time.

## Key Features

- 🧠 **Neuromorphic operator selection** using basal ganglia circuits (Nengo)
- ⚡ **13 built-in operators** covering exploration and exploitation strategies
- 🔄 **Adaptive utility-weight learning** with epsilon-greedy selection
- 🎯 **State-aware optimisation** with real-time 3-D feature extraction (diversity, improvement rate, convergence)
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
# include dev dependencies
uv pip install -e . --extra dev
```

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

NEVO uses a **basal ganglia** neural circuit to select between multiple optimisation operators based on the current search state.

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

```bash
python nevo/examples/basic_example.py
python nevo/examples/benchmark_experiment.py --suite cocoex --problems 1-24 --dimensions 2,3,5,10
```

## Neuromorphic Benefits

1. **Adaptive Selection**: Neural circuits learn which operators work best online
2. **Parallel Processing**: Population-based evaluation across dimensions
3. **Energy Efficiency**: Designed for neuromorphic hardware (e.g., Intel Loihi 2, SpiNNaker)
4. **Biological Inspiration**: Mimics Cortico-Basal Ganglia-Thalamic Loops

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

Contributions are welcome! Please feel free to submit a Pull Request.


