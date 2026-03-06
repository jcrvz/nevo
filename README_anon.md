# NEVO - Neuromorphic Evolutionary Optimisation
> This version of the README has been anonymized for review purposes. Many files and links have been removed to preserve anonymity.

**NEVO** (Neuromorphic EVolutionary Optimisation) is a Python framework that bridges evolutionary computation and neuromorphic computing. It uses spiking neural networks to adaptively select and coordinate optimisation operators in real-time.

## Key Features

- 🧠 **Neuromorphic operator selection** using basal ganglia circuits and adaptive utility learning
- ⚡ **Population-based parallel evaluation** for scalable performance
- 🔄 **Online learning of operator utility weights** for dynamic adaptation
- 🎯 **State-aware optimisation** with interpretable feature extraction (diversity, improvement rate, convergence)
- 🔌 **Loihi-compatible neural architectures** for neuromorphic hardware
- 🧩 **Extensible operator registry** supporting a wide range of metaheuristics
- 📊 **Comprehensive visualisation tools** for results and operator statistics
- 🧪 **Benchmarking integration** with IOHexperimenter and reproducible experiments
- 🛠️ **Modular design** for easy extension and customisation

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

### Using uv (recommended for modern Python projects)

If you have [uv](https://github.com/astral-sh/uv) installed, you can install dependencies from `pyproject.toml`:

```bash
uv pip install -e .
```

Or to install all dependencies (including dev):

```bash
uv pip install -e . --extra dev
```

NEVO uses a `pyproject.toml` for dependency management and build configuration. This enables modern, reproducible Python environments.

## Quick Start

```python
from nevo import NEVOptimiser
from ioh import get_problem

# Define optimisation problem
problem = get_problem(fid=1, dimension=10, instance=1)

# Create optimizer
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

NEVO uses a **basal ganglia** neural circuit to select between multiple optimisation operators:

- **LevyFlight**: Heavy-tailed exploration for escaping local minima
- **DifferentialEvolution**: Memory-based directed exploration
- **ParticleSwarm**: Velocity-based exploitation
- **SpiralOptimisation**: Anisotropic fine-tuning
- **RandomSearch**: Uniform random sampling
- **LocalRandomWalk**: Small-scale local exploration
- **GravitationalSearch**: Mass-based attraction dynamics
- **FireflyAlgorithm**: Light-based attraction and randomisation
- **CentralForce**: Physics-inspired global attraction
- **GeneticCrossover**: Recombination of memory solutions
- **GeneticMutation**: Random perturbation of solutions
- **SimulatedAnnealing**: Temperature-based local search
- **TabuSearch**: Memory-based local search avoiding revisits

Selection is based on **state features**:
- **Diversity**: Spread of solutions in search space
- **Improvement rate**: Recent success frequency
- **Convergence**: Fitness homogeneity

## Examples

See the `nevo/examples/` directory for complete examples:

```bash
python nevo/examples/basic_example.py
```

## Neuromorphic Benefits

1. **Adaptive Selection**: Neural circuits learn which operators work best online
2. **Parallel Processing**: Population-based evaluation across dimensions
3. **Energy Efficiency**: Designed for neuromorphic hardware (e.g., Intel Loihi 2, SpiNNaker)
4. **Biological Inspiration**: Mimics Cortico-Basal Ganglia-Thalamic Loops

## Citation

If you use NEVO in your research, please cite:

```bibtex
@software{nevo2025,
  title={NEVO: A Neuromorphic EVolutionary Optimiser with Spike-Driven Cortico-Basal-Thalamic Coordination},
  author={ANONYMOUS-AUTHOR(S)},
  year={2025},
  url={https://github.com/ANONYMOUS-AUTHOR(S)/nevo}
}
```

## License

MIT Licence - see (LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [ANOTHER OPTIMISER BASED ON SNNS](https://github.com/ANONYMOUS/TURBO-HYPER-OPTIMISER) - Original research implementation
- [Nengo](https://www.nengo.ai/) - Neural Engineering Framework
- [IOHexperimenter](https://iohprofiler.github.io/) - Benchmarking suite

