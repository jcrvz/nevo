# NEVO - Neuromorphic Evolutionary Optimization

**NEVO** (Neuromorphic EVolutionary Optimization) is a Python framework that bridges evolutionary computation and neuromorphic computing. It uses spiking neural networks to adaptively select and coordinate optimization operators in real-time.

## Key Features

- 🧠 **Neuromorphic operator selection** using basal ganglia circuits
- ⚡ **Population-based parallel evaluation** for massive speedup
- 🔄 **Adaptive learning** of operator utility weights
- 🎯 **State-aware optimization** with real-time feature extraction
- 🔌 **Loihi-compatible** neural architectures
- 📊 **Built-in visualization** tools

## Installation

```bash
pip install nevo
```

Or install from source:

```bash
git clone https://github.com/jcrvz/nevo.git
cd nevo
pip install -e .
```

## Quick Start

```python
from nevo import NEVOptimizer
from ioh import get_problem

# Define optimization problem
problem = get_problem(fid=1, dimension=10, instance=1)

# Create optimizer
optimizer = NEVOptimizer(
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
3. **Energy Efficiency**: Designed for neuromorphic hardware (Intel Loihi)
4. **Biological Inspiration**: Mimics cortico-basal ganglia-thalamic loops

## Citation

If you use NEVO in your research, please cite:

```bibtex
@software{nevo2024,
  title={NEVO: Neuromorphic Evolutionary Optimisation},
  author={Cruz-Duarte, Jorge Mario},
  year={2024},
  url={https://github.com/jcrvz/nevo}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [neuroptimiser](https://github.com/jcrvz/neuroptimiser) - Original research implementation
- [Nengo](https://www.nengo.ai/) - Neural Engineering Framework
- [IOHexperimenter](https://iohprofiler.github.io/) - Benchmarking suite

