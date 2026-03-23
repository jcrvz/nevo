# NEVO Setup Guide

This guide will help you set up the NEVO development environment.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jcrvz/nevo.git
cd nevo
```

### 2. Create Virtual Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install NEVO

For regular use:
```bash
pip install -e .
```

For development (includes testing and documentation tools):
```bash
pip install -e ".[dev,docs]"
```

### 4. Verify Installation

Run the test suite:
```bash
pytest tests/ -v
```

Run a basic example:
```bash
python nevo/examples/basic_example.py
```

## Project Structure

```
nevo/
├── nevo/                    # Main package
│   ├── core/               # Core optimization components
│   │   ├── optimizer.py    # Main NEVOptimizer class
│   │   ├── state.py        # State feature extraction
│   │   └── basal_ganglia.py # Operator selection network
│   ├── operators/          # Optimization operators
│   │   ├── base.py         # Base operator interface
│   │   ├── standard.py     # Standard operators (LF, DE, PSO, Spiral)
│   │   └── __init__.py     # Operator registry
│   ├── utils/              # Utilities
│   │   └── visualization.py # Plotting functions
│   └── examples/           # Example scripts
│       ├── basic_example.py
│       └── benchmark_experiment.py
├── tests/                  # Test suite
│   ├── test_optimizer.py
│   └── test_operators.py
├── docs/                   # Documentation
│   └── ARCHITECTURE.md
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # BSD 3-Clause License
```

## Quick Start

### Basic Usage

```python
from nevo import NEVOptimiser
from ioh import get_problem

# Define problem
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

### Visualisation

```python
from nevo.utils import plot_optimisation_results

plot_optimisation_results(
    optimiser,
    optimum=problem.optimum.y,
    title='My Optimisation',
    save_path='result.png'
)
```

### Custom Operators

```python
from nevo.operators.base import ExplorationOperator
import numpy as np

class MyOperator(ExplorationOperator):
    def __init__(self):
        super().__init__("MyOperator")
    
    def generate_population(self, centre, state, population_size):
        dim = len(centre)
        candidates = []
        for _ in range(population_size):
            noise = np.random.randn(dim) * 0.3
            candidate = centre + noise
            candidates.append(np.clip(candidate, -1.0, 1.0))
        return np.array(candidates)

# Use custom operator
optimizer = NEVOptimizer(
    objective_function=problem,
    bounds=(-5, 5),
    dimension=10,
    operators=[MyOperator()],
)
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=nevo tests/

# Run specific test file
pytest tests/test_operators.py -v
```

### Code Formatting

```bash
# Format code
black nevo/
isort nevo/

# Check formatting
black --check nevo/
```

### Building Documentation

```bash
cd docs
make html
```

## Troubleshooting

### Nengo Installation Issues

If you encounter issues with Nengo:
```bash
pip install --upgrade nengo
```

### IOHexperimenter Issues

If IOH fails to install:
```bash
pip install --upgrade ioh
```

### Import Errors

Make sure NEVO is installed in editable mode:
```bash
pip install -e .
```

Verify installation:
```python
import nevo
print(nevo.__version__)
```

### Matplotlib LaTeX Errors

If LaTeX rendering fails in plots:
```python
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False  # Disable LaTeX
```

## Next Steps

1. **Read the architecture guide**: See `docs/ARCHITECTURE.md` for detailed design principles
2. **Run examples**: Try `nevo/examples/basic_example.py`
3. **Run benchmarks**: See `nevo/examples/benchmark_experiment.py`
4. **Read contributing guide**: See `CONTRIBUTING.md` to add new features
5. **Explore operators**: Check `nevo/operators/standard.py` for implementation details

## Getting Help

- **Issues**: https://github.com/jcrvz/nevo/issues
- **Discussions**: https://github.com/jcrvz/nevo/discussions
- **Documentation**: https://nevo.readthedocs.io (coming soon)

## License

NEVO is released under the BSD 3-Clause License. See `LICENSE` for details.

