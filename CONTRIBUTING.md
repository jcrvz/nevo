# Contributing to NEVO

Thank you for your interest in contributing to NEVO! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/jcrvz/nevo.git
cd nevo
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**:
```bash
pip install -e ".[dev]"
```

4. **Run tests**:
```bash
pytest tests/
```

## Code Style

We follow standard Python conventions:

- **PEP 8** for code style
- **Black** for automatic formatting (line length: 88)
- **isort** for import sorting
- **Type hints** for all public functions

Format your code before committing:
```bash
black nevo/
isort nevo/
```

## Testing

All new features must include tests:

1. **Create test file**: `tests/test_yourfeature.py`
2. **Write tests**: Use pytest fixtures and assertions
3. **Run tests**: `pytest tests/test_yourfeature.py -v`
4. **Check coverage**: `pytest --cov=nevo tests/`

Example test:
```python
import pytest
from nevo.operators.base import Operator

def test_operator_initialization():
    """Test that operator initializes correctly."""
    # Your test here
    pass
```

## Adding New Operators

To add a new optimisation operator:

### 1. Create Operator Class

Create file `nevo/operators/myoperator.py`:

```python
from nevo.operators.base import ExplorationOperator
import numpy as np

class MyOperator(ExplorationOperator):
    """
    Brief description of your operator.
    
    Detailed explanation of the algorithm, when to use it,
    and any relevant references.
    
    Best used when: describe conditions
    """
    
    def __init__(self, param1: float = 1.0):
        """
        Parameters
        ----------
        param1 : float
            Description of parameter
        """
        super().__init__("MyOperator")
        self.param1 = param1
    
    def generate_population(
        self,
        centre: np.ndarray,
        state: Dict[str, Any],
        population_size: int
    ) -> np.ndarray:
        """Generate candidates using my algorithm."""
        dim = len(centre)
        candidates = []
        
        for _ in range(population_size):
            # Your algorithm here
            candidate = centre + np.random.randn(dim) * self.param1
            candidates.append(np.clip(candidate, -1.0, 1.0))
        
        return np.array(candidates)
```

### 2. Register Operator

Add to `nevo/operators/__init__.py`:

```python
from nevo.operators.myoperator import MyOperator

OPERATOR_REGISTRY: Dict[str, Type[Operator]] = {
    # ...existing operators...
    "MyOperator": MyOperator,
}
```

### 3. Define Utility Function

Add to `nevo/core/basal_ganglia.py`:

```python
def utility_my_operator(x: np.ndarray) -> float:
    """
    MyOperator utility: high when [condition].
    
    Input: [diversity, improvement_rate, convergence]
    """
    diversity, improvement, convergence = x
    # Your utility logic
    return diversity * 0.5 + convergence * 0.3

DEFAULT_UTILITY_FUNCTIONS = {
    # ...existing utilities...
    "MyOperator": utility_my_operator,
}
```

### 4. Add Tests

Create `tests/test_myoperator.py`:

```python
import pytest
import numpy as np
from nevo.operators.myoperator import MyOperator

def test_my_operator():
    """Test MyOperator generates valid candidates."""
    op = MyOperator(param1=0.5)
    centre = np.zeros(10)
    state = {
        "best_v": centre,
        "memory_vectors": np.random.randn(5, 10),
        "memory_fitness": np.random.rand(5),
        "f_default_worst": 1e10,
    }
    
    candidates = op.generate_population(centre, state, 20)
    
    assert candidates.shape == (20, 10)
    assert np.all(candidates >= -1.0)
    assert np.all(candidates <= 1.0)
```

### 5. Update Documentation

Add operator to `README.md` and `docs/ARCHITECTURE.md`.

## Adding New Features

### State Features

To add new state features:

1. **Extend `StateFeatures.compute()`** in `nevo/core/state.py`:
```python
def compute(self, state: Dict[str, Any]) -> np.ndarray:
    # ...existing features...
    
    # New feature
    my_feature = compute_my_feature(state)
    
    return np.array([
        features["diversity"],
        features["improvement_rate"],
        features["convergence"],
        my_feature,  # Add new feature
    ])
```

2. **Update ensemble dimensions** in `NEVOptimizer.build_model()`:
```python
state_ensemble = nengo.Ensemble(
    n_neurons=400,  # Increase for more dimensions
    dimensions=4,   # Was 3, now 4
    radius=1.5,
    label="StateEnsemble"
)
```

3. **Update utility functions** to use new feature.

### Visualisation

To add new plots:

1. **Create function** in `nevo/utils/visualisation.py`:
```python
def plot_my_visualisation(optimiser, save_path=None):
    """Create my custom visualisation."""
    # Your plotting code
    pass
```

2. **Add to example** in `nevo/examples/basic_example.py`.

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    One-line summary.
    
    Detailed explanation of what the function does,
    spanning multiple lines if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2 (default: "default")
        
    Returns
    -------
    result : bool
        Description of return value
        
    Examples
    --------
    >>> my_function(42, "test")
    True
    """
    pass
```

### README Updates

When adding features, update:
- Feature list in README.md
- Quick start example if API changes
- Architecture diagram if structure changes

## Pull Request Process

1. **Create feature branch**:
```bash
git checkout -b feature/my-feature
```

2. **Make changes**:
   - Write code
   - Add tests
   - Update documentation
   - Format code (`black`, `isort`)

3. **Commit changes**:
```bash
git add .
git commit -m "Add feature: brief description"
```

4. **Push to GitHub**:
```bash
git push origin feature/my-feature
```

5. **Create Pull Request**:
   - Go to GitHub repository
   - Click "New Pull Request"
   - Describe your changes
   - Link any related issues

6. **Address review feedback**:
   - Make requested changes
   - Push updates to same branch
   - PR will update automatically

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] Examples work
- [ ] CHANGELOG.md updated (if applicable)

## Code Review Guidelines

When reviewing PRs:

- **Functionality**: Does it work as intended?
- **Tests**: Are there sufficient tests?
- **Documentation**: Is it well documented?
- **Style**: Does it follow conventions?
- **Performance**: Any efficiency concerns?
- **Compatibility**: Works with existing code?

## Reporting Bugs

Use GitHub Issues with:

1. **Clear title**: Brief description of bug
2. **Description**: What happened vs. what you expected
3. **Reproduction**: Minimal code to reproduce
4. **Environment**: Python version, OS, package versions
5. **Output**: Error messages, tracebacks

Example:
```markdown
## Bug: Operator selection fails with 2 operators

**Description**: BasalGangliaSelector crashes when using only 2 operators.

**Reproduction**:
\```python
from nevo import NEVOptimizer
from nevo.operators.standard import LevyFlight, ParticleSwarm

optimizer = NEVOptimizer(
    objective_function=lambda x: sum(x**2),
    bounds=(-5, 5),
    dimension=10,
    operators=[LevyFlight(), ParticleSwarm()],
)
optimiser.run(1.0)
\```

**Error**:
\```
IndexError: index 2 is out of bounds for axis 0 with size 2
\```

**Environment**:
- Python 3.10
- nengo 3.2.0
- Ubuntu 22.04
```

## Feature Requests

Use GitHub Issues with:

1. **Clear title**: Brief feature description
2. **Motivation**: Why is this useful?
3. **Proposal**: How should it work?
4. **Examples**: Usage examples
5. **Alternatives**: Other approaches considered

## Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Email**: For private matters

Thank you for contributing to NEVO! 🎉

