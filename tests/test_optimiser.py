"""
Basic tests for NEVO optimiser.
"""

import numpy as np
import pytest
from nevo import NEVOptimiser
from nevo.operators import get_operator


def sphere(x):
    """Simple sphere function for testing."""
    return np.sum(x**2)


def test_optimiser_initialisation():
    """Test that optimiser initialises correctly."""
    optimiser = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=10,
        population_size=10,
        memory_size=5,
    )

    assert optimiser.dimension == 10
    assert optimiser.population_size == 10
    assert optimiser.memory_size == 5
    assert len(optimiser.operators) == 4  # Default operators


def test_optimiser_run():
    """Test that optimiser runs without errors."""
    optimiser = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=5,
        population_size=10,
        memory_size=5,
        seed=42,
    )

    # Run for short time
    optimiser.run(time=1.0, verbose=False)

    # Check that optimisation happened
    assert optimiser.state["total_evals"] > 0
    assert optimiser.state["best_f"] is not None
    assert optimiser.state["best_v"] is not None


def test_get_best_solution():
    """Test getting best solution."""
    optimiser = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=5,
        population_size=10,
        memory_size=5,
        seed=42,
    )

    optimiser.run(time=1.0, verbose=False)

    x_best, f_best = optimiser.get_best_solution()

    assert x_best is not None
    assert f_best is not None
    assert len(x_best) == 5
    assert isinstance(f_best, float)


def test_operator_loading():
    """Test loading operators from registry."""
    levy = get_operator("LevyFlight")
    assert levy.name == "LevyFlight"

    de = get_operator("DifferentialEvolution")
    assert de.name == "DifferentialEvolution"

    with pytest.raises(ValueError):
        get_operator("NonexistentOperator")


def test_custom_operators():
    """Test using custom operators."""
    from nevo.operators.standard import LevyFlight, ParticleSwarm

    custom_ops = [LevyFlight(), ParticleSwarm()]

    optimiser = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=5,
        population_size=10,
        memory_size=5,
        operators=custom_ops,
        seed=42,
    )

    assert len(optimiser.operators) == 2
    optimiser.run(time=0.5, verbose=False)

    # Both operators should have been used
    stats = optimiser.get_statistics()
    assert sum(stats["operator_counts"].values()) > 0

