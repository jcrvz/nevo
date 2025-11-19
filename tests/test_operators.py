"""
Tests for optimisation operators.
"""

import pytest
import numpy as np
from nevo.operators.standard import (
    LevyFlight,
    DifferentialEvolution,
    ParticleSwarm,
    SpiralOptimisation,
)


@pytest.fixture
def mock_state():
    """Create mock optimisation state."""
    return {
        "best_v": np.zeros(10),
        "best_f": 0.0,
        "memory_vectors": np.random.randn(5, 10),
        "memory_fitness": np.random.rand(5) * 10,
        "f_default_worst": 1e10,
    }


def test_levy_flight_operator(mock_state):
    """Test LevyFlight operator."""
    op = LevyFlight()
    centre = np.zeros(10)

    candidates = op.generate_population(centre, mock_state, population_size=10)

    assert candidates.shape == (10, 10)
    assert np.all(candidates >= -1.0)
    assert np.all(candidates <= 1.0)


def test_differential_evolution_operator(mock_state):
    """Test DifferentialEvolution operator."""
    op = DifferentialEvolution()
    centre = np.zeros(10)

    candidates = op.generate_population(centre, mock_state, population_size=10)

    assert candidates.shape == (10, 10)
    assert np.all(candidates >= -1.0)
    assert np.all(candidates <= 1.0)


def test_particle_swarm_operator(mock_state):
    """Test ParticleSwarm operator."""
    op = ParticleSwarm()
    centre = np.zeros(10)

    candidates = op.generate_population(centre, mock_state, population_size=10)

    assert candidates.shape == (10, 10)
    assert np.all(candidates >= -1.0)
    assert np.all(candidates <= 1.0)

    # Check velocity state
    assert len(op.velocities) == 10


def test_spiral_optimization_operator(mock_state):
    """Test SpiralOptimisation operator."""
    op = SpiralOptimisation()
    centre = np.zeros(10)

    candidates = op.generate_population(centre, mock_state, population_size=10)

    assert candidates.shape == (10, 10)
    assert np.all(candidates >= -1.0)
    assert np.all(candidates <= 1.0)


def test_operator_statistics():
    """Test operator statistics tracking."""
    op = LevyFlight()

    assert op.usage_count == 0
    assert op.success_count == 0

    # Update with success
    op.update_statistics(improved=True, improvement=1.5)
    assert op.usage_count == 1
    assert op.success_count == 1
    assert op.total_improvement == 1.5

    # Update with failure
    op.update_statistics(improved=False, improvement=0.0)
    assert op.usage_count == 2
    assert op.success_count == 1

    # Check parameters
    params = op.get_parameters()
    assert params["usage_count"] == 2
    assert params["success_rate"] == 0.5

    # Reset
    op.reset_statistics()
    assert op.usage_count == 0
    assert op.success_count == 0

