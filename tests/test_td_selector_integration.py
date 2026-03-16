#!/usr/bin/env python3
"""
Integration Test: TD Learning with BasalGangliaSelector
======================================================

Simple end-to-end test of TD learning integration.
"""

import numpy as np
from nevo.core.basal_ganglia import BasalGangliaSelector
from nevo.core.td_learning import (
    SimpleTDRule,
    ConservativeTDRule,
    LinearValueModel,
    BoundedValueModel,
)
from nevo.operators.standard import LevyFlight, ParticleSwarm, SpiralOptimisation


def test_td0_basic():
    """Test TD(0) learning basic functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: TD(0) Learning Basic")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm(), SpiralOptimisation()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        lambda_coeff=0.0,  # TD(0)
        learning_rate=0.1,
        gamma=0.99,
    )

    print(f"✓ Created selector with {len(operators)} operators")
    print(f"✓ TD learning enabled: {selector.td_enabled}")
    print(f"✓ λ = {selector.td_learner.lambda_coeff} (TD(0))")

    # Begin episode
    selector.begin_episode()

    # Simulate operator selections
    initial_values = selector.get_td_values().copy()
    print(f"  Initial TD values: {initial_values}")

    for i in range(5):
        # Random operator selection signal
        op_signal = np.random.rand(len(operators))
        op_signal /= np.sum(op_signal) + 1e-8

        fitness_improved = i > 0 and i % 2 == 0
        fitness = 10.0 - (i if fitness_improved else 0)

        op = selector.select_operator(op_signal, fitness)
        print(f"  Iteration {i}: Selected {op.name}")

    final_values = selector.get_td_values()
    print(f"  Final TD values: {final_values}")

    # Verify learning
    has_changed = not np.allclose(initial_values, final_values, atol=1e-6)
    assert has_changed, "TD values did not change!"

    print("✓ TEST 1 PASSED")


def test_td_lambda():
    """Test TD(λ) learning."""
    print("\n" + "=" * 70)
    print("TEST 2: TD(λ) Learning")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        lambda_coeff=0.9,  # TD(0.9)
        learning_rate=0.1,
    )

    selector.begin_episode()

    print(f"✓ TD(λ) with λ = {selector.td_learner.lambda_coeff}")

    initial_values = selector.get_td_values().copy()

    # Simulate operator sequence
    op_signals = [
        np.array([1.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([1.0, 0.5]),
        np.array([0.3, 1.0]),
    ]

    for i, op_signal in enumerate(op_signals):
        op_signal = op_signal / (np.sum(op_signal) + 1e-8)
        fitness = 10.0 - i * 0.5

        op = selector.select_operator(op_signal, fitness)
        print(f"  Iteration {i}: Selected {op.name}")

    final_values = selector.get_td_values()
    has_changed = not np.allclose(initial_values, final_values, atol=1e-6)
    assert has_changed, "TD(λ) values did not change!"

    print("✓ TEST 2 PASSED")


def test_rule_switching():
    """Test dynamic learning rule switching."""
    print("\n" + "=" * 70)
    print("TEST 3: Dynamic Rule Switching")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        learning_rule=SimpleTDRule(),
    )

    selector.begin_episode()

    print("  Phase 1: SimpleTDRule")
    for i in range(2):
        op_signal = np.random.rand(len(operators))
        selector.select_operator(op_signal, 10.0 - i)

    print("  Phase 2: ConservativeTDRule")
    selector.set_learning_rule(ConservativeTDRule(stability_weight=0.5))
    for i in range(2):
        op_signal = np.random.rand(len(operators))
        selector.select_operator(op_signal, 10.0 - i)

    td_values = selector.get_td_values()
    print(f"  Final TD values: {td_values}")

    print("✓ TEST 3 PASSED")


def test_value_model_switching():
    """Test dynamic value model switching."""
    print("\n" + "=" * 70)
    print("TEST 4: Dynamic Value Model Switching")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        value_model=LinearValueModel(len(operators)),
    )

    selector.begin_episode()

    print("  Phase 1: LinearValueModel")
    for i in range(2):
        op_signal = np.random.rand(len(operators))
        selector.select_operator(op_signal, 10.0 - i)

    linear_values = selector.get_td_values().copy()
    print(f"  Linear values: {linear_values}")

    print("  Phase 2: BoundedValueModel")
    selector.set_value_model(
        BoundedValueModel(
            n_operators=len(operators),
            min_bound=0.2,
            max_bound=2.0,
        )
    )

    for i in range(2):
        op_signal = np.random.rand(len(operators))
        selector.select_operator(op_signal, 10.0 - i)

    bounded_values = selector.get_td_values()
    print(f"  Bounded values: {bounded_values}")

    # Verify bounds
    assert np.all(bounded_values >= 0.2), "Values below minimum!"
    assert np.all(bounded_values <= 2.0), "Values above maximum!"

    print("✓ TEST 4 PASSED")


def test_parameter_monitoring():
    """Test monitoring TD parameters."""
    print("\n" + "=" * 70)
    print("TEST 5: Parameter Monitoring")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm(), SpiralOptimisation()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
    )

    selector.begin_episode()

    # Run a few iterations
    for i in range(3):
        op_signal = np.array([1.0, 0.5, 0.3])
        op_signal = op_signal / np.sum(op_signal)
        selector.select_operator(op_signal, 10.0 - i)

    # Get all available statistics
    td_values = selector.get_td_values()
    utility_weights = selector.get_utility_weights()
    stats = selector.get_td_statistics()

    print(f"  TD values: {td_values}")
    print(f"  Utility weights: {utility_weights}")
    print(f"  TD stats: {stats}")

    assert len(td_values) == len(operators)
    assert len(utility_weights) == len(operators)

    print("✓ TEST 5 PASSED")


def test_backward_compatibility():
    """Test that TD can be disabled."""
    print("\n" + "=" * 70)
    print("TEST 6: Backward Compatibility")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=False,  # Disable TD learning
    )

    print(f"✓ TD learning disabled: {not selector.td_enabled}")

    selector.begin_episode()

    # Should still work
    for i in range(3):
        op_signal = np.random.rand(len(operators))
        op = selector.select_operator(op_signal, 10.0 - i)
        print(f"  Iteration {i}: Selected {op.name}")

    # TD values should be defaults
    td_values = selector.get_td_values()
    assert np.allclose(td_values, 0.5), "TD should return defaults when disabled!"

    print("✓ TEST 6 PASSED - Backward compatible")


def test_lambda_switching():
    """Test dynamic λ switching."""
    print("\n" + "=" * 70)
    print("TEST 7: Dynamic Lambda Switching")
    print("=" * 70)

    operators = [LevyFlight(), ParticleSwarm()]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        lambda_coeff=0.0,  # Start with TD(0)
    )

    selector.begin_episode()

    print(f"  Initial λ = {selector.td_learner.lambda_coeff}")

    # Switch to TD(0.7)
    selector.set_td_lambda(0.7)
    print(f"  Switched to λ = {selector.td_learner.lambda_coeff}")
    assert selector.td_learner.lambda_coeff == 0.7

    # Switch to TD(λ)
    selector.set_td_lambda(0.95)
    print(f"  Switched to λ = {selector.td_learner.lambda_coeff}")
    assert selector.td_learner.lambda_coeff == 0.95

    print("✓ TEST 7 PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TD Learning Integration Tests - BasalGangliaSelector")
    print("=" * 70)

    try:
        test_td0_basic()
        test_td_lambda()
        test_rule_switching()
        test_value_model_switching()
        test_parameter_monitoring()
        test_backward_compatibility()
        test_lambda_switching()

        print("\n" + "=" * 70)
        print("✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
