#!/usr/bin/env python3
"""
Integration Test: TD Learning with NEVO Optimiser
================================================

Complete end-to-end test of TD learning integrated with NEVO.
"""

import numpy as np
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    ConservativeTDRule,
    LinearValueModel,
    BoundedValueModel,
)


def test_td0_with_optimiser():
    """Test TD(0) learning integrated with optimiser."""
    print("\n" + "="*70)
    print("TEST 1: TD(0) Learning with NEVOptimiser")
    print("="*70)

    def sphere_function(x):
        return np.sum(x**2)

    # Create optimiser with TD(0) learning
    optimizer = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        seed=42,
    )

    # Start episode
    optimizer.bg_selector.begin_episode()

    print(f"✓ Optimiser created with TD learning enabled")
    print(f"✓ Number of operators: {len(optimizer.bg_selector.operators)}")
    print(f"✓ TD learning enabled: {optimizer.bg_selector.td_enabled}")
    print(f"✓ λ coefficient: {optimizer.bg_selector.td_learner.lambda_coeff}")

    # Run optimization
    best_fitness_history = []
    for i in range(5):
        best_fitness = optimizer.run(time=0.1)
        best_fitness_history.append(best_fitness)

        if i % 2 == 0:
            td_values = optimizer.bg_selector.get_td_values()
            print(f"  Iteration {i}: fitness={best_fitness:.6f}, TD values={td_values}")

    # Verify TD learning occurred
    initial_values = np.full(len(optimizer.bg_selector.operators), 0.5)
    final_values = optimizer.bg_selector.get_td_values()

    has_changed = not np.allclose(initial_values, final_values, atol=1e-6)
    assert has_changed, "TD values did not change - learning not working!"

    print(f"\n✓ TD values changed: {initial_values} → {final_values}")
    print("✓ TEST 1 PASSED")


def test_td_lambda_with_optimiser():
    """Test TD(λ) learning integrated with optimiser."""
    print("\n" + "="*70)
    print("TEST 2: TD(λ) Learning with NEVOptimiser")
    print("="*70)

    def rosenbrock_function(x):
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                   for i in range(len(x) - 1))

    optimizer = NEVOptimiser(
        objective_function=rosenbrock_function,
        bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        seed=42,
    )

    # Switch to TD(λ)
    optimizer.bg_selector.set_td_lambda(0.9)
    optimizer.bg_selector.begin_episode()

    print(f"✓ Switched to TD(λ) learning")
    print(f"✓ λ = {optimizer.bg_selector.td_learner.lambda_coeff}")

    # Run optimization
    for i in range(5):
        best_fitness = optimizer.optimize(num_iterations=1)

        if i % 2 == 0:
            stats = optimizer.bg_selector.get_td_statistics()
            print(f"  Iteration {i}: fitness={best_fitness:.6f}, "
                  f"mean_td_error={stats.get('mean_td_error', 0.0):.6f}")

    print("✓ TEST 2 PASSED")


def test_dynamic_rule_switching():
    """Test dynamic rule switching during optimization."""
    print("\n" + "="*70)
    print("TEST 3: Dynamic Learning Rule Switching")
    print("="*70)

    def sphere_function(x):
        return np.sum(x**2)

    optimizer = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        seed=42,
    )

    optimizer.bg_selector.begin_episode()

    # Phase 1: Simple rule
    print("  Phase 1: SimpleTDRule")
    optimizer.bg_selector.set_learning_rule(SimpleTDRule())
    optimizer.optimize(num_iterations=1)
    print("  ✓ SimpleTDRule working")

    # Phase 2: Conservative rule
    print("  Phase 2: ConservativeTDRule")
    optimizer.bg_selector.set_learning_rule(
        ConservativeTDRule(stability_weight=0.5)
    )
    optimizer.optimize(num_iterations=1)
    print("  ✓ ConservativeTDRule working")

    # Verify both rules are functional
    td_values = optimizer.bg_selector.get_td_values()
    assert td_values is not None and len(td_values) > 0

    print("✓ TEST 3 PASSED")


def test_dynamic_value_model_switching():
    """Test dynamic value model switching during optimization."""
    print("\n" + "="*70)
    print("TEST 4: Dynamic Value Model Switching")
    print("="*70)

    def sphere_function(x):
        return np.sum(x**2)

    optimizer = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.2,  # Higher learning rate
        seed=42,
    )

    optimizer.bg_selector.begin_episode()

    # Phase 1: Linear model
    print("  Phase 1: LinearValueModel")
    optimizer.bg_selector.set_value_model(
        LinearValueModel(len(optimizer.bg_selector.operators))
    )
    for _ in range(2):
        optimizer.optimize(num_iterations=1)
    linear_values = optimizer.bg_selector.get_td_values()
    print(f"  ✓ Linear values: {linear_values}")

    # Phase 2: Bounded model
    print("  Phase 2: BoundedValueModel")
    optimizer.bg_selector.set_value_model(
        BoundedValueModel(
            n_operators=len(optimizer.bg_selector.operators),
            min_bound=0.2,
            max_bound=2.0,
            adapt_bounds=True,
        )
    )
    for _ in range(2):
        optimizer.optimize(num_iterations=1)
    bounded_values = optimizer.bg_selector.get_td_values()
    print(f"  ✓ Bounded values: {bounded_values}")

    # Verify bounds are respected
    assert np.all(bounded_values >= 0.2), "Values below minimum bound!"
    assert np.all(bounded_values <= 2.0), "Values above maximum bound!"

    print("✓ TEST 4 PASSED")


def test_parameter_monitoring():
    """Test monitoring TD learning parameters."""
    print("\n" + "="*70)
    print("TEST 5: Parameter Monitoring")
    print("="*70)

    def sphere_function(x):
        return np.sum(x**2)

    optimizer = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        seed=42,
    )

    optimizer.bg_selector.begin_episode()
    selector = optimizer.bg_selector

    # Run optimization and monitor
    for i in range(5):
        optimizer.optimize(num_iterations=1)

        if i % 2 == 0:
            # Get various statistics
            td_values = selector.get_td_values()
            utility_weights = selector.get_utility_weights()
            stats = selector.get_td_statistics()

            print(f"\n  Iteration {i}:")
            print(f"    TD values: {td_values}")
            print(f"    Utility weights: {utility_weights}")
            print(f"    TD stats: {stats}")

            # Verify all metrics are present and reasonable
            assert len(td_values) > 0
            assert len(utility_weights) > 0
            if stats:
                assert 'mean_td_error' in stats
                assert 'std_td_error' in stats

    print("\n✓ TEST 5 PASSED")


def test_backward_compatibility():
    """Test backward compatibility (can disable TD learning)."""
    print("\n" + "="*70)
    print("TEST 6: Backward Compatibility")
    print("="*70)

    def sphere_function(x):
        return np.sum(x**2)

    # Create optimiser and disable TD learning
    optimizer = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        seed=42,
    )

    # Disable TD learning
    optimizer.bg_selector.td_enabled = False
    optimizer.bg_selector.begin_episode()

    print(f"✓ TD learning disabled: {not optimizer.bg_selector.td_enabled}")

    # Run optimization (should still work)
    for i in range(3):
        best_fitness = optimizer.optimize(num_iterations=1)
        print(f"  Iteration {i}: fitness={best_fitness:.6f}")

    # TD values should be defaults
    td_values = optimizer.bg_selector.get_td_values()
    assert np.allclose(td_values, 0.5), "TD values should be default when disabled!"

    print("✓ TEST 6 PASSED - Backward compatible")


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("NEVO TD Learning Integration Tests")
    print("="*70)

    try:
        test_td0_with_optimiser()
        test_td_lambda_with_optimiser()
        test_dynamic_rule_switching()
        test_dynamic_value_model_switching()
        test_parameter_monitoring()
        test_backward_compatibility()

        print("\n" + "="*70)
        print("✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓")
        print("="*70)
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

