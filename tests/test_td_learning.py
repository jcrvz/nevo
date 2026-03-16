#!/usr/bin/env python3
"""
Test Temporal Difference (TD) Learning Implementation
=====================================================

Tests TD(0), TD(λ), and pluggable learning rules/value models in basal ganglia.
"""

import numpy as np
import sys

# Test imports
try:
    from nevo.core.td_learning import (
        TemporalDifferenceLearner,
        SimpleTDRule,
        DecayingTDRule,
        ConservativeTDRule,
        AdaptiveTDRule,
        LinearValueModel,
        BoundedValueModel,
        EligibilityTraceManager,
    )
    from nevo.core.basal_ganglia import BasalGangliaSelector
    from nevo.operators.base import Operator

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


# Dummy operator for testing
class DummyOperator(Operator):
    """Dummy operator for testing."""

    def __init__(self, name: str):
        super().__init__(name=name, operator_type="exploration")
        self.n_calls = 0

    def generate_population(self, centre, state, population_size):
        """Generate random population."""
        self.n_calls += 1
        return np.random.randn(population_size, centre.shape[0])

    def generate(self, state, memory_vectors, memory_fitness):
        """Legacy generate method."""
        return np.random.randn(memory_vectors.shape[0], memory_vectors.shape[1])


def test_simple_td_rule():
    """Test SimpleTDRule."""
    print("\n=== Test SimpleTDRule ===")
    rule = SimpleTDRule()

    td_error = 0.5
    learning_rate = 0.1
    current_value = 1.0

    update = rule.compute_update(td_error, learning_rate, current_value)
    expected = learning_rate * td_error  # 0.05

    assert abs(update - expected) < 1e-6, f"Expected {expected}, got {update}"
    print(f"✓ SimpleTDRule: TD error {td_error:.3f} → update {update:.6f}")


def test_decaying_td_rule():
    """Test DecayingTDRule with different decay types."""
    print("\n=== Test DecayingTDRule ===")

    # Exponential decay
    rule_exp = DecayingTDRule(decay_type="exponential", decay_rate=0.9)
    td_error = 1.0
    learning_rate = 0.1
    current_value = 0.5

    updates = []
    for t in range(5):
        update = rule_exp.compute_update(
            td_error, learning_rate, current_value, timestep=t
        )
        updates.append(update)
        print(f"  Timestep {t}: update = {update:.6f}")

    # Verify decay (updates decrease)
    assert updates[0] > updates[1] > updates[2], "Exponential decay not working"
    print("✓ DecayingTDRule exponential decay working")

    # Linear decay
    rule_lin = DecayingTDRule(decay_type="linear", decay_rate=0.2)
    update_lin = rule_lin.compute_update(
        td_error, learning_rate, current_value, timestep=2
    )
    print(f"✓ DecayingTDRule linear decay: update = {update_lin:.6f}")


def test_conservative_td_rule():
    """Test ConservativeTDRule."""
    print("\n=== Test ConservativeTDRule ===")
    rule = ConservativeTDRule(stability_weight=0.5)

    td_error = 2.0
    learning_rate = 0.5
    current_value = 1.0

    update = rule.compute_update(td_error, learning_rate, current_value)
    print(f"✓ ConservativeTDRule: TD error {td_error:.3f} → damped update {update:.6f}")

    # Verify dampening
    simple_rule = SimpleTDRule()
    simple_update = simple_rule.compute_update(td_error, learning_rate, current_value)
    assert abs(update) < abs(simple_update), "Dampening not working"
    print("✓ ConservativeTDRule dampening verified")


def test_adaptive_td_rule():
    """Test AdaptiveTDRule."""
    print("\n=== Test AdaptiveTDRule ===")
    rule = AdaptiveTDRule(window_size=5)

    td_error = 1.0
    learning_rate = 0.1
    current_value = 0.5

    # First update with small error
    update1 = rule.compute_update(td_error, learning_rate, current_value)
    print(f"  First update (error=1.0): {update1:.6f}")

    # Many updates with large error
    for _ in range(5):
        large_error = 10.0
        _ = rule.compute_update(large_error, learning_rate, current_value)

    # Update with same error but larger history
    update2 = rule.compute_update(td_error, learning_rate, current_value)
    print(f"  Later update (error=1.0): {update2:.6f}")

    # Verify adaptation (smaller update with high error history)
    assert abs(update2) < abs(update1), "Adaptation not working"
    print("✓ AdaptiveTDRule adaptation verified")


def test_linear_value_model():
    """Test LinearValueModel."""
    print("\n=== Test LinearValueModel ===")
    n_ops = 3
    model = LinearValueModel(n_ops, initial_value=0.5)

    # Get initial values
    values = model.get_values_array()
    assert np.allclose(values, 0.5), "Initial values incorrect"
    print(f"✓ Initial values: {values}")

    # Update value
    model.update(0, 0.3)
    values = model.get_values_array()
    print(f"✓ After update(0, 0.3): {values}")
    assert values[0] > 0.5, "Update not applied"

    # Test clipping
    model.set_value(1, 100.0)
    assert model.get_value(1) <= 5.0, "Clipping not working"
    print(f"✓ Value clipping works (100.0 → {model.get_value(1):.1f})")


def test_bounded_value_model():
    """Test BoundedValueModel."""
    print("\n=== Test BoundedValueModel ===")
    n_ops = 3
    model = BoundedValueModel(
        n_ops, initial_value=0.5, min_bound=0.2, max_bound=2.0, adapt_bounds=False
    )

    values = model.get_values_array()
    print(f"✓ Initial bounded values: {values}")

    # Try to set out of bounds
    model.set_value(0, 10.0)
    assert model.get_value(0) <= 2.0, "Bounds not enforced"
    print(f"✓ Out-of-bounds value clipped to {model.get_value(0):.1f}")


def test_eligibility_traces():
    """Test EligibilityTraceManager."""
    print("\n=== Test EligibilityTraceManager ===")
    n_ops = 4
    manager = EligibilityTraceManager(n_ops, lambda_coeff=0.9, trace_decay=0.95)

    # Visit operator 0
    manager.update_trace(0)
    traces = manager.get_traces()
    print(f"✓ After visiting op 0: {traces}")
    assert traces[0] == 1.0, "Trace not set"

    # Decay and visit operator 1
    manager.update_trace(1)
    traces = manager.get_traces()
    print(f"✓ After visiting op 1 (with decay): {traces}")

    # Check decay: op 0 trace should be less than 1.0
    assert 0.0 < traces[0] < 1.0, "Decay not working"
    assert traces[1] > 0.0, "New trace not set"
    print("✓ Eligibility trace decay working")


def test_temporal_difference_learner_td0():
    """Test TemporalDifferenceLearner with TD(0)."""
    print("\n=== Test TemporalDifferenceLearner TD(0) ===")
    n_ops = 3

    learner = TemporalDifferenceLearner(
        n_operators=n_ops,
        learning_rate=0.1,
        gamma=0.99,
        lambda_coeff=0.0,  # TD(0)
        learning_rule=SimpleTDRule(),
        value_model=LinearValueModel(n_ops),
    )

    learner.begin_episode()

    # Simulate operator selection and reward
    operator_idx = 0
    reward = 1.0
    next_values = learner.get_values()
    next_state_value = np.max(next_values)

    update_info = learner.update(
        operator_idx, reward, next_state_value, is_terminal=False
    )

    print("✓ TD(0) update info:")
    print(f"  TD error: {update_info['td_error']:.6f}")
    print(f"  TD target: {update_info['td_target']:.6f}")
    print(f"  Updated values: {learner.get_values()}")

    # Verify value changed
    assert learner.get_value(0) != 0.5, "Value not updated"
    print("✓ TD(0) value updated successfully")


def test_temporal_difference_learner_td_lambda():
    """Test TemporalDifferenceLearner with TD(λ)."""
    print("\n=== Test TemporalDifferenceLearner TD(λ) ===")
    n_ops = 3

    learner = TemporalDifferenceLearner(
        n_operators=n_ops,
        learning_rate=0.1,
        gamma=0.99,
        lambda_coeff=0.9,  # TD(0.9)
        learning_rule=SimpleTDRule(),
        value_model=LinearValueModel(n_ops),
    )

    learner.begin_episode()

    # Simulate sequence of operator visits
    ops_sequence = [0, 1, 0, 2, 0]
    reward = 1.0

    for op_idx in ops_sequence:
        next_values = learner.get_values()
        next_state_value = np.max(next_values)

        learner.update(op_idx, reward, next_state_value, is_terminal=False)

    values = learner.get_values()
    print(f"✓ TD(0.9) final values: {values}")

    # With λ=0.9, visited operators should have updated values
    assert values[0] != 0.5, "Visited operator 0 not updated"
    print("✓ TD(λ) eligibility traces working")


def test_basal_ganglia_selector_with_td():
    """Test BasalGangliaSelector with TD learning."""
    print("\n=== Test BasalGangliaSelector with TD Learning ===")

    # Create dummy operators
    operators = [DummyOperator(f"Op{i}") for i in range(3)]

    # Create selector with TD learning enabled
    selector = BasalGangliaSelector(
        operators=operators,
        neurons_per_ensemble=50,
        epsilon=0.05,
        learning_rate=0.1,
        gamma=0.99,
        lambda_coeff=0.0,  # TD(0)
        td_enabled=True,
    )

    print(f"✓ BasalGangliaSelector created with {len(operators)} operators")
    print(f"✓ TD learning enabled: {selector.td_enabled}")

    # Begin episode
    selector.begin_episode()

    # Simulate operator selection
    operator_selection = np.array([1.0, 0.5, 0.3])

    # First selection
    best_fitness = 10.0
    op1 = selector.select_operator(operator_selection, best_fitness)
    print(f"✓ Selected operator: {op1.name}")

    # Second selection with improvement
    better_fitness = 8.0
    op2 = selector.select_operator(operator_selection, better_fitness)
    print(f"✓ Selected operator: {op2.name}")

    # Get TD values
    td_values = selector.get_td_values()
    print(f"✓ TD values: {td_values}")

    # Get TD statistics
    stats = selector.get_td_statistics()
    print(f"✓ TD statistics: {stats}")

    # Dynamic lambda adjustment
    selector.set_td_lambda(0.9)
    print("✓ Set TD lambda to 0.9")


def test_dynamic_rule_switching():
    """Test switching learning rules dynamically."""
    print("\n=== Test Dynamic Learning Rule Switching ===")

    operators = [DummyOperator(f"Op{i}") for i in range(2)]

    selector = BasalGangliaSelector(
        operators=operators,
        epsilon=0.0,
        td_enabled=True,
        learning_rule=SimpleTDRule(),
    )

    print("✓ Initial rule: SimpleTDRule")

    # Switch to conservative rule
    conservative_rule = ConservativeTDRule(stability_weight=0.5)
    selector.set_learning_rule(conservative_rule)
    print("✓ Switched to: ConservativeTDRule")

    # Switch to adaptive rule
    adaptive_rule = AdaptiveTDRule(window_size=10)
    selector.set_learning_rule(adaptive_rule)
    print("✓ Switched to: AdaptiveTDRule")


def test_dynamic_value_model_switching():
    """Test switching value models dynamically."""
    print("\n=== Test Dynamic Value Model Switching ===")

    operators = [DummyOperator(f"Op{i}") for i in range(2)]

    selector = BasalGangliaSelector(
        operators=operators,
        td_enabled=True,
        value_model=LinearValueModel(2),
    )

    print("✓ Initial model: LinearValueModel")

    # Switch to bounded model
    bounded_model = BoundedValueModel(2, min_bound=0.3, max_bound=3.0)
    selector.set_value_model(bounded_model)
    print("✓ Switched to: BoundedValueModel")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Temporal Difference Learning Implementation")
    print("=" * 60)

    try:
        # Test learning rules
        test_simple_td_rule()
        test_decaying_td_rule()
        test_conservative_td_rule()
        test_adaptive_td_rule()

        # Test value models
        test_linear_value_model()
        test_bounded_value_model()

        # Test eligibility traces
        test_eligibility_traces()

        # Test TD learner
        test_temporal_difference_learner_td0()
        test_temporal_difference_learner_td_lambda()

        # Test basal ganglia integration
        test_basal_ganglia_selector_with_td()
        test_dynamic_rule_switching()
        test_dynamic_value_model_switching()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
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
    sys.exit(main())
