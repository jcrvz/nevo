#!/usr/bin/env python3
"""
Example: TD Learning in NEVO Optimiser
======================================

Demonstrates TD(0) and TD(λ) learning with the NEVO optimiser.
Neuromorphic Nengo networks are preserved throughout.
Fitness is read from ``optimiser.state["best_f"]`` after each run segment.
"""

import numpy as np
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    ConservativeTDRule,
    AdaptiveTDRule,
    BoundedValueModel,
)

_SEG = 0.1   # seconds per simulation segment

# Simple optimisation functions
def sphere_function(x):
    """Sphere function."""
    return float(np.sum(x**2))

def rosenbrock_function(x):
    """Rosenbrock function."""
    return float(sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(len(x) - 1)
    ))


def _best(opt: NEVOptimiser) -> float:
    """Return best fitness found so far (None-safe)."""
    f = opt.state.get("best_f")
    return f if f is not None else float("inf")


def example_td0_learning():
    """
    Example 1: TD(0) learning.

    TD(0) bootstraps from the immediate next state value.
    Only the current operator's value is updated per step.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: TD(0) Learning")
    print("=" * 70)

    optimiser = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        td_enabled=True,
        td_lambda=0.0,
        seed=42,
    )

    print("  Discount factor gamma = 0.99")
    print("  Learning rate alpha = 0.1")
    print("  lambda = 0.0 (TD(0), no eligibility traces)")

    for iteration in range(3):
        optimiser.run(time=_SEG, verbose=False)

        if iteration % 2 == 0:
            td_values = optimiser.bg_selector.get_td_values()
            stats = optimiser.bg_selector.get_td_statistics()
            print(f"\n  Segment {iteration}:")
            print(f"    Best fitness: {_best(optimiser):.6f}")
            print(f"    TD values: {td_values}")
            print(f"    Mean TD error: {stats['mean_td_error']:.6f}")

    print("\n  TD(0) example completed.")


def example_td_lambda_learning():
    """
    Example 2: TD(lambda=0.9) learning with eligibility traces.

    Credit propagates to previously visited operators.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: TD(lambda) Learning with Eligibility Traces")
    print("=" * 70)

    optimiser = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        td_enabled=True,
        td_lambda=0.9,
        seed=42,
    )

    print("  lambda = 0.9 (multi-step credit with eligibility traces)")

    for iteration in range(3):
        optimiser.run(time=_SEG, verbose=False)

        if iteration % 2 == 0:
            td_values = optimiser.bg_selector.get_td_values()
            print(f"\n  Segment {iteration}:")
            print(f"    Best fitness: {_best(optimiser):.6f}")
            print(f"    TD values: {td_values}")

    print("\n  TD(lambda) example completed.")


def example_dynamic_rule_switching():
    """
    Example 3: Dynamic learning rule switching.

    Switch between learning rules during optimisation to adapt
    to different phases of the search.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Dynamic Learning Rule Switching")
    print("=" * 70)

    optimiser = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        td_enabled=True,
        seed=42,
    )

    # Phase 1: simple rule (early exploration)
    print("\n  Phase 1: SimpleTDRule (exploration)")
    optimiser.bg_selector.set_learning_rule(SimpleTDRule())
    optimiser.run(time=_SEG, verbose=False)
    print(f"    Best fitness: {_best(optimiser):.6f}")

    # Phase 2: conservative rule (stability)
    print("\n  Phase 2: ConservativeTDRule (stability)")
    optimiser.bg_selector.set_learning_rule(ConservativeTDRule(stability_weight=0.5))
    optimiser.run(time=_SEG, verbose=False)
    print(f"    Best fitness: {_best(optimiser):.6f}")

    # Phase 3: adaptive rule (fine-tuning)
    print("\n  Phase 3: AdaptiveTDRule (fine-tuning)")
    optimiser.bg_selector.set_learning_rule(AdaptiveTDRule(window_size=10))
    optimiser.run(time=_SEG, verbose=False)
    print(f"    Best fitness: {_best(optimiser):.6f}")

    print("\n  Dynamic rule switching example completed.")


def example_dynamic_value_model_switching():
    """
    Example 4: Dynamic value model switching.

    Switch from LinearValueModel to BoundedValueModel to prevent
    value explosion in later stages of the search.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Dynamic Value Model Switching")
    print("=" * 70)

    optimiser = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.1,
        td_enabled=True,
        seed=42,
    )

    print(f"  Initial TD values: {optimiser.bg_selector.get_td_values()}")

    # Phase 1: linear model
    optimiser.run(time=_SEG, verbose=False)
    optimiser.run(time=_SEG, verbose=False)
    print(f"\n  After linear model (2 segments): {optimiser.bg_selector.get_td_values()}")

    # Phase 2: bounded model
    print("\n  Switching to BoundedValueModel [0.2, 2.0]")
    bounded_model = BoundedValueModel(
        n_operators=len(optimiser.bg_selector.operators),
        initial_value=0.5,
        min_bound=0.2,
        max_bound=2.0,
        adapt_bounds=True,
    )
    optimiser.bg_selector.set_value_model(bounded_model)

    optimiser.run(time=_SEG, verbose=False)
    optimiser.run(time=_SEG, verbose=False)
    print(f"  After bounded model (2 segments): {optimiser.bg_selector.get_td_values()}")

    print("\n  Dynamic value model switching example completed.")


def example_td_parameter_tuning():
    """
    Example 5: TD parameter tuning during optimisation.

    Adjust learning rate and lambda dynamically based on search progress.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: TD Parameter Tuning During Optimisation")
    print("=" * 70)

    optimiser = NEVOptimiser(
        objective_function=rosenbrock_function,
        bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.2,
        td_enabled=True,
        seed=42,
    )

    selector = optimiser.bg_selector
    print("  Initial learning rate: 0.2,  lambda: 0.0 (TD(0))")

    for segment in range(6):
        optimiser.run(time=_SEG, verbose=False)

        # Increase learning rate at segment 2
        if segment == 2:
            print("\n  Segment 2: increasing learning rate to 0.3")
            selector.td_learner.set_learning_rate(0.3)

        # Switch to TD(lambda) and reduce learning rate at segment 4
        if segment == 4:
            print("\n  Segment 4: switching to TD(lambda=0.7), learning rate 0.05")
            selector.set_td_lambda(0.7)
            selector.td_learner.set_learning_rate(0.05)

        if segment % 2 == 0:
            stats = selector.get_td_statistics()
            print(
                f"  Segment {segment}: fitness={_best(optimiser):.6f}"
                f"  mean_td_error={stats['mean_td_error']:.6f}"
            )

    print("\n  Parameter tuning example completed.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  TD Learning Examples for NEVO Optimiser")
    print("=" * 70)

    example_td0_learning()
    example_td_lambda_learning()
    example_dynamic_rule_switching()
    example_dynamic_value_model_switching()
    example_td_parameter_tuning()

    print("\n" + "=" * 70)
    print("  All examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
