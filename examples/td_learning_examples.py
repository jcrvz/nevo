#!/usr/bin/env python3
"""
Example: TD Learning in NEVO Optimiser
======================================

This example demonstrates how to use TD(0) and TD(λ) learning
with the NEVO optimiser while preserving Nengo neuromorphic networks.
"""

import numpy as np
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    DecayingTDRule,
    ConservativeTDRule,
    AdaptiveTDRule,
    LinearValueModel,
    BoundedValueModel,
)


def sphere_function(x):
    """Simple sphere function for testing."""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Rosenbrock function."""
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))


def example_td0_learning():
    """
    Example 1: TD(0) Learning - Basic temporal difference learning

    TD(0) uses bootstrapping from the immediate next state value.
    Updates only the current operator's value.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: TD(0) Learning")
    print("="*70)

    # Create optimiser with TD(0) learning
    optimiser = NEVOptimiser(
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

    # Configure TD(0) in basal ganglia
    optimiser.basal_ganglia_selector.begin_episode()

    print(f"✓ Created optimiser with TD(0) learning")
    print(f"✓ Discount factor γ = 0.99")
    print(f"✓ Learning rate α = 0.1")
    print(f"✓ λ = 0.0 (TD(0), no eligibility traces)")

    # Run optimization for 10 iterations
    for iteration in range(10):
        # Optimize
        best_fitness = optimiser.optimize(num_iterations=1)

        if iteration % 3 == 0:
            # Get TD values
            td_values = optimiser.basal_ganglia_selector.get_td_values()
            stats = optimiser.basal_ganglia_selector.get_td_statistics()

            print(f"\nIteration {iteration}:")
            print(f"  Best fitness: {best_fitness:.6f}")
            print(f"  TD values: {td_values}")
            print(f"  Mean TD error: {stats['mean_td_error']:.6f}")

    print("\n✓ TD(0) example completed")


def example_td_lambda_learning():
    """
    Example 2: TD(λ) Learning - Multi-step credit assignment

    TD(λ) uses eligibility traces to propagate credit to previously
    visited operators. λ=0.9 means heavy weight on recent history.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: TD(λ) Learning with Eligibility Traces")
    print("="*70)

    optimiser = NEVOptimiser(
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

    # Switch to TD(λ) learning
    optimiser.basal_ganglia_selector.set_td_lambda(0.9)
    optimiser.basal_ganglia_selector.begin_episode()

    print(f"✓ Created optimiser with TD(0.9) learning")
    print(f"✓ λ = 0.9 (multi-step credit with eligibility traces)")
    print(f"✓ Traces decay as: e_t = 0.9 * 0.99 * e_{t-1} + 1[visited]")

    # Run optimization
    for iteration in range(10):
        best_fitness = optimiser.optimize(num_iterations=1)

        if iteration % 3 == 0:
            td_values = optimiser.basal_ganglia_selector.get_td_values()

            print(f"\nIteration {iteration}:")
            print(f"  Best fitness: {best_fitness:.6f}")
            print(f"  TD values: {td_values}")

    print("\n✓ TD(λ) example completed")


def example_dynamic_rule_switching():
    """
    Example 3: Dynamic Learning Rule Switching

    Switch between different learning rules during optimization
    to adapt to different phases of the search.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Dynamic Learning Rule Switching")
    print("="*70)

    optimiser = NEVOptimiser(
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

    optimiser.basal_ganglia_selector.begin_episode()

    print("✓ Created optimiser")

    # Phase 1: Simple learning (early exploration)
    print("\nPhase 1: Simple TD Learning (exploration)")
    optimiser.basal_ganglia_selector.set_learning_rule(SimpleTDRule())
    for i in range(3):
        optimiser.optimize(num_iterations=1)
        print(f"  Iteration {i}: using SimpleTDRule")

    # Phase 2: Conservative learning (late exploration)
    print("\nPhase 2: Conservative TD Learning (stability)")
    conservative_rule = ConservativeTDRule(stability_weight=0.5)
    optimiser.basal_ganglia_selector.set_learning_rule(conservative_rule)
    for i in range(3):
        optimiser.optimize(num_iterations=1)
        print(f"  Iteration {i}: using ConservativeTDRule")

    # Phase 3: Adaptive learning (final convergence)
    print("\nPhase 3: Adaptive TD Learning (fine-tuning)")
    adaptive_rule = AdaptiveTDRule(window_size=10)
    optimiser.basal_ganglia_selector.set_learning_rule(adaptive_rule)
    for i in range(3):
        optimiser.optimize(num_iterations=1)
        print(f"  Iteration {i}: using AdaptiveTDRule")

    print("\n✓ Dynamic rule switching example completed")


def example_dynamic_value_model_switching():
    """
    Example 4: Dynamic Value Model Switching

    Switch between linear and bounded value models to manage
    value explosion or ensure stability.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Dynamic Value Model Switching")
    print("="*70)

    optimiser = NEVOptimiser(
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

    optimiser.basal_ganglia_selector.begin_episode()

    print("✓ Created optimiser with LinearValueModel")

    # Get initial values
    td_values = optimiser.basal_ganglia_selector.get_td_values()
    print(f"Initial TD values: {td_values}")

    # Run with linear model
    for i in range(5):
        optimiser.optimize(num_iterations=1)

    td_values = optimiser.basal_ganglia_selector.get_td_values()
    print(f"\nAfter 5 iterations: {td_values}")

    # Switch to bounded model
    print("\n→ Switching to BoundedValueModel")
    bounded_model = BoundedValueModel(
        n_operators=len(optimiser.basal_ganglia_selector.operators),
        initial_value=0.5,
        min_bound=0.2,
        max_bound=2.0,
        adapt_bounds=True,
    )
    optimiser.basal_ganglia_selector.set_value_model(bounded_model)

    # Continue optimization
    for i in range(5):
        optimiser.optimize(num_iterations=1)

    td_values = optimiser.basal_ganglia_selector.get_td_values()
    print(f"After 5 more iterations (bounded): {td_values}")

    print("\n✓ Dynamic value model switching example completed")


def example_td_parameter_tuning():
    """
    Example 5: TD Parameter Tuning During Optimization

    Dynamically adjust TD parameters based on optimization progress.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: TD Parameter Tuning During Optimization")
    print("="*70)

    optimiser = NEVOptimiser(
        objective_function=rosenbrock_function,
        bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        epsilon=0.1,
        learning_rate=0.2,
        seed=42,
    )

    optimiser.basal_ganglia_selector.begin_episode()
    selector = optimiser.basal_ganglia_selector

    print("✓ Optimizing Rosenbrock function")
    print(f"Initial learning rate: 0.2")
    print(f"Initial λ: 0.0 (TD(0))")

    best_fitness_history = []

    for iteration in range(15):
        best_fitness = optimiser.optimize(num_iterations=1)
        best_fitness_history.append(best_fitness)

        # Adjust learning rate based on progress
        if iteration == 5:
            print(f"\n→ Iteration 5: Increasing learning rate (faster adaptation)")
            selector.td_learner.set_learning_rate(0.3)

        if iteration == 10:
            print(f"\n→ Iteration 10: Switching to TD(λ) (multi-step learning)")
            selector.set_td_lambda(0.7)
            print(f"   Also decreasing learning rate (fine-tuning)")
            selector.td_learner.set_learning_rate(0.05)

        if iteration % 5 == 0:
            stats = selector.get_td_statistics()
            print(f"Iteration {iteration}: fitness={best_fitness:.6f}, "
                  f"mean_td_error={stats['mean_td_error']:.6f}")

    print("\n✓ Parameter tuning example completed")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TD Learning Examples for NEVO Optimiser".center(68) + "║")
    print("║" + "  Preserving Nengo Neuromorphic Networks".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    # Run examples
    example_td0_learning()
    example_td_lambda_learning()
    example_dynamic_rule_switching()
    example_dynamic_value_model_switching()
    example_td_parameter_tuning()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

