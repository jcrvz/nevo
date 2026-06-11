#!/usr/bin/env python3
"""
Quick Start: Using TD Learning in NEVO
======================================

Runnable quick-start examples for TD(0)/TD(lambda) with the current NEVO API.
"""

import numpy as np
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    ConservativeTDRule,
    AdaptiveTDRule,
    BoundedValueModel,
)

# Simple problem
def _sphere(x):
    return np.sum(x**2)


def _make_optimizer(learning_rate: float = 0.1) -> NEVOptimiser:
    return NEVOptimiser(
        objective_function=_sphere,
        bounds=(np.array([-5, -5]), np.array([5, 5])),
        dimension=2,
        population_size=20,
        memory_size=10,
        epsilon=0.1,
        learning_rate=learning_rate,
        seed=42,
    )


def _step(optimizer: NEVOptimiser, sim_time: float = 0.05) -> float:
    optimizer.run(time=sim_time, verbose=False)
    return float(optimizer.state["best_f"])


def quickstart_td0():
    print("Quick Start 1: TD(0) Learning")
    optimizer = _make_optimizer(learning_rate=0.1)
    for i in range(5):
        best_fitness = _step(optimizer)
        if i % 2 == 0:
            print(
                f"  iter={i} best={best_fitness:.4f} td={optimizer.bg_selector.get_td_values()}"
            )


def quickstart_td_lambda():
    print("\nQuick Start 2: TD(lambda) Learning")
    optimizer = _make_optimizer(learning_rate=0.1)
    optimizer.bg_selector.set_td_lambda(0.9)
    for i in range(5):
        best_fitness = _step(optimizer)
        if i % 2 == 0:
            stats = optimizer.bg_selector.get_td_statistics()
            print(
                f"  iter={i} best={best_fitness:.4f} td_err={stats.get('mean_td_error', 0.0):.4f}"
            )


def quickstart_rule_switching():
    print("\nQuick Start 3: Rule Switching")
    optimizer = _make_optimizer(learning_rate=0.15)
    optimizer.bg_selector.set_learning_rule(SimpleTDRule())
    _step(optimizer)
    optimizer.bg_selector.set_learning_rule(ConservativeTDRule(stability_weight=0.5))
    _step(optimizer)
    optimizer.bg_selector.set_learning_rule(AdaptiveTDRule(window_size=10))
    best_fitness = _step(optimizer)
    print(f"  best={best_fitness:.4f} td={optimizer.bg_selector.get_td_values()}")


def quickstart_bounded_values():
    print("\nQuick Start 4: Bounded Value Model")
    optimizer = _make_optimizer(learning_rate=0.2)
    optimizer.bg_selector.set_value_model(
        BoundedValueModel(
            n_operators=len(optimizer.bg_selector.operators),
            initial_value=0.5,
            min_bound=0.2,
            max_bound=2.0,
            adapt_bounds=True,
        )
    )
    for i in range(4):
        best_fitness = _step(optimizer)
        print(
            f"  iter={i} best={best_fitness:.4f} td={optimizer.bg_selector.get_td_values()}"
        )


def main():
    quickstart_td0()
    quickstart_td_lambda()
    quickstart_rule_switching()
    quickstart_bounded_values()


if __name__ == "__main__":
    main()
