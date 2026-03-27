#!/usr/bin/env python3
"""
Integration Test: TD Learning with NEVO Optimiser — Neuromorphic Modes
=======================================================================

All primary tests run in ``nm_dual`` or ``nm_softmix`` operator mode so that
TD learning is exercised alongside the actual Nengo LIF spike ensembles rather
than the traditional Python heuristics.

Architecture reminder
---------------------
The Nengo basal-ganglia (BG) circuit produces the action-selection signal
**inside** the simulation; the TD learner is a Python-side module that reads
that signal and biases utility weights fed back into the same BG circuit on
the next timestep.  The coupling point is ``BasalGangliaSelector.select_operator()``.

The traditional (``trad``) mode is exercised only in ``test_backward_compatibility``
so there is an explicit non-neuromorphic baseline.
"""

import numpy as np
from nevo.core.optimiser import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    ConservativeTDRule,
    LinearValueModel,
    BoundedValueModel,
)
from nevo.operators.standard import (
    NeuromorphicExplorationEnsemble,
    NeuromorphicExploitationEnsemble,
)

_SEG = 0.1   # seconds per simulation segment — short but covers many timesteps


def sphere_function(x):
    return float(np.sum(x**2))


def rosenbrock_function(x):
    return float(sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(len(x) - 1)
    ))


def _best(opt: NEVOptimiser) -> float:
    f = opt.state.get("best_f")
    return f if f is not None else float("inf")


def _nm_dual_opt(objective=sphere_function, dimension=2, seed=42, bounds=None, learning_rate=0.1, **extra) -> NEVOptimiser:
    """Factory: nm_dual optimiser with sensible defaults for fast testing."""
    if bounds is None:
        bounds = (np.full(dimension, -5.0), np.full(dimension, 5.0))
    return NEVOptimiser(
        objective_function=objective,
        bounds=bounds,
        dimension=dimension,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        operator_mode="nm_dual",
        td_enabled=True,
        learning_rate=learning_rate,
        seed=seed,
        **extra,
    )


# ---------------------------------------------------------------------------
# Helper: assert both LIF ensembles were wired by build_network()
# ---------------------------------------------------------------------------
def _assert_lif_ensembles_built(opt: NEVOptimiser) -> None:
    """Raise AssertionError if Nengo LIF ensembles were not built."""
    explore_op = next(
        (op for op in opt.operators if isinstance(op, NeuromorphicExplorationEnsemble)),
        None,
    )
    exploit_op = next(
        (op for op in opt.operators if isinstance(op, NeuromorphicExploitationEnsemble)),
        None,
    )
    assert explore_op is not None, "NeuromorphicExplorationEnsemble not found"
    assert exploit_op is not None, "NeuromorphicExploitationEnsemble not found"
    assert explore_op._nengo_ensemble is not None, \
        "Exploration LIF ensemble was never built (build_network not called)"
    assert exploit_op._nengo_ensemble is not None, \
        "Exploitation LIF ensemble was never built (build_network not called)"


# ===========================================================================
# Test 1 — TD(0) with nm_dual: values must deviate from their 0.5 initialisation
# ===========================================================================
def test_td0_nm_dual_mode():
    """TD(0) values change from 0.5 after spike-driven optimisation in nm_dual mode."""
    opt = _nm_dual_opt(td_lambda=0.0)

    assert opt.bg_selector.td_enabled
    assert opt.bg_selector.td_learner.lambda_coeff == 0.0
    assert len(opt.operators) == 2

    initial_values = opt.bg_selector.get_td_values().copy()

    fitness_history = []
    for _ in range(3):
        opt.run(time=_SEG, verbose=False)
        fitness_history.append(_best(opt))

    _assert_lif_ensembles_built(opt)

    final_values = opt.bg_selector.get_td_values()
    assert not np.allclose(initial_values, final_values, atol=1e-6), \
        "TD values did not change — learning not working in nm_dual mode!"
    assert all(f < float("inf") for f in fitness_history)


# ===========================================================================
# Test 2 — TD(λ=0.9) with nm_dual: eligibility traces through spike operators
# ===========================================================================
def test_td_lambda_nm_dual_mode():
    """TD(λ=0.9) accumulates statistics after running through LIF spike operators."""
    opt = _nm_dual_opt(
        objective=rosenbrock_function,
        bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        td_lambda=0.9,
    )

    assert opt.bg_selector.td_learner.lambda_coeff == 0.9

    for _ in range(3):
        opt.run(time=_SEG, verbose=False)

    _assert_lif_ensembles_built(opt)

    stats = opt.bg_selector.get_td_statistics()
    assert "mean_td_error" in stats
    assert "std_td_error" in stats
    assert _best(opt) < float("inf")


# ===========================================================================
# Test 3 — Dynamic rule switching: values remain finite across rule changes
# ===========================================================================
def test_dynamic_rule_switching_nm_dual():
    """TD values persist and stay finite when switching rules mid-run (nm_dual)."""
    opt = _nm_dual_opt()

    # Phase 1 — SimpleTDRule
    opt.bg_selector.set_learning_rule(SimpleTDRule())
    opt.run(time=_SEG, verbose=False)
    values_simple = opt.bg_selector.get_td_values().copy()

    # Phase 2 — switch to ConservativeTDRule on the live optimiser
    opt.bg_selector.set_learning_rule(ConservativeTDRule(stability_weight=0.5))
    opt.run(time=_SEG, verbose=False)
    values_conservative = opt.bg_selector.get_td_values()

    _assert_lif_ensembles_built(opt)

    assert len(values_simple) == 2
    assert np.all(np.isfinite(values_simple))
    assert len(values_conservative) == 2
    assert np.all(np.isfinite(values_conservative))


# ===========================================================================
# Test 4 — Dynamic value-model switching: BoundedValueModel respects [0.2, 2.0]
# ===========================================================================
def test_dynamic_value_model_switching_nm_dual():
    """BoundedValueModel keeps all 2 operator values within [0.2, 2.0] (nm_dual)."""
    opt = _nm_dual_opt(learning_rate=0.2)
    n_ops = len(opt.bg_selector.operators)  # 2 in nm_dual

    # Phase 1 — LinearValueModel
    opt.bg_selector.set_value_model(LinearValueModel(n_ops))
    opt.run(time=_SEG, verbose=False)
    opt.run(time=_SEG, verbose=False)
    assert np.all(np.isfinite(opt.bg_selector.get_td_values()))

    # Phase 2 — BoundedValueModel
    opt.bg_selector.set_value_model(
        BoundedValueModel(n_operators=n_ops, min_bound=0.2, max_bound=2.0, adapt_bounds=True)
    )
    opt.run(time=_SEG, verbose=False)
    opt.run(time=_SEG, verbose=False)
    bounded = opt.bg_selector.get_td_values()

    _assert_lif_ensembles_built(opt)

    assert np.all(bounded >= 0.2), f"Values below minimum bound: {bounded}"
    assert np.all(bounded <= 2.0), f"Values above maximum bound: {bounded}"


# ===========================================================================
# Test 5 — Monitoring APIs return complete, finite data in nm_dual mode
# ===========================================================================
def test_parameter_monitoring_nm_dual():
    """get_td_values / get_utility_weights / get_td_statistics all return valid data."""
    opt = _nm_dual_opt()

    for _ in range(3):
        opt.run(time=_SEG, verbose=False)

    _assert_lif_ensembles_built(opt)

    td_values = opt.bg_selector.get_td_values()
    utility_weights = opt.bg_selector.get_utility_weights()
    stats = opt.bg_selector.get_td_statistics()

    assert len(td_values) == 2
    assert np.all(np.isfinite(td_values))

    assert len(utility_weights) == 2
    assert all(isinstance(v, float) for v in utility_weights.values())

    assert "mean_td_error" in stats
    assert "std_td_error" in stats
    assert np.isfinite(stats["mean_td_error"])


# ===========================================================================
# Test 6 — nm_softmix: blended spike populations + TD run end-to-end
# ===========================================================================
def test_nm_softmix_td_integration():
    """nm_softmix mode with TD(0): softmax-blended LIF ensembles and learning."""
    opt = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(-5.0, 5.0),
        dimension=2,
        population_size=20,
        memory_size=10,
        operator_mode="nm_softmix",
        td_enabled=True,
        td_lambda=0.0,
        learning_rate=0.1,
        seed=7,
    )

    assert len(opt.operators) == 2

    initial_values = opt.bg_selector.get_td_values().copy()

    for _ in range(3):
        opt.run(time=_SEG, verbose=False)

    _assert_lif_ensembles_built(opt)

    # Both operators should be called (softmix always blends both populations)
    counts = opt.state["operator_counts"]
    assert all(c > 0 for c in counts.values()), \
        f"Some operators never called in nm_softmix: {counts}"

    # TD values must have shifted from their initial state
    final_values = opt.bg_selector.get_td_values()
    assert not np.allclose(initial_values, final_values, atol=1e-6), \
        "TD values did not change in nm_softmix mode"


# ===========================================================================
# Test 7 — nm_dual operator usage: BG circuit drives non-trivial selection
# ===========================================================================
def test_nm_dual_operator_selection_is_non_trivial():
    """Both LIF operators must be selected at least once over the run."""
    opt = _nm_dual_opt(epsilon=0.2, seed=99)

    opt.run(time=0.3, verbose=False)

    counts = opt.state["operator_counts"]
    assert all(c > 0 for c in counts.values()), \
        f"Basal ganglia never selected one of the operators: {counts}"


# ===========================================================================
# Test 8 — Backward compatibility: trad mode + td_enabled=False (non-neuromorphic)
# ===========================================================================
def test_backward_compatibility_trad_no_td():
    """Traditional operators with td_enabled=False still run and produce a result."""
    opt = NEVOptimiser(
        objective_function=sphere_function,
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        dimension=2,
        population_size=20,
        memory_size=10,
        neurons_per_ensemble=100,
        operator_mode="trad",
        td_enabled=False,
        seed=42,
    )

    assert not opt.bg_selector.td_enabled

    for _ in range(3):
        opt.run(time=_SEG, verbose=False)

    assert _best(opt) < float("inf"), "Optimiser must produce a finite solution"

    # TD values stay at their sentinel 0.5 because learning is off
    td_values = opt.bg_selector.get_td_values()
    assert np.allclose(td_values, 0.5), \
        "TD values should remain 0.5 when TD is disabled!"


# ---------------------------------------------------------------------------
def main():
    tests = [
        test_td0_nm_dual_mode,
        test_td_lambda_nm_dual_mode,
        test_dynamic_rule_switching_nm_dual,
        test_dynamic_value_model_switching_nm_dual,
        test_parameter_monitoring_nm_dual,
        test_nm_softmix_td_integration,
        test_nm_dual_operator_selection_is_non_trivial,
        test_backward_compatibility_trad_no_td,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"✓  {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"✗  {fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
