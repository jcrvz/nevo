"""
Verify that TD(0) and TD(lambda) are actually updating values inside
the neuromorphic run loop (i.e. inside the Nengo simulation).
Prints operator-level TD values after each run so it is visually clear.

Covers:
  1. Basic configurations  – no TD / TD(0) / TD(λ=0.9)
  2. Custom learning rules – ConservativeTDRule, AdaptiveTDRule, DecayingTDRule
  3. Custom value model    – BoundedValueModel
  4. Dynamic rule switching (SimpleTDRule → ConservativeTDRule mid-run)
  5. Dynamic λ switching   (TD(0) → TD(λ=0.7) mid-run)
"""

import numpy as np
from nevo import NEVOptimiser
from nevo.core.td_learning import (
    SimpleTDRule,
    DecayingTDRule,
    ConservativeTDRule,
    AdaptiveTDRule,
    BoundedValueModel,
)

RUN_TIME = 0.1   # seconds per segment
DIM = 5

COMMON = dict(
    bounds=(-5.0, 5.0),
    dimension=DIM,
    population_size=20,
    memory_size=10,
    operator_mode="trad",
    seed=69,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    return float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                     for i in range(len(x) - 1)))


# ---------------------------------------------------------------------------
# Helper: print TD state for an optimiser
# ---------------------------------------------------------------------------
def print_td_state(opt: NEVOptimiser, indent: str = "  ") -> None:
    if opt.bg_selector.td_enabled:
        td_vals = opt.bg_selector.get_td_values()
        td_stats = opt.bg_selector.get_td_statistics()
        weights = opt.bg_selector.get_utility_weights()
        print(f"\n{indent}TD values & utility weights per operator:")
        for i, op in enumerate(opt.operators):
            print(
                f"{indent}  {op.name:25s}  V={td_vals[i]:.6f}"
                f"  w={weights[op.name]:.4f}"
            )
        print(
            f"\n{indent}TD stats: mean_δ={td_stats.get('mean_td_error', 0.0):.6f}"
            f"  std_δ={td_stats.get('std_td_error', 0.0):.6f}"
        )
    else:
        print(f"\n{indent}(TD disabled — pure epsilon-greedy)")


# ===========================================================================
# Section 1 – Basic configurations
# ===========================================================================
print("\n" + "#" * 65)
print("# SECTION 1 — Basic TD configurations")
print("#" * 65)

basic_configs = [
    ("epsilon-greedy (no TD)",    dict(td_enabled=False)),
    ("epsilon-greedy + TD(0)",    dict(td_enabled=True, td_lambda=0.0, learning_rate=0.1)),
    ("epsilon-greedy + TD(λ=0.9)", dict(td_enabled=True, td_lambda=0.9, learning_rate=0.1)),
]

for label, td_kwargs in basic_configs:
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)
    opt = NEVOptimiser(objective_function=sphere, **COMMON, **td_kwargs)
    opt.run(time=RUN_TIME, verbose=True)
    print_td_state(opt)


# ===========================================================================
# Section 2 – Custom learning rules (from td_learning_examples.py)
# ===========================================================================
print("\n\n" + "#" * 65)
print("# SECTION 2 — Custom learning rules")
print("#" * 65)

rule_configs = [
    (
        "TD(0) + SimpleTDRule (explicit)",
        dict(td_enabled=True, td_lambda=0.0, learning_rate=0.1),
        SimpleTDRule(),
    ),
    (
        "TD(0) + ConservativeTDRule (stability_weight=0.5)",
        dict(td_enabled=True, td_lambda=0.0, learning_rate=0.1),
        ConservativeTDRule(stability_weight=0.5),
    ),
    (
        "TD(0) + AdaptiveTDRule (window_size=10)",
        dict(td_enabled=True, td_lambda=0.0, learning_rate=0.2),
        AdaptiveTDRule(window_size=10),
    ),
    (
        "TD(λ=0.7) + DecayingTDRule (exponential, decay=0.9)",
        dict(td_enabled=True, td_lambda=0.7, learning_rate=0.1),
        DecayingTDRule(decay_type="exponential", decay_rate=0.9),
    ),
]

for label, td_kwargs, rule in rule_configs:
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)
    opt = NEVOptimiser(objective_function=sphere, **COMMON, **td_kwargs)
    opt.set_td_learning_rule(rule)
    opt.run(time=RUN_TIME, verbose=True)
    print_td_state(opt)


# ===========================================================================
# Section 3 – Custom value model: BoundedValueModel
# ===========================================================================
print("\n\n" + "#" * 65)
print("# SECTION 3 — Custom value model (BoundedValueModel)")
print("#" * 65)

print("\n" + "=" * 65)
print("  TD(0) + BoundedValueModel [0.2, 2.0]  (adapt_bounds=True)")
print("=" * 65)

opt = NEVOptimiser(
    objective_function=sphere,
    **COMMON,
    td_enabled=True,
    td_lambda=0.0,
    learning_rate=0.1,
    td_value_model=BoundedValueModel(
        n_operators=13,       # trad mode has 13 operators
        initial_value=0.5,
        min_bound=0.2,
        max_bound=2.0,
        adapt_bounds=True,
    ),
)
opt.run(time=RUN_TIME, verbose=True)
print_td_state(opt)


# ===========================================================================
# Section 4 – Dynamic learning-rule switching mid-optimisation
#   Phase A: SimpleTDRule  → run RUN_TIME/2 s
#   Phase B: switch to ConservativeTDRule → run RUN_TIME/2 s more
# ===========================================================================
print("\n\n" + "#" * 65)
print("# SECTION 4 — Dynamic learning-rule switching")
print("#" * 65)

print("\n" + "=" * 65)
print("  Phase A — SimpleTDRule")
print("=" * 65)

opt = NEVOptimiser(
    objective_function=rosenbrock,
    **COMMON,
    td_enabled=True,
    td_lambda=0.0,
    learning_rate=0.2,
)
opt.set_td_learning_rule(SimpleTDRule())
opt.run(time=RUN_TIME / 2, verbose=True)
print_td_state(opt)

print("\n" + "=" * 65)
print("  Phase B — switched to ConservativeTDRule (same optimiser)")
print("=" * 65)

opt.set_td_learning_rule(ConservativeTDRule(stability_weight=0.4))
opt.run(time=RUN_TIME / 2, verbose=True)
print_td_state(opt)

print("\n  ↳ TD values should reflect accumulated learning across both phases")


# ===========================================================================
# Section 5 – Dynamic λ switching mid-optimisation
#   Phase A: TD(0) (λ=0.0)  → run RUN_TIME/2 s
#   Phase B: switch to TD(λ=0.7) → run RUN_TIME/2 s more
# ===========================================================================
print("\n\n" + "#" * 65)
print("# SECTION 5 — Dynamic λ switching (TD(0) → TD(λ=0.7))")
print("#" * 65)

print("\n" + "=" * 65)
print("  Phase A — TD(0)  λ=0.0")
print("=" * 65)

opt = NEVOptimiser(
    objective_function=rosenbrock,
    **COMMON,
    td_enabled=True,
    td_lambda=0.0,
    learning_rate=0.1,
)
opt.run(time=RUN_TIME / 2, verbose=True)
print_td_state(opt)

print("\n" + "=" * 65)
print("  Phase B — TD(λ=0.7)  (same optimiser, λ changed in-place)")
print("=" * 65)

opt.set_td_lambda(0.7)
opt.run(time=RUN_TIME / 2, verbose=True)
print_td_state(opt)

print("\n  ↳ Eligibility traces now propagate credit over multiple steps")
