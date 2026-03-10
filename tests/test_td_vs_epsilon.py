"""
Verify that TD(0) and TD(lambda) are actually updating values inside
the neuromorphic run loop (i.e. inside the Nengo simulation).
Prints operator-level TD values after each run so it is visually clear.
"""
import numpy as np
from nevo import NEVOptimiser


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


configs = [
    ("epsilon-greedy (no TD)", dict(td_enabled=False)),
    ("epsilon-greedy + TD(0)",  dict(td_enabled=True, td_lambda=0.0, learning_rate=0.1)),
    ("epsilon-greedy + TD(λ=0.9)", dict(td_enabled=True, td_lambda=0.9, learning_rate=0.1)),
]

for label, td_kwargs in configs:
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)

    opt = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5.0, 5.0),
        dimension=5,
        population_size=20,
        memory_size=10,
        operator_mode="default",
        seed=69,
        **td_kwargs,
    )
    opt.run(time=0.5, verbose=True)

    # Show TD state after run
    if opt.bg_selector.td_enabled:
        td_vals = opt.bg_selector.get_td_values()
        td_stats = opt.bg_selector.get_td_statistics()
        print(f"\n  TD values per operator:")
        for i, op in enumerate(opt.operators):
            print(f"    {op.name:25s}  V={td_vals[i]:.6f}")
        print(f"\n  TD stats: mean_delta={td_stats['mean_td_error']:.6f}  "
              f"std_delta={td_stats['std_td_error']:.6f}")
    else:
        print("\n  (TD disabled — pure epsilon-greedy)")

