# Quick Reference: Spike-Driven vs. Continuous Ensembles

## What Changed?

### Before (Pseudo-Neuromorphic)
```python
# Continuous state updates with hand-tuned decay
state[i] = (1 - decay) * state[i] + decay * target + noise
```
- Linear differential equations, no spikes.
- Noise injected manually.
- No biological grounding.

### Now (Truly Neuromorphic)
```python
# Poisson spikes -> synaptic filtering -> NEF decoding
rates_hz = sigmoid(encoders @ x + intercepts)
spikes = poisson(rates_hz * dt)
filtered = synaptic_filter(spikes, tau)
candidate = filtered @ decoders
```
- Stochastic Poisson spike generation.
- Natural synaptic dynamics.
- Learning via reward-modulated STDP.
- Compatible with neuromorphic hardware (Loihi, SpiNNaker).

---

## When to Use Each Mode?

### `operator_mode="trad"` (13 heuristic operators)
- **Best for**: Smooth, well-behaved, low-dimensional problems.
- **When**: You want proven, tuned metaheuristics.
- **Not for**: Real neuromorphic hardware targets.

### `operator_mode="nm_dual"` (Hard WTA switching)
- **Best for**: Multimodal landscapes and neuromorphic hardware targets.
- **When**: You want spike-driven exploration/exploitation with online learning.
- **Not for**: Smooth convex problems (spike noise increases near-optimum error).

### `operator_mode="nm_softmix"` (Beta-blended populations)
- **Best for**: Continuous balancing of exploration and exploitation.
- **When**: You want smooth, per-candidate interpolation between the two ensemble modes (weights drawn from `Beta(α, β)`, `soft_mix_concentration=6.0`).
- **Not for**: When clear mode switching is needed.

---

## Performance Expectations

### Sphere (Convex, 5D)
- `"trad"`: Best (7e-4). No local optima; heuristics excel.
- `"nm_dual"`: 1.8e-3. Spike noise adds penalty near the optimum.
- `"nm_softmix"`: 1.8e-3. Similar to dual mode.

### Rastrigin (Multimodal, 5D)
- `"trad"`: Struggles (1.0e+0).
- `"nm_dual"`: **Wins (0.124)**. Spike stochasticity and STDP learning help.
- `"nm_softmix"`: 0.372.

**Lesson**: Neuromorphic modes excel when exploration and learning matter more than near-optimum precision.

---

## Under the Hood: Spiking Mechanics

### Exploration Ensemble
- **Poisson firing rate**: 100 Hz max.
- **Fast synapses**: τ = 5 ms (responsive).
- **Broad encoders**: Random, high-variance (explores all directions).
- **Dopamine-gated learning**: Decoders update on improvement.
- **Purpose**: Escape local minima via stochastic spike events.

### Exploitation Ensemble
- **Poisson firing rate**: 80 Hz max.
- **Slow synapses**: τ = 20 ms (attractor-like).
- **Sharp encoders**: Low-variance, concentrated (stable convergence).
- **Local STDP**: Reinforces moves towards the best solution.
- **Purpose**: Refine solutions in the trust region near the attractor.

---

## Integration with Nengo

The dual ensembles can run directly on Nengo-enabled neuromorphic hardware:

```python
# Standard CPU simulation
with nengo.Simulator(model, dt=0.001) as sim:
    sim.run(5.0)

# Loihi 2 deployment (via nengo_loihi)
import nengo_loihi
with nengo_loihi.Simulator(model) as sim:
    sim.run(5.0)
```

All spike-driven mechanics are compatible with Intel Loihi 2 and SpiNNaker.

---

## Code Examples

### Example 1: Hard Switching — Neuromorphic Dual

```python
from nevo import NEVOptimiser

opt = NEVOptimiser(
    objective_function=lambda x: np.sum(x**2),
    bounds=(-5, 5),
    dimension=10,
    operator_mode="nm_dual",
    population_size=30,
    seed=42,
)
opt.run(time=3.0, verbose=True)
x_best, f_best = opt.get_best_solution()
```

### Example 2: Compare All Three Modes

```python
from nevo import NEVOptimiser

for mode in ["trad", "nm_dual", "nm_softmix"]:
    opt = NEVOptimiser(
        objective_function=rastrigin,
        bounds=(-5.12, 5.12),
        dimension=5,
        operator_mode=mode,
        seed=123,
    )
    opt.run(time=1.0, verbose=False)
    stats = opt.get_statistics()
    print(f"{mode:12s}: best_f = {stats['best_fitness']:.6e}")
```

### Example 3: Benchmark with `benchmark_operator_modes.py`

```bash
python examples/benchmark_operator_modes.py \
  --dimension 10 \
  --time 2.0 \
  --reps 5 \
  --out my_results.csv
```

Output CSV columns: `problem`, `mode`, `mean_best_f`, `std_best_f`, `min_best_f`, `max_best_f`.

Benchmarked modes: `trad_eps_greedy`, `trad_td0`, `trad_td_lambda`, `nm_dual_eps_greedy`,
`nm_dual_td0`, `nm_dual_td_lambda`, `nm_softmix_eps_greedy`, `nm_softmix_td0`, `nm_softmix_td_lambda`.

---

## Next Steps

1. Deploy on Loihi 2 via `nengo_loihi`.
2. Extend to SpiNNaker.
3. Learn NEF decoder training via Nengo tutorials.
4. Add more spike-based operators (e.g., liquid state machines for recurrent exploration).
