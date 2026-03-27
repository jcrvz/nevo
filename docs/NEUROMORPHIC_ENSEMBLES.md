# Neuromorphic Candidate Ensembles

The `NeuromorphicExplorationEnsemble` and `NeuromorphicExploitationEnsemble` operators are grounded in spike-driven neural dynamics.

## Neuromorphic Mechanisms

### 1. Poisson Spike Generation

Each neural population encodes input via stochastic spiking:

```python
rates_hz = max_rate / (1 + exp(-(encoders @ x + intercepts)))
spike_raster = poisson(rates_hz * dt)
```

- **Biological basis**: Cortical neurons fire Poisson-distributed spikes around a rate governed by tuning curves.
- **Stochasticity**: Intrinsic spike-time jitter provides natural exploration noise without hand-tuned Gaussian parameters.
- **Energy relevance**: Sparse spike codes are computationally efficient on neuromorphic hardware (Loihi, SpiNNaker).

### 2. Synaptic Filtering

Spike trains are filtered through exponential synapses (τ-dependent):

```
τ dS/dt = -S + spikes  →  S(t+1) = (1-α)S(t) + α*spikes(t)
```

where α = dt / (τ + dt).

- **Fast synapses (τ=5 ms, exploration)**: Rapid response to input changes; high-frequency passband.
- **Slow synapses (τ=20 ms, exploitation)**: Smooth, stable state evolution; attractor-like dynamics.
- **Biological basis**: AMPA (fast), NMDA (slow), and GABA receptors have distinct timescales.

### 3. Neural Population Coding (NEF)

Candidates are decoded from time-averaged firing rates via learned decoders:

```
candidate = mean_spike_rate @ decoders
```

- **Basis**: Neural Engineering Framework (Eliasmith & Anderson, 2003).
- **Emergent representation**: Multi-dimensional information encoded in population vectors.
- **Noise resilience**: Population coding naturally averages out single-neuron noise.

### 4. Reward-Modulated Hebbian Learning (STDP-like)

Decoders update on improvements using dopamine-gated plasticity:

```python
if delta_reward > 0:
    decoders += learning_rate * delta_reward * (mean_rates outer target)
```

- **Biological basis**: Reward-modulated synaptic plasticity via dopamine neurotransmitter.
- **Online learning**: Adapts decoder weights to reinforce successful search directions.
- **Locus**: Basal ganglia and dopaminergic midbrain (SNc/VTA).

### 5. Attractor Dynamics (Exploitation)

The exploitation ensemble maintains a slow attractor via:
- Slow synaptic filtering (τ=20 ms).
- Sharp tuning curves concentrated around the attractor centre.
- Implicit recurrence through synaptic state persistence.

```
attractor(t+1) = 0.9 * attractor(t) + 0.1 * new_target
```

- **Biological basis**: Persistent neural activity in prefrontal cortex and superior colliculus.
- **Computational benefit**: Natural drift towards optimal solutions without explicit gradient information.
- **Hardware efficiency**: Asynchronous spike events reduce power relative to continuous updates.

### 6. Broad vs. Sharp Tuning

**Exploration**: Random broad encoders (all dimensions equally represented).
- Encoders ~ N(0, 1) / ||·||
- Intercepts ~ Uniform(-1, 1)
- Result: Wide receptive fields, diverse exploration directions.

**Exploitation**: Sharply tuned encoders concentrated around the current solution.
- Encoders ~ N(0, sharp_tuning^2) / ||·|| (smaller variance)
- Intercepts ~ Uniform(-0.5, 0.5) (narrow range)
- Result: Narrow, aligned tuning; stable convergence.

---

## Benchmark Comparison

### Sphere Function (5D, 0.5 s sim time)

| Mode | Mean Best F | Interpretation |
|------|-------------|----------------|
| `"trad"` | 7.0e-4 | Hand-crafted operators, strong baseline |
| `"nm_dual"` | 1.8e-3 | Spike stochasticity adds exploration noise |
| `"nm_softmix"` | 1.8e-3 | Blended spike populations |

**Note**: Sphere is convex, so heuristic operators excel. Neuromorphic ensembles trade some near-optimum precision for spike-driven robustness.

### Rastrigin Function (5D, 0.5 s sim time, highly multimodal)

| Mode | Mean Best F | Interpretation |
|------|-------------|----------------|
| `"trad"` | 1.0e+0 | Heuristics struggle on multimodal landscapes |
| `"nm_dual"` | 0.124 | **Spike stochasticity + STDP learning outperform** |
| `"nm_softmix"` | 0.372 | Soft blending is less aggressive than hard switching |

**Key insight**: On hard multimodal problems, spike-driven dynamics outperform tuned heuristics.

---

## Usage

### Dual-Ensemble Mode (`"nm_dual"` — Hard WTA Switching)

```python
from nevo import NEVOptimiser

optimiser = NEVOptimiser(
    objective_function=my_function,
    bounds=(-5, 5),
    dimension=10,
    operator_mode="nm_dual",
    population_size=30,
    memory_size=15,
)
optimiser.run(time=5.0)
```

### Soft-Mix Mode (`"nm_softmix"` — Continuous Blending)

```python
optimiser = NEVOptimiser(
    objective_function=my_function,
    bounds=(-5, 5),
    dimension=10,
    operator_mode="nm_softmix",
    population_size=30,
    memory_size=15,
)
optimiser.run(time=5.0)
```

---

## Constructor Parameters

### `NeuromorphicExplorationEnsemble`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neurons` | int | 150 | Number of LIF neurons |
| `tau_synapse` | float | 0.005 | Synaptic time constant (seconds; 5 ms) |
| `max_rates` | tuple | (100, 200) | Min/max peak firing rates (Hz) |
| `intercepts` | tuple | (-1.0, 1.0) | Neuron activation thresholds |
| `use_numpy_fallback` | bool | False | Use NumPy path (no Nengo simulator required) |

### `NeuromorphicExploitationEnsemble`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neurons` | int | 200 | Number of LIF neurons |
| `tau_synapse` | float | 0.020 | Synaptic time constant (seconds; 20 ms) |
| `max_rates` | tuple | (50, 100) | Min/max peak firing rates (Hz) |
| `intercepts` | tuple | (-0.5, 0.5) | Neuron activation thresholds |
| `trust_radius` | float | 0.2 | Radius of the trust region in v-space |
| `use_numpy_fallback` | bool | False | Use NumPy path (no Nengo simulator required) |

**Important**: Both operators require access to `state["simulator"]`, which is injected automatically during `NEVOptimiser.run()`. Setting `use_numpy_fallback=True` removes this requirement.

---

## References

1. **Neural Engineering Framework**: Eliasmith, C., & Anderson, C. H. (2003). *Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems*.
2. **Reward-Modulated Plasticity**: Florian, R. V. (2007). Reinforcement learning through modulation of spike-timing-dependent synaptic plasticity.
3. **Population Coding**: Georgopoulos, A. P. (2000). Neural aspects of cognitive control.
4. **Neuromorphic Hardware**: Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning.
