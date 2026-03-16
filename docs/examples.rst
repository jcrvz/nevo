Examples
========

All runnable scripts live in the ``examples/`` directory.  Each section below
shows the key usage pattern alongside the command to run it.

----

Basic Optimisation
------------------

``examples/basic_example.py`` runs NEVO on any IOH benchmark problem using the
full 13-operator traditional suite.

.. code-block:: bash

   python examples/basic_example.py --time 5.0 --dimensions 10
   python examples/basic_example.py --problem-ids 1,6,10 --time 3.0 --dimensions 5

.. code-block:: python

   from nevo import NEVOptimiser
   from ioh import get_problem

   problem = get_problem(fid=1, instance=1, dimension=10)
   problem.reset()

   optimiser = NEVOptimiser(
       objective_function=problem,
       bounds=(problem.bounds.lb, problem.bounds.ub),
       dimension=10,
       population_size=50,
       memory_size=25,
       dt=0.001,
       epsilon=0.1,
       learning_rate=0.4,
       seed=42,
   )

   optimiser.run(time=5.0)
   x_best, f_best = optimiser.get_best_solution()
   print(f"Best fitness: {f_best:.6e}")

After ``run()`` the fitness curves and operator usage can be plotted:

.. code-block:: python

   from nevo.utils import plot_optimisation_results, plot_operator_statistics

   plot_optimisation_results(optimiser, optimum=problem.optimum.y)
   plot_operator_statistics(optimiser)

.. note::
   ``plot_optimisation_results()`` calls ``setup_plotting_style()`` which
   enables LaTeX rendering (``text.usetex=True``).  A working LaTeX
   installation is required for publication-quality output.

----

Neuromorphic Operator Modes
----------------------------

NEVO ships three operator modes.  Pass ``operator_mode`` to switch:

.. code-block:: python

   # Hard winner-take-all between two LIF ensembles
   opt_dual = NEVOptimiser(
       objective_function=my_fn,
       bounds=(-5, 5),
       dimension=10,
       operator_mode="nm_dual",
       population_size=30,
   )
   opt_dual.run(time=5.0)

   # Per-candidate Beta-blended mix of the two ensembles
   opt_mix = NEVOptimiser(
       objective_function=my_fn,
       bounds=(-5, 5),
       dimension=10,
       operator_mode="nm_softmix",   # soft_mix_concentration=6.0, temperature=0.35
       population_size=30,
   )
   opt_mix.run(time=5.0)

See :doc:`NEUROMORPHIC_ENSEMBLES` for a detailed description of the spiking
mechanics and benchmark results.

----

TD Learning — Quick Start
--------------------------

``examples/td_quickstart.py`` covers TD(0), TD(λ), and runtime rule/model
swapping in under 100 lines.

.. code-block:: bash

   python examples/td_quickstart.py

**TD(0)**

.. code-block:: python

   from nevo import NEVOptimiser

   optimiser = NEVOptimiser(
       objective_function=lambda x: float(sum(xi**2 for xi in x)),
       bounds=(-5, 5),
       dimension=2,
       population_size=20,
       td_enabled=True,
       td_lambda=0.0,       # TD(0) — no eligibility traces
       learning_rate=0.1,
       seed=42,
   )

   for _ in range(5):
       optimiser.run(time=0.05, verbose=False)

   print(optimiser.bg_selector.get_td_values())

**TD(λ) with eligibility traces**

.. code-block:: python

   optimiser.set_td_lambda(0.9)          # switch to TD(0.9)
   optimiser.run(time=0.1, verbose=False)

**Swap learning rule at runtime**

.. code-block:: python

   from nevo.core.td_learning import ConservativeTDRule, AdaptiveTDRule

   optimiser.set_td_learning_rule(ConservativeTDRule(stability_weight=0.5))
   optimiser.run(time=0.05, verbose=False)

   optimiser.set_td_learning_rule(AdaptiveTDRule(window_size=10))
   optimiser.run(time=0.05, verbose=False)

**Swap value model at runtime**

.. code-block:: python

   from nevo.core.td_learning import BoundedValueModel

   n_ops = len(optimiser.operators)
   optimiser.set_td_value_model(
       BoundedValueModel(n_ops, min_bound=0.2, max_bound=2.0)
   )
   optimiser.run(time=0.05, verbose=False)

----

Extended TD Learning Patterns
-------------------------------

``examples/td_learning_examples.py`` demonstrates five usage patterns:

.. code-block:: bash

   python examples/td_learning_examples.py

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - #
     - Pattern
     - Key idea
   * - 1
     - TD(0)
     - Standard single-step bootstrapping
   * - 2
     - TD(λ=0.9)
     - Multi-step credit via eligibility traces
   * - 3
     - Dynamic rule switching
     - ``SimpleTDRule`` → ``ConservativeTDRule`` → ``AdaptiveTDRule`` per phase
   * - 4
     - Dynamic value model switching
     - Start with ``LinearValueModel``; switch to ``BoundedValueModel`` mid-run
   * - 5
     - Parameter tuning
     - Change ``learning_rate`` and ``lambda_coeff`` inline without rebuilding

Multi-run pattern (model is built lazily on first ``run()`` and reused):

.. code-block:: python

   optimiser.run(time=0.1, verbose=False)   # builds Nengo model
   optimiser.run(time=0.1, verbose=False)   # reuses model, continues from last state
   optimiser.reset_td_episode()             # clear eligibility traces, keep value estimates
   optimiser.run(time=0.1, verbose=False)   # fresh TD episode

----

Benchmark: Operator Mode Comparison
-------------------------------------

``examples/benchmark_operator_modes.py`` benchmarks all operator/TD
combinations on Sphere and Rastrigin, printing a table and writing a CSV.

.. code-block:: bash

   python examples/benchmark_operator_modes.py
   python examples/benchmark_operator_modes.py --reps 10 --time 1.0 --dimension 5
   python examples/benchmark_operator_modes.py --seed 42 --out results.csv

CLI arguments:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Flag
     - Default
     - Description
   * - ``--dimension``
     - ``10``
     - Problem dimension
   * - ``--time``
     - ``1.0``
     - Simulation time per run (seconds)
   * - ``--reps``
     - ``3``
     - Number of independent repetitions (seeds 0…reps-1)
   * - ``--seed``
     - —
     - Single seed override (overrides ``--reps``)
   * - ``--out``
     - ``operator_mode_benchmark.csv``
     - Output CSV path

Benchmarked modes:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Mode string
     - Description
   * - ``trad_eps_greedy``
     - 13 standard operators, ε-greedy only, no TD
   * - ``trad_td0``
     - 13 standard operators + TD(0)
   * - ``trad_td_lambda``
     - 13 standard operators + TD(λ=0.9)
   * - ``nm_dual_eps_greedy``
     - 2 LIF ensembles, hard WTA, ε-greedy only
   * - ``nm_dual_td0``
     - 2 LIF ensembles, hard WTA + TD(0)
   * - ``nm_dual_td_lambda``
     - 2 LIF ensembles, hard WTA + TD(λ=0.9)
   * - ``nm_softmix_eps_greedy``
     - 2 LIF ensembles, Beta-blended, ε-greedy only
   * - ``nm_softmix_td0``
     - 2 LIF ensembles, Beta-blended + TD(0)
   * - ``nm_softmix_td_lambda``
     - 2 LIF ensembles, Beta-blended + TD(λ=0.9)

Output CSV columns: ``problem``, ``mode``, ``mean_best_f``, ``std_best_f``,
``min_best_f``, ``max_best_f``.

----

Accessing Probe Data
---------------------

After ``run()``, every Nengo probe is available through ``optimiser.simulator``:

.. code-block:: python

   import numpy as np

   optimiser.run(time=5.0, verbose=False)

   sim = optimiser.simulator

   # Shape (T, 3): [best_f, mean_f, operator_idx] per timestep
   stats = sim.data[optimiser.stats_probe]
   best_f_trace   = stats[:, 0]
   mean_f_trace   = stats[:, 1]
   operator_trace = stats[:, 2]

   # Shape (T, 3): [diversity, improvement_rate, convergence]
   features = sim.data[optimiser.state_features_probe]

   # Shape (T, N_operators): raw thalamus output
   op_signal = sim.data[optimiser.operator_probe]

   t = np.arange(len(best_f_trace)) * optimiser.dt
   print(f"Final best fitness: {best_f_trace[-1]:.6e}")

