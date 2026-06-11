Getting Started
===============

Installation
------------

From source::

   git clone https://github.com/jcrvz/nevo.git
   cd nevo
   pip install -e .


Using pip (not available on PyPI yet, but will be soon)::

   pip install nevo

Using uv (recommended)::

   uv pip install -e .
   uv pip install -e . --extra dev    # includes dev and docs dependencies

Requirements
------------

- Python 3.10 or later
- `nengo >= 4.1`
- `numpy >= 1.20, < 2.0`
- `ioh >= 0.3.18` (benchmark problems)

For documentation building::

   uv pip install -e . --extra docs

Quick Start
-----------

.. code-block:: python

   from nevo import NEVOptimiser
   from ioh import get_problem

   problem = get_problem(fid=1, dimension=10, instance=1)

   optimiser = NEVOptimiser(
       objective_function=problem,
       bounds=(problem.bounds.lb, problem.bounds.ub),
       dimension=10,
       population_size=50,
       memory_size=25,
   )

   optimiser.run(time=20.0)
   x_best, f_best = optimiser.get_best_solution()
   print(f"Best fitness: {f_best:.6e}")

Operator Modes
--------------

Three modes control which operators are used:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - ``operator_mode``
     - Description
   * - ``"trad"``
     - 13 standard heuristic operators (default)
   * - ``"nm_dual"``
     - 2 neuromorphic LIF ensembles, hard winner-take-all switching
   * - ``"nm_softmix"``
     - 2 neuromorphic LIF ensembles, Beta-distribution-blended (``soft_mix_concentration=6.0``)

Neuromorphic mode example:

.. code-block:: python

   optimiser = NEVOptimiser(
       objective_function=my_function,
       bounds=(-5, 5),
       dimension=10,
       operator_mode="nm_dual",
       population_size=30,
   )
   optimiser.run(time=10.0)

Running Tests
-------------

::

   pytest tests/                       # full suite
   pytest tests/test_td_integration.py # TD learning integration

Running Examples
----------------

::

   python examples/basic_example.py --time 2.0 --dimensions 5
   python examples/td_learning_examples.py

----

Accessing Results
-----------------

Best solution and fitness:

.. code-block:: python

   x_best, f_best = optimiser.get_best_solution()

Operator usage and TD values:

.. code-block:: python

   stats = optimiser.get_statistics()
   # {total_evaluations, best_fitness, operator_counts,
   #  operator_weights, operator_success_rates}

Time-series probe data (after ``run()``):

.. code-block:: python

   import numpy as np

   sim = optimiser.simulator
   stats_data = sim.data[optimiser.stats_probe]   # shape (T, 3)
   best_f_trace   = stats_data[:, 0]              # best fitness per step
   mean_f_trace   = stats_data[:, 1]              # mean population fitness
   operator_trace = stats_data[:, 2]              # selected operator index

   features = sim.data[optimiser.state_features_probe]  # shape (T, 3)
   # columns: [diversity, improvement_rate, convergence]

----

TD Learning
-----------

TD learning is enabled by default (``td_enabled=True``).  The discount factor,
lambda, and learning rule can all be configured at construction or at runtime:

.. code-block:: python

   from nevo import NEVOptimiser
   from nevo.core.td_learning import ConservativeTDRule, BoundedValueModel

   optimiser = NEVOptimiser(
       objective_function=my_fn,
       bounds=(-5, 5),
       dimension=10,
       td_enabled=True,
       td_lambda=0.0,         # TD(0); set to 0.9 for TD(λ)
       learning_rate=0.1,
       td_gamma=0.99,
   )

   # Swap rule/model at runtime without rebuilding the Nengo network:
   optimiser.set_td_lambda(0.9)
   optimiser.set_td_learning_rule(ConservativeTDRule(stability_weight=0.5))
   optimiser.set_td_value_model(BoundedValueModel(len(optimiser.operators)))

   optimiser.run(time=10.0)

Multi-run workflow — calling ``run()`` multiple times reuses the same Nengo model:

.. code-block:: python

   optimiser.run(time=5.0, verbose=False)     # builds model lazily
   optimiser.run(time=5.0, verbose=False)     # continues from previous state
   optimiser.reset_td_episode()               # clear traces; keep value estimates
   optimiser.run(time=5.0, verbose=False)     # fresh TD episode

----

What to Read Next
-----------------

.. list-table::
   :widths: 30 70

   * - :doc:`examples`
     - Annotated code for every example script.
   * - :doc:`ARCHITECTURE`
     - Component diagram, data-flow, and design principles.
   * - :doc:`NEUROMORPHIC_ENSEMBLES`
     - Spiking mechanics, spike decoding, and benchmark comparisons.
   * - :doc:`td_learning_guide`
     - Full TD learning API — rules, value models, eligibility traces.
   * - :doc:`QUICK_REFERENCE_NEUROMORPHIC`
     - One-page cheat-sheet for operator modes and spiking parameters.
   * - :doc:`api/index`
     - Complete auto-generated API reference.

