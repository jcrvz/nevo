# Changelog

All notable changes to NEVO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-23

### Added
- **9 new standard operators** (expanded library from 4 to 13 total):
  - Exploration: `RandomSearch`, `GravitationalSearch`, `FireflyAlgorithm`, `CentralForce`, `GeneticCrossover`
  - Exploitation: `GeneticMutation`, `LocalRandomWalk`, `SimulatedAnnealing`, `TabuSearch`
- **Utility functions** for all 9 new operators in `basal_ganglia.py` (`utility_random_search`, `utility_gravitational_search`, `utility_firefly`, `utility_central_force`, `utility_genetic_crossover`, `utility_genetic_mutation`, `utility_local_random_walk`, `utility_simulated_annealing`, `utility_tabu_search`)
- **Nengo probes**: `stats_probe` (best/mean fitness + operator index), `state_features_probe`, `state_probe`, and `operator_probe` for post-hoc analysis
- **GPU-accelerated backend**: `run(use_dl=True)` leverages `nengo-dl`/TensorFlow when available, with automatic CPU fallback
- **Fitness-weighted centre** computation (`compute_fitness_weighted_centre`) for smarter population seeding
- **Operator metadata**: `short_name` and `complexity` (1–10 scale) fields added to the `Operator` base class
- **Benchmark experiment system** (`nevo/examples/benchmark_experiment.py`) with full IOH Experimenter and COCO post-processing integration and support for resuming interrupted runs
- **Experiment utilities**: `run_experiment.sh` (OAR job script), `setup.sh` (uv environment bootstrap), `check_progress.py` (live progress monitor), `fix_info_files.py` (repair malformed COCO `.info` files)
- **Paper-ready visualisation** (`plot_results.py`): convergence curves, operator usage charts, and per-function performance plots with LaTeX rendering
- **Basal ganglia fairness test** (`tests/test_basal_ganglia_fairness.py`) verifying that no single operator dominates selection under uniform utilities
- **Operator unit tests** (`tests/test_operators.py`) covering all 13 registered operators
- `setuptools` added as an explicit runtime dependency

### Changed
- **License changed from MIT to BSD 3-Clause**
- **Operator registry** updated in `nevo/operators/__init__.py` to include all 13 operators
- **Default operator list** in `NEVOptimiser.__init__` expanded to all 13 operators
- **Epsilon-greedy selection** now breaks ties randomly instead of always defaulting to index 0, eliminating selection bias in the basal ganglia output
- **Nengo model** built lazily on first `run()` call and reused across subsequent calls, enabling multi-run workflows without rebuilding the network
- `ioh` dependency pinned to `>=0.3.18`; `requires-python` relaxed back to `>=3.10`
- Visualisation module (`nevo/utils/visualisation.py`) extended with SVG particle animation helpers and additional plot utilities

### Fixed
- Index-0 selection bias when all basal ganglia thalamus outputs are equal (now uses uniform random tie-breaking)

## [0.1.0] - 2025-11-19

### Added
- Initial project structure
- Base operator interface (`ExplorationOperator`, `ExploitationOperator`)
- `NEVOptimiser` main class with Nengo-based simulation loop
- `BasalGangliaSelector` for neuromorphic winner-take-all operator selection
- `StateFeatures` for 3-D feature extraction (diversity, improvement rate, convergence)
- Four standard operators: `LevyFlight`, `DifferentialEvolution`, `ParticleSwarm`, `SpiralOptimisation`
- Adaptive utility weight learning with epsilon-greedy exploration
- v-space normalisation (`[-1, 1]^D`) with `trs2o` conversion at evaluation
- Population-based parallel candidate evaluation
- Solution memory archive (competitive replacement)
- Visualisation utilities (`nevo/utils/visualisation.py`) with LaTeX-enabled plotting
- `OPERATOR_REGISTRY` with `get_operator()` and `list_operators()` helpers
- Basic example script (`nevo/examples/basic_example.py`)
- Core test suite (`tests/test_optimiser.py`)
- README, ARCHITECTURE.md, CONTRIBUTING.md, and MIT License
