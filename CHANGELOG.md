# Changelog

All notable changes to NEVO will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-23

### Added
- `nm_dual` operator mode: hard winner-take-all switching between `NeuromorphicExplorationEnsemble` and `NeuromorphicExploitationEnsemble`.
- `nm_softmix` operator mode: softmax-blended populations with configurable `temperature` and `concentration`.
- `NeuromorphicExplorationEnsemble`: Nengo LIF ensemble with Poisson spike generation, fast synapses (τ=5 ms), and broad encoders.
- `NeuromorphicExploitationEnsemble`: Nengo LIF ensemble with slow synapses (τ=20 ms), sharp tuning, and attractor dynamics.
- Modular TD learning system (`nevo/core/td_learning.py`): `SimpleTDRule`, `DecayingTDRule`, `ConservativeTDRule`, `AdaptiveTDRule`, `LinearValueModel`, `BoundedValueModel`, `EligibilityTraceManager`, `TemporalDifferenceLearner`.
- TD(λ) support via eligibility traces.
- Dynamic rule/model switching at runtime: `opt.bg_selector.set_learning_rule()`, `set_value_model()`, `set_td_lambda()`.
- Pass-through helpers on `NEVOptimiser`: `set_td_learning_rule()`, `set_td_value_model()`, `set_td_lambda()`.
- Nine additional standard operators: `RandomSearch`, `GravitationalSearch`, `FireflyAlgorithm`, `CentralForce`, `GeneticCrossover`, `GeneticMutation`, `LocalRandomWalk`, `SimulatedAnnealing`, `TabuSearch`.
- `OPERATOR_REGISTRY` and `get_operator()` factory in `nevo/operators/__init__.py`.
- Comprehensive integration test suite (`tests/test_td_integration.py`) covering `nm_dual` and `nm_softmix` modes.

### Fixed
- All documentation operator mode strings updated to match code (`"trad"`, `"nm_dual"`, `"nm_softmix"`).
- `td_learning_examples.py`: replaced non-existent `optimize()` and `basal_ganglia_selector` with `run()` and `bg_selector`.

## [0.1.0] — 2025-11-19

### Added
- Initial project structure.
- `NEVOptimiser` main class with Nengo-based simulation loop.
- `BasalGangliaSelector` for neuromorphic operator selection.
- `StateFeatures` for feature extraction (diversity, improvement rate, convergence).
- Four core operators: `LevyFlight`, `DifferentialEvolution`, `ParticleSwarm`, `SpiralOptimisation`.
- v-space normalisation (`trs2o()`) and competitive memory archive.
- Visualisation utilities: `plot_optimisation_results()`, `plot_operator_statistics()`.
- IOHexperimenter integration.
- Basic example script and README.
- MIT Licence.
