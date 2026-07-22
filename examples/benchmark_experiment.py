"""
Benchmark Experiment Runner
===========================

Run NEVO on multiple benchmark problems using either IOH or COCO/cocoex suites.

Supported COCO/cocoex suites (pass with ``--coco-suite``):

+----------------------+----------+------------------------------------------+
| Suite name           | # funcs  | Valid dimensions                         |
+======================+==========+==========================================+
| bbob (default)       | 24       | 2, 3, 5, 10, 20, 40                      |
+----------------------+----------+------------------------------------------+
| bbob-noisy           | 30       | 2, 3, 5, 10, 20, 40                      |
+----------------------+----------+------------------------------------------+
| bbob-biobj           | 55       | 2, 3, 5, 10, 20, 40  (multi-objective)   |
+----------------------+----------+------------------------------------------+
| bbob-biobj-ext       | 92       | 2, 3, 5, 10, 20, 40  (multi-objective)   |
+----------------------+----------+------------------------------------------+
| bbob-largescale      | 24       | 20, 40, 80, 160, 320, 640                |
+----------------------+----------+------------------------------------------+
| bbob-mixint          | 24       | 5, 10, 20, 40, 80, 160 (mixed-integer)   |
+----------------------+----------+------------------------------------------+
| bbob-constrained     | 48       | 2, 3, 5, 10, 20, 40                      |
+----------------------+----------+------------------------------------------+
| sbox-cost            | 24       | 2, 3, 5, 10, 20, 40                      |
+----------------------+----------+------------------------------------------+

.. note::
   Multi-objective suites (bbob-biobj, bbob-biobj-ext) return vector fitness values.
   NEVO treats the *first* objective as the scalar to minimise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Literal, Tuple, Any

from nevo import NEVOptimiser
from nevo.utils import plot_optimisation_results

# All COCO/cocoex suite names supported by this runner
COCO_SUITES = [
    "bbob",
    "bbob-noisy",
    "bbob-biobj",
    "bbob-biobj-ext",
    "bbob-largescale",
    "bbob-mixint",
    "bbob-constrained",
    "sbox-cost",
]

# Recommended valid dimensions per COCO suite
COCO_SUITE_DIMS = {
    "bbob":             [2, 3, 5, 10, 20, 40],
    "bbob-noisy":       [2, 3, 5, 10, 20, 40],
    "bbob-biobj":       [2, 3, 5, 10, 20, 40],
    "bbob-biobj-ext":   [2, 3, 5, 10, 20, 40],
    "bbob-largescale":  [20, 40, 80, 160, 320, 640],
    "bbob-mixint":      [5, 10, 20, 40, 80, 160],
    "bbob-constrained": [2, 3, 5, 10, 20, 40],
    "sbox-cost":        [2, 3, 5, 10, 20, 40],
}

# Multi-objective suites — NEVO uses first objective only
COCO_MULTIOBJECTIVE_SUITES = {"bbob-biobj", "bbob-biobj-ext"}


class BenchmarkProblem:
    """
    Wrapper class to provide a unified interface for IOH and COCO problems.
    """

    def __init__(
        self,
        problem_id: int,
        instance: int,
        dimension: int,
        suite: Literal["ioh", "cocoex"] = "ioh",
        observer: Any = None,
        coco_suite_name: str = "bbob",
    ):
        """
        Initialise a benchmark problem.

        Parameters
        ----------
        problem_id : int
            Problem/function ID
        instance : int
            Problem instance
        dimension : int
            Problem dimension
        suite : str
            Benchmark suite to use: "ioh" or "cocoex"
        observer : cocoex.Observer, optional
            COCO observer for logging results (only used with cocoex suite)
        coco_suite_name : str
            Which COCO suite to use when ``suite="cocoex"``
            (e.g. "bbob", "bbob-noisy", "bbob-largescale", …).
            Defaults to "bbob".
        """
        self.problem_id = problem_id
        self.instance = instance
        self.dimension = dimension
        self.suite = suite
        self.coco_suite_name = coco_suite_name
        self._problem = None
        self._optimum = None
        self._bounds = None
        self._observer = observer
        self._suite = None  # Keep suite alive for COCO

        self._load_problem()

    def _load_problem(self):
        """Load the problem from the selected suite."""
        if self.suite == "ioh":
            self._load_ioh_problem()
        elif self.suite in ["cocoex", "coco"]:
            self._load_coco_problem()
        else:
            raise ValueError(
                f"Unknown suite: {self.suite}. Use 'ioh' or 'coco|cocoex'."
            )

    def _load_ioh_problem(self):
        """Load problem from IOH suite."""
        from ioh import get_problem

        self._problem = get_problem(
            fid=self.problem_id,
            instance=self.instance,
            dimension=self.dimension,
        )
        self._problem.reset()
        self._bounds = (self._problem.bounds.lb, self._problem.bounds.ub)
        self._optimum = self._problem.optimum.y

    def _load_coco_problem(self):
        """Load problem from COCO/cocoex suite."""
        import cocoex

        suite_name = self.coco_suite_name

        # Create the suite with the specified dimension and instance
        self._suite = cocoex.Suite(
            suite_name,
            f"instances: {self.instance}",
            f"function_indices: {self.problem_id} dimensions: {self.dimension}",
        )

        # Get the first (and only) problem matching our criteria
        for problem in self._suite:
            self._problem = problem
            # Attach observer AFTER getting the problem but BEFORE any evaluation
            if self._observer is not None:
                self._problem.observe_with(self._observer)
            break

        if self._problem is None:
            raise ValueError(
                f"Could not find COCO problem f{self.problem_id} "
                f"instance {self.instance} dimension {self.dimension} "
                f"in suite '{suite_name}'"
            )

        lb = np.array(self._problem.lower_bounds)
        ub = np.array(self._problem.upper_bounds)
        self._bounds = (lb, ub)

        # COCO does not expose the optimal value directly for postprocessing suites
        self._optimum = None

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the objective function.

        For multi-objective COCO suites (bbob-biobj, bbob-biobj-ext) the
        returned vector is scalarised by taking the *first* objective so that
        NEVO's single-objective optimiser can still be applied.
        """
        result = self._problem(x)
        # Multi-objective: scalarise to first objective
        if hasattr(result, "__len__"):
            return float(result[0])
        return result

    def reset(self):
        """Reset the problem (if supported)."""
        if self.suite == "ioh" and hasattr(self._problem, "reset"):
            self._problem.reset()

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return problem bounds as (lower, upper) tuple."""
        return self._bounds

    @property
    def optimum(self) -> float:
        """Return the optimal fitness value."""
        return self._optimum

    @property
    def name(self) -> str:
        """Return problem name."""
        if self.suite == "ioh":
            return self._problem.meta_data.name
        else:
            return self._problem.name

    @property
    def evaluations(self) -> int:
        """Return the number of evaluations so far."""
        if self.suite == "ioh":
            return self._problem.state.evaluations
        else:
            return self._problem.evaluations

    def finalize(self):
        """Finalize the problem (important for COCO to flush data)."""
        if self.suite in ["cocoex", "coco"]:
            # Free the problem to ensure data is flushed
            if self._problem is not None:
                self._problem.free()
                self._problem = None


def _run_coco_batch(args_tuple):
    """
    Worker function to run a COCO batch (for multiprocessing).

    Each worker runs a complete batch with its own observer.
    COCO observer automatically appends to existing data files, enabling resume.
    """
    (
        problems,
        instances,
        dimensions,
        simulation_time,
        n_runs,
        algorithm_name,
        output_dir,
        total_batches,
        batch_number,
        use_dl,
        coco_suite_name,
        operator_mode,
        td_enabled,
        td_lambda,
    ) = args_tuple

    import cocoex
    import nengo

    # Disable Nengo cache to avoid multiprocessing lock issues
    nengo.rc.set("decoder_cache", "enabled", "False")

    all_results = []

    # Create observer for this batch
    # Use unique folder name for each batch to avoid conflicts during parallel execution
    batch_folder = (
        f"{algorithm_name}_batch{batch_number:03d}"
        if total_batches > 1
        else algorithm_name
    )
    observer = cocoex.Observer(
        coco_suite_name,
        f"result_folder: {batch_folder} "
        f"algorithm_name: {algorithm_name} "
        f'algorithm_info: "NEVO neuromorphic optimiser"',
    )

    # Suite configuration strings (COCO uses comma-separated values for lists)
    instances_str = ",".join(str(i) for i in instances)
    problems_str = ",".join(str(p) for p in problems)
    dimensions_str = ",".join(str(d) for d in dimensions)

    # Use BatchScheduler to determine which problems this batch handles
    batcher = cocoex.BatchScheduler(total_batches, batch_number)

    # Run all experiments for this batch
    for run in range(n_runs):
        # Create a fresh suite for each run
        suite = cocoex.Suite(
            coco_suite_name,
            f"instances: {instances_str}",
            f"function_indices: {problems_str} dimensions: {dimensions_str}",
        )

        for problem in suite:
            # Skip if not in current batch
            if not batcher.is_in_batch(problem):
                continue

            problem.observe_with(observer)

            problem_id = problem.id_function
            instance = problem.id_instance
            dimension = problem.dimension

            seed = problem_id * 1000 + instance * 100 + run

            print(
                f"[Batch {batch_number + 1}/{total_batches}] "
                f"f{problem_id:02d}, i{instance}, {dimension}D, run {run + 1}/{n_runs}"
            )

            # Check if this specific run already exists (for resuming interrupted experiments)
            if output_dir is not None:
                run_results_file = (
                    Path(output_dir)
                    / f"results_f{problem_id:02d}_i{instance:02d}_{dimension}D_run{run + 1:02d}.csv"
                )
                if run_results_file.exists():
                    try:
                        existing_result = pd.read_csv(run_results_file)
                        if len(existing_result) > 0:
                            print("  -> Skipping (already completed)")
                            all_results.append(existing_result.iloc[0].to_dict())
                            continue
                    except Exception:
                        pass  # Ignore corrupted files, re-run

            # Get bounds
            lb = np.array(problem.lower_bounds)
            ub = np.array(problem.upper_bounds)

            # Wrap in BenchmarkProblem to get multi-obj scalarisation if needed
            def _eval(x, _p=problem):
                result = _p(x)
                if hasattr(result, "__len__"):
                    return float(result[0])
                return result

            # Create optimiser
            optimiser = NEVOptimiser(
                objective_function=_eval,
                bounds=(lb, ub),
                dimension=dimension,
                population_size=50,
                memory_size=25,
                neurons_per_ensemble=50,
                dt=0.001,
                epsilon=0.15,
                learning_rate=0.3,
                seed=seed,
                operator_mode=operator_mode,
                td_enabled=td_enabled,
                td_lambda=td_lambda,
            )

            # Run optimisation
            start_time = time.time()
            optimiser.run(time=simulation_time, verbose=False, use_dl=use_dl)
            elapsed = time.time() - start_time

            # Get results
            x_best, f_best = optimiser.get_best_solution()
            stats = optimiser.get_statistics()

            # Store results
            result = {
                "suite": coco_suite_name,
                "problem_id": problem_id,
                "problem_name": problem.name,
                "instance": instance,
                "dimension": dimension,
                "run": run + 1,
                "batch": batch_number + 1,
                "seed": seed,
                "best_fitness": f_best,
                "optimal_fitness": None,
                "error": None,
                "relative_error": None,
                "total_evaluations": stats["total_evaluations"],
                "wall_time": elapsed,
                "evals_per_second": stats["total_evaluations"] / elapsed,
            }

            # Add operator statistics
            for op_name, count in stats["operator_counts"].items():
                result[f"op_count_{op_name}"] = count
                result[f"op_weight_{op_name}"] = stats["operator_weights"][op_name]
                result[f"op_success_{op_name}"] = stats["operator_success_rates"][
                    op_name
                ]

            all_results.append(result)

            # Save intermediate results
            if output_dir is not None:
                pd.DataFrame([result]).to_csv(
                    Path(output_dir)
                    / f"results_f{problem_id:02d}_i{instance:02d}_{dimension}D_run{run + 1:02d}.csv",
                    index=False,
                )

    return all_results


def _merge_coco_batch_folders(
    algorithm_name: str, n_batches: int, exdata_dir: str = "exdata"
):
    """
    Merge COCO batch folders into a single folder.

    After parallel execution, each batch creates its own folder (e.g., NEVO_batch000,
    NEVO_batch001). This function merges all batch data into a single folder for
    cocopp postprocessing.

    Parameters
    ----------
    algorithm_name : str
        Base algorithm name
    n_batches : int
        Number of batches that were run
    exdata_dir : str
        Path to exdata directory
    """
    import shutil

    exdata_path = Path(exdata_dir)
    target_folder = exdata_path / algorithm_name

    # If only one batch, just rename if needed
    if n_batches == 1:
        return

    # Create target folder if it doesn't exist
    target_folder.mkdir(parents=True, exist_ok=True)

    # Merge each batch folder
    for batch_num in range(n_batches):
        batch_folder = exdata_path / f"{algorithm_name}_batch{batch_num:03d}"

        if not batch_folder.exists():
            print(f"Warning: Batch folder {batch_folder} not found")
            continue

        # Copy all contents from batch folder to target folder
        for item in batch_folder.iterdir():
            target_item = target_folder / item.name

            if item.is_dir():
                # For data directories (e.g., data_f1), merge contents
                if target_item.exists():
                    # Merge files into existing directory
                    for subitem in item.iterdir():
                        target_subitem = target_item / subitem.name
                        if target_subitem.exists():
                            # Append data to existing file
                            if subitem.suffix in [".dat", ".tdat", ".rdat", ".mdat"]:
                                with open(target_subitem, "a") as dst:
                                    with open(subitem, "r") as src:
                                        dst.write(src.read())
                        else:
                            shutil.copy2(subitem, target_subitem)
                else:
                    shutil.copytree(item, target_item)
            else:
                # For .info files, merge them
                if item.suffix == ".info":
                    if target_item.exists():
                        # Append to existing .info file (skip header for subsequent batches)
                        with open(target_item, "a") as dst:
                            with open(item, "r") as src:
                                content = src.read()
                                # Skip the first line (header) if target already has content
                                lines = content.split("\n")
                                # Find where data lines start (after % comments)
                                data_start = 0
                                for i, line in enumerate(lines):
                                    if line.startswith("data_"):
                                        data_start = i
                                        break
                                dst.write("\n" + "\n".join(lines[data_start:]))
                    else:
                        shutil.copy2(item, target_item)
                else:
                    if not target_item.exists():
                        shutil.copy2(item, target_item)

        # Remove batch folder after merging
        shutil.rmtree(batch_folder)

    print(f"Merged {n_batches} batch folders into exdata/{algorithm_name}")


def run_coco_benchmark(
    problems: list,
    instances: list,
    dimensions: list,
    simulation_time: float = 20.0,
    n_runs: int = 1,
    algorithm_name: str = "NEVO",
    output_dir: Path = None,
    n_cores: int = 1,
    use_dl: bool = False,
    coco_suite_name: str = "bbob",
    operator_mode: str = "trad",
    td_enabled: bool = True,
    td_lambda: float = 0.0,
):
    """
    Run NEVO on COCO benchmark problems using proper batch system.

    Uses cocoex.BatchScheduler for distributing experiments across cores.
    Each core runs a batch of problems with its own observer, ensuring
    proper COCO data file organisation for cocopp postprocessing.

    Parameters
    ----------
    problems : list
        List of problem/function IDs
    instances : list
        List of problem instances
    dimensions : list
        List of problem dimensions
    simulation_time : float
        Simulation time in seconds
    n_runs : int
        Number of independent runs
    algorithm_name : str
        Algorithm name for COCO logging
    output_dir : Path
        Output directory for CSV results
    n_cores : int
        Number of CPU cores to use (each core runs a batch)
    use_dl : bool
        Use NengoDL backend for GPU acceleration
    coco_suite_name : str
        COCO suite to use (e.g. "bbob", "bbob-noisy", "bbob-largescale", …).
    operator_mode : str
        Operator selection mode: "trad", "nm_dual", or "nm_softmix".
    td_enabled : bool
        Whether to enable TD learning for operator selection.
    td_lambda : float
        TD(λ) eligibility trace coefficient (0.0 = TD(0)).

    Returns
    -------
    results : pd.DataFrame
        Results from all runs
    """
    import multiprocessing as mp

    # Use spawn to avoid fork issues with Nengo on macOS/Linux
    ctx = mp.get_context("spawn")

    # Number of batches equals number of cores
    total_batches = n_cores

    # Build batch arguments
    batch_args = []
    for batch_number in range(total_batches):
        batch_args.append(
            (
                problems,
                instances,
                dimensions,
                simulation_time,
                n_runs,
                algorithm_name,
                str(output_dir) if output_dir else None,
                total_batches,
                batch_number,
                use_dl,
                coco_suite_name,
                operator_mode,
                td_enabled,
                td_lambda,
            )
        )

    # Run batches
    all_results = []
    if n_cores == 1:
        # Sequential execution - single batch
        results = _run_coco_batch(batch_args[0])
        all_results.extend(results)
    else:
        # Parallel execution - multiple batches
        with ctx.Pool(processes=n_cores) as pool:
            batch_results = pool.map(_run_coco_batch, batch_args)
            for results in batch_results:
                all_results.extend(results)

        # Merge batch folders into a single folder for cocopp
        _merge_coco_batch_folders(algorithm_name, n_cores)

    return pd.DataFrame(all_results)


def run_benchmark(
    problem_id: int,
    instance: int,
    dimension: int,
    simulation_time: float = 20.0,
    n_runs: int = 5,
    seed_offset: int = 0,
    suite: Literal["ioh", "cocoex"] = "ioh",
    observer: Any = None,
    algorithm_name: str = "NEVO",
    output_dir: Path = None,
    coco_suite_name: str = "bbob",
    operator_mode: str = "trad",
    td_enabled: bool = True,
    td_lambda: float = 0.0,
):
    """
    Run NEVO on a benchmark problem multiple times.

    Parameters
    ----------
    problem_id : int
        Problem/function ID
    instance : int
        Problem instance
    dimension : int
        Problem dimension
    simulation_time : float
        Simulation time in seconds
    n_runs : int
        Number of independent runs
    seed_offset : int
        Offset for random seeds
    suite : str
        Benchmark suite to use: "ioh" or "cocoex"
    observer : cocoex.Observer, optional
        COCO observer for logging results (only used with cocoex suite)
    output_dir : Path, optional
        Output directory for saving per-run results (enables resume capability)
    coco_suite_name : str
        COCO suite name to use when ``suite="cocoex"`` (default: "bbob").
    operator_mode : str
        Operator selection mode: "trad", "nm_dual", or "nm_softmix".
    td_enabled : bool
        Whether to enable TD learning for operator selection.
    td_lambda : float
        TD(λ) eligibility trace coefficient (0.0 = TD(0)).

    Returns
    -------
    results : pd.DataFrame
        Results from all runs
    """
    results = []

    for run in range(n_runs):
        print(f"\n{'=' * 70}")
        print(
            f"[{suite.upper()}] Problem f{problem_id:02d}, Instance {instance}, Dimension {dimension}D, Run {run + 1}/{n_runs}"
        )
        print(f"{'=' * 70}")

        # Check if this specific run already exists (for resuming interrupted experiments)
        if output_dir is not None:
            run_results_file = (
                Path(output_dir)
                / f"results_f{problem_id:02d}_i{instance:02d}_{dimension}D_run{run + 1:02d}.csv"
            )
            if run_results_file.exists():
                try:
                    existing_result = pd.read_csv(run_results_file)
                    if len(existing_result) > 0:
                        print(f"  -> Skipping run {run + 1} (already completed)")
                        results.append(existing_result.iloc[0].to_dict())
                        continue
                except Exception:
                    pass  # Ignore corrupted files, re-run

        # Get problem using the wrapper
        problem = BenchmarkProblem(
            problem_id=problem_id,
            instance=instance,
            dimension=dimension,
            suite=suite,
            observer=observer,
            coco_suite_name=coco_suite_name,
        )

        # Create optimiser
        optimiser = NEVOptimiser(
            objective_function=problem,
            bounds=problem.bounds,
            dimension=dimension,
            population_size=100,  # Increased for better exploration
            memory_size=50,  # Increased for better diversity
            neurons_per_ensemble=100,  # Increased for better action selection
            dt=0.001,
            epsilon=0.15,  # Slightly more exploration
            learning_rate=0.3,  # Slightly lower for stability
            seed=seed_offset + run,
            operator_mode=operator_mode,
            td_enabled=td_enabled,
            td_lambda=td_lambda,
        )

        # Run optimisation
        start_time = time.time()
        optimiser.run(time=simulation_time, verbose=False)
        elapsed = time.time() - start_time

        # Get results
        x_best, f_best = optimiser.get_best_solution()
        stats = optimiser.get_statistics()

        # Store results
        result = {
            "suite": coco_suite_name if suite in ["cocoex", "coco"] else suite,
            "problem_id": problem_id,
            "problem_name": problem.name,
            "instance": instance,
            "dimension": dimension,
            "run": run + 1,
            "seed": seed_offset + run,
            "best_fitness": f_best,
            "optimal_fitness": problem.optimum,
            "total_evaluations": stats["total_evaluations"],
            "wall_time": elapsed,
            "evals_per_second": stats["total_evaluations"] / elapsed,
        }

        # Add error metrics only if optimum is known (not for COCO)
        if problem.optimum is not None:
            result["error"] = f_best - problem.optimum
            result["relative_error"] = (
                (f_best - problem.optimum) / abs(problem.optimum)
                if problem.optimum != 0
                else f_best
            )
        else:
            result["error"] = None
            result["relative_error"] = None

        # Add operator statistics
        for op_name, count in stats["operator_counts"].items():
            result[f"op_count_{op_name}"] = count
            result[f"op_weight_{op_name}"] = stats["operator_weights"][op_name]
            result[f"op_success_{op_name}"] = stats["operator_success_rates"][op_name]

        results.append(result)

        # Save plot for first run (only if optimum is known)
        if run == 0 and problem.optimum is not None:
            # Use actual COCO suite name for filename (not just "cocoex")
            suite_for_filename = coco_suite_name if suite in ["cocoex", "coco"] else suite
            plot_optimisation_results(
                optimiser,
                optimum=problem.optimum,
                title=f"[{suite.upper() if suite != 'cocoex' else coco_suite_name.upper()}] f{problem_id:02d} i{instance:02d} {dimension}D",
                save_path=f"benchmark_{suite_for_filename}_f{problem_id:02d}_i{instance:02d}_{dimension}D.png",
            )

        # Finalize the problem to flush COCO data
        problem.finalize()

    return pd.DataFrame(results)


def run_single_experiment(args_tuple):
    """
    Run a single experiment (for multiprocessing).

    Parameters
    ----------
    args_tuple : tuple
        Tuple of (problem_id, instance, dimension, simulation_time, n_runs,
                  seed_offset, suite, output_dir, algorithm_name, coco_suite_name,
                  operator_mode, td_enabled, td_lambda)

    Returns
    -------
    pd.DataFrame
        Results from the experiment
    """
    import nengo

    # Disable Nengo cache to avoid multiprocessing lock issues
    nengo.rc.set("decoder_cache", "enabled", "False")

    (
        problem_id,
        instance,
        dimension,
        simulation_time,
        n_runs,
        seed_offset,
        suite,
        output_dir,
        algorithm_name,
        coco_suite_name,
        operator_mode,
        td_enabled,
        td_lambda,
    ) = args_tuple

    # Check if results already exist (for resuming interrupted experiments)
    results_file = (
        output_dir / f"results_f{problem_id:02d}_i{instance:02d}_{dimension}D.csv"
    )
    if results_file.exists():
        try:
            existing_df = pd.read_csv(results_file)
            if len(existing_df) >= n_runs:
                print(
                    f"[{suite.upper()}] Skipping f{problem_id:02d}, i{instance}, {dimension}D (already completed)"
                )
                return existing_df
        except Exception:
            pass  # Ignore corrupted files, re-run

    # Create COCO observer if using cocoex suite
    observer = None
    if suite in ["cocoex", "coco"]:
        import cocoex

        # Create observer for COCO postprocessing with cocopp
        # COCO automatically saves to exdata/<result_folder>
        observer = cocoex.Observer(
            coco_suite_name,
            f"result_folder: {algorithm_name} "
            f"algorithm_name: {algorithm_name} "
            f'algorithm_info: "NEVO neuromorphic optimiser"',
        )

    results_df = run_benchmark(
        problem_id=problem_id,
        instance=instance,
        dimension=dimension,
        simulation_time=simulation_time,
        n_runs=n_runs,
        seed_offset=seed_offset,
        suite=suite,
        observer=observer,
        coco_suite_name=coco_suite_name,
        operator_mode=operator_mode,
        td_enabled=td_enabled,
        td_lambda=td_lambda,
    )

    # Save intermediate results
    results_df.to_csv(
        output_dir / f"results_f{problem_id:02d}_i{instance:02d}_{dimension}D.csv",
        index=False,
    )

    return results_df


def main():
    """Run benchmark experiments."""
    import argparse
    from multiprocessing import cpu_count

    def parse_int_list(value: str) -> list:
        """
        Parse a string into a list of integers.

        Supports:
        - Single integers: "5" -> [5]
        - Comma-separated: "1,2,3" -> [1, 2, 3]
        - Ranges: "1-5" -> [1, 2, 3, 4, 5]
        - Mixed: "1-3,5,7-9" -> [1, 2, 3, 5, 7, 8, 9]
        - Space-separated (from nargs="+"): handled by argparse
        """
        result = []
        # Handle space-separated values from nargs="+"
        parts = value.replace(" ", ",").split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part and not part.startswith("-"):
                # It's a range like "1-5"
                try:
                    start, end = part.split("-")
                    result.extend(range(int(start), int(end) + 1))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            else:
                # It's a single number
                try:
                    result.append(int(part))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid integer: {part}")
        return result

    def parse_int_list_arg(values):
        """Parse multiple values that may contain ranges."""
        result = []
        for v in values:
            result.extend(parse_int_list(str(v)))
        return sorted(set(result))  # Remove duplicates and sort

    parser = argparse.ArgumentParser(
        description="Run NEVO benchmark experiments on IOH or COCO suites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--suite",
        type=str,
        choices=["ioh", "cocoex"],
        default="ioh",
        help="Benchmark suite to use: 'ioh' or 'cocoex'. "
             "For cocoex, specify which COCO suite with --coco-suite.",
    )
    parser.add_argument(
        "--coco-suite",
        type=str,
        choices=COCO_SUITES,
        default="bbob",
        metavar="COCO_SUITE",
        help=(
            "Which COCO/cocoex suite to run (only used when --suite cocoex). "
            "Choices: " + ", ".join(COCO_SUITES) + ". "
            "Recommended dimensions per suite — "
            "bbob/bbob-noisy/bbob-biobj/bbob-biobj-ext/bbob-constrained/sbox-cost: 2,3,5,10,20,40; "
            "bbob-largescale: 20,40,80,160,320,640; "
            "bbob-mixint: 5,10,20,40,80,160. "
            "Multi-objective suites (bbob-biobj, bbob-biobj-ext) are scalarised "
            "to their first objective."
        ),
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["1"],
        help="Problem/function IDs. Supports ranges: 1-24, 1,2,5, or 1-3,5,7-9",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=["1-5"],
        help="Problem instances. Supports ranges: 1-15, 1,2,3, or 1-5,10",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=["2"],
        help=(
            "Problem dimensions. Supports ranges: 2,5,10 or 2-10. "
            "Valid dims depend on the chosen --coco-suite "
            "(bbob default: 2,3,5,10,20,40; largescale: 20,40,80,160,320,640)."
        ),
    )
    parser.add_argument(
        "--time",
        type=float,
        default=20.0,
        help="Simulation time in seconds",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs per problem",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to use (1 = sequential, 0 = all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: benchmark_results_<suite>_<coco-suite>)",
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default="NEVO",
        help="Algorithm name for COCO logging (used with cocoex suite)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Total number of batches for COCO (for parallel execution on different machines)",
    )
    parser.add_argument(
        "--current-batch",
        type=int,
        default=1,
        help="Current batch number (1-indexed, used with --batch for COCO)",
    )
    parser.add_argument(
        "--use-dl",
        action="store_true",
        help="Use NengoDL backend for GPU acceleration (requires nengo-dl and tensorflow)",
    )
    parser.add_argument(
        "--operator-mode",
        type=str,
        choices=["trad", "traditional", "nm_dual", "nm_softmix"],
        default="trad",
        help=(
            "Operator selection mode: "
            "'trad'/'traditional' = 13 standard operators (default), "
            "'nm_dual' = 2 neuromorphic ensembles, "
            "'nm_softmix' = neuromorphic ensembles with soft blending."
        ),
    )
    parser.add_argument(
        "--no-td",
        action="store_true",
        default=False,
        help="Disable TD learning for operator selection (enabled by default).",
    )
    parser.add_argument(
        "--td-lambda",
        type=float,
        default=0.0,
        help="TD(λ) eligibility trace coefficient. 0.0 = TD(0), e.g. 0.9 = TD(λ).",
    )

    args = parser.parse_args()

    # Auto-detect cocoex suite if --coco-suite was explicitly set (not default)
    # This allows users to just pass --coco-suite bbob-largescale without --suite cocoex
    if args.coco_suite != "bbob" and args.suite == "ioh":
        print(f"[INFO] --coco-suite {args.coco_suite} specified but --suite not set to cocoex.")
        print(f"[INFO] Automatically switching to --suite cocoex")
        args.suite = "cocoex"

    # Parse range notation for problems, instances, and dimensions
    args.problems = parse_int_list_arg(args.problems)
    args.instances = parse_int_list_arg(args.instances)
    args.dimensions = parse_int_list_arg(args.dimensions)

    # Warn if chosen dimensions are unusual for the selected COCO suite
    if args.suite == "cocoex":
        recommended = COCO_SUITE_DIMS.get(args.coco_suite, [])
        if recommended:
            unusual = [d for d in args.dimensions if d not in recommended]
            if unusual:
                print(
                    f"[WARNING] Dimensions {unusual} are not in the recommended set "
                    f"for suite '{args.coco_suite}': {recommended}. "
                    "cocoex may raise an error if a dimension is unsupported."
                )

    # Resolve td_enabled from --no-td flag
    td_enabled = not args.no_td

    # Set output directory — include operator_mode suffix to avoid collisions
    # across different configuration runs
    if args.output_dir is None:
        if args.suite == "cocoex":
            base = f"benchmark_results_{args.coco_suite.replace('-', '_')}"
        else:
            base = f"benchmark_results_{args.suite}"
        output_dir = Path(f"{base}_{args.operator_mode}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Determine number of cores
    n_cores = args.cores
    if n_cores == 0:
        n_cores = cpu_count()
    elif n_cores < 0:
        n_cores = max(1, cpu_count() + n_cores)

    print(f"\n{'=' * 70}")
    print("NEVO BENCHMARK EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Suite:        {args.suite.upper()}", end="")
    if args.suite == "cocoex":
        print(f" / {args.coco_suite}")
    else:
        print()
    print(f"Problems:     {args.problems}")
    print(f"Instances:    {args.instances}")
    print(f"Dimensions:   {args.dimensions}")
    print(f"Time:         {args.time}s")
    print(f"Runs:         {args.runs}")
    print(f"Cores:        {n_cores}")
    print(f"Output:       {output_dir}")
    print(f"Op. mode:     {args.operator_mode}")
    print(f"TD learning:  {'enabled' if td_enabled else 'disabled'}")
    print(f"TD lambda:    {args.td_lambda}")
    if args.suite == "cocoex":
        print(f"Algorithm:    {args.algorithm_name}")
        print(f"Batches:      {n_cores} (1 per core)")
        print(f"COCO data:    exdata/{args.algorithm_name}")
        if args.coco_suite in COCO_MULTIOBJECTIVE_SUITES:
            print(
                f"[NOTE] '{args.coco_suite}' is a multi-objective suite. "
                "NEVO will optimise the first objective only."
            )
    if args.use_dl:
        print("Backend:      NengoDL (GPU accelerated)")
    print(f"{'=' * 70}\n")

    # Build list of experiments
    experiments = []
    for problem_id in args.problems:
        for instance in args.instances:
            for dimension in args.dimensions:
                seed_offset = problem_id * 1000 + instance * 100
                experiments.append(
                    (
                        problem_id,
                        instance,
                        dimension,
                        args.time,
                        args.runs,
                        seed_offset,
                        args.suite,
                        output_dir,
                        args.algorithm_name,
                        args.coco_suite,
                        args.operator_mode,
                        td_enabled,
                        args.td_lambda,
                    )
                )

    print(f"Total experiments: {len(experiments)}")

    # Run experiments
    all_results = []

    if args.suite == "cocoex":
        # Use COCO batch system - each core runs a batch
        # COCO observer automatically appends to existing data, enabling resume
        combined_df = run_coco_benchmark(
            problems=args.problems,
            instances=args.instances,
            dimensions=args.dimensions,
            simulation_time=args.time,
            n_runs=args.runs,
            algorithm_name=args.algorithm_name,
            output_dir=output_dir,
            n_cores=n_cores,
            use_dl=args.use_dl,
            coco_suite_name=args.coco_suite,
            operator_mode=args.operator_mode,
            td_enabled=td_enabled,
            td_lambda=args.td_lambda,
        )
    else:
        # IOH suite - use original approach
        if n_cores == 1:
            # Sequential execution
            for exp in experiments:
                results_df = run_single_experiment(exp)
                all_results.append(results_df)
        else:
            # Parallel execution with spawn context to avoid Nengo issues
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_cores) as pool:
                all_results = pool.map(run_single_experiment, experiments)

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / "all_results.csv", index=False)

    # Summary statistics (handle COCO case where error is None)
    if args.suite == "cocoex":
        # For COCO, only aggregate non-error metrics
        summary = combined_df.groupby(["suite", "problem_id", "dimension"]).agg(
            {
                "best_fitness": ["mean", "std", "min", "max"],
                "total_evaluations": "mean",
                "wall_time": "mean",
            }
        )
    else:
        summary = combined_df.groupby(["suite", "problem_id", "dimension"]).agg(
            {
                "error": ["mean", "std", "min", "max"],
                "total_evaluations": "mean",
                "wall_time": "mean",
            }
        )

    summary.to_csv(output_dir / "summary_statistics.csv")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nSummary statistics:")
    print(summary)

    # Print cocopp postprocessing instructions for COCO suite
    if args.suite == "cocoex":
        print("\n" + "-" * 70)
        print("COCO POSTPROCESSING")
        print("-" * 70)
        print(f"\nSuite used: {args.coco_suite}")
        print("\nTo postprocess results with cocopp, run:")
        print(f"  python -m cocopp exdata/{args.algorithm_name}")
        print("\nOr to compare with other algorithms:")
        print(
            f"  python -m cocopp exdata/{args.algorithm_name} <other_algorithm_folder>"
        )
        if args.coco_suite == "bbob":
            print("\nExample with public BBOB baselines:")
            print(
                f"  python -m cocopp exdata/{args.algorithm_name} "
                "bbob/2009/RANDOMSEARCH bbob/2009/PSO! bbob/2012/DE!"
            )
        elif args.coco_suite == "bbob-noisy":
            print("\nExample with public bbob-noisy baselines:")
            print(
                f"  python -m cocopp exdata/{args.algorithm_name} "
                "bbob-noisy/2009/RANDOMSEARCH"
            )
        elif args.coco_suite == "bbob-largescale":
            print("\nExample with public bbob-largescale baselines:")
            print(
                f"  python -m cocopp exdata/{args.algorithm_name} "
                "bbob-largescale/2019/LMCMA"
            )
        print("\nResults will be generated in the 'ppdata' folder.")
        print("-" * 70)


if __name__ == "__main__":
    main()
