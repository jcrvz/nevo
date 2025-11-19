"""
Benchmark Experiment Runner
===========================

Run NEVO on multiple benchmark problems and compare with v7 implementation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ioh import get_problem
import time

from nevo import NEVOptimiser
from nevo.utils import plot_optimisation_results, plot_operator_statistics


def run_benchmark(
    problem_id: int,
    instance: int,
    dimension: int,
    simulation_time: float = 20.0,
    n_runs: int = 5,
    seed_offset: int = 0,
):
    """
    Run NEVO on a benchmark problem multiple times.

    Parameters
    ----------
    problem_id : int
        IOH problem ID
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

    Returns
    -------
    results : pd.DataFrame
        Results from all runs
    """
    results = []

    for run in range(n_runs):
        print(f"\n{'='*70}")
        print(f"Problem f{problem_id:02d}, Instance {instance}, Dimension {dimension}D, Run {run+1}/{n_runs}")
        print(f"{'='*70}")

        # Get problem
        problem = get_problem(
            fid=problem_id,
            instance=instance,
            dimension=dimension
        )
        problem.reset()

        # Create optimiser
        optimiser = NEVOptimiser(
            objective_function=problem,
            bounds=(problem.bounds.lb, problem.bounds.ub),
            dimension=dimension,
            population_size=50,
            memory_size=25,
            neurons_per_ensemble=100,
            dt=0.001,
            epsilon=0.1,
            learning_rate=0.4,
            seed=seed_offset + run,
        )

        # Run optimization
        start_time = time.time()
        optimiser.run(time=simulation_time, verbose=True)
        elapsed = time.time() - start_time

        # Get results
        x_best, f_best = optimiser.get_best_solution()
        stats = optimiser.get_statistics()

        # Store results
        result = {
            'problem_id': problem_id,
            'instance': instance,
            'dimension': dimension,
            'run': run + 1,
            'seed': seed_offset + run,
            'best_fitness': f_best,
            'optimal_fitness': problem.optimum.y,
            'error': f_best - problem.optimum.y,
            'relative_error': (f_best - problem.optimum.y) / abs(problem.optimum.y),
            'total_evaluations': stats['total_evaluations'],
            'wall_time': elapsed,
            'evals_per_second': stats['total_evaluations'] / elapsed,
        }

        # Add operator statistics
        for op_name, count in stats['operator_counts'].items():
            result[f'op_count_{op_name}'] = count
            result[f'op_weight_{op_name}'] = stats['operator_weights'][op_name]
            result[f'op_success_{op_name}'] = stats['operator_success_rates'][op_name]

        results.append(result)

        # Save plot for first run
        if run == 0:
            plot_optimisation_results(
                optimiser,
                optimum=problem.optimum.y,
                title=f'f{problem_id:02d} i{instance:02d} {dimension}D',
                save_path=f'benchmark_f{problem_id:02d}_i{instance:02d}_{dimension}D.png'
            )

    return pd.DataFrame(results)


def main():
    """Run benchmark experiments."""

    # Configuration
    PROBLEMS = [1, 2, 10, 15, 20]  # Sphere, Ellipsoid, Rosenbrock, Rastrigin, Schwefel
    INSTANCES = [1, 2]
    DIMENSIONS = [2, 5, 10]
    SIMULATION_TIME = 20.0
    N_RUNS = 5

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for problem_id in PROBLEMS:
        for instance in INSTANCES:
            for dimension in DIMENSIONS:
                results_df = run_benchmark(
                    problem_id=problem_id,
                    instance=instance,
                    dimension=dimension,
                    simulation_time=SIMULATION_TIME,
                    n_runs=N_RUNS,
                    seed_offset=problem_id * 1000 + instance * 100,
                )

                all_results.append(results_df)

                # Save intermediate results
                results_df.to_csv(
                    output_dir / f'results_f{problem_id:02d}_i{instance:02d}_{dimension}D.csv',
                    index=False
                )

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / 'all_results.csv', index=False)

    # Summary statistics
    summary = combined_df.groupby(['problem_id', 'dimension']).agg({
        'error': ['mean', 'std', 'min', 'max'],
        'total_evaluations': 'mean',
        'wall_time': 'mean',
    })

    summary.to_csv(output_dir / 'summary_statistics.csv')

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nSummary statistics:")
    print(summary)


if __name__ == "__main__":
    main()

