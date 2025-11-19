"""
Basic NEVO Example
==================

This example demonstrates basic usage of NEVO optimiser on
standard benchmark functions from IOHexperimenter.
"""

import numpy as np
import matplotlib.pyplot as plt
from ioh import get_problem

from nevo import NEVOptimiser
from nevo.utils import plot_optimisation_results, plot_operator_statistics


def main():
    """Run basic NEVO optimisation example."""

    # Problem setup
    PROBLEM_ID = 1  # Sphere function
    PROBLEM_INS = 1
    NUM_DIMS = 10
    SIMULATION_TIME = 20.0

    # Get problem from IOHexperimenter
    problem = get_problem(
        fid=PROBLEM_ID,
        instance=PROBLEM_INS,
        dimension=NUM_DIMS
    )
    problem.reset()

    print("=" * 70)
    print("NEVO - Neuromorphic Evolutionary Optimisation")
    print("=" * 70)
    print(f"Problem: {problem.meta_data.name} (f{PROBLEM_ID:02d})")
    print(f"Dimension: {NUM_DIMS}D")
    print(f"Optimal fitness: {problem.optimum.y:.6f}")
    print("=" * 70)
    print()

    # Create optimiser
    optimiser = NEVOptimiser(
        objective_function=problem,
        bounds=(problem.bounds.lb, problem.bounds.ub),
        dimension=NUM_DIMS,
        population_size=50,
        memory_size=25,
        neurons_per_ensemble=100,
        dt=0.001,
        epsilon=0.1,
        learning_rate=0.4,
        seed=42,
    )

    # Run optimisation
    optimiser.run(time=SIMULATION_TIME, verbose=True)

    # Get best solution
    x_best, f_best = optimiser.get_best_solution()

    print("\n" + "=" * 70)
    print("SOLUTION QUALITY")
    print("=" * 70)
    print(f"Best fitness found: {f_best:.6e}")
    print(f"Optimal fitness:    {problem.optimum.y:.6e}")
    print(f"Error from optimum: {f_best - problem.optimum.y:.6e}")
    print(f"Relative error:     {(f_best - problem.optimum.y) / abs(problem.optimum.y):.6e}")

    print(f"\nBest solution found:")
    print(f"  {x_best}")
    print(f"\nOptimal solution:")
    print(f"  {problem.optimum.x}")

    # Visualise results
    plot_optimisation_results(
        optimiser,
        optimum=problem.optimum.y,
        title=f'{problem.meta_data.name} (f{PROBLEM_ID:02d}) {NUM_DIMS}D',
        save_path=f'nevo_example_f{PROBLEM_ID:02d}_{NUM_DIMS}D.png'
    )

    plot_operator_statistics(
        optimiser,
        save_path=f'nevo_operators_f{PROBLEM_ID:02d}_{NUM_DIMS}D.png'
    )

    plt.show()


if __name__ == "__main__":
    main()

