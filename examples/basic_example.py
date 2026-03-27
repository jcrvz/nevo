"""
Basic NEVO Example
==================

This example demonstrates basic usage of NEVO optimiser on
standard benchmark functions from IOHexperimenter.
"""

import argparse
import matplotlib.pyplot as plt
from ioh import get_problem

from nevo import NEVOptimiser
from nevo.utils import plot_optimisation_results, plot_operator_statistics
import nevo.operators.standard as nev_ops


def main(args=None):
    """Run basic NEVO optimisation example."""

    parser = argparse.ArgumentParser(
        description="Run a basic NEVO optimisation on an IOH benchmark problem."
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        default=1,
        help="IOH problem identifier (default: 1 for Sphere).",
    )
    parser.add_argument(
        "--problem-ids",
        type=str,
        default=None,
        help="Comma list or ranges of IOH problem ids (e.g. 1,6,10 or 1-15). Overrides --problem-id if set.",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=1,
        help="IOH problem instance (default: 1).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10,
        help="Number of dimensions for the problem (default: 10).",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="Simulation time in seconds (default: 10.0).",
    )
    opts = parser.parse_args(args=args)

    def parse_id_list(spec: str):
        if not spec:
            return []
        ids = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                ids.extend(range(int(start), int(end) + 1))
            else:
                ids.append(int(part))
        return ids

    problem_ids = (
        parse_id_list(opts.problem_ids) if opts.problem_ids else [opts.problem_id]
    )
    problem_ins = opts.instance
    num_dims = opts.dimensions
    simulation_time = opts.time

    print(
        f"Selected problem_ids={problem_ids}, instance={problem_ins}, dimensions={num_dims}, time={simulation_time}s"
    )

    for problem_id in problem_ids:
        # Get problem from IOHexperimenter
        problem = get_problem(fid=problem_id, instance=problem_ins, dimension=num_dims)
        problem.reset()

        print("=" * 70)
        print("NEVO - Neuromorphic Evolutionary Optimisation")
        print("=" * 70)
        print(f"Problem: {problem.meta_data.name} (f{problem_id:02d})")
        print(f"Dimension: {num_dims}D")
        print(f"Optimal fitness: {problem.optimum.y:.6f}")
        print("=" * 70)
        print()

        operators = [
            nev_ops.LevyFlight(),
            nev_ops.DifferentialEvolution(),
            nev_ops.ParticleSwarm(),
            nev_ops.SpiralOptimisation(),
            nev_ops.RandomSearch(),
            nev_ops.LocalRandomWalk(),
            nev_ops.GravitationalSearch(),
            nev_ops.FireflyAlgorithm(),
            nev_ops.CentralForce(),
            nev_ops.GeneticCrossover(),
            nev_ops.GeneticMutation(),
            nev_ops.SimulatedAnnealing(),
            nev_ops.TabuSearch(),
        ]

        # Create optimiser
        optimiser = NEVOptimiser(
            objective_function=problem,
            bounds=(problem.bounds.lb, problem.bounds.ub),
            dimension=num_dims,
            population_size=50,
            memory_size=25,
            neurons_per_ensemble=100,
            dt=0.001,
            epsilon=0.1,
            learning_rate=0.4,
            seed=69,
            operators=operators,
        )

        # Run optimisation
        optimiser.run(time=simulation_time, verbose=True)

        # Get best solution
        x_best, f_best = optimiser.get_best_solution()

        print("\n" + "=" * 70)
        print("SOLUTION QUALITY")
        print("=" * 70)
        print(f"Best fitness found: {f_best:.6e}")
        print(f"Optimal fitness:    {problem.optimum.y:.6e}")
        print(f"Error from optimum: {f_best - problem.optimum.y:.6e}")
        print(
            f"Relative error:     {(f_best - problem.optimum.y) / abs(problem.optimum.y):.6e}"
        )

        print("\nBest solution found:")
        print(f"  {x_best}")
        print("\nOptimal solution:")
        print(f"  {problem.optimum.x}")

        # Visualise results
        plot_optimisation_results(
            optimiser,
            optimum=problem.optimum.y,
            title=f"f{problem_id:02d}-i{problem_ins:02d} {num_dims}D",
            save_path=f"nevo_example_f{problem_id:02d}_i{problem_ins:02d}_{num_dims}D.png",
        )

        plot_operator_statistics(
            optimiser,
            save_path=f"nevo_operators_f{problem_id:02d}_i{problem_ins:02d}_{num_dims}D.png",
        )

    plt.show()


if __name__ == "__main__":
    main()
