"""
Small benchmark harness for comparing NEVO operator modes.

Compares:
- default
- neuromorphic_dual
- neuromorphic_soft_mix

on simple analytic functions with fixed reps.
"""

import argparse
import csv
import math
from statistics import mean, pstdev
from typing import Callable, Dict, List

import numpy as np

from nevo import NEVOptimiser


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    n = x.shape[0]
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * math.pi * x)))


def run_single(
    objective: Callable[[np.ndarray], float],
    mode: str,
    dimension: int,
    sim_time: float,
    seed: int,
) -> float:
    optimiser = NEVOptimiser(
        objective_function=objective,
        bounds=(-5.0, 5.0),
        dimension=dimension,
        population_size=30,
        memory_size=15,
        operator_mode=mode,
        seed=seed,
    )
    optimiser.run(time=sim_time, verbose=False)
    return float(optimiser.state["best_f"]) if optimiser.state["best_f"] is not None else float("inf")


def benchmark(
    problems: Dict[str, Callable[[np.ndarray], float]],
    modes: List[str],
    reps: List[int],
    dimension: int,
    sim_time: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for problem_name, objective in problems.items():
        for mode in modes:
            scores = [run_single(objective, mode, dimension, sim_time, s) for s in reps]
            rows.append(
                {
                    "problem": problem_name,
                    "mode": mode,
                    "mean_best_f": mean(scores),
                    "std_best_f": pstdev(scores),
                    "min_best_f": min(scores),
                    "max_best_f": max(scores),
                }
            )
    return rows


def write_csv(rows: List[Dict[str, float]], out_csv: str) -> None:
    if not rows:
        return
    fieldnames = ["problem", "mode", "mean_best_f", "std_best_f", "min_best_f", "max_best_f"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, float]]) -> None:
    print("\nBenchmark Results")
    print("-" * 92)
    print(f"{'problem':12s} {'mode':24s} {'mean_best_f':14s} {'std_best_f':14s} {'min_best_f':14s}")
    print("-" * 92)
    for r in rows:
        print(
            f"{r['problem']:12s} {r['mode']:24s} "
            f"{r['mean_best_f']:14.6e} {r['std_best_f']:14.6e} {r['min_best_f']:14.6e}"
        )
    print("-" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NEVO operator modes.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--time", type=float, default=1.0, dest="sim_time")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=str, default="operator_mode_benchmark.csv")
    args = parser.parse_args()

    problems = {
        "sphere": sphere,
        "rastrigin": rastrigin,
    }
    modes = ["default", "neuromorphic_dual", "neuromorphic_soft_mix"]
    reps = list(range(args.reps))

    rows = benchmark(
        problems=problems,
        modes=modes,
        reps=reps,
        dimension=args.dimension,
        sim_time=args.sim_time,
    )
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nSaved CSV: {args.out}")


if __name__ == "__main__":
    main()

