#!/usr/bin/env python
"""
Check progress of benchmark experiments based on saved CSV files.

Usage:
    python check_progress.py [output_dir] [--expected-problems N] [--expected-instances N] [--expected-dimensions D1,D2,...] [--expected-runs N]

Examples:
    python check_progress.py benchmark_results_cocoex
    python check_progress.py benchmark_results_cocoex --expected-problems 24 --expected-instances 15 --expected-dimensions 2,3,5,10 --expected-runs 1
"""

import argparse
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict


def parse_results_filename(filename: str) -> dict:
    """
    Parse a results filename to extract problem, instance, dimension, and run.

    Expected format: results_f{problem:02d}_i{instance:02d}_{dimension}D_run{run:02d}.csv
    """
    pattern = r"results_f(\d+)_i(\d+)_(\d+)D_run(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        return {
            "problem": int(match.group(1)),
            "instance": int(match.group(2)),
            "dimension": int(match.group(3)),
            "run": int(match.group(4)),
        }
    return None


def check_progress(
    output_dir: Path,
    expected_problems: list = None,
    expected_instances: list = None,
    expected_dimensions: list = None,
    expected_runs: int = 1,
):
    """
    Check the progress of benchmark experiments.

    Parameters
    ----------
    output_dir : Path
        Directory containing result CSV files
    expected_problems : list, optional
        List of expected problem IDs (default: 1-24)
    expected_instances : list, optional
        List of expected instances (default: 1-15)
    expected_dimensions : list, optional
        List of expected dimensions (default: [2, 3, 5, 10])
    expected_runs : int
        Number of expected runs per experiment (default: 1)
    """
    if expected_problems is None:
        expected_problems = list(range(1, 25))  # f1-f24
    if expected_instances is None:
        expected_instances = list(range(1, 16))  # i1-i15
    if expected_dimensions is None:
        expected_dimensions = [2, 3, 5, 10]

    # Find all per-run result files
    result_files = list(output_dir.glob("results_f*_i*_*D_run*.csv"))

    # Parse all filenames
    completed = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )  # problem -> instance -> dimension -> {runs}

    for f in result_files:
        parsed = parse_results_filename(f.name)
        if parsed:
            # Verify file is not empty/corrupted
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    completed[parsed["problem"]][parsed["instance"]][
                        parsed["dimension"]
                    ].add(parsed["run"])
            except Exception:
                pass  # Skip corrupted files

    # Calculate statistics
    total_expected = (
        len(expected_problems)
        * len(expected_instances)
        * len(expected_dimensions)
        * expected_runs
    )
    total_completed = sum(
        len(runs)
        for prob_data in completed.values()
        for inst_data in prob_data.values()
        for runs in inst_data.values()
    )

    # Print summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK PROGRESS REPORT")
    print(f"{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    print("Expected configuration:")
    print(
        f"  Problems:   {len(expected_problems)} (f{min(expected_problems):02d}-f{max(expected_problems):02d})"
    )
    print(
        f"  Instances:  {len(expected_instances)} (i{min(expected_instances):02d}-i{max(expected_instances):02d})"
    )
    print(f"  Dimensions: {expected_dimensions}")
    print(f"  Runs:       {expected_runs}")
    print(f"  Total:      {total_expected} experiments\n")

    print(
        f"Completed: {total_completed}/{total_expected} ({100 * total_completed / total_expected:.1f}%)"
    )
    print(f"Remaining: {total_expected - total_completed}\n")

    # Breakdown by dimension
    print(f"{'=' * 70}")
    print("BREAKDOWN BY DIMENSION")
    print(f"{'=' * 70}")

    for dim in sorted(expected_dimensions):
        dim_expected = len(expected_problems) * len(expected_instances) * expected_runs
        dim_completed = sum(
            len(runs)
            for prob_data in completed.values()
            for inst_data in prob_data.values()
            for d, runs in inst_data.items()
            if d == dim
        )
        pct = 100 * dim_completed / dim_expected if dim_expected > 0 else 0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {dim:3d}D: {bar} {dim_completed:4d}/{dim_expected:4d} ({pct:5.1f}%)")

    # Breakdown by problem
    print(f"\n{'=' * 70}")
    print("BREAKDOWN BY PROBLEM")
    print(f"{'=' * 70}")

    for prob in sorted(expected_problems):
        prob_expected = (
            len(expected_instances) * len(expected_dimensions) * expected_runs
        )
        prob_completed = sum(
            len(runs)
            for inst_data in completed.get(prob, {}).values()
            for runs in inst_data.values()
        )
        pct = 100 * prob_completed / prob_expected if prob_expected > 0 else 0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  f{prob:02d}: {bar} {prob_completed:4d}/{prob_expected:4d} ({pct:5.1f}%)"
        )

    # Find missing experiments
    print(f"\n{'=' * 70}")
    print("MISSING EXPERIMENTS")
    print(f"{'=' * 70}")

    missing = []
    for prob in expected_problems:
        for inst in expected_instances:
            for dim in expected_dimensions:
                for run in range(1, expected_runs + 1):
                    if run not in completed.get(prob, {}).get(inst, {}).get(dim, set()):
                        missing.append((prob, inst, dim, run))

    if missing:
        print(f"\nTotal missing: {len(missing)} experiments")
        if len(missing) <= 20:
            print("\nMissing experiments:")
            for prob, inst, dim, run in missing:
                print(f"  f{prob:02d}, i{inst:02d}, {dim}D, run {run}")
        else:
            print("\nFirst 20 missing experiments:")
            for prob, inst, dim, run in missing[:20]:
                print(f"  f{prob:02d}, i{inst:02d}, {dim}D, run {run}")
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("\n✓ All experiments completed!")

    print(f"\n{'=' * 70}\n")

    return {
        "total_expected": total_expected,
        "total_completed": total_completed,
        "missing": missing,
        "completed": completed,
    }


def main():
    def parse_int_list(value: str) -> list:
        """
        Parse a string into a list of integers.

        Supports:
        - Single integers: "5" -> [5]
        - Comma-separated: "1,2,3" -> [1, 2, 3]
        - Ranges: "1-5" -> [1, 2, 3, 4, 5]
        - Mixed: "1-3,5,7-9" -> [1, 2, 3, 5, 7, 8, 9]
        """
        result = []
        for part in value.split(","):
            part = part.strip()
            if "-" in part and not part.startswith("-"):
                try:
                    start, end = part.split("-")
                    result.extend(range(int(start), int(end) + 1))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            else:
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
        return sorted(set(result))

    parser = argparse.ArgumentParser(
        description="Check progress of NEVO benchmark experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["1-24"],
        help="Problem/function IDs. Supports ranges: 1-24, 1,2,5, or 1-3,5,7-9",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=["1-15"],
        help="Problem instances. Supports ranges: 1-15, 1,2,3, or 1-5,10",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=["2,3,5,10"],
        help="Problem dimensions. Supports ranges: 2,5,10 or 2-10",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of expected runs per experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results_cocoex",
        help="Output directory containing result CSV files",
    )

    args = parser.parse_args()

    # Parse range notation for problems, instances, and dimensions
    expected_problems = parse_int_list_arg(args.problems)
    expected_instances = parse_int_list_arg(args.instances)
    expected_dimensions = parse_int_list_arg(args.dimensions)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory '{output_dir}' does not exist.")
        return 1

    check_progress(
        output_dir=output_dir,
        expected_problems=expected_problems,
        expected_instances=expected_instances,
        expected_dimensions=expected_dimensions,
        expected_runs=args.runs,
    )

    return 0


if __name__ == "__main__":
    exit(main())
