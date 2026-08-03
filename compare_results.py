#!/usr/bin/env python
"""
NEVO Benchmark Comparison
==========================

Merges result CSVs from multiple experiment sub-folders and produces
side-by-side comparison figures suitable for IEEE two-column format.

Typical usage
-------------
  python compare_results.py                            # auto-discover benchmark_results_bbob/*
  python compare_results.py --input-dir my_results     # custom parent directory
  python compare_results.py --output-dir figs_cmp      # custom output
  python compare_results.py --experiments trad_td0,nm_dual_td0   # subset of folders
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.distutils.fcompiler import none
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE — matches plot_results.py
# =============================================================================

import shutil

USE_LATEX = shutil.which("latex") is not None

if USE_LATEX:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times", "Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        }
    )
else:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.facecolor": "white",
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "black",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
    }
)

sns.set_style(
    "ticks",
    {
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
    },
)

# IEEE column widths
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16
DPI = 300

FIGSIZE_SINGLE = (SINGLE_COL_WIDTH, 2.6)
FIGSIZE_SINGLE_TALL = (SINGLE_COL_WIDTH, 3.4)
FIGSIZE_DOUBLE = (DOUBLE_COL_WIDTH, 2.8)
FIGSIZE_DOUBLE_TALL = (DOUBLE_COL_WIDTH, 4.2)

# Small epsilon for log-safe transformations
LOG_EPS = 1e-10

# Success threshold: run is "solved" when perf (absolute error f(x)−f*) ≤ this value.
# 1e-6 is the standard BBOB precision target used by COCO.
SUCCESS_THRESHOLD = 1e-6

# Column used as the primary performance metric throughout the script.
# Set dynamically in add_performance_metric().
_PERF_COL: str = "relative_error"
_PERF_LABEL: str = r"$\delta_{\mathrm{rel}}$"
_PERF_LOG_LABEL: str = r"$\log(\delta_{\mathrm{rel}} + \varepsilon)$"
_PERF_LOWER_IS_BETTER: bool = True

# Distribution plot style — set to True via --violin CLI flag.
_USE_VIOLIN: bool = False

# BBOB categories
BBOB_CATEGORIES = {
    "separ\n(f1-f5)": [1, 2, 3, 4, 5],
    "lcond\n(f6-f9)": [6, 7, 8, 9],
    "hcond\n(f10-14)": [10, 11, 12, 13, 14],
    "multi\n(f15-19)": [15, 16, 17, 18, 19],
    "multi2\n(f20-24)": [20, 21, 22, 23, 24],
}

FUNCTION_TO_CATEGORY = {
    fid: cat for cat, fids in BBOB_CATEGORIES.items() for fid in fids
}

# =============================================================================
# LABEL HELPERS
# =============================================================================

_MODE_LABELS = {
    "trad": "Trad.",
    "nm_dual": "NM-Dual",
    "nm_softmix": "NM-SoftMix",
}

_RULE_LABELS_LATEX = {
    "td0": r"TD(0)",
    "td_lambda": r"TD($\lambda$)",
    "eps_greedy": r"$\varepsilon$-greedy",
}

_RULE_LABELS_PLAIN = {
    "td0": "TD(0)",
    "td_lambda": "TD(lam)",
    "eps_greedy": "eps-greedy",
}


def _parse_label(folder_name: str, use_latex: bool = True) -> str:
    """
    Derive a human-readable experiment label from a folder name.

    Label order: RL strategy first, then operator mode.

    Examples
    --------
    benchmark_results_bbob_trad_td0          → 'TD(0) / Trad.'
    benchmark_results_bbob_nm_dual_td_lambda → 'TD(λ) / NM-Dual'
    my_exp_nm_softmix_eps_greedy             → 'ε-greedy / NM-SoftMix'
    unknown_name                             → 'unknown_name'
    """
    rule_labels = _RULE_LABELS_LATEX if use_latex else _RULE_LABELS_PLAIN

    # Try to match known mode tokens (longest first)
    mode_key = None
    for key in sorted(_MODE_LABELS, key=len, reverse=True):
        if key in folder_name:
            mode_key = key
            break

    # Try to match known rule tokens (longest first)
    rule_key = None
    for key in sorted(rule_labels, key=len, reverse=True):
        if key in folder_name:
            rule_key = key
            break

    if mode_key and rule_key:
        return f"{rule_labels[rule_key]}+{_MODE_LABELS[mode_key]}"
    if mode_key:
        return _MODE_LABELS[mode_key]
    if rule_key:
        return rule_labels[rule_key]

    # Fallback: strip common prefix and return remainder
    cleaned = re.sub(r"^benchmark_results_[^_]+_", "", folder_name)
    return cleaned or folder_name


# =============================================================================
# DATA LOADING & MERGING
# =============================================================================


def discover_experiments(parent: Path) -> dict[str, Path]:
    """
    Return {folder_name: path} for every sub-folder that contains results.
    A folder is valid if it contains all_results.csv or at least one
    results_*.csv file.
    """
    experiments = {}
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        has_all = (child / "all_results.csv").exists()
        has_individual = any(child.glob("results_*.csv"))
        if has_all or has_individual:
            experiments[child.name] = child
    return experiments


def load_one_experiment(folder: Path) -> pd.DataFrame:
    """Load a single experiment folder into a DataFrame."""
    all_csv = folder / "all_results.csv"
    if all_csv.exists():
        try:
            df = pd.read_csv(all_csv)
            if len(df) > 0:
                return df
        except Exception as e:
            print(f"  Warning: could not read {all_csv}: {e}")

    # Fall back to individual files
    frames = []
    for f in sorted(folder.glob("results_*.csv")):
        try:
            tmp = pd.read_csv(f)
            if len(tmp) > 0:
                frames.append(tmp)
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(f"No usable CSV files found in {folder}")


def load_all_experiments(
    parent: Path,
    selected: list[str] | None = None,
) -> pd.DataFrame:
    """
    Discover, load, and merge all experiments under *parent*.

    Each row gains an `experiment` column (human-readable label) and an
    `experiment_key` column (raw folder name, useful for sorting).

    Parameters
    ----------
    parent:
        Root directory that contains one sub-folder per experiment.
    selected:
        If provided, only load sub-folders whose names appear in this list
        (matched as substring or exact).
    """
    available = discover_experiments(parent)
    if not available:
        raise ValueError(f"No experiment folders found in {parent}")

    if selected:
        filtered = {}
        for key in selected:
            matches = {k: v for k, v in available.items() if key in k}
            if not matches:
                print(f"  Warning: no folder matched '{key}', skipping")
            filtered.update(matches)
        available = filtered

    frames = []
    print(f"\nLoading {len(available)} experiment(s):")
    for folder_name, folder_path in available.items():
        label = _parse_label(folder_name, use_latex=USE_LATEX)
        try:
            df = load_one_experiment(folder_path)
            df["experiment_key"] = folder_name
            df["experiment"] = label
            frames.append(df)
            print(f"  [{label}]  {len(df)} rows  ←  {folder_path.name}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    if not frames:
        raise ValueError("No data loaded — check your --input-dir or --experiments.")

    merged = pd.concat(frames, ignore_index=True)

    # Canonical numeric cleanup
    for col in ("relative_error", "best_fitness", "error"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Clip relative_error to [0, ∞) — negative values arise from numerical noise
    if "relative_error" in merged.columns:
        merged["relative_error"] = merged["relative_error"].clip(lower=0.0)

    # Add BBOB category
    if "problem_id" in merged.columns:
        merged["category"] = merged["problem_id"].map(FUNCTION_TO_CATEGORY)

    return merged


# =============================================================================
# UTILITIES
# =============================================================================


def ensure_box_spines(ax):
    ax.set_facecolor("white")
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")


def experiment_palette(experiments: list[str]) -> dict:
    """Map experiment labels to distinct colours."""
    colors = sns.color_palette("tab10", n_colors=len(experiments))
    return dict(zip(experiments, colors))


def save_fig(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.pdf/png")


def _distplot(ax, *, data, x, y, hue, hue_order, palette, **kwargs):
    """
    Draw either a violin plot or a box plot depending on the ``_USE_VIOLIN`` flag.

    Keyword arguments common to both are forwarded directly; violin-specific
    and box-specific defaults are applied automatically so callers don't need
    to branch.
    """
    if _USE_VIOLIN:
        # Strip box-only kwargs and add violin defaults
        kwargs.pop("fliersize", None)
        kwargs.setdefault("inner", "box")
        kwargs.setdefault("linewidth", 0.7)
        sns.violinplot(
            ax=ax, data=data, x=x, y=y,
            hue=hue, hue_order=hue_order, palette=palette,
            **kwargs,
        )
    else:
        kwargs.setdefault("linewidth", 0.7)
        kwargs.setdefault("fliersize", 1.5)
        sns.boxplot(
            ax=ax, data=data, x=x, y=y,
            hue=hue, hue_order=hue_order, palette=palette,
            linecolor="auto",
            **kwargs,
        )


def _log_perf(series: pd.Series) -> pd.Series:
    """log10(perf + LOG_EPS) — works for both relative_error and perf_norm."""
    return np.log10(series.clip(lower=0) + LOG_EPS)


def sorted_experiments(df: pd.DataFrame) -> list[str]:
    """Stable experiment order (alphabetical by key, then label)."""
    mapping = df[["experiment_key", "experiment"]].drop_duplicates()
    mapping = mapping.sort_values("experiment_key")
    return mapping["experiment"].tolist()


def _build_ioh_optimum_lookup(df: pd.DataFrame) -> dict:
    """
    Return a dict mapping (problem_id, instance, dimension) → optimum.y
    by instantiating each unique BBOB problem via the IOH library.

    Uses the same ``ioh.get_problem(fid, instance, dimension)`` call as
    ``benchmark_experiment.py``, so the optimal values are guaranteed to match
    the fitness values recorded during the experiments.
    """
    try:
        from ioh import get_problem
    except ImportError:
        return {}

    combos = (
        df[["problem_id", "instance", "dimension"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    lookup = {}
    for row in combos:
        fid, iid, dim = int(row.problem_id), int(row.instance), int(row.dimension)
        if (fid, iid, dim) not in lookup:
            p = get_problem(fid=fid, instance=iid, dimension=dim)
            lookup[(fid, iid, dim)] = p.optimum.y
    return lookup


def add_performance_metric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a unified ``perf`` column to *df* and update the module-level
    ``_PERF_*`` globals so all plot functions use consistent labels.

    Priority
    --------
    1. ``relative_error`` — used as-is when at least 10 % of values are non-null.
    2. IOH lookup        — queries ``ioh.get_problem`` to obtain the true BBOB
       optimum ``f*`` for every ``(problem_id, instance, dimension)`` triple,
       then sets ``perf = best_fitness − f*`` (absolute gap; ≥ 0, lower = better).
       This is the standard BBOB performance metric.
    3. Relative fallback — per-group min-max normalisation of ``best_fitness``
       when IOH is not installed (0 = best across experiments, 1 = worst).
    """
    global _PERF_COL, _PERF_LABEL, _PERF_LOG_LABEL, _PERF_LOWER_IS_BETTER

    # --- Option 1: pre-computed relative_error in CSV ---
    if "relative_error" in df.columns:
        valid_frac = df["relative_error"].notna().mean()
        if valid_frac >= 0.10:
            df["perf"] = df["relative_error"].clip(lower=0.0)
            _PERF_COL = "perf"
            _PERF_LABEL = r"$\delta_{\mathrm{rel}}$"
            _PERF_LOG_LABEL = r"$\log(\delta_{\mathrm{rel}})$"
            _PERF_LOWER_IS_BETTER = True
            print(f"  Performance metric: relative_error ({valid_frac:.0%} non-null)")
            return df

    if "best_fitness" not in df.columns:
        raise ValueError(
            "No usable performance column found (need relative_error or best_fitness)."
        )

    # --- Option 2: IOH lookup → absolute gap f(x) − f* ---
    print("  Querying IOH for BBOB optima (f* per problem/instance/dimension)…")
    lookup = _build_ioh_optimum_lookup(df)

    if lookup:
        key_cols = ["problem_id", "instance", "dimension"]
        key_cols = [c for c in key_cols if c in df.columns]
        df["optimum_y"] = df.apply(
            lambda r: lookup.get(
                (int(r["problem_id"]), int(r["instance"]), int(r["dimension"])),
                float("nan"),
            ),
            axis=1,
        )
        df["perf"] = (df["best_fitness"] - df["optimum_y"]).clip(lower=0.0)
        n_solved = (df["perf"] < 1e-6).mean()
        _PERF_COL = "perf"
        _PERF_LABEL = r"$|f(\pmb{x}_\text{best}) - f_*|$"
        _PERF_LOG_LABEL = r"$\log(|f(\pmb{x}_\text{best}) - f_*|)$"
        _PERF_LOWER_IS_BETTER = True
        print(
            f"  Performance metric: absolute error f(x)−f* via IOH  "
            f"({n_solved:.1%} of runs solved to within 1e-6)"
        )
        return df

    # --- Option 3: cross-experiment relative fallback (no IOH) ---
    print("  Warning: IOH not available — falling back to relative normalisation.")
    group_cols = [c for c in ["problem_id", "instance", "dimension"] if c in df.columns]
    g = df.groupby(group_cols)["best_fitness"]
    g_min = g.transform("min")
    g_range = (g.transform("max") - g_min).clip(lower=LOG_EPS)
    df["perf"] = (df["best_fitness"] - g_min) / g_range
    _PERF_COL = "perf"
    _PERF_LABEL = r"Norm. excess $\tilde{f}$"
    _PERF_LOG_LABEL = r"$\log(\tilde{f})$"
    _PERF_LOWER_IS_BETTER = True
    print("  Performance metric: normalised excess (0=best, 1=worst across experiments)")
    return df


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================


def plot_performance_by_dimension(df: pd.DataFrame, output_dir: Path):
    """
    Box plots of performance metric grouped by dimension, one hue per experiment.

    Caption: Distribution of the performance metric (relative error when available,
    otherwise normalised excess fitness) across all BBOB runs grouped by problem
    dimension. Lower is better. Box shows median and IQR.
    """
    if "perf" not in df.columns:
        print("  Skipping performance_by_dimension: perf column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    plot_df = df.copy()
    plot_df["__lp__"] = _log_perf(plot_df["perf"])

    _distplot(
        ax, data=plot_df, x="dimension", y="__lp__",
        hue="experiment", hue_order=exps, palette=palette,
    )
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.legend(title=None, bbox_to_anchor=(0.0, 1.0), loc="upper left",
              ncol=min(len(exps), 3), frameon=False, columnspacing=0.5, handlelength=0.5)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_performance_by_dimension")


def plot_performance_by_category(df: pd.DataFrame, output_dir: Path):
    """
    Box plots of performance metric per BBOB category, hue = experiment.

    Caption: Performance metric per BBOB problem category. Categories encode
    different structural difficulty: separable, ill-conditioned, multi-modal,
    and weak global structure.
    """
    if "perf" not in df.columns or "category" not in df.columns:
        print("  Skipping performance_by_category: column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)
    cat_order = [c for c in BBOB_CATEGORIES if c in df["category"].unique()]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    plot_df = df.copy()
    plot_df["__lp__"] = _log_perf(plot_df["perf"])

    _distplot(
        ax, data=plot_df, x="category", y="__lp__",
        hue="experiment", hue_order=exps, order=cat_order,
        palette=palette,
    )
    ax.set_xlabel("BBOB Category")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.set_xticklabels(ax.get_xticklabels(), ha="center")
    ax.legend(title=None, bbox_to_anchor=(1.0, 1.00), loc="upper right",
              ncol=min(len(exps), 3), frameon=False, columnspacing=0.5, handlelength=0.5)
    ensure_box_spines(ax)
    ax.set_ylim(top=plot_df["__lp__"].max() * 1.4)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_performance_by_category")


def plot_dimension_scaling(df: pd.DataFrame, output_dir: Path):
    """
    Line plot: median performance vs dimension per experiment.

    Caption: Median performance metric as a function of problem dimension for
    each NEVO configuration. Shaded band shows the 25th–75th percentile range.
    Lower values indicate better optimisation.
    """
    if "perf" not in df.columns:
        print("  Skipping dimension_scaling: perf column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)
    dims = sorted(df["dimension"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for exp in exps:
        sub = df[df["experiment"] == exp]
        stats = (
            sub.groupby("dimension")["perf"]
            .agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
        ax.semilogy(
            stats["dimension"], stats["median"] + LOG_EPS,
            marker="o", markersize=4, linewidth=1.2,
            label=exp, color=palette[exp],
        )
        ax.fill_between(
            stats["dimension"],
            stats["q25"] + LOG_EPS, stats["q75"] + LOG_EPS,
            alpha=0.12, color=palette[exp],
        )

    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel(r"Median(" + _PERF_LABEL + r")")
    ax.set_xticks(dims)
    ax.legend(bbox_to_anchor=(1.0, 0.0), loc="lower right",
              ncol=1, frameon=False, fontsize=7)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_dimension_scaling")


def plot_performance_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × problem_id (columns) — median log performance.

    Caption: Median log performance per BBOB function and experiment, averaged
    over all dimensions and instances. Darker colour = worse performance.
    """
    if "perf" not in df.columns:
        print("  Skipping performance_heatmap: perf column missing.")
        return

    exps = sorted_experiments(df)
    func_ids = sorted(df["problem_id"].unique())

    records = [
        {"experiment": exp, "problem_id": fid,
         "log_perf": np.log10(
             df[(df["experiment"] == exp) & (df["problem_id"] == fid)]["perf"].median()
             + LOG_EPS
         )}
        for exp in exps for fid in func_ids
    ]
    pivot = (
        pd.DataFrame(records)
        .pivot(index="experiment", columns="problem_id", values="log_perf")
        .reindex(exps)
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 0.55 * len(exps) + 0.6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".1f", annot_kws={"size": 6},
        linewidths=0.3,
        cbar_ax=cax,
        cbar_kws={"label": r"median$(\log(|f(\pmb{x}_\text{best}) - f_*|)$"},
        square=True,
    )
    ax.set_xlabel("Function ID")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_performance_heatmap")


def plot_summary_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × dimension (columns) — median log performance.

    Caption: Summary of median log performance by experiment and problem dimension.
    """
    if "perf" not in df.columns:
        print("  Skipping summary_heatmap: perf column missing.")
        return

    exps = sorted_experiments(df)
    dims = sorted(df["dimension"].unique())

    records = [
        {"experiment": exp, "dimension": dim,
         "log_perf": np.log10(
             df[(df["experiment"] == exp) & (df["dimension"] == dim)]["perf"].median()
             + LOG_EPS
         )}
        for exp in exps for dim in dims
    ]
    pivot = (
        pd.DataFrame(records)
        .pivot(index="experiment", columns="dimension", values="log_perf")
        .reindex(exps)
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 0.55 * len(exps) + 0.6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.4,
        cbar_ax=cax,
        cbar_kws={"label": r"median$(\log(|f(\pmb{x}_\text{best}) - f_*|)$"},
        square=True,
    )
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_summary_heatmap")


def plot_success_rate(df: pd.DataFrame, output_dir: Path,
                      threshold: float = SUCCESS_THRESHOLD):
    """
    Bar chart: fraction of runs with perf ≤ threshold per experiment × dimension.

    Meaningful only when ``perf`` is relative_error; when it is a normalised
    excess it indicates runs that reached a near-best solution.

    Caption: Proportion of runs with performance metric ≤ {threshold} by
    dimension and experiment. Higher bars indicate more reliable convergence.
    """
    if "perf" not in df.columns:
        print("  Skipping success_rate: perf column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)

    records = [
        {"experiment": exp, "dimension": dim,
         "success_rate": (df[(df["experiment"] == exp) & (df["dimension"] == dim)]["perf"]
                          <= threshold).mean()}
        for exp in exps
        for dim in sorted(df["dimension"].unique())
    ]
    sr_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    sns.barplot(
        data=sr_df, x="dimension", y="success_rate",
        hue="experiment", hue_order=exps, palette=palette,
        ax=ax, edgecolor="black", linewidth=0.5,
    )
    thresh_str = f"{int(np.log10(threshold))}"
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel(rf"Success rate (perf $\leq 10^{{{thresh_str}}}$)")
    ax.set_ylim(0, 1.0)
    ax.legend(title=None, bbox_to_anchor=(1.0, 1.0), loc="upper right",
              ncol=min(len(exps), 3), frameon=False, columnspacing=0.5, handlelength=0.5)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_success_rate")


def plot_wall_time(df: pd.DataFrame, output_dir: Path):
    """
    Box plots of wall time per dimension and experiment.

    Caption: Wall-clock time (seconds) per optimisation run grouped by dimension.
    Differences reflect algorithm overhead and problem dimensionality.
    """
    if "wall_time" not in df.columns:
        print("  Skipping wall_time: column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    _distplot(
        ax, data=df, x="dimension", y="wall_time",
        hue="experiment", hue_order=exps, palette=palette,
    )
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel(r"Wall Time (s)")
    ax.set_yscale("log")
    ax.legend(title=None, bbox_to_anchor=(0.0, 0.0), loc="lower left",
              ncol=min(len(exps), 3), frameon=False, columnspacing=0.5, handlelength=0.5)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_wall_time")


def plot_evals_per_second(df: pd.DataFrame, output_dir: Path):
    """
    Box plots of evaluation throughput per dimension and experiment.

    Caption: Objective function evaluation throughput (evaluations / second)
    by dimension and configuration. Higher is better. Exposes computational
    overhead of neuromorphic vs traditional operator modes.
    """
    if "evals_per_second" not in df.columns:
        print("  Skipping evals_per_second: column missing.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    _distplot(
        ax, data=df, x="dimension", y="evals_per_second",
        hue="experiment", hue_order=exps, palette=palette,
    )
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel(r"Evals / second")
    ax.set_yscale("log")
    ax.legend(title=None, bbox_to_anchor=(0.0, 1.00), loc="upper left",
              ncol=min(len(exps), 3), frameon=False, columnspacing=0.5, handlelength=0.5)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_evals_per_second")


def plot_timing_summary(df: pd.DataFrame, output_dir: Path):
    """
    Two-panel figure: median wall_time and median evals/s per experiment.

    Caption: Per-experiment timing summary averaged over all problems and
    dimensions. Left: median wall-clock time per run. Right: median evaluation
    throughput.
    """
    has_time = "wall_time" in df.columns
    has_eps = "evals_per_second" in df.columns
    if not has_time and not has_eps:
        print("  Skipping timing_summary: no timing columns.")
        return

    exps = sorted_experiments(df)
    palette = experiment_palette(exps)
    ncols = int(has_time) + int(has_eps)

    fig, axes = plt.subplots(1, ncols, figsize=(ncols * SINGLE_COL_WIDTH, 2.2))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_time:
        ax = axes[idx]; idx += 1
        summary = df.groupby("experiment")["wall_time"].median().reindex(exps)
        colors = [palette[e] for e in exps]
        ax.barh(exps, summary.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(r"Median Wall Time (s)")
        ax.invert_yaxis()
        ensure_box_spines(ax)

    if has_eps:
        ax = axes[idx]
        summary = df.groupby("experiment")["evals_per_second"].median().reindex(exps)
        colors = [palette[e] for e in exps]
        ax.barh(exps, summary.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(r"Median Evals / second")
        ax.invert_yaxis()
        ensure_box_spines(ax)

    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_timing_summary")


def plot_category_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × BBOB category (columns) — median log performance.

    Caption: Median log performance per BBOB category and experiment, averaged
    across all dimensions and instances. Reveals which configurations excel on
    specific problem types.
    """
    if "perf" not in df.columns or "category" not in df.columns:
        print("  Skipping category_heatmap: column missing.")
        return

    exps = sorted_experiments(df)
    cat_order = [c for c in BBOB_CATEGORIES if c in df["category"].unique()]

    records = [
        {"experiment": exp, "category": cat,
         "log_perf": np.log10(
             df[(df["experiment"] == exp) & (df["category"] == cat)]["perf"].median()
             + LOG_EPS
         )}
        for exp in exps for cat in cat_order
    ]
    pivot = (
        pd.DataFrame(records)
        .pivot(index="experiment", columns="category", values="log_perf")
        .reindex(exps)[cat_order]
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 0.55 * len(exps) + 0.6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.4,
        square=True,
        cbar_ax=cax,
        cbar_kws={"label": r"median$(\log(|f(\pmb{x}_\text{best}) - f_*|)$"},
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_category_heatmap")


def plot_operator_mode_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Stacked bar of mean operator usage fractions per experiment.

    Only experiments that carry ``op_count_*`` columns are included.

    Caption: Mean operator usage fraction per experiment. Bars sum to 1. Only
    configurations with operator-count data are shown.
    """
    op_cols = [c for c in df.columns if c.startswith("op_count_")]
    if not op_cols:
        print("  Skipping operator_breakdown: no op_count columns.")
        return

    exps = sorted_experiments(df)
    valid_exps = [
        e for e in exps
        if df[df["experiment"] == e][op_cols].notna().any().any()
    ]
    if not valid_exps:
        print("  Skipping operator_breakdown: all op_count columns are null.")
        return

    records = []
    for exp in valid_exps:
        sub = df[df["experiment"] == exp]
        for col in op_cols:
            if sub[col].notna().any():
                op = col.replace("op_count_", "")
                records.append({"experiment": exp, "operator": op,
                                 "mean_count": sub[col].mean()})

    op_df = pd.DataFrame(records)
    op_pivot = (
        op_df.pivot(index="experiment", columns="operator", values="mean_count")
        .fillna(0)
        .reindex(valid_exps)
    )
    # Normalise to fractions
    op_pivot = op_pivot.div(op_pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(
        figsize=(max(SINGLE_COL_WIDTH, 0.8 * len(valid_exps)), 2.8)
    )
    colors = sns.color_palette("tab20", n_colors=len(op_pivot.columns))
    op_pivot.plot(
        kind="bar", stacked=True, ax=ax,
        color=colors, edgecolor="black", linewidth=0.3, width=0.6,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Operator usage fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Operator", bbox_to_anchor=(1.02, 1), loc="upper left",
              ncol=1, fontsize=6, frameon=False)
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_operator_breakdown")


# =============================================================================
# SUMMARY TABLE
# =============================================================================


def print_summary_table(df: pd.DataFrame):
    """Print a quick summary table to stdout."""
    exps = sorted_experiments(df)
    perf_col = "perf" if "perf" in df.columns else None
    cols = ([perf_col] if perf_col else []) + [
        "best_fitness", "wall_time", "evals_per_second", "total_evaluations"
    ]
    avail = [c for c in cols if c in df.columns]

    rows = []
    for exp in exps:
        sub = df[df["experiment"] == exp]
        row = {"Experiment": exp}
        for c in avail:
            row[c] = sub[c].median()
        row["n_runs"] = len(sub)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Experiment")
    print("\nMedian summary per experiment:")
    print(summary.to_string(float_format="{:.4g}".format))
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare NEVO benchmark results across multiple experiment folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmark_results_bbob",
        help="Parent directory that contains one sub-folder per experiment "
             "(default: benchmark_results_bbob)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures_comparison",
        help="Directory to write figures into (default: figures_comparison)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of sub-folder names (or name substrings) to include. "
             "Omit to include all discovered folders.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=SUCCESS_THRESHOLD,
        help=f"Relative error threshold for the success-rate plot (default: {SUCCESS_THRESHOLD})",
    )
    parser.add_argument(
        "--no-operator-breakdown",
        action="store_true",
        help="Skip the operator usage breakdown plot (useful when all experiments "
             "use different operator sets)",
    )
    parser.add_argument(
        "--violin",
        action="store_true",
        help="Use violin plots instead of box plots for distribution figures.",
    )
    args = parser.parse_args()

    global _USE_VIOLIN
    _USE_VIOLIN = args.violin

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("NEVO Benchmark Comparison")
    print(f"{'=' * 60}")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Plot style: {'violin' if _USE_VIOLIN else 'box'}")
    print(f"{'=' * 60}")

    selected = (
        [s.strip() for s in args.experiments.split(",")]
        if args.experiments
        else None
    )

    df = load_all_experiments(Path(args.input_dir), selected=selected)

    print(f"\nMerged dataset: {len(df)} rows")
    print(f"  Experiments : {sorted(df['experiment'].unique())}")
    print(f"  Dimensions  : {sorted(df['dimension'].unique())}")
    print(f"  Functions   : f{sorted(df['problem_id'].unique())[0]}–"
          f"f{sorted(df['problem_id'].unique())[-1]}")

    print("\nSelecting performance metric...")
    df = add_performance_metric(df)

    print_summary_table(df)

    print("Generating figures...")

    plot_performance_by_dimension(df, output_dir)
    plot_performance_by_category(df, output_dir)
    plot_dimension_scaling(df, output_dir)
    plot_performance_heatmap(df, output_dir)
    plot_summary_heatmap(df, output_dir)
    plot_category_heatmap(df, output_dir)
    plot_success_rate(df, output_dir, threshold=args.success_threshold)
    plot_wall_time(df, output_dir)
    plot_evals_per_second(df, output_dir)
    plot_timing_summary(df, output_dir)

    if not args.no_operator_breakdown:
        plot_operator_mode_breakdown(df, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
