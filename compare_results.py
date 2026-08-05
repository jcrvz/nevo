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

import matplotlib.patches as mpatches
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
    "separ": [1, 2, 3, 4, 5],
    "lcond": [6, 7, 8, 9],
    "hcond": [10, 11, 12, 13, 14],
    "multi": [15, 16, 17, 18, 19],
    "multi2": [20, 21, 22, 23, 24],
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


def _darken_color(rgba, factor=0.65):
    """Return a darkened RGBA by reducing lightness in HLS space."""
    import colorsys
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    rd, gd, bd = colorsys.hls_to_rgb(h, max(0.0, l * factor), s)
    return (rd, gd, bd)


def colorize_boxplot(ax):
    """
    Color box edges, whiskers, caps, and fliers darker than their fill.
    Median lines are left untouched so they remain visible.
    Legend handle edges are updated to match.
    """
    box_info = []
    for p in ax.patches:
        if isinstance(p, mpatches.PathPatch):
            verts = p.get_path().vertices
            xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
            box_width = xmax - xmin
            fc = p.get_facecolor()
            dark = _darken_color(fc)
            p.set_edgecolor(dark)
            box_info.append((xmin, xmax, box_width, dark))

    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 0:
            continue
        x_mean = float(np.mean(xdata))
        x_span = float(max(xdata) - min(xdata))
        has_marker = line.get_marker() not in ("", "None", None)

        for xmin, xmax, box_width, dark in box_info:
            if xmin - 0.01 <= x_mean <= xmax + 0.01:
                # Median: no marker, x_span ≈ box_width → skip, keep visible
                is_median = (not has_marker) and (box_width > 0) and (
                    abs(x_span - box_width) / box_width < 0.05
                )
                if not is_median:
                    line.set_color(dark)
                    line.set_markerfacecolor(dark)
                    line.set_markeredgecolor(dark)
                break

    # Keep legend handle edges consistent with the darkened box edges
    legend = ax.get_legend()
    if legend:
        for handle in legend.legend_handles:
            if isinstance(handle, mpatches.Patch):
                handle.set_edgecolor(_darken_color(handle.get_facecolor()))


def experiment_palette(experiments: list[str]) -> dict:
    """Map experiment labels to distinct colours."""
    colors = sns.color_palette("tab10", n_colors=len(experiments))
    return dict(zip(experiments, colors))


def save_fig(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.pdf/png")


# Marker cycle for scatter plots — one shape per experiment
_SCATTER_MARKERS = ["o", "s", "^", "D", "v", "P", "h", "*", "X", "<"]


def _scatter_perf_vs_walltime(
    ax,
    agg: pd.DataFrame,
    *,
    exps: list,
    group_col: str,
    group_palette: dict,
    group_title: str,
    x_col: str = "wall_med",
):
    """
    Core scatter logic shared by the perf-vs-walltime plots.
    *agg* must contain columns: experiment, log_perf, <group_col>, and x_col.
    Markers encode experiment; colours encode group_col.
    Returns (marker_handles, colour_handles) for legend construction.
    
    Parameters
    ----------
    x_col : str
        Column name to use for x-axis (default: "wall_med" for wall time,
        can also be "time_per_eval" for efficiency).
    """
    exp_markers = {exp: _SCATTER_MARKERS[i % len(_SCATTER_MARKERS)] for i, exp in enumerate(exps)}
    groups = [g for g in group_palette if g in agg[group_col].unique()]

    for exp in exps:
        for grp in groups:
            sub = agg[(agg["experiment"] == exp) & (agg[group_col] == grp)]
            if sub.empty:
                continue
            fc = group_palette[grp]
            ec = _darken_color(fc)
            ax.scatter(
                sub[x_col], sub["log_perf"],
                marker=exp_markers[exp],
                color=fc,
                edgecolors=ec,
                s=20, alpha=0.78, linewidths=0.5,
                zorder=3,
            )

    marker_handles = [
        plt.Line2D(
            [0], [0], marker=exp_markers[exp], color="none",
            markerfacecolor="#555555", markeredgecolor="#222222",
            markeredgewidth=0.5, markersize=5, linestyle="None", label=exp,
        )
        for exp in exps
    ]
    colour_handles = [
        mpatches.Patch(
            facecolor=group_palette[g],
            edgecolor=_darken_color(group_palette[g]),
            linewidth=0.8, label=str(g),
        )
        for g in groups
    ]
    return marker_handles, colour_handles



def plot_perf_vs_time_per_eval_by_dimension(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plot: median performance (error) vs median time per evaluation.
    Each point = median over all instances of one (experiment, dimension, problem_id) triple.
    Marker shape encodes experiment; fill colour encodes problem dimension.

    Time per evaluation = wall_time / total_evaluations (seconds per evaluation).
    Lower values indicate more efficient algorithms.

    Caption: Performance–efficiency trade-off. Each point summarises one (configuration,
    dimension, function) triple (median over instances). Lower-left is ideal
    (better performance, faster per-evaluation speed).
    """
    if "perf" not in df.columns or "wall_time" not in df.columns or "total_evaluations" not in df.columns:
        print("  Skipping perf_vs_time_per_eval_by_dimension: columns missing.")
        return

    exps = sorted_experiments(df)
    dims = sorted(df["dimension"].unique())
    dim_palette = dict(zip(dims, sns.color_palette("Set2", n_colors=len(dims))))

    agg = (
        df.groupby(["experiment", "dimension", "problem_id"])
        .agg(
            perf_med=("perf", "median"),
            wall_med=("wall_time", "median"),
            evals_med=("total_evaluations", "median"),
        )
        .reset_index()
    )
    agg["time_per_eval"] = agg["wall_med"] / agg["evals_med"]
    agg["log_perf"] = _log_perf(agg["perf_med"])
    agg["__grp__"] = agg["dimension"]
    dim_palette_generic = {d: dim_palette[d] for d in dims}

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    marker_handles, colour_handles = _scatter_perf_vs_walltime(
        ax, agg, exps=exps,
        group_col="__grp__", group_palette=dim_palette_generic,
        group_title=r"$D$",
        x_col="time_per_eval",
    )

    leg1 = ax.legend(
        handles=marker_handles, title="Implementation",
        bbox_to_anchor=(1.0, 1.0), loc="upper left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=1,
        alignment="left",
        handlelength=0.8, borderpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=colour_handles, title=r"Dimensionality",
        bbox_to_anchor=(1.0, 0.0), loc="lower left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=2,
        handlelength=0.5, borderpad=0.1,
    )

    ax.set_xlabel(r"Median Time per Evaluation (s)")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.set_xscale("log")
    ensure_box_spines(ax)
    fig.subplots_adjust(left=0.09, right=0.67, top=0.96, bottom=0.16)
    save_fig(fig, output_dir, "comparison_perf_vs_time_per_eval_by_dimension")


def plot_perf_vs_time_per_eval_by_category(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plot: median performance (error) vs median time per evaluation.
    Each point = median over all instances of one (experiment, dimension, problem_id) triple.
    Marker shape encodes experiment; fill colour encodes BBOB function category.

    Time per evaluation = wall_time / total_evaluations (seconds per evaluation).
    Lower values indicate more efficient algorithms.

    Caption: Performance–efficiency trade-off coloured by BBOB category. Reveals which
    problem structures drive the perf/efficiency relationship for each configuration.
    """
    if "perf" not in df.columns or "wall_time" not in df.columns or "total_evaluations" not in df.columns or "category" not in df.columns:
        print("  Skipping perf_vs_time_per_eval_by_category: columns missing.")
        return

    exps = sorted_experiments(df)
    cats = [c for c in BBOB_CATEGORIES if c in df["category"].unique()]
    cat_palette = dict(zip(cats, sns.color_palette("Set2", n_colors=len(cats))))

    agg = (
        df.groupby(["experiment", "dimension", "problem_id", "category"])
        .agg(
            perf_med=("perf", "median"),
            wall_med=("wall_time", "median"),
            evals_med=("total_evaluations", "median"),
        )
        .reset_index()
    )
    agg["time_per_eval"] = agg["wall_med"] / agg["evals_med"]
    agg["log_perf"] = _log_perf(agg["perf_med"])
    agg["__grp__"] = agg["category"]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    marker_handles, colour_handles = _scatter_perf_vs_walltime(
        ax, agg, exps=exps,
        group_col="__grp__", group_palette=cat_palette,
        group_title="Category",
        x_col="time_per_eval",
    )

    leg1 = ax.legend(
        handles=marker_handles, title="Implementation",
        bbox_to_anchor=(1.0, 1.0), loc="upper left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=1,
        handlelength=0.8, borderpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=colour_handles, title="Category",
        bbox_to_anchor=(1.0, 0.0), loc="lower left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=2,
        handlelength=0.5, borderpad=0.1, columnspacing=1.0,
    )

    ax.set_xlabel(r"Median Time per Evaluation (s)")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.set_xscale("log")
    ensure_box_spines(ax)
    fig.subplots_adjust(left=0.09, right=0.67, top=0.96, bottom=0.16)
    save_fig(fig, output_dir, "comparison_perf_vs_time_per_eval_by_category")


def plot_perf_vs_walltime_by_dimension(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plot: median performance (error) vs median wall time.
    Each point = median over all instances of one (experiment, dimension, problem_id) triple.
    Marker shape encodes experiment; fill colour encodes problem dimension.

    Caption: Performance–cost trade-off. Each point summarises one (configuration,
    dimension, function) triple (median over instances). Lower-left is ideal.
    """
    if "perf" not in df.columns or "wall_time" not in df.columns:
        print("  Skipping perf_vs_walltime_by_dimension: columns missing.")
        return

    exps = sorted_experiments(df)
    dims = sorted(df["dimension"].unique())
    dim_palette = dict(zip(dims, sns.color_palette("Set2", n_colors=len(dims))))

    agg = (
        df.groupby(["experiment", "dimension", "problem_id"])
        .agg(perf_med=("perf", "median"), wall_med=("wall_time", "median"))
        .reset_index()
    )
    agg["log_perf"] = _log_perf(agg["perf_med"])
    agg["__grp__"] = agg["dimension"]
    dim_palette_generic = {d: dim_palette[d] for d in dims}

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    marker_handles, colour_handles = _scatter_perf_vs_walltime(
        ax, agg, exps=exps,
        group_col="__grp__", group_palette=dim_palette_generic,
        group_title=r"$D$",
    )

    leg1 = ax.legend(
        handles=marker_handles, title="Implementation",
        bbox_to_anchor=(1.0, 1.0), loc="upper left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=1,
        alignment="left",
        handlelength=0.8, borderpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=colour_handles, title=r"Dimensionality",
        bbox_to_anchor=(1.0, 0.0), loc="lower left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=2,
        handlelength=0.5, borderpad=0.1,
    )

    ax.set_xlabel(r"Median Wall Time (s)")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.set_xscale("log")
    ensure_box_spines(ax)
    fig.subplots_adjust(left=0.09, right=0.67, top=0.96, bottom=0.16)
    save_fig(fig, output_dir, "comparison_perf_vs_walltime_by_dimension")


def plot_perf_vs_walltime_by_category(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plot: median performance (error) vs median wall time.
    Each point = median over all instances of one (experiment, dimension, problem_id) triple.
    Marker shape encodes experiment; fill colour encodes BBOB function category.

    Caption: Performance–cost trade-off coloured by BBOB category. Reveals which
    problem structures drive the perf/wall-time relationship for each configuration.
    """
    if "perf" not in df.columns or "wall_time" not in df.columns or "category" not in df.columns:
        print("  Skipping perf_vs_walltime_by_category: columns missing.")
        return

    exps = sorted_experiments(df)
    cats = [c for c in BBOB_CATEGORIES if c in df["category"].unique()]
    cat_palette = dict(zip(cats, sns.color_palette("Set2", n_colors=len(cats))))

    agg = (
        df.groupby(["experiment", "dimension", "problem_id", "category"])
        .agg(perf_med=("perf", "median"), wall_med=("wall_time", "median"))
        .reset_index()
    )
    agg["log_perf"] = _log_perf(agg["perf_med"])
    agg["__grp__"] = agg["category"]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    marker_handles, colour_handles = _scatter_perf_vs_walltime(
        ax, agg, exps=exps,
        group_col="__grp__", group_palette=cat_palette,
        group_title="Category",
    )

    leg1 = ax.legend(
        handles=marker_handles, title="Implementation",
        bbox_to_anchor=(1.0, 1.0), loc="upper left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=1,
        handlelength=0.8, borderpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=colour_handles, title="Category",
        bbox_to_anchor=(1.0, 0.0), loc="lower left",
        alignment="left",
        frameon=False, fontsize=7, title_fontsize=7, ncol=2,
        handlelength=0.5, borderpad=0.1, columnspacing=1.0,
    )

    ax.set_xlabel(r"Median Wall Time (s)")
    ax.set_ylabel(_PERF_LOG_LABEL)
    ax.set_xscale("log")
    ensure_box_spines(ax)
    fig.subplots_adjust(left=0.09, right=0.67, top=0.96, bottom=0.16)
    save_fig(fig, output_dir, "comparison_perf_vs_walltime_by_category")


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
        df["optimal_fitness"] = df.apply(
            lambda r: lookup.get(
                (int(r["problem_id"]), int(r["instance"]), int(r["dimension"])),
                float("nan"),
            ),
            axis=1,
        )
        df["perf"] = (df["best_fitness"] - df["optimal_fitness"]).clip(lower=0.0)
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

    colorize_boxplot(ax)

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

    colorize_boxplot(ax)

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

    # Compute figsize: cell size ≈ 0.35 inches (square), plus margins for labels/colorbar
    cell_size_in = 0.35
    n_rows, n_cols = len(pivot), len(pivot.columns)
    fig_w = n_cols * cell_size_in + 1.2  # Add space for labels and colorbar
    fig_h = n_rows * cell_size_in + 0.8  # Add space for title and labels
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
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

    colorize_boxplot(ax)

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

    colorize_boxplot(ax)

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

    # Compute figsize: cell size ≈ 0.35 inches (square), plus margins for labels/colorbar
    cell_size_in = 0.35
    n_rows, n_cols = len(pivot), len(pivot.columns)
    fig_w = n_cols * cell_size_in + 1.2  # Add space for labels and colorbar
    fig_h = n_rows * cell_size_in + 0.8  # Add space for title and labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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


def _operator_entropy(probs: np.ndarray) -> float:
    """Shannon entropy (bits) of a probability distribution. Zeros are ignored."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0


def _entropy_pivot(
    df: pd.DataFrame,
    exps: list[str],
    group_col: str,
    group_vals,
    op_cols: list[str],
) -> pd.DataFrame:
    """
    Build a (experiment × group_val) pivot of operator Shannon entropy (bits).

    For each cell the operator counts are averaged over all rows in the subset,
    then normalised to a probability distribution before computing entropy.
    """
    records = []
    for exp in exps:
        for gval in group_vals:
            sub = df[(df["experiment"] == exp) & (df[group_col] == gval)]
            if sub.empty or not any(sub[c].notna().any() for c in op_cols):
                records.append({"experiment": exp, group_col: gval, "entropy": np.nan})
                continue
            means = np.array([sub[c].mean() for c in op_cols], dtype=float)
            means = np.nan_to_num(means, nan=0.0)
            total = means.sum()
            if total == 0:
                records.append({"experiment": exp, group_col: gval, "entropy": 0.0})
                continue
            records.append({
                "experiment": exp,
                group_col: gval,
                "entropy": _operator_entropy(means / total),
            })
    return (
        pd.DataFrame(records)
        .pivot(index="experiment", columns=group_col, values="entropy")
        .reindex(exps)
    )


def plot_operator_mode_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Two-column figure: left panel shows the stacked operator usage fractions
    per experiment; right panel shows Shannon entropy of operator usage.
    Both panels share the same y-axis for experiment names. The legend from
    the left panel covers operator colors.

    Only experiments that carry ``op_count_*`` columns are included.

    Caption: (Left) Mean operator usage fraction per experiment. Bars sum to 1.
    (Right) Shannon entropy of the operator usage distribution per experiment —
    higher values indicate more diverse operator selection.
    Only configurations with operator-count data are shown.
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

    # Shannon entropy per experiment
    entropies = [
        _operator_entropy(op_pivot.loc[exp].values) for exp in valid_exps
    ]

    y = np.arange(len(valid_exps))
    colors = sns.color_palette("tab20", n_colors=len(op_pivot.columns))
    fig_w = max(DOUBLE_COL_WIDTH, 4.5)
    fig_h = max(2.8, 0.35 * len(valid_exps) + 0.6)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [0.6, 0.2], "wspace": 0.05},
    )

    # ── Left: stacked operator fractions (horizontal bars) ──────────────────
    bottom = np.zeros(len(valid_exps))
    for i, col in enumerate(op_pivot.columns):
        vals = op_pivot[col].values
        ax_left.barh(y, vals, left=bottom, color=colors[i],
                     edgecolor="black", linewidth=0.3, height=0.6, label=col)
        bottom += vals
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(valid_exps)
    ax_left.set_xlabel("Operator usage fraction")
    ax_left.set_xlim(0, 1.05)
    legend = ax_left.legend(
        title="Operator", bbox_to_anchor=(0.95, 0.0),
        loc="lower right", ncol=1, fontsize=6, frameon=True,
        framealpha=0.8,
    )
    legend.get_frame().set_linewidth(0.5)
    ensure_box_spines(ax_left)

    # ── Right: entropy bar chart ──────────────────────────────────────────
    exp_palette = experiment_palette(valid_exps)
    bar_colors = [exp_palette[e] for e in valid_exps]
    ax_right.barh(y, entropies, color=bar_colors, edgecolor="black", linewidth=0.4,
                  height=0.6)
    w_max = max(entropies) if entropies else 1.0
    for yi, H in zip(y, entropies):
        ax_right.text(H + 0.03 * w_max, yi, f"{H:.2f}",
                      ha="left", va="center", fontsize=6)
    ax_right.set_xlabel(r"Entropy (bits)")
    ax_right.set_xlim(0, w_max * 1.35)
    ax_right.tick_params(left=False)
    ensure_box_spines(ax_right)

    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_operator_breakdown")


def plot_entropy_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × problem_id (columns) — operator Shannon entropy.

    For each cell the operator counts are averaged over all runs in that
    (experiment, function) subset, normalised to a probability distribution,
    and the Shannon entropy (bits) is computed.  Higher entropy = more
    diverse operator usage.

    Caption: Operator selection entropy (bits) per BBOB function and
    experiment, averaged over all dimensions and instances.  Higher values
    indicate more balanced use of the operator repertoire.
    """
    op_cols = [c for c in df.columns if c.startswith("op_count_")]
    if not op_cols:
        print("  Skipping entropy_heatmap: no op_count columns.")
        return

    exps = sorted_experiments(df)
    func_ids = sorted(df["problem_id"].unique())

    pivot = _entropy_pivot(df, exps, "problem_id", func_ids, op_cols)
    h_max = np.nanmax(pivot.values) if not np.all(np.isnan(pivot.values)) else 1.0

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 0.55 * len(exps) + 0.6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlGnBu",
        annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.3,
        vmin=0, vmax=h_max,
        cbar_ax=cax,
        cbar_kws={"label": r"Entropy $H$ (bits)"},
        square=True,
    )
    ax.set_xlabel("Function ID")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_entropy_heatmap")


def plot_entropy_summary_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × dimension (columns) — operator Shannon entropy.

    Caption: Operator selection entropy (bits) by experiment and problem
    dimension.  Higher entropy indicates more balanced use of the operator
    repertoire at that dimensionality.
    """
    op_cols = [c for c in df.columns if c.startswith("op_count_")]
    if not op_cols:
        print("  Skipping entropy_summary_heatmap: no op_count columns.")
        return

    exps = sorted_experiments(df)
    dims = sorted(df["dimension"].unique())

    pivot = _entropy_pivot(df, exps, "dimension", dims, op_cols)
    h_max = np.nanmax(pivot.values) if not np.all(np.isnan(pivot.values)) else 1.0

    # Compute figsize: cell size ≈ 0.35 inches (square), plus margins for labels/colorbar
    cell_size_in = 0.35
    n_rows, n_cols = len(pivot), len(pivot.columns)
    fig_w = n_cols * cell_size_in + 1.2  # Add space for labels and colorbar
    fig_h = n_rows * cell_size_in + 0.8  # Add space for title and labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlGnBu",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4,
        vmin=0, vmax=h_max,
        cbar_ax=cax,
        cbar_kws={"label": r"Entropy $H$ (bits)"},
        square=True,
    )
    ax.set_xlabel(r"Dimension, $D$")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_entropy_summary_heatmap")


def plot_entropy_category_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: experiment (rows) × BBOB category (columns) — operator Shannon entropy.

    Caption: Operator selection entropy (bits) per BBOB category and
    experiment, averaged across all dimensions and instances.  Reveals which
    configurations diversify their operator use on specific problem types.
    """
    op_cols = [c for c in df.columns if c.startswith("op_count_")]
    if not op_cols or "category" not in df.columns:
        print("  Skipping entropy_category_heatmap: column missing.")
        return

    exps = sorted_experiments(df)
    cat_order = [c for c in BBOB_CATEGORIES if c in df["category"].unique()]

    pivot = _entropy_pivot(df, exps, "category", cat_order, op_cols)[cat_order]
    h_max = np.nanmax(pivot.values) if not np.all(np.isnan(pivot.values)) else 1.0

    # Compute figsize: cell size ≈ 0.35 inches (square), plus margins for labels/colorbar
    cell_size_in = 0.35
    n_rows, n_cols = len(pivot), len(pivot.columns)
    fig_w = n_cols * cell_size_in + 1.2  # Add space for labels and colorbar
    fig_h = n_rows * cell_size_in + 0.8  # Add space for title and labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        pivot, ax=ax, cmap="YlGnBu",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4,
        vmin=0, vmax=h_max,
        square=True,
        cbar_ax=cax,
        cbar_kws={"label": r"Entropy $H$ (bits)"},
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ensure_box_spines(ax)
    plt.tight_layout()
    save_fig(fig, output_dir, "comparison_entropy_category_heatmap")


def plot_perf_vs_time_per_eval(df: pd.DataFrame, output_dir: Path):
    """
    Faceted scatter plot with KDE contours: rows=experiments, columns=dimensions.
     
    Each subplot shows the relationship between performance (log scale, y-axis)
    and time per evaluation (log scale, x-axis), with points coloured by BBOB 
    category. KDE contours show the approximate density distribution per category.
     
    This eliminates double-encoding: experiments on rows, dimensions on columns,
    categories via colours.
     
    Caption: Performance–efficiency landscape across experiments and problem
    dimensions. Each cell shows all runs for that (experiment, dimension) pair,
    coloured by BBOB category. KDE contours reveal problem-structure clustering.
    """
    if "perf" not in df.columns or "wall_time" not in df.columns or \
       "total_evaluations" not in df.columns or "category" not in df.columns or \
       "dimension" not in df.columns:
        print("  Skipping perf_vs_time_per_eval: columns missing.")
        return

    # Prepare data
    plot_df = df.copy()
    plot_df["time_per_eval"] = plot_df["wall_time"] / plot_df["total_evaluations"]
    plot_df["log_perf"] = _log_perf(plot_df["perf"])
     
    exps = sorted_experiments(plot_df)
    dims = sorted(plot_df["dimension"].unique())
    cats = [c for c in BBOB_CATEGORIES if c in plot_df["category"].unique()]
     
    # Prepare category colour palette
    cat_palette = dict(zip(cats, sns.color_palette("Set2", n_colors=len(cats))))
     
    # Create facet grid
    n_exp = len(exps)
    n_dim = len(dims)
    fig_w = max(DOUBLE_COL_WIDTH, n_dim * 2.0)
    fig_h = max(3.0, n_exp * 2.2)
     
    fig, axes = plt.subplots(n_exp, n_dim, figsize=(fig_w, fig_h), 
                             sharex=True, sharey=True)
     
    # Ensure axes is 2D even for single row/column
    if n_exp == 1:
        axes = axes.reshape(1, -1)
    if n_dim == 1:
        axes = axes.reshape(-1, 1)
     
    # Plot each cell
    for i, exp in enumerate(exps):
        for j, dim in enumerate(dims):
            ax = axes[i, j]
             
            # Filter data for this cell
            cell_data = plot_df[(plot_df["experiment"] == exp) & 
                                (plot_df["dimension"] == dim)]
             
            if cell_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8, color="gray")
                continue
             
            # Plot scatter points, one per category
            for cat in cats:
                cat_data = cell_data[cell_data["category"] == cat]
                if not cat_data.empty:
                    ax.scatter(
                        cat_data["time_per_eval"], cat_data["log_perf"],
                        color=cat_palette[cat], label=cat,
                        s=15, alpha=0.5, edgecolors="none", zorder=2,
                    )
             
            # Add KDE contours for each category
            for cat in cats:
                cat_data = cell_data[cell_data["category"] == cat]
                if len(cat_data) >= 3:  # Need at least 3 points for KDE
                    try:
                        x = np.log10(cat_data["time_per_eval"] + 1e-10)
                        y = cat_data["log_perf"].values
                         
                        # Skip if too few unique values
                        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                            continue
                         
                        from scipy.stats import gaussian_kde
                         
                        # Create 2D KDE
                        xy = np.vstack([x, y])
                        try:
                            kde = gaussian_kde(xy, bw_method=0.25)
                             
                            # Create grid for contours
                            x_min, x_max = x.min() - 0.5, x.max() + 0.5
                            y_min, y_max = y.min() - 0.5, y.max() + 0.5
                            xx, yy = np.meshgrid(
                                np.linspace(x_min, x_max, 30),
                                np.linspace(y_min, y_max, 30),
                            )
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            zz = kde(positions).reshape(xx.shape)
                             
                            # Plot contours
                            ax.contour(
                                10**xx, yy, zz, levels=2,
                                colors=cat_palette[cat], alpha=0.4,
                                linewidths=0.8,
                            )
                        except (np.linalg.LinAlgError, ValueError):
                            pass  # Skip KDE if singular or other error
                    except ImportError:
                        pass  # scipy not available
             
            ax.set_xscale("log")
            ensure_box_spines(ax)
             
            # Column labels (dimensions)
            if i == 0:
                ax.set_title(f"$D = {dim}$", fontsize=8, fontweight="bold")
             
            # Row labels (experiments)
            if j == 0:
                ax.set_ylabel(exp, fontsize=8, fontweight="bold")
            else:
                ax.set_ylabel("")
             
            # Axis labels only on edges
            if i == n_exp - 1:
                ax.set_xlabel(r"Time/Eval (s)", fontsize=7)
            else:
                ax.set_xlabel("")
             
            if j > 0:
                ax.set_ylabel("")
     
    # Create shared legend for categories (bottom of figure)
    handles = [
        mpatches.Patch(facecolor=cat_palette[c], label=c, edgecolor="black", linewidth=0.5)
        for c in cats
    ]
    fig.legend(
        handles, cats, title="Category",
        bbox_to_anchor=(0.5, -0.02), loc="upper center",
        ncol=len(cats), frameon=False, fontsize=7, title_fontsize=7,
    )
     
    # Set shared y-label
    fig.text(0.02, 0.5, _PERF_LOG_LABEL, va="center", rotation="vertical",
             fontsize=8, fontweight="bold")
     
    plt.tight_layout(rect=[0.04, 0.05, 1, 1])
    save_fig(fig, output_dir, "comparison_perf_vs_time_per_eval_facet")




def print_summary_table(df: pd.DataFrame):
    """Print a quick summary table to stdout."""
    exps = sorted_experiments(df)
    perf_col = "perf" if "perf" in df.columns else None
    cols = ([perf_col] if perf_col else []) + [
        "wall_time", "evals_per_second", "total_evaluations"
    ]
    avail = [c for c in cols if c in df.columns]

    rows = []
    for exp in exps:
        sub = df[df["experiment"] == exp]
        row = {"Experiment": exp}
        for c in avail:
            median = sub[c].median()
            row[c + "-median"] = median
            #row[c + "-mean"] = sub[c].mean()
            #row[c + "-std"] = sub[c].std()

            quantile_25 = sub[c].quantile(0.25)
            quantile_75 = sub[c].quantile(0.75)
            iqr = quantile_75 - quantile_25
            row[c + "-iqr"] = iqr

            row[c + "-qcv"] = iqr / median if median != 0 else None

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
    plot_perf_vs_walltime_by_dimension(df, output_dir)
    plot_perf_vs_walltime_by_category(df, output_dir)
    plot_perf_vs_time_per_eval_by_dimension(df, output_dir)
    plot_perf_vs_time_per_eval_by_category(df, output_dir)
    # plot_perf_vs_time_per_eval(df, output_dir)

    if not args.no_operator_breakdown:
        plot_operator_mode_breakdown(df, output_dir)
        plot_entropy_heatmap(df, output_dir)
        plot_entropy_summary_heatmap(df, output_dir)
        plot_entropy_category_heatmap(df, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
