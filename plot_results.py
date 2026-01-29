#!/usr/bin/env python
"""
NEVO Benchmark Results Visualization
=====================================

Paper-ready figures using seaborn for analyzing COCO benchmark results.
Focuses on operator weights, success rates, and performance by dimension/category.

Designed for IEEE two-column format (10pt font).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION (IEEE Paper-ready)
# =============================================================================

# Check if LaTeX is available, fallback to mathtext if not
import shutil
USE_LATEX = shutil.which('latex') is not None

if USE_LATEX:
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times', 'Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    })
else:
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # Computer Modern
    })

# Common settings
plt.rcParams.update({
    # Font sizes for IEEE 10pt
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # Text color - ensure pure black
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',

    # Remove grid
    'axes.grid': False,

    # Box/spines - well delimited with black lines
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.facecolor': 'white',

    # Figure settings
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'black',

    # Legend
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0,
    'legend.fancybox': False,
    'legend.borderpad': 0.4,
})

# Disable seaborn grid
sns.set_style("ticks", {
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
})

# Custom color palette
OPERATOR_COLORS = {
    'LevyFlight': '#e41a1c',
    'DifferentialEvolution': '#377eb8',
    'ParticleSwarm': '#4daf4a',
    'SpiralOptimisation': '#984ea3',
    'RandomSearch': '#ff7f00',
    'LocalRandomWalk': '#ffff33',
    'GravitationalSearch': '#a65628',
    'FireflyAlgorithm': '#f781bf',
    'CentralForce': '#999999',
    'GeneticCrossover': '#66c2a5',
    'GeneticMutation': '#fc8d62',
    'SimulatedAnnealing': '#8da0cb',
    'TabuSearch': '#e78ac3',
}

# Short names for operators (for better readability in plots)
OPERATOR_SHORT_NAMES = {
    'LevyFlight': 'LF',
    'DifferentialEvolution': 'DE',
    'ParticleSwarm': 'PS',
    'SpiralOptimisation': 'SO',
    'RandomSearch': 'RS',
    'LocalRandomWalk': 'LW',
    'GravitationalSearch': 'GS',
    'FireflyAlgorithm': 'FA',
    'CentralForce': 'CF',
    'GeneticCrossover': 'GX',
    'GeneticMutation': 'GM',
    'SimulatedAnnealing': 'SA',
    'TabuSearch': 'TS',
}

# Operator order: Exploitation (simple→complex) then Exploration (simple→complex)
# Exploitation: SO, PSO, TS, SA, GM, LW
# Exploration: GSA, FA, CFO, DE, GX, LF, RS
OPERATOR_ORDER = ['SO', 'PS', 'TS', 'SA', 'GM', 'LW', 'GS', 'FA', 'CF', 'DE', 'GX', 'LF', 'RS']

# Operator type classification
OPERATOR_TYPE = {
    'SO': 'Exploitation', 'PS': 'Exploitation', 'TS': 'Exploitation',
    'SA': 'Exploitation', 'GM': 'Exploitation', 'LW': 'Exploitation',
    'GS': 'Exploration', 'FA': 'Exploration', 'CF': 'Exploration',
    'DE': 'Exploration', 'GX': 'Exploration', 'LF': 'Exploration',
    'RS': 'Exploration',
}

# BBOB function categories
BBOB_CATEGORIES = {
    r'\texttt{separ} (f1-f5)': [1, 2, 3, 4, 5],
    r'\texttt{lcond} (f6-f10)': [6, 7, 8, 9],
    r'\texttt{hcond} (f10-14)': [10, 11, 12, 13, 14],
    r'\texttt{multi} (f15-19)': [15, 16, 17, 18, 19],
    r'\texttt{mult2} (f20-f24)': [20, 21, 22, 23, 24],
}

# Reverse mapping
FUNCTION_TO_CATEGORY = {}
for cat, funcs in BBOB_CATEGORIES.items():
    for f in funcs:
        FUNCTION_TO_CATEGORY[f] = cat

# IEEE column widths (inches)
# Single column: 3.5 inches, Double column: 7.16 inches
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16

# Figure sizes for IEEE format
FIGSIZE_SINGLE = (SINGLE_COL_WIDTH, 2.6)           # Single column
FIGSIZE_SINGLE_TALL = (SINGLE_COL_WIDTH, 3.2)      # Single column, tall
FIGSIZE_DOUBLE = (DOUBLE_COL_WIDTH, 2.8)           # Double column
FIGSIZE_DOUBLE_TALL = (DOUBLE_COL_WIDTH, 4.0)      # Double column, tall
FIGSIZE_DOUBLE_WIDE = (DOUBLE_COL_WIDTH, 5.0)      # Double column, very tall

DPI = 300


def ensure_box_spines(ax):
    """Ensure all four spines are visible and properly styled with black border."""
    # Set background
    ax.set_facecolor('white')

    # Make all spines visible with black lines
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')

def load_results(results_dir: str = 'benchmark_results_cocoex') -> pd.DataFrame:
    """Load all results from CSV files."""
    results_path = Path(results_dir)

    # Try loading all_results.csv first
    all_results_file = results_path / 'all_results.csv'
    if all_results_file.exists():
        try:
            df = pd.read_csv(all_results_file)
            if len(df) > 0:
                print(f"Loaded {len(df)} results from all_results.csv")
                return df
        except Exception as e:
            print(f"Error loading all_results.csv: {e}")

    # Otherwise, load from individual files
    print("Loading from individual result files...")
    all_data = []
    csv_files = list(results_path.glob('results_f*_run*.csv'))

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                all_data.append(df)
        except Exception:
            pass

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(df)} results from {len(csv_files)} files")
        return df

    raise ValueError(f"No results found in {results_dir}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for analysis."""
    # Add category column
    df['category'] = df['problem_id'].map(FUNCTION_TO_CATEGORY)

    # Extract operator names
    operators = [col.replace('op_weight_', '') for col in df.columns if col.startswith('op_weight_')]

    return df, operators


def melt_operator_data(df: pd.DataFrame, operators: list, metric: str = 'weight') -> pd.DataFrame:
    """Convert operator data to long format for plotting."""
    id_vars = ['problem_id', 'instance', 'dimension', 'category', 'best_fitness']

    if metric == 'weight':
        value_vars = [f'op_weight_{op}' for op in operators]
    elif metric == 'count':
        value_vars = [f'op_count_{op}' for op in operators]
    elif metric == 'success':
        value_vars = [f'op_success_{op}' for op in operators]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Filter to existing columns
    value_vars = [v for v in value_vars if v in df.columns]
    id_vars = [v for v in id_vars if v in df.columns]

    melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='operator_metric',
        value_name='value'
    )

    # Extract operator name
    melted['operator'] = melted['operator_metric'].str.replace(f'op_{metric}_', '')
    melted['operator_short'] = melted['operator'].map(OPERATOR_SHORT_NAMES)

    return melted


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_weights_by_dimension(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Plot operator weights grouped by dimension.

    Caption: Mean operator weights across problem dimensions (2D to 40D).
    Operators are ordered by type: exploitation (SO to LW) followed by
    exploration (GSA to RS), from simple to complex within each category.
    Higher weights indicate stronger preference by the basal ganglia selector.
    """
    melted = melt_operator_data(df, operators, 'weight')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Calculate mean weights per dimension and operator
    summary = melted.groupby(['dimension', 'operator_short'])['value'].mean().reset_index()

    # Pivot for grouped bar chart
    pivot = summary.pivot(index='operator_short', columns='dimension', values='value')

     # Sort operators by predefined order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    # Plot with custom colors
    pivot.plot(kind='bar', ax=ax, width=0.75, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'Operator, $h_o$')
    ax.set_ylabel(r'Mean Weight')
    # Legend at top, outside the plot
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center',
              frameon=False, ncol=len(pivot.columns), labels=[f"{val}D" for val in pivot.columns],
              fontsize=7, columnspacing=0.5, handlelength=1, handleheight=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'weights_by_dimension.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_by_dimension.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_by_dimension.pdf/png")


def plot_weights_by_category(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Plot operator weights grouped by BBOB category.

    Caption: Mean operator weights across BBOB problem categories. Categories
    represent different problem characteristics: Separable (f1-f5), Low/Moderate
    Conditioning (f6-f9), High Conditioning (f10-f14), Multi-modal (f15-f19),
    and Multi-modal with Weak Global Structure (f20-f24).
    """
    melted = melt_operator_data(df, operators, 'weight')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL)

    # Calculate mean weights per category and operator
    summary = melted.groupby(['category', 'operator_short'])['value'].mean().reset_index()

    # Pivot for grouped bar chart
    pivot = summary.pivot(index='operator_short', columns='category', values='value')

    # Order categories
    cat_order = list(BBOB_CATEGORIES.keys())
    pivot = pivot[[c for c in cat_order if c in pivot.columns]]

    # Sort operators by predefined order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    # Plot
    pivot.plot(kind='bar', ax=ax, width=0.75, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'Operator, $h_o$')
    ax.set_ylabel(r'Mean Weight')
    # Legend at top, outside the plot
    ax.legend(loc='upper left', #bbox_to_anchor=(0.5, 1.25),
              frameon=False, ncol=1, fontsize=7, columnspacing=0.5, handlelength=1, handleheight=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'weights_by_category.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_by_category.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_by_category.pdf/png")


def plot_weights_heatmap_dimension(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Heatmap of operator weights across dimensions.

    Caption: Heatmap showing mean operator weights for each problem dimension.
    Darker colors indicate higher weights (stronger operator preference).
    Exploitation operators (SO-LW) are shown first, followed by exploration
    operators (GSA-RS). Values represent the final adapted weight after
    neuromorphic learning.
    """
    melted = melt_operator_data(df, operators, 'weight')

    # Calculate mean weights
    summary = melted.groupby(['dimension', 'operator_short'])['value'].mean().reset_index()
    pivot = summary.pivot(index='operator_short', columns='dimension', values='value')

    # Sort operators by predefined order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL)

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': r'Weight'},
        linewidths=0.5,
        annot_kws={'size': 7},
    )

    ax.set_xlabel(r'Dimension, $D$')
    ax.set_ylabel(r'Operator, $h_o$')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'weights_heatmap_dimension.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_heatmap_dimension.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_heatmap_dimension.pdf/png")


def plot_weights_heatmap_category_dimension(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Heatmap of operator weights across categories and dimensions (faceted).

    Caption: Operator weight heatmaps faceted by problem dimension, showing
    how operator preferences vary across BBOB problem categories. Each panel
    corresponds to a different dimensionality (2D to 40D). This visualization
    reveals dimension-dependent operator selection strategies learned by the
    basal ganglia circuit.
    """
    melted = melt_operator_data(df, operators, 'weight')

    dimensions = sorted(df['dimension'].unique())
    n_dims = len(dimensions)

    # Create figure with 2 rows x 3 cols layout for 6 dimensions
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5*SINGLE_COL_WIDTH, 1.5*SINGLE_COL_WIDTH),
                              squeeze=False, sharex=True, sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})

    axes_flat = axes.flatten()

    for idx, dim in enumerate(dimensions):
        ax = axes_flat[idx]
        subset = melted[melted['dimension'] == dim]
        summary = subset.groupby(['category', 'operator_short'])['value'].mean().reset_index()
        pivot = summary.pivot(index='operator_short', columns='category', values='value')

        # Order categories
        cat_order = [c for c in BBOB_CATEGORIES.keys() if c in pivot.columns]
        pivot = pivot[cat_order]

        # Sort operators by predefined order
        ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
        pivot = pivot.loc[ordered_ops]

        # Only show colorbar on rightmost column
        show_cbar = (idx % n_cols == n_cols - 1)

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='flare', #'YlOrRd',
            ax=ax,
            cbar=False, #show_cbar,
            cbar_kws={}, #{'shrink': 0.8} if show_cbar else {},
            linewidths=0.5,
            annot_kws={'size': 6},
        )

        ax.set_title(f'${dim}$D')
        if idx % n_cols != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(r'Operator, $h_o$')

        if idx < (n_rows - 1) * n_cols:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(r'Category')

        ax.set_xticklabels(axes_flat[idx].get_xticklabels(), rotation=0, ha='right')

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)

        # Ensure box is well delimited
        ensure_box_spines(ax)

    # Hide unused axes
    for idx in range(len(dimensions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.savefig(output_dir / 'weights_heatmap_category_dimension.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_heatmap_category_dimension.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_heatmap_category_dimension.pdf/png")


def plot_success_rate_by_dimension(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Plot operator success rates by dimension.

    Caption: Mean operator success rates (proportion of iterations that improved
    the best solution) across problem dimensions. Higher success rates indicate
    operators that consistently produce fitness improvements. Note that high
    success does not always correlate with high weight, as the basal ganglia
    also considers exploration-exploitation balance.
    """
    melted = melt_operator_data(df, operators, 'success')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    summary = melted.groupby(['dimension', 'operator_short'])['value'].mean().reset_index()
    pivot = summary.pivot(index='operator_short', columns='dimension', values='value')

    # Sort operators by predefined order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    pivot.plot(kind='bar', ax=ax, width=0.75, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'Operator, $h_o$')
    ax.set_ylabel(r'Mean Success Rate')
    ax.set_yscale('log')

    # Legend at top, outside the plot
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center',
              frameon=False, ncol=len(pivot.columns), labels=[f"{val}D" for val in pivot.columns],
              fontsize=7, columnspacing=0.5, handlelength=1, handleheight=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_dimension.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'success_by_dimension.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: success_by_dimension.pdf/png")


def plot_operator_usage_distribution(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Box plot of operator usage counts.

    Caption: Distribution of operator usage counts across all experiments.
    Box plots show median, interquartile range, and outliers. Operators with
    higher median counts were selected more frequently by the basal ganglia
    decision circuit. The variance indicates consistency of selection across
    different problems and dimensions.
    """
    melted = melt_operator_data(df, operators, 'count')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL)

    # Use predefined operator order
    order = [op for op in OPERATOR_ORDER if op in melted['operator_short'].unique()]

    sns.boxplot(
        data=melted,
        x='operator_short',
        y='value',
        order=order,
        ax=ax,
        palette='Set2',
        linewidth=0.8,
        fliersize=3
    )

    ax.set_xlabel(r'Operator, $h_o$')
    ax.set_ylabel(r'Usage Count')
    ax.set_yscale('log')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'usage_distribution.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'usage_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: usage_distribution.pdf/png")


def plot_weights_per_function(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Detailed heatmap of weights per function.

    Caption: Heatmap of operator weights for each BBOB function (f1-f24),
    averaged across all dimensions and instances. Functions are grouped by
    category: Separable (f1-f5), Low/Moderate Conditioning (f6-f9), High
    Conditioning (f10-f14), Multi-modal (f15-f19), and Multi-modal Weak
    Structure (f20-f24). This reveals function-specific operator preferences.
    """
    melted = melt_operator_data(df, operators, 'weight')

    # Average across instances
    summary = melted.groupby(['problem_id', 'operator_short'])['value'].mean().reset_index()
    pivot = summary.pivot(index='operator_short', columns='problem_id', values='value')

    # Sort operators by predefined order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    sns.heatmap(
        pivot,
        annot=False,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': r'Weight'},
        linewidths=0.3
    )

    ax.set_xlabel(r'Function ID')
    ax.set_ylabel(r'Operator, $h_o$')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'weights_per_function.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_per_function.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_per_function.pdf/png")


def plot_top_operators_by_category(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Bar chart showing top 5 operators per category.

    Caption: Top 5 operators ranked by mean weight for each BBOB problem
    category. This visualization highlights which operators are most effective
    for different problem types. Color coding matches operator identity across
    panels. Values indicate mean weight after adaptation.
    """
    melted = melt_operator_data(df, operators, 'weight')

    categories = list(BBOB_CATEGORIES.keys())
    fig, axes = plt.subplots(1, len(categories), figsize=(DOUBLE_COL_WIDTH, 2.2), sharey=True)

    for ax, cat in zip(axes, categories):
        subset = melted[melted['category'] == cat]
        summary = subset.groupby('operator_short')['value'].mean().sort_values(ascending=False)
        top5 = summary.head(5)

        colors = [OPERATOR_COLORS.get(
            [k for k, v in OPERATOR_SHORT_NAMES.items() if v == op][0], '#999999'
        ) for op in top5.index]

        bars = ax.barh(range(len(top5)), top5.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top5)))
        ax.set_yticklabels(top5.index)
        ax.invert_yaxis()
        ax.set_xlabel(r'Weight')
        # Use short category name
        short_cat = cat.replace('Low/Moderate ', 'L/M ').replace('Multi-modal Weak Structure', 'MM Weak')
        ax.set_xlabel(short_cat, fontsize=8)

        # Add value labels
        for bar, val in zip(bars, top5.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=6)

        # Ensure box is well delimited
        ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_operators_by_category.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'top_operators_by_category.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: top_operators_by_category.pdf/png")


def plot_weight_vs_success_correlation(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Scatter plot showing correlation between weight and success rate, with error bars (std) for each operator.
    Caption: Relationship between mean operator weight and mean success rate.
    Each point represents an operator, with color indicating operator identity.
    The Pearson correlation coefficient ($r$) quantifies the linear relationship.
    Operators with high weight but low success may be valued for exploration,
    while high-success operators contribute to exploitation.
    Error bars show standard deviation for both weight (horizontal) and success (vertical).
    """
    weight_melted = melt_operator_data(df, operators, 'weight')
    success_melted = melt_operator_data(df, operators, 'success')

    # Compute mean and std for each operator
    weight_agg = weight_melted.groupby('operator_short')['value'].agg(['mean', 'std']).reset_index()
    weight_agg.columns = ['operator_short', 'mean_weight', 'std_weight']
    success_agg = success_melted.groupby('operator_short')['value'].agg(['mean', 'std']).reset_index()
    success_agg.columns = ['operator_short', 'mean_success', 'std_success']
    merged = weight_agg.merge(success_agg, on='operator_short')

    fig, ax = plt.subplots(figsize=(1.5*SINGLE_COL_WIDTH,SINGLE_COL_WIDTH/2))

    colors = [OPERATOR_COLORS.get(
        [k for k, v in OPERATOR_SHORT_NAMES.items() if v == op][0], '#999999'
    ) for op in merged['operator_short']]
    shapes = ['o' if OPERATOR_TYPE.get(op) == 'Exploitation' else 's' for op in merged['operator_short']]

    # Plot error bars (std) as +
    # for i, row in merged.iterrows():
    #     ax.errorbar(
    #         row['mean_weight'], row['mean_success'],
    #         xerr=row['std_weight'], yerr=row['std_success'],
    #         fmt='none', ecolor='black', elinewidth=0.8, capsize=2, alpha=0.7, zorder=1
    #     )

    # Plot points
    for i, row in merged.iterrows():
        ax.scatter(row['mean_weight'], row['mean_success'], c=colors[i], s=50,
                   marker=shapes[i], alpha=0.9, edgecolors='black', linewidth=0.5, zorder=2)
        if row['operator_short'] in ['LF', 'DE']:
            xytext_ = (0, -10)
        else:
            xytext_ = (5, -2)
        ax.annotate(row['operator_short'],
                   (row['mean_weight'], row['mean_success']),
                   xytext=xytext_, textcoords='offset points', fontsize=7)

    ax.set_xlabel(r'Mean Weight')
    ax.set_ylabel(r'Mean Success Rate')
    ax.set_yscale('log')
    ax.set_aspect('auto')

    # Add correlation
    corr = merged['mean_weight'].corr(merged['mean_success'])
    ax.text(0.05, 0.95, f'$r = {corr:.3f}$', transform=ax.transAxes, fontsize=9,
            verticalalignment='top')

    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'weight_vs_success.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weight_vs_success.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weight_vs_success.pdf/png")


def plot_performance_summary(df: pd.DataFrame, output_dir: Path):
    """
    Summary statistics of performance.

    Caption: Performance summary across all experiments. (a) Evaluation
    throughput by problem dimension, showing computational efficiency.
    (b) Wall-clock time required per experiment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(1.5*SINGLE_COL_WIDTH, SINGLE_COL_WIDTH/2.0), sharey=False)

    # 1. Evaluations per second by dimension
    ax = axes[0]
    sns.boxplot(data=df, x='dimension', y='evals_per_second', ax=ax,
                palette='Set2', linewidth=0.8, fliersize=2)
    ax.set_xlabel(r'Dimension, $D$')
    ax.set_ylabel(r'Evals./second')
    ax.set_yscale('log')
    # ax.text(0.02, 0.98, r'(a)', transform=ax.transAxes, fontsize=9,
    #         fontweight='bold', va='top')
    ensure_box_spines(ax)

    # 2. Wall time by dimension
    ax = axes[1]
    sns.boxplot(data=df, x='dimension', y='wall_time', ax=ax,
                palette='Set2', linewidth=0.8, fliersize=2)
    ax.set_xlabel(r'Dimension, $D$')
    ax.set_ylabel(r'Wall Time, (s)')
    # ax.set_yscale('log')
    # ax.text(0.02, 0.98, r'(b)', transform=ax.transAxes, fontsize=9,
    #         fontweight='bold', va='top')
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_summary.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: performance_summary.pdf/png")

def plot_dimension_scaling(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    How operator preferences change with dimension.

    Caption: Operator weight trajectories across problem dimensions. Lines
    show mean weight for each operator as dimensionality increases from 2D
    to 40D. Exploitation operators (SO-LW) are shown with solid markers,
    exploration operators (GSA-RS) with open markers. This reveals how the
    basal ganglia adapts its strategy for higher-dimensional problems.
    """
    melted = melt_operator_data(df, operators, 'weight')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Line plot for each operator in predefined order
    for i, op in enumerate(OPERATOR_ORDER):
        if op not in melted['operator_short'].unique():
            continue
        subset = melted[melted['operator_short'] == op]
        means = subset.groupby('dimension')['value'].mean()

        full_op_name = [k for k, v in OPERATOR_SHORT_NAMES.items() if v == op][0]
        color = OPERATOR_COLORS.get(full_op_name, '#999999')

        # Different markers for exploitation vs exploration
        marker = 'o' if OPERATOR_TYPE.get(op) == 'Exploitation' else 's'
        fillstyle = 'full' if OPERATOR_TYPE.get(op) == 'Exploitation' else 'none'

        ax.plot(means.index, means.values, marker=marker, label=op, color=color,
                linewidth=1.5, markersize=5, fillstyle=fillstyle, markeredgewidth=1)

    ax.set_xlabel(r'Dimension, $D$')
    ax.set_ylabel(r'Mean Weight')
    # Legend at top, outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, frameon=False, fontsize=7)
    ax.set_xticks(sorted(df['dimension'].unique()))

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'dimension_scaling.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'dimension_scaling.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: dimension_scaling.pdf/png")


def plot_operator_ranking_by_dimension(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Rank operators within each dimension.

    Caption: Operator ranking heatmap by problem dimension. Rank 1 indicates
    the operator with highest mean weight for that dimension. Green colors
    indicate top-ranked operators, red indicates lower ranks. Operators are
    ordered by type (exploitation then exploration). This visualization
    summarizes which operators dominate at each dimensionality.
    """
    melted = melt_operator_data(df, operators, 'weight')

    dimensions = sorted(df['dimension'].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL)

    rank_data = []
    for dim in dimensions:
        subset = melted[melted['dimension'] == dim]
        means = subset.groupby('operator_short')['value'].mean().sort_values(ascending=False)
        for rank, (op, _) in enumerate(means.items(), 1):
            rank_data.append({'dimension': dim, 'operator': op, 'rank': rank})

    rank_df = pd.DataFrame(rank_data)
    pivot = rank_df.pivot(index='operator', columns='dimension', values='rank')

    # Sort by predefined operator order
    ordered_ops = [op for op in OPERATOR_ORDER if op in pivot.index]
    pivot = pivot.loc[ordered_ops]

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        ax=ax,
        cbar_kws={'label': r'Rank'},
        linewidths=0.5,
        annot_kws={'size': 7}
    )

    ax.set_xlabel(r'Dimension, $D$')
    ax.set_ylabel(r'Operator, $h_o$')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

    # Ensure box is well delimited
    ensure_box_spines(ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'operator_ranking.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'operator_ranking.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: operator_ranking.pdf/png")


def plot_weights_by_dimension_boxswarm(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Boxplot + swarmplot of operator weights grouped by dimension.
    Shows the distribution of weights for each operator and dimension.
    """
    melted = melt_operator_data(df, operators, 'weight')
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    order = [op for op in OPERATOR_ORDER if op in melted['operator_short'].unique()]
    dims = sorted(melted['dimension'].unique())
    palette = sns.color_palette('Set2', n_colors=len(dims))

    sns.boxplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='dimension',
        order=order,
        palette=palette,
        ax=ax,
        linewidth=0.8,
        fliersize=2,
        dodge=True,
        width=0.7
    )
    sns.swarmplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='dimension',
        order=order,
        palette=palette,
        ax=ax,
        dodge=True,
        size=2,
        alpha=0.7,
        linewidth=0.2
    )
    ax.set_xlabel(r'Operator, $h_o$')
    ax.set_ylabel(r'Weight')
    ax.legend(title=r'$D$', bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=False, ncol=len(dims))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ensure_box_spines(ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'weights_by_dimension_boxswarm.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_by_dimension_boxswarm.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_by_dimension_boxswarm.pdf/png")


def plot_weights_by_category_boxswarm(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Boxplot + swarmplot of operator weights grouped by BBOB category.
    Shows the distribution of weights for each operator and category.
    """
    melted = melt_operator_data(df, operators, 'weight')
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL)
    order = [op for op in OPERATOR_ORDER if op in melted['operator_short'].unique()]
    cats = [c for c in BBOB_CATEGORIES.keys() if c in melted['category'].unique()]
    palette = sns.color_palette('Set2', n_colors=len(cats))

    sns.boxplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='category',
        order=order,
        hue_order=cats,
        palette=palette,
        ax=ax,
        linewidth=0.8,
        fliersize=2,
        dodge=True,
        width=0.7
    )
    sns.swarmplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='category',
        order=order,
        hue_order=cats,
        palette=palette,
        ax=ax,
        dodge=True,
        size=2,
        alpha=0.7,
        linewidth=0.2
    )
    ax.set_xlabel(r'Operator')
    ax.set_ylabel(r'Weight')
    ax.legend(title='Category', bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=True, ncol=len(cats))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ensure_box_spines(ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'weights_by_category_boxswarm.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'weights_by_category_boxswarm.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: weights_by_category_boxswarm.pdf/png")


def plot_success_rate_by_dimension_boxswarm(df: pd.DataFrame, operators: list, output_dir: Path):
    """
    Boxplot + swarmplot of operator success rates grouped by dimension.
    Shows the distribution of success rates for each operator and dimension.
    """
    melted = melt_operator_data(df, operators, 'success')
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    order = [op for op in OPERATOR_ORDER if op in melted['operator_short'].unique()]
    dims = sorted(melted['dimension'].unique())
    palette = sns.color_palette('Set2', n_colors=len(dims))

    sns.boxplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='dimension',
        order=order,
        palette=palette,
        ax=ax,
        linewidth=0.8,
        fliersize=2,
        dodge=True,
        width=0.7
    )
    sns.swarmplot(
        data=melted,
        x='operator_short',
        y='value',
        hue='dimension',
        order=order,
        palette=palette,
        ax=ax,
        dodge=True,
        size=2,
        alpha=0.7,
        linewidth=0.2
    )
    ax.set_xlabel(r'Operator')
    ax.set_ylabel(r'Success Rate')
    ax.set_yscale('log')
    ax.legend(title=r'$D$', bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=True, ncol=len(dims))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ensure_box_spines(ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_dimension_boxswarm.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'success_by_dimension_boxswarm.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: success_by_dimension_boxswarm.pdf/png")


def main():
    """Generate all plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper-ready plots from NEVO benchmark results")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='benchmark_results_cocoex',
        help='Directory containing result CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save figures'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("NEVO Results Visualization")
    print(f"{'='*60}")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    df = load_results(args.input_dir)
    df, operators = prepare_data(df)

    print(f"\nData summary:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Dimensions: {sorted(df['dimension'].unique())}")
    print(f"  Functions: {sorted(df['problem_id'].unique())}")
    print(f"  Categories: {df['category'].unique().tolist()}")
    print(f"  Operators: {len(operators)}")

    print(f"\nGenerating figures...")

    # Generate all plots

    # plot_weights_heatmap_dimension(df, operators, output_dir)
    plot_weights_heatmap_category_dimension(df, operators, output_dir)
    # plot_success_rate_by_dimension(df, operators, output_dir)
    # plot_dimension_scaling(df, operators, output_dir)
    # plot_weights_per_function(df, operators, output_dir)

    # plot_performance_summary(df, output_dir)
    # plot_weight_vs_success_correlation(df, operators, output_dir)

    # -x-
    # plot_weights_by_dimension(df, operators, output_dir)
    # plot_weights_by_category(df, operators, output_dir)
    # plot_operator_usage_distribution(df, operators, output_dir)
    # plot_top_operators_by_category(df, operators, output_dir)
    # plot_operator_ranking_by_dimension(df, operators, output_dir)
    # plot_weights_by_dimension_boxswarm(df, operators, output_dir)
    # plot_weights_by_category_boxswarm(df, operators, output_dir)
    # plot_success_rate_by_dimension_boxswarm(df, operators, output_dir)

    print(f"\n{'='*60}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
