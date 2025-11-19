"""
Visualisation Utilities
=======================

Tools for visualising NEVO optimisation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any


def setup_plotting_style(fontsize: int = 14):
    """
    Set up publication-quality plotting style.

    Parameters
    ----------
    fontsize : int
        Base font size for plots
    """
    plt.rcParams.update({
        # LaTeX rendering
        'text.usetex': True,

        # Fonts
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,

        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 0.8,

        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.color': '0.9',
        'axes.grid': True,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


def plot_optimisation_results(
    optimiser,
    optimum: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Create comprehensive visualisation of optimisation results.

    Parameters
    ----------
    optimiser : NEVOptimiser
        Optimiser instance after running
    optimum : float, optional
        Known optimal fitness value (for error computation)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    """
    if optimiser.simulator is None:
        raise ValueError("Optimiser must be run before plotting")

    setup_plotting_style()

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    # Extract data
    sim = optimiser.simulator
    stats_data = sim.data[optimiser.stats_probe]
    best_f_trace = stats_data[:, 0]
    mean_f_trace = stats_data[:, 1]
    operator_trace = stats_data[:, 2]

    # Plot 1: Fitness evolution
    ax1 = axes[0]
    valid_mask = best_f_trace < optimiser.f_default_worst
    t_valid = sim.trange()[valid_mask]
    best_valid = best_f_trace[valid_mask]
    mean_valid = mean_f_trace[valid_mask]

    if optimum is not None:
        # Plot error from optimum
        ax1.plot(t_valid, best_valid - optimum, 'b-', linewidth=2,
                label='Best-so-far error')
        ax1.plot(t_valid, mean_valid - optimum, 'gray', alpha=0.3,
                linewidth=0.5, label='Population mean error')
        ax1.set_ylabel('Fitness Error')
        ax1.set_yscale('log')
    else:
        # Plot absolute fitness
        ax1.plot(t_valid, best_valid, 'b-', linewidth=2,
                label='Best-so-far')
        ax1.plot(t_valid, mean_valid, 'gray', alpha=0.3,
                linewidth=0.5, label='Population mean')
        ax1.set_ylabel('Fitness')
        ax1.set_yscale('log')

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Operator selection (histogram style)
    ax2 = axes[1]
    operator_names = [op.name for op in optimiser.operators]
    colors = {
        'LevyFlight': 'red',
        'DifferentialEvolution': 'orange',
        'ParticleSwarm': 'green',
        'SpiralOptimisation': 'blue',
    }

    for i, op_name in enumerate(operator_names):
        mask = operator_trace == i
        times = sim.trange()[mask]

        if len(times) > 0:
            # Vertical lines at each activation
            ax2.vlines(times, i - 0.3, i + 0.3,
                      colors=colors.get(op_name, 'gray'), alpha=0.3, linewidths=0.5)
            # Density visualization
            hist, edges = np.histogram(times, bins=50)
            centers = (edges[:-1] + edges[1:]) / 2
            if hist.max() > 0:
                density = hist / hist.max() * 0.4
                ax2.fill_between(centers, i - density, i + density,
                                color=colors.get(op_name, 'gray'), alpha=0.5,
                                label=op_name)

    ax2.set_ylabel('Active Operator')
    ax2.set_yticks(range(len(operator_names)))
    ax2.set_yticklabels(operator_names)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_ylim(-0.5, len(operator_names) - 0.5)

    # Plot 3: State features
    ax3 = axes[2]
    raw_features = sim.data[optimiser.state_features_probe]

    # Downsample for visibility
    downsample = 10
    time_ds = sim.trange()[::downsample]
    diversity_raw = raw_features[::downsample, 0]
    improvement_raw = raw_features[::downsample, 1]
    convergence_raw = raw_features[::downsample, 2]

    ax3.plot(time_ds, diversity_raw, 'r-',
            label=r'Diversity ($\phi_d$)', alpha=0.8, linewidth=1.5)
    ax3.plot(time_ds, improvement_raw, 'g-',
            label=r'Improvement Rate ($\phi_i$)', alpha=0.8, linewidth=1.5)
    ax3.plot(time_ds, convergence_raw, 'b-',
            label=r'Convergence ($\phi_c$)', alpha=0.8, linewidth=1.5)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Feature Value')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc='center right')

    # Title
    if title is None:
        title = f'nevo ({optimiser.dimension}D)'

    if optimum is not None:
        title += f', Best: {optimiser.state["best_f"]:.3g}, Error: {optimiser.state["best_f"] - optimum:.3g}'
    else:
        title += f', Best: {optimiser.state["best_f"]:.3g}'

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes


def plot_operator_statistics(
    optimiser,
    save_path: Optional[str] = None,
):
    """
    Plot operator usage and performance statistics.

    Parameters
    ----------
    optimiser : NEVOptimiser
        Optimiser instance after running
    save_path : str, optional
        Path to save figure
    """
    setup_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Get statistics
    stats = optimiser.get_statistics()
    operator_names = [op.name for op in optimiser.operators]

    # Plot 1: Usage counts
    ax1 = axes[0]
    counts = [stats["operator_counts"][name] for name in operator_names]
    colors = ['red', 'orange', 'green', 'blue'][:len(operator_names)]

    ax1.bar(range(len(operator_names)), counts, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(operator_names)))
    ax1.set_xticklabels(operator_names, rotation=45, ha='right')
    ax1.set_ylabel('Usage Count')
    ax1.set_title('Operator Usage')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success rates and weights
    ax2 = axes[1]
    success_rates = [stats["operator_success_rates"][name] for name in operator_names]
    weights = [stats["operator_weights"][name] for name in operator_names]

    x = np.arange(len(operator_names))
    width = 0.35

    ax2.bar(x - width/2, success_rates, width, label='Success Rate',
           color='green', alpha=0.7)
    ax2.bar(x + width/2, weights, width, label='Utility Weight',
           color='blue', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(operator_names, rotation=45, ha='right')
    ax2.set_ylabel('Value')
    ax2.set_title('Operator Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes

