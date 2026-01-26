"""
Visualisation Utilities
=======================

Tools for visualising NEVO optimisation results.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Optional, Dict, Any
from PIL import Image
import random


def setup_plotting_style(fontsize: int = 12):
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
        'axes.labelsize': fontsize -2,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize -2,
        'ytick.labelsize': fontsize -2,
        'legend.fontsize': fontsize - 2,
        'figure.titlesize': fontsize,

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
        'axes.grid': False,

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
    show_legend: bool = True,
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

    fig, axes = plt.subplots(3, 1, figsize=(4, 7), sharex=True)

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

    if show_legend:
        ax1.legend(loc='upper right')

    # Plot 2: Operator selection (histogram style)
    ax2 = axes[1]

    # Separate operators by type and sort by complexity (read from operator)
    exploration_ops = []
    exploitation_ops = []
    for i, op in enumerate(optimiser.operators):
        if op.operator_type == "exploration":
            exploration_ops.append((i, op))
        else:
            exploitation_ops.append((i, op))

    # Sort by complexity within each group
    exploration_ops.sort(key=lambda x: x[1].complexity)
    exploitation_ops.sort(key=lambda x: x[1].complexity)

    # Combine: exploration first, then exploitation
    sorted_ops = exploration_ops + exploitation_ops
    n_exploration = len(exploration_ops)
    n_exploitation = len(exploitation_ops)
    n_operators = len(sorted_ops)

    # Create colormaps for each type (oranges for exploration, blues for exploitation)
    cmap_exploration = cm.get_cmap('Oranges')
    cmap_exploitation = cm.get_cmap('Blues')

    # Build mapping from original index to display position and colour
    display_names = []
    colors = {}
    original_to_display = {}

    for display_idx, (orig_idx, op) in enumerate(sorted_ops):
        original_to_display[orig_idx] = display_idx
        display_names.append(op.short_name)

        if op.operator_type == "exploration":
            # Position within exploration group
            pos_in_group = [x[1].name for x in exploration_ops].index(op.name)
            color = cmap_exploration(0.3 + 0.6 * pos_in_group / max(1, n_exploration - 1))
        else:
            # Position within exploitation group
            pos_in_group = [x[1].name for x in exploitation_ops].index(op.name)
            color = cmap_exploitation(0.3 + 0.6 * pos_in_group / max(1, n_exploitation - 1))

        colors[op.short_name] = color

    # Plot each operator
    for display_idx, (orig_idx, op) in enumerate(sorted_ops):
        mask = operator_trace == orig_idx
        times = sim.trange()[mask]

        if len(times) > 0:
            op_name = op.short_name
            # Vertical lines at each activation
            ax2.vlines(times, display_idx - 0.3, display_idx + 0.3,
                      colors=colors.get(op_name, 'gray'), alpha=0.3, linewidths=0.5)
            # Density visualisation
            hist, edges = np.histogram(times, bins=50)
            centers = (edges[:-1] + edges[1:]) / 2
            if hist.max() > 0:
                density = hist / hist.max() * 0.4
                ax2.fill_between(centers, display_idx - density, display_idx + density,
                                color=colors.get(op_name, 'gray'), alpha=0.5,
                                label=op_name)

    # Add separator line between exploration and exploitation
    if n_exploration > 0 and n_exploitation > 0:
        separator_y = n_exploration - 0.5
        ax2.axhline(y=separator_y, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add group labels on the right
    if n_exploration > 0:
        ax2.text(sim.trange()[-1] * 1.02, (n_exploration - 1) / 2, 'Exploration',
                fontsize=8, va='center', ha='left', rotation=90, alpha=0.7)
    if n_exploitation > 0:
        ax2.text(sim.trange()[-1] * 1.02, n_exploration + (n_exploitation - 1) / 2, 'Exploitation',
                fontsize=8, va='center', ha='left', rotation=90, alpha=0.7)

    ax2.set_ylabel('Active Operator')
    ax2.set_yticks(range(len(display_names)))
    ax2.set_yticklabels(display_names)
    ax2.set_ylim(-0.5, n_operators - 0.5)
    ax2.set_xlim(sim.trange()[0], sim.trange()[-1] * 1.05)

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

    if show_legend:
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
    show_legend: bool = True,
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

    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

    # Get statistics
    stats = optimiser.get_statistics()

    # Separate and sort operators by type and complexity (read from operator)
    exploration_ops = []
    exploitation_ops = []
    for op in optimiser.operators:
        if op.operator_type == "exploration":
            exploration_ops.append(op)
        else:
            exploitation_ops.append(op)

    # Sort by complexity within each group
    exploration_ops.sort(key=lambda x: x.complexity)
    exploitation_ops.sort(key=lambda x: x.complexity)

    # Combine: exploration first, then exploitation
    sorted_ops = exploration_ops + exploitation_ops
    n_exploration = len(exploration_ops)
    n_exploitation = len(exploitation_ops)

    operator_names = [op.name for op in sorted_ops]
    operator_short_names = [op.short_name for op in sorted_ops]

    # Create colormaps for each type (oranges for exploration, blues for exploitation)
    cmap_exploration = cm.get_cmap('Oranges')
    cmap_exploitation = cm.get_cmap('Blues')

    colors = []
    for i, op in enumerate(sorted_ops):
        if op.operator_type == "exploration":
            pos_in_group = i
            color = cmap_exploration(0.3 + 0.6 * pos_in_group / max(1, n_exploration - 1))
        else:
            pos_in_group = i - n_exploration
            color = cmap_exploitation(0.3 + 0.6 * pos_in_group / max(1, n_exploitation - 1))
        colors.append(color)

    # Plot 1: Usage counts (as percentage)
    ax1 = axes[0]
    # ax1.set_title('Operator Usage')

    counts = [stats["operator_counts"][name] for name in operator_names]
    total_counts = sum(counts)
    percentages = [100.0 * c / total_counts if total_counts > 0 else 0.0 for c in counts]

    # Protect zero values for log scale by using a small epsilon
    epsilon = 0.01
    percentages_safe = [max(p, epsilon) for p in percentages]

    ax1.bar(range(len(operator_names)), percentages_safe, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(operator_names)))
    ax1.set_xticklabels(operator_short_names, rotation=0, ha='center')
    ax1.set_ylabel('Usage (\\%)')
    ax1.set_yscale('log')

    # Add annotation showing what 100% means (total calls)
    ax1.text(0.02, 0.90, f'100\\% = {total_counts:,} calls',
             transform=ax1.transAxes, fontsize=7, va='bottom', ha='left',
             alpha=0.7, style='italic')

    # Add separator line
    if n_exploration > 0 and n_exploitation > 0:
        ax1.axvline(x=n_exploration - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Plot 2: Success rates and weights
    ax2 = axes[1]
    # ax2.set_title('Operator Performance')

    success_rates = [stats["operator_success_rates"][name] for name in operator_names]
    weights = [stats["operator_weights"][name] for name in operator_names]

    # Protect zero values for log scale
    success_rates_safe = [max(s, epsilon) for s in success_rates]
    weights_safe = [max(w, epsilon) for w in weights]

    x = np.arange(len(operator_names))
    width = 0.35

    ax2.bar(x - width/2, success_rates_safe, width, label='Success Rate',
           color='green', alpha=0.7)
    ax2.bar(x + width/2, weights_safe, width, label='Utility Weight',
           color='blue', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(operator_short_names, rotation=0, ha='center')
    ax2.set_ylabel('Value')
    ax2.set_yscale('log')

    # Add separator line
    if n_exploration > 0 and n_exploitation > 0:
        ax2.axvline(x=n_exploration - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    if show_legend:
        ax2.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes


def generate_particle_svg(
    text: str,
    output_svg: str = "neuroptim_particles.svg",
    num_particles: int = 10000,
    radius: float = 0.5,
    fill_colour: str = "#1e4c15",
    font_size: int = 100,
    font_path: Optional[str] = None,
    padding: int = 20,
    blur_radius: int = 2,
    min_probability: float = 0.0,
    curl_particles: int = 1000,
) -> str:
    """
    Generate an SVG with particles sampled to form text.

    Particles are distributed with higher density inside the letters and
    a smooth circular falloff outside, creating a natural scattered effect.
    If the text contains the uppercase letter 'O', subtle spiral trajectories
    will be added, representing particles being attracted towards it.

    Parameters
    ----------
    text : str
        The text string to render as particles
    output_svg : str
        Output SVG file path
    num_particles : int
        Total number of particles to generate
    radius : float
        Fixed particle radius
    fill_colour : str
        SVG fill colour for particles
    font_size : int
        Font size for rendering the text
    font_path : str, optional
        Path to a TrueType font file. If None, uses a serif font
    padding : int
        Padding around the text in pixels (allows particles to scatter outside)
    blur_radius : int
        Radius for the circular windowing to smooth probability transition
    min_probability : float
        Minimum probability outside the letter area (0 to 1)
    curl_particles : int
        Number of particles to use for the curl attractor trajectories (letter O)

    Returns
    -------
    str
        Path to the saved SVG file
    """
    from PIL import ImageDraw, ImageFont
    from scipy.ndimage import distance_transform_edt
    import math


    # Load serif font
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # Try common serif fonts
        serif_fonts = [
            "Times New Roman.ttf",
            "TimesNewRoman.ttf",
            "Georgia.ttf",
            "DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
        font = None
        for font_name in serif_fonts:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

    # Create a temporary image to measure text size
    temp_img = Image.new("L", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Find positions of letter 'O' (uppercase only) for curl attractor replacement
    o_centres = []
    o_radii = []
    o_char_width = 0
    if 'O' in text:
        # Measure each character to find O positions
        x_offset = padding - bbox[0]
        for i, char in enumerate(text):
            char_bbox = temp_draw.textbbox((0, 0), text[:i+1], font=font)
            char_start_bbox = temp_draw.textbbox((0, 0), text[:i], font=font) if i > 0 else (0, 0, 0, 0)

            if char == 'O':
                # Calculate centre of this character
                char_left = x_offset + char_start_bbox[2]
                char_right = x_offset + char_bbox[2]
                char_width = char_right - char_left
                centre_x = (char_left + char_right) / 2
                centre_y = padding + text_height / 2
                # The radius is the circumference radius of the O (half the character width)
                o_radius = char_width / 2.2
                o_centres.append((centre_x, centre_y))
                o_radii.append(o_radius)

                o_char_width = char_width

    # Create text WITHOUT 'O' for the mask (replace O with space to preserve spacing)
    text_for_mask = ''.join(3 * ' ' if c == 'O' else c for c in text)

    # Create the mask image with the text (excluding O)
    W = text_width + 2 * padding  # Extra space for O replacements
    H = text_height + 2 * padding
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.text((padding - bbox[0], padding - bbox[1]), text_for_mask, font=font, fill=255)

    # Convert to array
    mask_arr = np.array(mask_img) / 255.0  # 1 = inside letter, 0 = outside

    # Create smooth probability map using distance transform
    # Calculate distance from each pixel to the nearest letter pixel
    binary_mask = mask_arr > 0.5
    distance_outside = distance_transform_edt(~binary_mask)

    # Apply circular windowing: smooth exponential falloff based on distance
    # Probability = 1 inside letters, exponentially decreasing to min_probability outside
    prob_arr = np.where(
        binary_mask,
        1.0,  # Full probability inside letters
        min_probability + (1.0 - min_probability) * np.exp(-distance_outside / blur_radius)
    )

    H, W = prob_arr.shape

    # Particle sampling for ALL letters (including O)
    points = []

    for _ in range(num_particles):
        max_attempts = 1000
        for _ in range(max_attempts):
            # Random position (float coordinates)
            x = random.uniform(0, W)
            y = random.uniform(0, H)

            # Apply probability map (use int indices for array lookup)
            ix = min(int(x), W - 1)
            iy = min(int(y), H - 1)
            if random.random() < prob_arr[iy, ix]:
                points.append((x, y, radius))
                break

    # Generate curl attractor particles for each uppercase 'O'
    # Spiral trajectories coming from outside and converging to the O's circumference (the cycle)
    curl_points = []
    for (cx, cy), o_r in zip(o_centres, o_radii):
        # Generate several spiral trajectories
        num_trajectories = random.randint(4, 7)
        particles_per_trajectory = curl_particles // num_trajectories

        for _ in range(num_trajectories):
            # Random starting angle for this trajectory
            theta_start = random.uniform(0, 2 * math.pi)

            for _ in range(particles_per_trajectory):
                # Parameter along the spiral (0 = far away, 1 = at the cycle)
                t = random.uniform(0, 1)

                # Start from outside and spiral towards the O's circumference (the cycle)
                start_r = o_r * random.uniform(1.1, 2.0)
                # Spiral inward towards the O's circumference (not the centre)
                current_r = o_r + (start_r - o_r) * (1 - t)

                # Rotation as particle approaches the cycle
                rotation = t * math.pi * 1.2
                theta = theta_start + rotation

                # Small spread for natural look
                current_r += random.gauss(0, o_r * 0.04)
                theta += random.gauss(0, 0.03)

                # Convert to Cartesian
                px = cx + current_r * math.cos(theta)
                py = cy + current_r * math.sin(theta)

                curl_points.append((px, py, radius))

        # Add particles on the cycle itself (the O's circumference)
        for _ in range(curl_particles // 2):
            theta = random.uniform(0, 2 * math.pi)
            # Small perturbation around the cycle
            r_noise = random.gauss(o_r, o_r * 0.04)
            px = cx + r_noise * math.cos(theta)
            py = cy + r_noise * math.sin(theta)
            curl_points.append((px, py, radius))

    # SVG generation
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    )
    svg_content = ""

    for (x, y, r) in points:
        svg_content += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill_colour}" />\n'

    # Add curl attractor particles
    for (x, y, r) in curl_points:
        svg_content += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill_colour}" />\n'

    svg_footer = "</svg>"

    svg_full = svg_header + svg_content + svg_footer

    # Save primary SVG with the specified fill colour
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_full)

    # Derive a filename for the white-filled variant and save it
    if "." in output_svg:
        base, ext = output_svg.rsplit(".", 1)
        white_svg = f"{base}_white.{ext}"
    else:
        white_svg = f"{output_svg}_white"

    svg_white = svg_full.replace(f'fill="{fill_colour}"', 'fill="white"')

    with open(white_svg, "w", encoding="utf-8") as f:
        f.write(svg_white)

    return output_svg
