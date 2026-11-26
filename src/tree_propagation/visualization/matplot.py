import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tree_propagation.models import SeedShadowResult
from tree_propagation.types import DispersalType
from tree_propagation.visualization.common import (
    compass_label_positions,
    compute_exponential_pdf_curve,
    compute_plot_extent,
    compute_sector_boundary_endpoints,
    exp_pdf_label,
)


def _new_axes(size: tuple[int, int], title: str) -> plt.Axes:  # type: ignore
    """Creates a standalone single-Axes figure with a white background and a title."""
    _fig, ax = plt.subplots(1, figsize=size, facecolor="#FFFFFF")
    ax.set_title(title)
    return ax


def draw_wind_rose_boundaries(ax: plt.Axes, max_extent: float, **style_kwargs) -> None:  # type: ignore
    """Draws radial wind sector boundary rays from (0, 0) out past max_extent."""
    style = {
        "color": "gray",
        "linewidth": 1,
        "linestyle": "--",
        "alpha": 0.6,
        "zorder": 1,
        **style_kwargs,
    }

    for x1, y1 in compute_sector_boundary_endpoints(max_extent, scale=1.1):
        ax.plot([0, x1], [0, y1], **style)


def draw_compass_labels(ax: plt.Axes, margin: float = 0.03, **style_kwargs) -> None:  # type: ignore
    """Pins N/S/E/W labels to the inner edges of an Axes frame."""
    style = {
        "ha": "center",
        "va": "center",
        "fontweight": "bold",
        "color": "dimgray",
        "fontsize": 11,
        "zorder": 5,
        **style_kwargs,
    }
    for label, (lx, ly) in compass_label_positions(margin=margin):
        ax.text(lx, ly, label, transform=ax.transAxes, **style)


def draw_parent(ax: plt.Axes, origin: tuple[float, float] = (0.0, 0.0), **style_kwargs) -> None:  # type: ignore
    x0, y0 = origin
    style = {
        "marker": "X",
        "s": 10,
        "facecolor": "deeppink",
        "edgecolor": "black",
        "linewidth": 1.2,
        "label": f"Parent Tree ({x0:g}, {y0:g})",
        "zorder": 4,
        **style_kwargs,
    }
    ax.scatter([x0], [y0], **style)


def draw_mean_radius(ax: plt.Axes, mean_radius: float, origin: tuple[float, float] = (0.0, 0.0)) -> None:  # type: ignore
    ax.add_patch(
        patches.Circle(
            origin,
            radius=mean_radius,
            fill=False,
            edgecolor="red",
            alpha=0.75,
            linewidth=2,
            label=f"Theoretical Mean ({mean_radius:.2f} m)",
            zorder=3,
        )
    )


def plot_seed_shadow(
    result: SeedShadowResult,
    ax_scatter: plt.Axes | None = None,  # type: ignore
    size: tuple[int, int] = (12, 12),
    max_extent: float | None = None,
    **style_kwargs,
) -> plt.Axes:  # type: ignore
    style = {
        "s": 3,
        "alpha": 0.15,
        "linewidth": 1,
        "edgecolors": "black",
        "color": "green",
        **style_kwargs,
    }

    if ax_scatter is None:
        ax_scatter = _new_axes(size, f"Seed shadow for species {result.species}")

    if max_extent is None:
        max_extent = compute_plot_extent(result)

    ax_scatter.set_facecolor("#FDF1D6")
    if not result.is_empty:
        sns.scatterplot(
            x=result.xs,
            y=result.ys,
            ax=ax_scatter,
            label="seeds",
            **style,
        )

    ax_scatter.set_xlim(-max_extent, max_extent)
    ax_scatter.set_ylim(-max_extent, max_extent)
    ax_scatter.set_xlabel("X\n(W <- -> E)", fontweight="bold")
    ax_scatter.set_ylabel(
        "Y\n(S ↓ ↑ N)",
        rotation=0,
        fontweight="bold",
        labelpad=22,
        va="center",
    )

    if result.dispersal_type == DispersalType.WIND:
        draw_wind_rose_boundaries(ax=ax_scatter, max_extent=max_extent)
    draw_compass_labels(ax=ax_scatter)
    draw_parent(ax=ax_scatter, s=20)
    draw_mean_radius(ax=ax_scatter, mean_radius=result.theoretical_mean_radius)
    ax_scatter.set_aspect("equal", "box")

    return ax_scatter


def plot_distance_histogram(
    result: SeedShadowResult,
    ax_distance: plt.Axes | None = None,  # type: ignore
    size: tuple[int, int] = (12, 12),
    max_dist: float | None = None,
    **style_kwargs,
) -> plt.Axes:  # type: ignore
    if ax_distance is None:
        ax_distance = _new_axes(size, f"Dispersal distance histogram for species {result.species}")

    dists = result.distances
    r_bar = result.theoretical_mean_radius

    if not result.is_empty:
        empirical_mean = float(np.mean(dists))
        sns.histplot(
            x=dists,
            bins=60,
            binrange=(0, float(dists.max())),
            stat="density",
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label="Simulated",
            ax=ax_distance,
        )

        ax_distance.axvline(
            x=empirical_mean,
            color="black",
            linewidth=2,
            linestyle=":",
        )
        ax_distance.annotate(
            f"Sample mean = {empirical_mean:.2f} m",
            xy=(empirical_mean, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -6),
            textcoords="offset points",
            ha="left",
            va="top",
            color="black",
        )

    x_grid, pdf = compute_exponential_pdf_curve(result)
    ax_distance.plot(
        x_grid,
        pdf,
        color="red",
        linewidth=2,
        linestyle="--",
        label=exp_pdf_label(r_bar),
    )

    if max_dist is not None:
        ax_distance.set_xlim(0, max_dist)
    else:
        ax_distance.set_xlim(left=0)

    ax_distance.set_xlabel("Distance r [m]", fontweight="bold")
    ax_distance.set_ylabel("Probability Density", fontweight="bold")
    ax_distance.legend(loc="upper right")

    return ax_distance


def plot_simulated_seed_shadow_and_distance_histogram(
    result: SeedShadowResult,
    size: tuple[float, float] = (10, 4.8),
    max_extent: float | None = None,
    max_dist: float | None = None,
) -> plt.Figure:  # type: ignore
    fig, (ax_scatter, ax_distance) = plt.subplots(
        1,
        2,
        figsize=size,
        facecolor="#FFFFFF",
        gridspec_kw={"wspace": 0.28},
    )
    fig.suptitle(
        f"{result.species} ({result.dispersal_type.value}, n = {result.num_instances} simulations)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plot_seed_shadow(result=result, ax_scatter=ax_scatter, max_extent=max_extent)
    plot_distance_histogram(result=result, ax_distance=ax_distance, max_dist=max_dist)
    ax_distance.set_box_aspect(1)

    # Remove per-axes legends that overlap "N" and "Sample mean"
    if ax_scatter.get_legend():
        ax_scatter.get_legend().remove()
    if ax_distance.get_legend():
        ax_distance.get_legend().remove()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="green",
            markeredgecolor="black",
            alpha=0.5,
            markersize=5,
            label="Seeds",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="deeppink",
            markeredgecolor="black",
            markersize=7,
            label="Parent (0, 0)",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            alpha=0.75,
            label=f"Mean Radius ({result.theoretical_mean_radius:.2f} m)",
        ),
        Patch(
            facecolor="steelblue",
            edgecolor="white",
            alpha=0.6,
            label="Simulated",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Exp PDF (r̄ = {result.theoretical_mean_radius:.2f} m)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            linestyle=":",
            label="Sample Mean",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True,
        fontsize=8.5,
    )

    return fig


def plot_species_seed_shadow_grid(
    results: list[SeedShadowResult],
    max_extent: float | None = None,
    max_dist: float | None = None,
    num_rows: int | None = None,
    row_height: float = 3.1,
    fig_width: float = 9.0,
) -> plt.Figure:  # type: ignore
    """Plots a multi-row comparison grid of seed shadows and radial distance histograms."""
    if not results:
        raise ValueError("results must contain at least one SeedShadowResult.")

    if max_extent is None:
        max_extent = max(compute_plot_extent(r) for r in results)
    if max_dist is None:
        max_dist = max((float(r.distances.max()) for r in results if not r.is_empty), default=1.0)

    n = len(results)
    total_rows = max(n, num_rows or n)
    fig, axes = plt.subplots(
        total_rows,
        2,
        figsize=(fig_width, row_height * total_rows),
        facecolor="#FFFFFF",
        gridspec_kw={"wspace": 0.25, "hspace": 0.28},
    )
    if total_rows == 1:
        axes = axes.reshape(1, 2)

    for idx in range(total_rows):
        ax_scatter, ax_dist = axes[idx]
        if idx >= n:
            ax_scatter.axis("off")
            ax_dist.axis("off")
            continue

        result = results[idx]
        plot_seed_shadow(result=result, ax_scatter=ax_scatter, max_extent=max_extent)
        plot_distance_histogram(result=result, ax_distance=ax_dist, max_dist=max_dist)
        ax_dist.set_box_aspect(1)

        ax_scatter.set_title(
            f"{result.species} ({result.dispersal_type.value})",
            fontweight="bold",
            fontsize=11,
            loc="left",
        )

        # Remove all per-subplot legends so nothing overlaps N or "Sample mean"
        if ax_scatter.get_legend():
            ax_scatter.get_legend().remove()
        if ax_dist.get_legend():
            ax_dist.get_legend().remove()

        # Keep x-axis labels only on the last populated row
        if idx < n - 1:
            ax_scatter.set_xlabel("")
            ax_dist.set_xlabel("")

    # Single shared legend centered above the grid
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green",
               markeredgecolor="black", alpha=0.5, markersize=5, label="Seeds"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="deeppink",
               markeredgecolor="black", markersize=7, label="Parent Tree (0, 0)"),
        Line2D([0], [0], color="red", linewidth=2, alpha=0.75, label="Theoretical Mean Radius"),
        Patch(facecolor="steelblue", edgecolor="white", alpha=0.6, label="Simulated Density"),
        Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Theoretical Exp PDF"),
        Line2D([0], [0], color="black", linewidth=2, linestyle=":", label="Sample Mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    return fig