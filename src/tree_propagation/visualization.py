import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import patches
from plotly.subplots import make_subplots

from src.tree_propagation.dispersal import DispersalType
from src.tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from src.tree_propagation.wind_config import WindRose


def _truncated_cmap(cmap_name: str = "YlGnBu", minval: float = 0.25, maxval: float = 1.0):
    """Clips the pale bottom end of a colormap so low-count bins remain clearly visible."""
    base = plt.get_cmap(cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}",
        base(np.linspace(minval, maxval, 256)),
    )

def plot_wind_rose(wind_rose: WindRose) -> go.Figure:
    """Creates a polar wedge chart showing incoming wind probabilities by compass sector."""
    directions = [d.name for d in wind_rose.probabilities]
    angles = [d.value for d in wind_rose.probabilities]
    probs = list(wind_rose.probabilities.values())
    max_prob = max(probs)

    fig = go.Figure(
        go.Barpolar(
            r=probs,
            theta=angles,
            
            width=[45] * len(angles),
            customdata=directions,
            marker={
                "color": probs,
                "colorscale": "Blues",
                "cmin": 0,
                "line": {"color": "white", "width": 1.5},
            },
            opacity=0.88,
            hovertemplate="From <b>%{customdata}</b> (%{theta}°)<br>Probability: %{r:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Wind Rose (Blowing From)", "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        height=300,
        margin={"t": 55, "b": 30, "l": 40, "r": 40},
        template="plotly_white",
        polar={
            "angularaxis": {
                "direction": "clockwise",
                "rotation": 90,  # Places 0° (North) at the top
                "tickmode": "array",
                "tickvals": [0, 45, 90, 135, 180, 225, 270, 315],
                "ticktext": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            },
            "radialaxis": {
                "range": [0, max_prob * 1.12],
                "tickmode": "linear",
                "tick0": 0.10,
                "dtick": 0.10,
                "tickformat": ".0%",
                "angle": 45,
                "tickangle": 45,
                "tickfont": {"size": 10, "color": "gray"},
                "gridcolor": "lightgray",
            },
        },
    )
    return fig

def _nice_colorbar_ticks(cmax: float, target_ticks: int = 6) -> list[float]:
    """Generates clean, human-readable tick values from 0 up to cmax for any scale."""
    if cmax <= 0:
        return [0.0]

    raw_step = cmax / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude

    if residual <= 1.5:
        nice_step = 1.0 * magnitude
    elif residual <= 3.0:
        nice_step = 2.0 * magnitude
    elif residual <= 7.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    ticks = np.arange(0, cmax + nice_step * 0.5, nice_step)
    return [round(float(t), 10) for t in ticks if t <= cmax * 1.001]

def draw_wind_rose_boundaries(ax: plt.Axes, max_extent: float, **style_kwargs) -> None: # type: ignore
    """Draws radial wind sector boundary rays from (0, 0) out past max_extent."""
    style = {
        "color": "gray",
        "linewidth": 1,
        "linestyle": "--",
        "alpha": 0.6,
        "zorder": 1,
        **style_kwargs,
    }

    for boundary_deg in np.arange(22.5, 360.0, 45.0):
        rad = np.deg2rad(boundary_deg)
        ax.plot(
            [0, max_extent * 1.1 * np.sin(rad)],
            [0, max_extent * 1.1 * np.cos(rad)],
            **style
        )

def draw_compass_labels(ax: plt.Axes, margin: float = 0.03, **style_kwargs) -> None: # type: ignore
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
    positions = [
        ("N", (0.5, 1.0 - margin)),
        ("S", (0.5, margin)),
        ("E", (1.0 - margin, 0.5)),
        ("W", (margin, 0.5)),
    ]
    for label, (lx, ly) in positions:
        ax.text(lx, ly, label, transform=ax.transAxes, **style)

def draw_parent(ax: plt.Axes, origin: tuple = (0.0, 0.0), **style_kwargs) -> None: # type: ignore
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
     ax.scatter(
        [x0],
        [y0],
        **style
    )

def draw_mean_radius(ax: plt.Axes, mean_radius: float, origin: tuple = (0.0, 0.0)) -> None: # type: ignore
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

def plot_seed_shadow_and_distance(
    result: SeedShadowResult, metrics: SeedShadowMetrics, max_scatter_points: int = 100_000
) -> go.Figure:
    """Creates an interactive, zoomable 2D seed shadow scatterplot and distance distribution."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{result.species} ({result.dispersal_type}) — 2D Seed Shadow",
            "Dispersal Distance Distribution",
        ),
        horizontal_spacing=0.1,
    )

    xs, ys = result.xs, result.ys
    dists = result.distances
    r_bar = result.theoretical_mean_radius

    empirical_mean = (
        metrics.empirical_r_bar
        if metrics is not None
        else float(np.mean(result.distances))
    )

    # Subsample only if total seeds exceed max_scatter_points so browser WebGL stays responsive
    if len(xs) > max_scatter_points:
        idx = np.linspace(0, len(xs) - 1, max_scatter_points, dtype=int)
        plot_xs, plot_ys, plot_dists = xs[idx], ys[idx], dists[idx]
    else:
        plot_xs, plot_ys, plot_dists = xs, ys, dists

    power = 0.75
    cmax_m = float(np.percentile(plot_dists, 98))
    color_vals = np.clip(plot_dists, 0, cmax_m) ** power
    tick_vals_m = _nice_colorbar_ticks(cmax_m)

    tick_candidates = [0, 1, 2, 4, 7, 10, 14, 20, 30]
    tick_vals_m = [t for t in tick_candidates if t <= cmax_m]

    # Gradual 8-stop scale
    gradual_greens = [
        [0.00, "#041A0F"],  # deep pine-black (0 m)
        [0.16, "#06311C"],  # dark forest
        [0.32, "#145331"],  # rich emerald
        [0.50, "#33965B"],  # vibrant medium green
        [0.68, "#5BB876"],  # fresh sage
        [0.85, "#8FD692"],  # light spring green
        [1.00, "#C7ECB0"],  # pale lime (outer edges)
    ]

    #marker_opacity = 0.38 if len(plot_xs) > 5_000 else 0.65

    # Interactive WebGL Seed Scatterplot
    fig.add_trace(
        go.Scattergl(
            x=plot_xs,
            y=plot_ys,
            mode="markers",
            name="Seeds",
            marker={
                "size": 4,
                "color": color_vals,
                "cmin": 0,
                "cmax": cmax_m**power,
                "colorscale": gradual_greens,
                "opacity": 0.60,
                "line": {"width": 0.35, "color": "rgba(13, 66, 40, 0.3)"},
                "colorbar": {
                    "title": "Distance [m]",
                    "x": 0.44,
                    "len": 0.85,
                    "tickvals": [t**power for t in tick_vals_m],
                    "ticktext": [f"{t:g}" for t in tick_vals_m],
                },
            },
            customdata=plot_dists,
            hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>r: %{customdata:.2f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Parent tree at (0, 0)
    fig.add_trace(
        go.Scattergl(
            x=[0],
            y=[0],
            mode="markers",
            name="Parent Tree (0, 0)",
            marker={"symbol": "x", "size": 12, "color": "deeppink", "line": {"width": 1.5, "color": "black"}},
            hovertemplate="Parent Tree (0, 0)<extra></extra>",
        ),
        row=1,
        col=1,
    )

    max_extent = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05

    # Wind sector boundary rays
    if result.dispersal_type == DispersalType.WIND:
        for boundary_deg in np.arange(22.5, 360.0, 45.0):
            rad = np.deg2rad(boundary_deg)
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_extent * 1.5 * np.sin(rad),
                y1=max_extent * 1.5 * np.cos(rad),
                line={"color": "gray", "width": 1, "dash": "dash"},
                opacity=0.6,
                layer="below",
                row=1,
                col=1,
            )

    # Theoretical mean dispersal radius circle + legend entry
    fig.add_shape(
        type="circle",
        x0=-r_bar,
        y0=-r_bar,
        x1=r_bar,
        y1=r_bar,
        line={"color": "red", "width": 2},
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Theoretical Mean ({r_bar:.2f} m)",
            line={"color": "red", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Compass markers pinned to the 2D subplot frame
    for label, (lx, ly) in [("N", (0.5, 0.98)), ("S", (0.5, 0.02)), ("E", (0.98, 0.5)), ("W", (0.02, 0.5))]:
        fig.add_annotation(
            x=lx,
            y=ly,
            xref="x domain",
            yref="y domain",
            text=f"<b>{label}</b>",
            showarrow=False,
            font={"color": "dimgray", "size": 12},
        )

    # Lock 1:1 aspect ratio so zooming keeps circles circular
    fig.update_xaxes(title_text="x [m] (East →)", range=[-max_extent, max_extent], row=1, col=1)
    fig.update_yaxes(
        title_text="y [m] (North ↑)",
        range=[-max_extent, max_extent],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=1,
    )

    # Dispersal Distance Histogram vs Theoretical Exponential PDF
    fig.add_trace(
        go.Histogram(
            x=dists,
            xbins={"start": 0},
            nbinsx=60,
            histnorm="probability density",
            name="Simulated",
            marker_color="steelblue",
            opacity=0.6,
        ),
        row=1,
        col=2,
    )

    x_grid = np.linspace(0, np.percentile(dists, 99.5), 300)
    pdf = (1.0 / r_bar) * np.exp(-x_grid / r_bar)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf,
            mode="lines",
            name=f"Exp PDF (r̄ = {r_bar:.2f} m)",
            line={"color": "red", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=2,
    )

    fig.add_vline(
        x=empirical_mean,
        line_width=2,
        line_dash="dot",
        line_color="black",
        annotation_text=f"Sample mean = {empirical_mean:.2f} m",
        annotation_position="top right",
        row="1",
        col="2",
    )

    fig.update_xaxes(title_text="Distance r [m]", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)

    fig.update_layout(
        height=550,
        template="plotly_white",
        plot_bgcolor="#DFE3E8",
        paper_bgcolor="#F8F9FA",
        dragmode="zoom",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
        margin={"t": 60, "b": 90, "l": 50, "r": 30},
    )

    return fig

def plot_simulated_seed_shadow_and_distance_histogram(
        result: SeedShadowResult, size: tuple[int, int] = (12, 8)
) -> plt.Figure: # type: ignore
    fig, (ax_scatter, ax_distance) = plt.subplots(
            1, 2,
            figsize=size,
            facecolor="#FFFFFF",
            gridspec_kw={"wspace": 0.25},
    )
    fig.suptitle(
        f"Seed shadow and dispersal distance\n(Species={result.species}, Simulations={result.num_instances})",
        fontsize=14,
        fontweight="bold",
        y=0.96,  # Adjust vertical spacing slightly above the plots
    )

    plot_seed_shadow(result=result, ax_scatter=ax_scatter)
    plot_distance_histogram(result=result, ax_distance=ax_distance)

    ax_distance.set_box_aspect(1)

    return fig

def plot_distance_histogram(
        result: SeedShadowResult, ax_distance: plt.Axes | None, size: tuple[int, int] = (12, 12), **style_kwargs # type: ignore
) -> plt.Axes: # type: ignore
    if ax_distance is None:
        _fig, ax_distance = plt.subplots(
            1,
            figsize=size,
            facecolor="#FFFFFF",
        )
        plt.title(f"Dispersal distance histogram for species {result.species}")

    dists = np.hypot(result.xs, result.ys)
    r_bar = result.theoretical_mean_radius
    empirical_mean = float(np.mean(dists))

    # Histogram (replicates go.Histogram)
    sns.histplot(
        x=dists,
        bins=60,
        binrange=(0, dists.max()),
        stat="density",
        color="steelblue",
        alpha=0.6,
        edgecolor="white",  # Optional: gives clean separation like Plotly bars
        linewidth=0.5,
        label="Simulated",
        ax=ax_distance,
    )

    # Theoretical Exponential PDF
    x_grid = np.linspace(0, np.percentile(dists, 99.5), 300)
    pdf = (1.0 / r_bar) * np.exp(-x_grid / r_bar)
    ax_distance.plot(
        x_grid,
        pdf,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Exp PDF (r̄ = {r_bar:.2f} m)",
    )

    # Empirical mean vertical line + annotation (replicates fig.add_vline)
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

    # Axes labels, limits, and legend
    ax_distance.set_xlim(left=0)
    ax_distance.set_xlabel("Distance r [m]",
                           fontweight="bold",
    )
    ax_distance.set_ylabel("Probability Density",
                           fontweight="bold",
    )
    ax_distance.legend()

    return ax_distance

def plot_seed_shadow(
        result: SeedShadowResult, ax_scatter: plt.Axes | None, size: tuple[int, int] = (12, 12), **style_kwargs # type: ignore
) -> plt.Axes: # type: ignore
    style = {
        "s": 3,
        "alpha": 0.15,
        "linewidth": 1,
        "edgecolors": "black",
        "color": "green",
        **style_kwargs
    }
    
    if ax_scatter is None:
        _fig, ax_scatter = plt.subplots(
            1,
            figsize=size,
            facecolor="#FFFFFF",
            gridspec_kw={"wspace": 0.25},
        )
        plt.title(f"Seed shadow for species {result.species}")

    xs, ys = result.xs, result.ys
    max_extent = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05

    ax_scatter.set_facecolor("#FDF1D6")
    sns.scatterplot(x=result.xs,
                    y=result.ys,
                    ax=ax_scatter,
                    label="seeds",
                    **style
    )
    ax_scatter.set_xlim(-max_extent, max_extent)
    ax_scatter.set_ylim(-max_extent, max_extent)
    ax_scatter.set_xlabel("X\n(W <- -> E)",
                          fontweight="bold",
    )
    ax_scatter.set_ylabel("Y\n(S ↓ ↑ N)",
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
    ax_scatter.set_aspect('equal', 'box')

    return ax_scatter