import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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
    r_bar = result.theoretical_r_bar

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

    # Interactive WebGL Seed Scatterplot
    fig.add_trace(
        go.Scattergl(
            x=plot_xs,
            y=plot_ys,
            mode="markers",
            name="Seeds",
            marker={
                "size": 4,
                "color": plot_dists,
                "colorscale": "Plasma_r",
                #"colorscale": "Viridis_r",
                "reversescale": True,
                "opacity": 0.65,
                "colorbar": {"title": "Distance [m]", "x": 0.44, "len": 0.85},
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
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Distance r [m]", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)

    fig.update_layout(
        height=550,
        template="plotly_white",
        dragmode="zoom",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
        margin={"t": 60, "b": 90, "l": 50, "r": 30},
    )

    return fig

#def plot_seed_shadow_and_distance_static(result: SeedShadowResult) -> plt.Figure:
#    """Creates a side-by-side 2D seed shadow (log-scaled hexbin) and exponential distance check."""
#    fig, (ax_map, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))
#    
#
#    # 2D Seed Shadow with log color scale so outer plumes are clearly visible
#    hb = ax_map.hexbin(
#        result.xs,
#        result.ys,
#        gridsize=100,
#        #cmap="YlGnBu",
#        #cmap="plasma",
#        cmap=_truncated_cmap("Greens", minval=0.25, maxval=1.10),
#        bins="log",
#        mincnt=1,
#    )
#    ax_map.scatter([0], [0], color="pink", marker="X", s=10, label="Parent Tree (0, 0)", zorder=5)
#
#    # Center (0, 0) symmetrically so visual orientation isn't skewed
#    max_extent = max(np.abs(result.xs).max(), np.abs(result.ys).max()) * 1.05
#    if result.dispersal_type == DispersalType.WIND:
#        for boundary_deg in np.arange(22.5, 360.0, 45.0):
#            rad = np.deg2rad(boundary_deg)
#            # Extend slightly past max_extent so rays cleanly reach the plot frame
#            x_end = max_extent * 1.5 * np.sin(rad)
#            y_end = max_extent * 1.5 * np.cos(rad)
#            ax_map.plot(
#                [0, x_end],
#                [0, y_end],
#                color="gray",
#                linestyle="--",
#                linewidth=1.0,
#                alpha=0.75,
#                zorder=3,
#            )
#    ax_map.set_xlim(-max_extent, max_extent)
#    ax_map.set_ylim(-max_extent, max_extent)
#    ax_map.set_aspect("equal", adjustable="box")
#
#    mean_ring = plt.Circle(
#        (0, 0),
#        result.theoretical_r_bar,
#        color="red",
#        fill=False,
#        linestyle="-",
#        linewidth=2.0,
#        alpha=0.9,
#        zorder=4,
#        label=f"Theoretical Mean ({result.theoretical_r_bar:.2f}m)"
#    )
#    ax_map.add_patch(mean_ring)
#
#    # Add explicit compass markers (North = Up)
#    #ax_map.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
#    #ax_map.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
#    for label, (lx, ly) in [("N", (0.5, 0.96)), ("S", (0.5, 0.04)), ("E", (0.96, 0.5)), ("W", (0.04, 0.5))]:
#        ax_map.text(
#            lx, ly, label, transform=ax_map.transAxes,
#            ha="center", va="center", fontweight="bold", color="dimgray",
#        )
#
#    ax_map.set_title(f"{result.species} ({result.dispersal_type}) — 2D Seed Shadow")
#    ax_map.set_xlabel("x [m]\n(East →)")
#    ax_map.set_ylabel("y [m]\n(North ↑)", rotation=0, labelpad=30)
#    ax_map.legend(loc="upper right")
#    fig.colorbar(hb, ax=ax_map, label="Seed count (log10)")
#
#    # Distance Histogram vs Theoretical Exponential PDF
#    dists = result.distances
#    r_bar = result.theoretical_r_bar
#    ax_dist.hist(dists, bins=60, density=True, alpha=0.6, color="steelblue", label="Simulated")
#
#    x_grid = np.linspace(0, np.percentile(dists, 99.5), 300)
#    pdf = (1.0 / r_bar) * np.exp(-x_grid / r_bar)
#    ax_dist.plot(x_grid, pdf, "r--", lw=2, label=f"Exp PDF ($\\bar{{r}}={r_bar:.2f}$ m)")
#    ax_dist.axvline(
#        result.empirical_r_bar, color="black", linestyle=":",
#        label=f"Sample mean = {result.empirical_r_bar:.2f} m",
#    )
#
#    ax_dist.set_title("Dispersal Distance Distribution")
#    ax_dist.set_xlabel("Distance r [m]")
#    ax_dist.set_ylabel("Probability Density")
#    ax_dist.legend()
#
#    plt.tight_layout()
#    plt.close(fig)
#    return fig