import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from src.tree_propagation.types import DispersalType
from src.tree_propagation.visualization_common import (
    compass_label_positions,
    compute_exponential_pdf_curve,
    compute_plot_extent,
    compute_sector_boundary_endpoints,
)
from src.tree_propagation.wind_config import WindRose

GRADUAL_GREENS = [
    [0.00, "#041A0F"],  # deep pine-black (0 m)
    [0.16, "#06311C"],  # dark forest
    [0.32, "#145331"],  # rich emerald
    [0.50, "#33965B"],  # vibrant medium green
    [0.68, "#5BB876"],  # fresh sage
    [0.85, "#68AE6A"],  # light spring green
    [1.00, "#93B87C"],  # pale lime (outer edges)
]


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


def plot_wind_rose(wind_rose: WindRose) -> go.Figure:
    """Creates a polar wedge chart showing incoming wind probabilities by compass sector."""
    directions = [d.name for d in wind_rose.probabilities]
    angles = [d.value for d in wind_rose.probabilities]
    probs = list(wind_rose.probabilities.values())
    max_prob = max(probs)
    sector_width = 360.0 / len(angles)

    fig = go.Figure(
        go.Barpolar(
            r=probs,
            theta=angles,
            width=[sector_width] * len(angles),
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
                "tickvals": angles,
                "ticktext": directions,
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


def plot_simulated_seed_shadow_and_distance_histogram(
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
    max_extent = compute_plot_extent(result)

    if not result.is_empty:
        # Subsample only if total seeds exceed max_scatter_points so browser WebGL stays responsive
        if len(xs) > max_scatter_points:
            idx = np.linspace(0, len(xs) - 1, max_scatter_points, dtype=int)
            plot_xs, plot_ys, plot_dists = xs[idx], ys[idx], dists[idx]
        else:
            plot_xs, plot_ys, plot_dists = xs, ys, dists

        power = 0.75
        cmax_m = max(float(np.percentile(plot_dists, 98)), 1e-6)
        color_vals = np.clip(plot_dists, 0, cmax_m) ** power
        tick_vals_m = _nice_colorbar_ticks(cmax_m)

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
                    "colorscale": GRADUAL_GREENS,
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

    # Wind sector boundary rays
    if result.dispersal_type == DispersalType.WIND:
        for x1, y1 in compute_sector_boundary_endpoints(max_extent, scale=1.1):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=x1,
                y1=y1,
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

    for label, (lx, ly) in compass_label_positions(margin=0.02):
        fig.add_annotation(
            x=lx,
            y=ly,
            xref="x domain",
            yref="y domain",
            text=f"<b>{label}</b>",
            showarrow=False,
            font={"color": "dimgray", "size": 12},
        )

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
    if not result.is_empty:
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

        empirical_mean = metrics.empirical_mean_radius
        fig.add_vline(
            x=empirical_mean,
            line_width=2,
            line_dash="dot",
            line_color="black",
            annotation_text=f"Sample mean = {empirical_mean:.2f} m",
            annotation_position="top right",
            row=1,  # type: ignore
            col=2,  # type: ignore
        )

    x_grid, pdf = compute_exponential_pdf_curve(result)
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