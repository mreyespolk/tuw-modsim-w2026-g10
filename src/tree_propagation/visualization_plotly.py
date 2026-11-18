from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from tree_propagation.tree_config import SPECIES_TRAIT_LABELS, TreeConfig
from tree_propagation.types import DispersalType
from tree_propagation.visualization_common import (
    compass_label_positions,
    compute_exponential_pdf_curve,
    compute_plot_extent,
    compute_sector_boundary_endpoints,
)
from tree_propagation.wind_config import WindRose

DISPERSAL_COLORS = {
    DispersalType.WIND: "#2B7BBA",
    DispersalType.ANIMAL: "#D95F02",
    DispersalType.GRAVITY: "#7570B3",
}

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

def plot_dispersal_reach_by_height_class(
        species_config: Mapping[str, TreeConfig], wind_strength: int = 10
) -> go.Figure:
    """Line chart comparing theoretical mean dispersal distance r_bar across height classes k=1..K."""
    fig = go.Figure()

    for cfg in sorted(species_config.values(), key=lambda c: c.species_name):
        ks = list(range(1, cfg.maximal_height_class + 1))
        if cfg.dispersal_type == DispersalType.WIND:
            assert cfg.wind_spreading_factor is not None
            r_bars = [wind_strength * cfg.wind_spreading_factor * k for k in ks]
            dash = "solid"
        else:
            assert cfg.mean_dispersal_distance is not None
            r_bars = [cfg.mean_dispersal_distance for _ in ks]
            dash = "dash" if cfg.dispersal_type == DispersalType.ANIMAL else "dot"

        fig.add_trace(
            go.Scatter(
                x=ks,
                y=r_bars,
                mode="lines+markers",
                name=f"{cfg.species_name} ({cfg.dispersal_type.value})",
                line={"color": DISPERSAL_COLORS[cfg.dispersal_type], "width": 2, "dash": dash},
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{cfg.species_name}</b> ({cfg.dispersal_type.value})<br>"
                    "Height class k: %{x}<br>"
                    "Theoretical mean r̄: %{y:.1f} m<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={
            "text": f"Theoretical Mean Dispersal Distance (r̄) by Height Class (Wind Strength s = {wind_strength:g})",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={"title": "Height Class (k)", "dtick": 1},
        yaxis={"title": "Theoretical Mean Dispersal Radius r̄ [m]", "rangemode": "tozero"},
        height=480,
        template="plotly_white",
        margin={"t": 60, "b": 50, "l": 60, "r": 30},
    )
    return fig

def plot_pioneer_climax_tradeoff(traits_df: pd.DataFrame) -> go.Figure:
    """4D scatter comparing shade tolerance, maturity age, peak fecundity, and dispersal mode."""
    fig = go.Figure()

    for dispersal_type in DispersalType:
        sub = traits_df[traits_df["dispersal_type"] == dispersal_type.value]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["shade_tolerance"],
                y=sub["maturity_age"],
                mode="markers+text",
                name=dispersal_type.value.capitalize(),
                text=sub["species_name"],
                textposition="top center",
                textfont={"size": 11},
                marker={
                    "size": 14 + sub["peak_seed_rate"] * 1.8,
                    "color": DISPERSAL_COLORS[dispersal_type],
                    "opacity": 0.85,
                    "line": {"width": 1.5, "color": "white"},
                },
                customdata=sub[["mean_life_span", "maximal_height_class", "peak_seed_rate"]].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Shade tolerance (τ): %{x:.2f}<br>"
                    "Maturity age (A): %{y} yr<br>"
                    "Mean lifespan (L): %{customdata[0]} yr<br>"
                    "Max height class (K): %{customdata[1]}<br>"
                    "Peak seed rate (φ·K): %{customdata[2]:.1f} / yr"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_annotation(
        x=0.03, y=1.04, xref="paper", yref="paper", text="<b>Pioneer / Light-Demanding</b>",
        showarrow=False, font={"size": 11, "color": "dimgray"},
    )
    fig.add_annotation(
        x=0.97, y=1.04, xref="paper", yref="paper", text="<b>Climax / Shade-Tolerant</b>",
        showarrow=False, font={"size": 11, "color": "dimgray"},
    )

    fig.update_layout(
        title={"text": "Successional Strategy: Shade Tolerance vs. Maturity Age", "x": 0.5, "xanchor": "center"},
        xaxis={"title": SPECIES_TRAIT_LABELS["shade_tolerance"], "range": [0.05, 1.0]},
        yaxis={"title": SPECIES_TRAIT_LABELS["maturity_age"], "range": [0, 35]},
        height=480,
        template="plotly_white",
        legend={"title": "Dispersal Type", "orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        margin={"t": 75, "b": 80, "l": 60, "r": 30},
    )
    return fig


def plot_species_lifecycle_timelines(traits_df: pd.DataFrame) -> go.Figure:
    """Horizontal timeline showing juvenile phase, reproductive span, and max-canopy age."""
    ordered = traits_df.sort_values(["mean_life_span", "maturity_age"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=ordered["species_name"],
            x=ordered["maturity_age"],
            name="Juvenile (Non-reproductive)",
            orientation="h",
            marker_color="#B0BEC5",
            hovertemplate="<b>%{y}</b><br>Juvenile phase: 0–%{x} yr<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=ordered["species_name"],
            x=ordered["reproductive_span"],
            name="Reproductive Span (A → L)",
            orientation="h",
            marker_color="#33965B",
            customdata=ordered["mean_life_span"],
            hovertemplate="<b>%{y}</b><br>Reproductive duration: %{x} yr<br>Mean lifespan (L): %{customdata} yr<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=ordered["species_name"],
            x=ordered["time_to_max_height"],
            mode="markers",
            name="Reaches Max Height Class (K · T)",
            marker={"symbol": "diamond", "size": 10, "color": "deeppink", "line": {"width": 1, "color": "black"}},
            customdata=ordered["maximal_height_class"],
            hovertemplate="<b>%{y}</b><br>Reaches K=%{customdata} at %{x} yr<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Species Lifespan, Reproductive Window & Canopy Maturity", "x": 0.5, "xanchor": "center"},
        barmode="stack",
        xaxis_title="Age [years]",
        height=460,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        margin={"t": 60, "b": 80, "l": 100, "r": 30},
    )
    return fig


def plot_species_trait_heatmap(traits_df: pd.DataFrame) -> go.Figure:
    """Normalized multi-trait comparison heatmap with exact raw values annotated in each cell."""
    ordered = traits_df.sort_values("shade_tolerance", ascending=False)
    trait_keys = list(SPECIES_TRAIT_LABELS.keys())
    trait_labels = [SPECIES_TRAIT_LABELS[k] for k in trait_keys]

    raw_matrix = ordered[trait_keys].to_numpy(dtype=float)
    col_min = raw_matrix.min(axis=0)
    col_span = np.where(raw_matrix.max(axis=0) - col_min > 0, raw_matrix.max(axis=0) - col_min, 1.0)
    norm_matrix = (raw_matrix - col_min) / col_span

    text_matrix = np.vectorize(lambda v: f"{v:g}")(np.round(raw_matrix, 2))

    fig = go.Figure(
        go.Heatmap(
            z=norm_matrix,
            x=trait_labels,
            y=ordered["species_name"],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Greens",
            showscale=True,
            colorbar={"title": "Relative<br>Scale", "tickvals": [0, 0.5, 1], "ticktext": ["Min", "Mid", "Max"]},
            hovertemplate="<b>%{y}</b><br>%{x}: <b>%{text}</b><br>Normalized: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Species Ecological Trait Matrix (Sorted by Shade Tolerance)", "x": 0.5, "xanchor": "center"},
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed"},
        height=520,
        template="plotly_white",
        margin={"t": 120, "b": 30, "l": 100, "r": 40},
    )
    return fig


def plot_species_3d_scatter(
    traits_df: pd.DataFrame,
    x_key: str = "shade_tolerance",
    y_key: str = "maturity_age",
    z_key: str = "mean_life_span",
) -> go.Figure:
    """Interactive 3D scatterplot of any three species trait columns, colored by dispersal type."""
    x_label = SPECIES_TRAIT_LABELS[x_key]
    y_label = SPECIES_TRAIT_LABELS[y_key]
    z_label = SPECIES_TRAIT_LABELS[z_key]

    fig = go.Figure()

    for dispersal_type in DispersalType:
        sub = traits_df[traits_df["dispersal_type"] == dispersal_type.value]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=sub[x_key],
                y=sub[y_key],
                z=sub[z_key],
                mode="markers+text",
                name=dispersal_type.value.capitalize(),
                text=sub["species_name"],
                textposition="top center",
                textfont={"size": 10},
                marker={
                    "size": 7,
                    "color": DISPERSAL_COLORS[dispersal_type],
                    "opacity": 0.9,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate=(
                    "<b>%{text}</b> (" + dispersal_type.value + ")<br>"
                    f"{x_label}: %{{x:.2f}}<br>"
                    f"{y_label}: %{{y:.2f}}<br>"
                    f"{z_label}: %{{z:.2f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": f"3D Species Trait Space: {x_label} × {y_label} × {z_label}", "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "zaxis_title": z_label,
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        },
        height=620,
        template="plotly_white",
        legend={"title": "Dispersal Type", "orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"},
        margin={"t": 60, "b": 30, "l": 10, "r": 10},
    )
    return fig