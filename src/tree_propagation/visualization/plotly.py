import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tree_propagation.dispersal import compute_mean_dispersal_distance
from tree_propagation.environment import EnvironmentState
from tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from tree_propagation.species_config import SPECIES_TRAIT_LABELS, SpeciesConfig
from tree_propagation.types import DispersalType
from tree_propagation.visualization.common import (
    compass_label_positions,
    compute_directional_transects,
    compute_distance_rings,
    compute_exponential_pdf_curve,
    compute_plot_extent,
    compute_sector_boundary_endpoints,
    default_distance_limit,
    exp_pdf_label,
    exponential_pdf,
)
from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose


@dataclass(frozen=True)
class DispersalStyle:
    """Consistent per-dispersal-type styling shared by all species charts."""

    color: str
    dash: str


DISPERSAL_STYLES: dict[DispersalType, DispersalStyle] = {
    DispersalType.WIND: DispersalStyle(color="#2B7BBA", dash="solid"),
    DispersalType.GRAVITY: DispersalStyle(color="#7570B3", dash="dot"),
    DispersalType.ANIMAL: DispersalStyle(color="#D95F02", dash="dash"),
}

# Left-to-right column order of the faceted (one subplot per dispersal type) charts.
FACET_ORDER: tuple[DispersalType, ...] = (DispersalType.WIND, DispersalType.GRAVITY, DispersalType.ANIMAL)

GRADUAL_GREENS = [
    [0.00, "#041A0F"],  # deep pine-black (0 m)
    [0.16, "#06311C"],  # dark forest
    [0.32, "#145331"],  # rich emerald
    [0.50, "#33965B"],  # vibrant medium green
    [0.68, "#5BB876"],  # fresh sage
    [0.85, "#68AE6A"],  # light spring green
    [1.00, "#93B87C"],  # pale lime (outer edges)
]

# Seed colors use distance**SEED_COLOR_POWER so the dense near-parent region keeps visible contrast.
SEED_COLOR_POWER = 0.75
SIMULATED_COLOR = "steelblue"
PAPER_BG = "#F8F9FA"
SHADOW_PLOT_BG = "#DFE3E8"

MARKER_SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "cross",
    "x",
    "star",
    "pentagon",
    "triangle-down",
    "hexagon",
)


def _make_dispersal_facets() -> go.Figure:
    """Creates a 1×N subplot grid with one shared-y column per dispersal type in FACET_ORDER."""
    return make_subplots(
        rows=1,
        cols=len(FACET_ORDER),
        shared_yaxes=True,
        subplot_titles=[dtype.value.capitalize() for dtype in FACET_ORDER],
        horizontal_spacing=0.04,
    )


def _facet_column(dispersal_type: DispersalType) -> int:
    """1-based subplot column of a dispersal type in the faceted charts."""
    return FACET_ORDER.index(dispersal_type) + 1


def _facet_marker_symbols(dispersal_types: Iterable[DispersalType]) -> list[str]:
    """Assigns marker symbols that cycle independently within each facet, in input order."""
    seen: Counter[DispersalType] = Counter()
    symbols: list[str] = []
    for dtype in dispersal_types:
        symbols.append(MARKER_SYMBOLS[seen[dtype] % len(MARKER_SYMBOLS)])
        seen[dtype] += 1
    return symbols


def _lifespan_step_trace(
    name: str,
    dispersal_type: DispersalType,
    symbol: str,
    ages: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    hovertemplate: str,
    customdata: Sequence[int] | np.ndarray | None = None,
) -> go.Scatter:
    """Step line over a species' lifespan, styled by dispersal type, with a marker at the final age only."""
    style = DISPERSAL_STYLES[dispersal_type]
    return go.Scatter(
        x=ages,
        y=values,
        customdata=customdata,
        mode="lines+markers",
        name=name,
        line={"color": style.color, "width": 2, "dash": style.dash, "shape": "hv"},
        marker={
            "symbol": symbol,
            "size": [0] * (len(ages) - 1) + [9],
            "line": {"width": 1, "color": "white"},
        },
        hovertemplate=hovertemplate,
    )


def _apply_facet_layout(fig: go.Figure, title: str) -> None:
    """Shared title/legend/size layout for the faceted lifespan charts."""
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        legend={"itemsizing": "constant"},
        height=480,
        template="plotly_white",
        margin={"t": 80, "b": 50, "l": 60, "r": 30},
    )


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


@dataclass(frozen=True)
class _ShadowPanelStyle:
    """Styling that differs between the standalone 2D seed shadow and the combined figure's panel."""

    colorbar: dict
    colorbar_ticks: int
    parent_trace: type[go.Scatter] | type[go.Scattergl]
    parent_name: str
    parent_size: int
    parent_line_width: float
    ray_opacity: float
    # Legend label templates for the (median, mean, 95% quantile) rings, formatted with the radius.
    ring_labels: tuple[str, str, str]
    ring_width: float
    compass_margin: float
    compass_font_size: int
    x_title: str
    y_title: str
    x_axis_extra: dict = field(default_factory=dict)


_COMBINED_PANEL_STYLE = _ShadowPanelStyle(
    colorbar={"title": "Distance [m]", "x": 0.288, "xanchor": "left", "len": 0.85},
    colorbar_ticks=6,
    parent_trace=go.Scattergl,
    parent_name="Parent Tree (0, 0)",
    parent_size=12,
    parent_line_width=1.5,
    ray_opacity=0.6,
    ring_labels=("50% Median ({:.1f} m)", "Theoretical Mean ({:.1f} m)", "95% Quantile ({:.1f} m)"),
    ring_width=1.5,
    compass_margin=0.02,
    compass_font_size=12,
    x_title="x [m]<br>(East →)",
    y_title="y [m]<br>(North ↑)",
    x_axis_extra={"constraintoward": "right"},
)

_STANDALONE_PANEL_STYLE = _ShadowPanelStyle(
    colorbar={"title": "r [m]", "thickness": 12, "len": 0.8},
    colorbar_ticks=5,
    parent_trace=go.Scatter,
    parent_name="Parent (0, 0)",
    parent_size=10,
    parent_line_width=1.3,
    ray_opacity=0.55,
    ring_labels=("50% ({:.1f} m)", "Mean r̄ ({:.1f} m)", "95% ({:.1f} m)"),
    ring_width=1.3,
    compass_margin=0.03,
    compass_font_size=11,
    x_title="x [m] (East →)",
    y_title="y [m] (North ↑)",
)


def _subsample_seeds(result: SeedShadowResult, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evenly thins (xs, ys, distances) to at most `max_points` so browser WebGL stays responsive."""
    xs, ys, dists = result.xs, result.ys, result.distances
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points, dtype=int)
        return xs[idx], ys[idx], dists[idx]
    return xs, ys, dists


def _seed_scatter_trace(
    xs: np.ndarray, ys: np.ndarray, dists: np.ndarray, colorbar: dict, target_ticks: int
) -> go.Scattergl:
    """Seed landing points colored by distance on a power scale, with a colorbar labeled in meters."""
    cmax_m = max(float(np.percentile(dists, 98)), 1e-6)
    tick_vals_m = _nice_colorbar_ticks(cmax_m, target_ticks=target_ticks)
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        name="Seeds",
        marker={
            "size": 4,
            "color": np.clip(dists, 0, cmax_m) ** SEED_COLOR_POWER,
            "cmin": 0,
            "cmax": cmax_m**SEED_COLOR_POWER,
            "colorscale": GRADUAL_GREENS,
            "opacity": 0.60,
            "line": {"width": 0.35, "color": "rgba(13, 66, 40, 0.3)"},
            "colorbar": {
                **colorbar,
                "tickvals": [t**SEED_COLOR_POWER for t in tick_vals_m],
                "ticktext": [f"{t:g}" for t in tick_vals_m],
            },
        },
        customdata=dists,
        hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>r: %{customdata:.2f} m<extra></extra>",
    )


def _add_seed_shadow_panel(
    fig: go.Figure,
    result: SeedShadowResult,
    extent: float,
    max_scatter_points: int,
    style: _ShadowPanelStyle,
    **subplot: int,
) -> None:
    """Draws seeds, parent, wind-sector rays, Exp(r̄) rings, compass labels and square axes.

    `subplot` is `row=..., col=...` when drawing into a subplot grid, and empty for a plain figure.
    """
    r_bar = result.theoretical_mean_radius

    if not result.is_empty:
        xs, ys, dists = _subsample_seeds(result, max_scatter_points)
        fig.add_trace(_seed_scatter_trace(xs, ys, dists, style.colorbar, style.colorbar_ticks), **subplot)

    fig.add_trace(
        style.parent_trace(
            x=[0],
            y=[0],
            mode="markers",
            name=style.parent_name,
            marker={
                "symbol": "x",
                "size": style.parent_size,
                "color": "deeppink",
                "line": {"width": style.parent_line_width, "color": "black"},
            },
            hovertemplate="Parent Tree (0, 0)<extra></extra>",
        ),
        **subplot,
    )

    if result.dispersal_type == DispersalType.WIND:
        for x1, y1 in compute_sector_boundary_endpoints(extent, scale=1.1):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=x1,
                y1=y1,
                line={"color": "gray", "width": 1, "dash": "dash"},
                opacity=style.ray_opacity,
                layer="below",
                **subplot,
            )

    ring_line_styles = (
        ("#D32F2F", "dot", style.ring_width),
        ("red", "solid", 2.0),
        ("#B71C1C", "dash", style.ring_width),
    )
    for radius, label, (color, dash, width) in zip(
        compute_distance_rings(r_bar), style.ring_labels, ring_line_styles, strict=True
    ):
        if radius <= 0.0:
            continue
        line = {"color": color, "width": width, "dash": dash}
        fig.add_shape(type="circle", x0=-radius, y0=-radius, x1=radius, y1=radius, line=line, **subplot)
        # Shapes have no legend entry, so add an invisible line trace for one.
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label.format(radius), line=line), **subplot)

    for label, (lx, ly) in compass_label_positions(margin=style.compass_margin):
        fig.add_annotation(
            x=lx,
            y=ly,
            xref="x domain",
            yref="y domain",
            text=f"<b>{label}</b>",
            showarrow=False,
            font={"color": "dimgray", "size": style.compass_font_size},
        )

    fig.update_xaxes(
        title_text=style.x_title,
        range=[-extent, extent],
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        **style.x_axis_extra,
        **subplot,
    )
    # TO DO: Figure out how we can rotate the y-axis title so it is horizontal
    fig.update_yaxes(title_text=style.y_title, range=[-extent, extent], constrain="domain", **subplot)


def _exp_pdf_trace(x: np.ndarray, pdf: np.ndarray, r_bar: float, **trace_kwargs) -> go.Scatter:
    """Red dashed theoretical Exp(r̄) density curve."""
    return go.Scatter(
        x=x,
        y=pdf,
        mode="lines",
        name=exp_pdf_label(r_bar),
        line={"color": "red", "width": 2, "dash": "dash"},
        **trace_kwargs,
    )


def _density_bar_trace(centers: np.ndarray, density: np.ndarray, bin_width: float, **trace_kwargs) -> go.Bar:
    """Pre-binned simulated density bars."""
    return go.Bar(
        x=centers,
        y=density,
        width=bin_width * 0.92,
        name="Simulated",
        marker_color=SIMULATED_COLOR,
        opacity=0.65,
        **trace_kwargs,
    )


def _add_mean_vline(fig: go.Figure, mean: float, label: str, **subplot: int) -> None:
    """Dotted vertical line marking the simulated mean distance."""
    fig.add_vline(
        x=mean,
        line_width=2,
        line_dash="dot",
        line_color="black",
        annotation_text=f"{label} = {mean:.2f} m",
        annotation_position="top right",
        **subplot,  # type: ignore[arg-type]
    )


def plot_simulated_seed_shadow_and_distance_histogram(
    result: SeedShadowResult, metrics: SeedShadowMetrics, max_scatter_points: int = 100_000
) -> go.Figure:
    """Creates an interactive, zoomable 2D seed shadow scatterplot and distance distribution."""
    title_left = f"{result.species} ({result.dispersal_type}) — 2D Seed Shadow"
    title_right = "Dispersal Distance Distribution"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.32, 0.68],
        subplot_titles=(title_left, title_right),
        horizontal_spacing=0.12,
    )

    _add_seed_shadow_panel(
        fig, result, compute_plot_extent(result), max_scatter_points, _COMBINED_PANEL_STYLE, row=1, col=1
    )

    # Dispersal Distance Histogram vs Theoretical Exponential PDF
    if not result.is_empty:
        fig.add_trace(
            go.Histogram(
                x=result.distances,
                xbins={"start": 0},
                nbinsx=60,
                histnorm="probability density",
                name="Simulated",
                marker_color=SIMULATED_COLOR,
                opacity=0.6,
            ),
            row=1,
            col=2,
        )
        _add_mean_vline(fig, metrics.empirical_mean_radius, "Sample mean", row=1, col=2)

    x_grid, pdf = compute_exponential_pdf_curve(result)
    fig.add_trace(_exp_pdf_trace(x_grid, pdf, result.theoretical_mean_radius), row=1, col=2)

    fig.update_xaxes(title_text="Distance r [m]", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)

    fig.update_annotations(yshift=8, selector={"text": title_left})
    fig.update_annotations(yshift=8, selector={"text": title_right})

    fig.update_layout(
        height=550,
        template="plotly_white",
        plot_bgcolor=SHADOW_PLOT_BG,
        paper_bgcolor=PAPER_BG,
        dragmode="zoom",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
        margin={"t": 70, "b": 90, "l": 50, "r": 30},
    )

    return fig

def plot_seed_shadow_2d(
    result: SeedShadowResult,
    max_extent: float | None = None,
    max_scatter_points: int = 30_000,
) -> go.Figure:
    """Single-panel square 2D seed shadow plot with optional shared axis extent."""
    extent = max_extent if max_extent is not None else compute_plot_extent(result)
    fig = go.Figure()
    _add_seed_shadow_panel(fig, result, extent, max_scatter_points, _STANDALONE_PANEL_STYLE)

    fig.update_layout(
        title={"text": f"{result.species} ({result.dispersal_type.value})", "x": 0.5, "xanchor": "center"},
        height=400,
        template="plotly_white",
        plot_bgcolor=SHADOW_PLOT_BG,
        paper_bgcolor=PAPER_BG,
        dragmode="zoom",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
        margin={"t": 45, "b": 85, "l": 50, "r": 20},
    )
    return fig


def plot_distance_distribution(
    result: SeedShadowResult,
    metrics: SeedShadowMetrics,
    max_dist: float | None = None,
    bins: int = 50,
) -> go.Figure:
    """Single-panel pre-binned dispersal distance histogram + Exp(r̄) PDF overlay."""
    r_bar = result.theoretical_mean_radius
    if max_dist is not None:
        upper_x = max_dist
    elif not result.is_empty:
        upper_x = float(result.distances.max())
    else:
        upper_x = default_distance_limit(r_bar)
    upper_x = max(upper_x, 1e-6)

    fig = go.Figure()

    if not result.is_empty:
        counts, edges = np.histogram(result.distances, bins=bins, range=(0.0, upper_x), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig.add_trace(_density_bar_trace(centers, counts, float(edges[1] - edges[0])))
        _add_mean_vline(fig, metrics.empirical_mean_radius, "Mean")

    x_grid = np.linspace(0.0, upper_x, 300)
    fig.add_trace(_exp_pdf_trace(x_grid, exponential_pdf(x_grid, r_bar), r_bar))

    fig.update_xaxes(title_text="Distance r [m]", range=[0.0, upper_x])
    fig.update_yaxes(title_text="Probability Density")
    fig.update_layout(
        title={"text": "Dispersal Distance Distribution", "x": 0.5, "xanchor": "center"},
        height=300,
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
        margin={"t": 40, "b": 70, "l": 50, "r": 20},
    )
    return fig

def plot_dispersal_reach_by_age(
    species_config: Mapping[str, SpeciesConfig],
    wind_strength: int = 1,
) -> go.Figure:
    """Faceted step chart of the theoretical mean dispersal distance r̄ over each species' lifespan.

    r̄ is computed with the same `compute_mean_dispersal_distance` used by the simulation, so the
    chart cannot drift from the dispersal model.
    """
    environment = EnvironmentState(wind_strength=wind_strength)
    configs = sorted(species_config.values(), key=lambda c: c.species_name)
    symbols = _facet_marker_symbols(cfg.dispersal_type for cfg in configs)

    max_lifespan = 0
    max_r_bar = 0.0
    fig = _make_dispersal_facets()

    for cfg, symbol in zip(configs, symbols, strict=True):
        lifespan = cfg.mean_life_span
        ages = np.arange(0, lifespan + 1)
        height_classes = cfg.height_classes_at_ages(ages)
        r_bars = [
            compute_mean_dispersal_distance(config=cfg, height_class=int(k), environment=environment)
            for k in height_classes
        ]
        max_lifespan = max(max_lifespan, lifespan)
        max_r_bar = max(max_r_bar, *r_bars)

        fig.add_trace(
            _lifespan_step_trace(
                name=cfg.species_name,
                dispersal_type=cfg.dispersal_type,
                symbol=symbol,
                ages=ages,
                values=r_bars,
                customdata=height_classes,
                hovertemplate=(
                    f"<b>{cfg.species_name}</b> ({cfg.dispersal_type.value})<br>"
                    "Age: %{x} yr (Height class k = %{customdata})<br>"
                    f"Mean lifespan: {lifespan} yr<br>"
                    "Theoretical mean r̄: %{y:.1f} m<extra></extra>"
                ),
            ),
            row=1,
            col=_facet_column(cfg.dispersal_type),
        )

    fig.update_xaxes(title_text="Tree Age [years]", range=[0, max_lifespan * 1.04])
    fig.update_yaxes(range=[0, max(max_r_bar, 1.0) * 1.08])
    fig.update_yaxes(title_text="Theoretical Mean Dispersal Radius r̄ [m]", row=1, col=1)
    _apply_facet_layout(
        fig, f"Theoretical Mean Dispersal Distance (r̄) by Tree Age (Wind Strength s = {wind_strength:g})"
    )
    return fig

def plot_pioneer_climax_tradeoff(traits_df: pd.DataFrame) -> go.Figure:
    """4D scatter comparing shade tolerance, maturity age, peak fecundity, and dispersal mode."""
    # Headroom above the oldest maturity age keeps the largest markers and their labels in view.
    max_maturity_age = float(traits_df["maturity_age"].max()) if not traits_df.empty else 0.0
    y_max = max(max_maturity_age * 1.15 + 3.0, 10.0)

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
                    "color": DISPERSAL_STYLES[dispersal_type].color,
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
        # Shade tolerance is validated to [0, 1]; small padding keeps edge markers fully visible.
        xaxis={"title": SPECIES_TRAIT_LABELS["shade_tolerance"], "range": [-0.03, 1.03]},
        yaxis={"title": SPECIES_TRAIT_LABELS["maturity_age"], "range": [0, y_max]},
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
            x=ordered["mean_reproductive_span"],
            name="Reproductive Span (A → L)",
            orientation="h",
            marker_color="#33965B",
            customdata=ordered["mean_life_span"],
            hovertemplate=("<b>%{y}</b><br>Reproductive duration: %{x} yr<br>"
                           "Mean lifespan (L): %{customdata} yr<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=ordered["species_name"],
            x=ordered["years_to_max_height"],
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
                    "color": DISPERSAL_STYLES[dispersal_type].color,
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

def plot_growth_over_time(traits_df: pd.DataFrame) -> go.Figure:
    """Faceted step-line chart comparing discrete height-class trajectories k(a) across species lifespans."""
    ordered = traits_df.sort_values("species_name")
    dispersal_types = [DispersalType(d) for d in ordered["dispersal_type"]]
    symbols = _facet_marker_symbols(dispersal_types)

    max_lifespan = int(traits_df["mean_life_span"].max())
    max_height_class = int(traits_df["maximal_height_class"].max())
    fig = _make_dispersal_facets()

    for row, dtype, symbol in zip(ordered.itertuples(), dispersal_types, symbols, strict=True):
        lifespan = int(row.mean_life_span) # type: ignore
        ages = np.arange(0, lifespan + 1)
        height_classes = SpeciesConfig.height_classes_from_params(
            ages, int(row.years_per_height_class), int(row.maximal_height_class) # type: ignore
        )

        fig.add_trace(
            _lifespan_step_trace(
                name=str(row.species_name),
                dispersal_type=dtype,
                symbol=symbol,
                ages=ages,
                values=height_classes,
                hovertemplate=(
                    f"<b>{row.species_name}</b> ({dtype.value})<br>"
                    "Age: %{x} yr<br>"
                    "Height class k: %{y} "
                    f"(K = {row.maximal_height_class}, T = {row.years_per_height_class} yr)<br>"
                    f"Mean lifespan: {lifespan} yr"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=_facet_column(dtype),
        )

    fig.update_xaxes(title_text="Tree Age [years]", range=[0, max_lifespan * 1.04])
    fig.update_yaxes(dtick=1, range=[0, max_height_class + 0.5])
    fig.update_yaxes(title_text="Height Class (k)", row=1, col=1)
    _apply_facet_layout(fig, "Tree Height Class Growth Over Mean Lifespan")
    return fig

def plot_directional_transects(
    result: SeedShadowResult,
    wind_rose: WindRose = DEFAULT_WIND_ROSE,
    half_bins: int = 30,
    max_dist: float | None = None,
    cols: int = 1,
) -> go.Figure:
    """Creates a faceted signed-distance histogram across the four compass transects."""
    transects = compute_directional_transects(
        result=result, wind_rose=wind_rose, half_bins=half_bins, max_dist=max_dist
    )
    rows = math.ceil(len(transects) / cols)
    is_vertical = cols == 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"{t.title} ({t.seed_count:,})" for t in transects],
        horizontal_spacing=0.06,
        vertical_spacing=0.08 if is_vertical else 0.14,
    )

    canopy_scale = 1.2 if is_vertical else 1.8
    canopy_lobes = (
        (-15 * canopy_scale, 2 * canopy_scale, -8 * canopy_scale, 8 * canopy_scale, "#2E7D32"),
        (-2 * canopy_scale, 15 * canopy_scale, -8 * canopy_scale, 8 * canopy_scale, "#388E3C"),
        (-11 * canopy_scale, 11 * canopy_scale, -3 * canopy_scale, 14 * canopy_scale, "#4CAF50"),
    )

    for idx, t in enumerate(transects):
        row_idx, col_idx = divmod(idx, cols)
        row_idx += 1
        col_idx += 1
        show_legend = idx == 0

        fig.add_trace(
            _density_bar_trace(
                t.bin_centers,
                t.empirical_density,
                t.bin_width,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{t.title}</b><br>"
                    "Signed distance: %{x:.1f} m<br>"
                    "Density: %{y:.5f}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=col_idx,
        )

        for xs_branch, pdf_branch, is_first_branch in (
            (t.x_neg, t.pdf_neg, True),
            (t.x_pos, t.pdf_pos, False),
        ):
            fig.add_trace(
                _exp_pdf_trace(
                    xs_branch,
                    pdf_branch,
                    result.theoretical_mean_radius,
                    showlegend=show_legend and is_first_branch,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=col_idx,
            )

        fig.add_vline(
            x=0,
            y0=0.05,
            y1=1.0,
            line_width=8 if is_vertical else 10,
            line_color="#6D4C41",
            row=row_idx,  # type: ignore[arg-type]
            col=col_idx,  # type: ignore[arg-type]
        )

        subplot_num = idx + 1
        x_ref = "x" if subplot_num == 1 else f"x{subplot_num}"
        y_dom_ref = "y domain" if subplot_num == 1 else f"y{subplot_num} domain"

        for x0, x1, y0, y1, fill_color in canopy_lobes:
            fig.add_shape(
                type="circle",
                xref=x_ref,
                yref=y_dom_ref,
                xsizemode="pixel",
                ysizemode="pixel",
                xanchor=0,
                yanchor=0.87 if is_vertical else 0.89,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=fill_color,
                line={"color": "#1B5E20", "width": 1.2},
            )

    fig.update_xaxes(title_text="Signed Distance [m]", row=rows)
    fig.add_annotation(
        text="Probability Density",
        xref="paper",
        yref="paper",
        x=-0.14 if is_vertical else -0.07,
        y=0.5,
        textangle=-90,
        showarrow=False,
        font={"size": 12},
    )
    fig.update_layout(
        title={
            "text": "Directional Transects (Parent at 0 m)",
            "x": 0.5,
            "xanchor": "center",
        },
        height=640 if is_vertical else 540,
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        bargap=0.05,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.10 if is_vertical else -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
        margin={"t": 55, "b": 70, "l": 55 if is_vertical else 65, "r": 15},
    )
    return fig