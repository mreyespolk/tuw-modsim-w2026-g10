from dataclasses import dataclass

import numpy as np

from tree_propagation.models import SeedShadowResult
from tree_propagation.types import DispersalType
from tree_propagation.wind_config import (
    DEFAULT_WIND_ROSE,
    NUM_SECTORS,
    WindDirection,
    WindRose,
    bearing_to_sector_index,
)

# Bidirectional compass transects as (positive_direction, negative_direction) travel sectors.
TRANSECT_SPECS: tuple[tuple[WindDirection, WindDirection], ...] = (
    (WindDirection.N, WindDirection.S),
    (WindDirection.E, WindDirection.W),
    (WindDirection.NE, WindDirection.SW),
    (WindDirection.SE, WindDirection.NW),
)


@dataclass(frozen=True)
class TransectSeries:
    """Binned signed-distance histogram and theoretical density curve for one compass transect."""

    title: str
    pos_label: str
    neg_label: str
    bin_centers: np.ndarray
    bin_width: float
    empirical_density: np.ndarray
    seed_count: int
    x_neg: np.ndarray
    pdf_neg: np.ndarray
    x_pos: np.ndarray
    pdf_pos: np.ndarray


def default_distance_limit(r_bar: float) -> float:
    """Distance-axis upper limit used when there are no seeds to derive it from (5·r̄, at least 1 m)."""
    return max(r_bar * 5.0, 1.0)


def exponential_pdf(x: np.ndarray, r_bar: float, weight: float = 1.0) -> np.ndarray:
    """Density of Exp(r̄) at |x|, scaled by `weight`; all zeros when r̄ <= 0."""
    if r_bar <= 0.0:
        return np.zeros_like(x, dtype=float)
    return (weight / r_bar) * np.exp(-np.abs(x) / r_bar)


def compute_distance_rings(r_bar: float) -> tuple[float, float, float]:
    """Radii (median, mean, 95% quantile) of the Exp(r̄) distance distribution."""
    return r_bar * float(np.log(2.0)), r_bar, r_bar * float(np.log(20.0))


def exp_pdf_label(r_bar: float) -> str:
    """Legend label for the theoretical Exp(r̄) density curve."""
    return f"Exp PDF (r̄ = {r_bar:.2f} m)"


def compass_label_positions(margin: float = 0.03) -> list[tuple[str, tuple[float, float]]]:
    """Returns normalized (x, y) axis-domain coordinates for N, S, E, W labels."""
    return [
        ("N", (0.5, 1.0 - margin)),
        ("S", (0.5, margin)),
        ("E", (1.0 - margin, 0.5)),
        ("W", (margin, 0.5)),
    ]


def compute_plot_extent(result: SeedShadowResult, padding: float = 1.05) -> float:
    """Computes a safe symmetric axis limit, falling back when no seeds exist."""
    fallback = max(result.theoretical_mean_radius * 2.0, 1.0)
    if result.is_empty:
        return fallback
    raw_max = float(max(np.abs(result.xs).max(), np.abs(result.ys).max()))
    return raw_max * padding if raw_max > 0 else fallback


def compute_sector_boundary_endpoints(
    max_extent: float, scale: float = 1.1, num_sectors: int = NUM_SECTORS
) -> list[tuple[float, float]]:
    """Computes (x1, y1) endpoints from (0, 0) for wind-sector boundary rays."""
    step_deg = 360.0 / num_sectors
    half_step_deg = step_deg / 2.0
    radius = max_extent * scale
    endpoints: list[tuple[float, float]] = []
    for boundary_deg in np.arange(half_step_deg, 360.0, step_deg):
        rad = np.deg2rad(boundary_deg)
        endpoints.append((float(radius * np.sin(rad)), float(radius * np.cos(rad))))
    return endpoints


def compute_exponential_pdf_curve(
    result: SeedShadowResult, num_points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Computes (x_grid, pdf) for the theoretical Exp(r_bar) density overlay."""
    r_bar = result.theoretical_mean_radius
    if not result.is_empty:
        grid_max = float(np.percentile(result.distances, 99.5))
    else:
        grid_max = default_distance_limit(r_bar)

    x_grid = np.linspace(0.0, max(grid_max, 1e-6), num_points)
    return x_grid, exponential_pdf(x_grid, r_bar)


def compute_directional_transects(
    result: SeedShadowResult,
    wind_rose: WindRose = DEFAULT_WIND_ROSE,
    half_bins: int = 30,
    max_dist: float | None = None,
) -> list[TransectSeries]:
    """Splits seeds into 4 bidirectional compass transects and computes signed-distance densities."""
    r_bar = result.theoretical_mean_radius
    if max_dist is not None:
        upper_dist = max(max_dist, 1e-6)
    elif result.is_empty:
        upper_dist = default_distance_limit(r_bar)
    else:
        upper_dist = max(float(np.percentile(result.distances, 99.5)), r_bar * 2.0, 1.0)

    # Symmetric edges around 0.0 so positive and negative directions never share a center bin
    bin_edges = np.linspace(-upper_dist, upper_dist, 2 * half_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])

    if not result.is_empty:
        # Travel bearing clockwise from North (+x = East, +y = North)
        sectors = bearing_to_sector_index(np.rad2deg(np.arctan2(result.xs, result.ys)))
    else:
        sectors = np.array([], dtype=int)

    x_pos = np.linspace(0.0, upper_dist, 150)
    x_neg = np.linspace(-upper_dist, 0.0, 150)
    total_seeds = result.total_seeds

    # Non-wind dispersal is isotropic: every travel sector is equally likely.
    travel_rose = wind_rose if result.dispersal_type == DispersalType.WIND else WindRose.uniform()

    transects: list[TransectSeries] = []
    for pos_dir, neg_dir in TRANSECT_SPECS:
        pdf_pos = exponential_pdf(x_pos, r_bar, weight=travel_rose.downwind_probability(pos_dir))
        pdf_neg = exponential_pdf(x_neg, r_bar, weight=travel_rose.downwind_probability(neg_dir))

        if total_seeds > 0:
            pos_dists = result.distances[sectors == pos_dir.sector_index]
            neg_dists = -result.distances[sectors == neg_dir.sector_index]
            signed_dists = np.concatenate([neg_dists, pos_dists])
            counts, _ = np.histogram(signed_dists, bins=bin_edges)
            empirical_density = counts / (total_seeds * bin_width)
            seed_count = int(signed_dists.size)
        else:
            empirical_density = np.zeros_like(bin_centers)
            seed_count = 0

        transects.append(
            TransectSeries(
                title=f"{neg_dir.name} ← 0 → {pos_dir.name}",
                pos_label=pos_dir.name,
                neg_label=neg_dir.name,
                bin_centers=bin_centers,
                bin_width=bin_width,
                empirical_density=empirical_density,
                seed_count=seed_count,
                x_neg=x_neg,
                pdf_neg=pdf_neg,
                x_pos=x_pos,
                pdf_pos=pdf_pos,
            )
        )

    return transects