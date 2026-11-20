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
        grid_max = max(r_bar * 5.0, 1.0)

    x_grid = np.linspace(0.0, max(grid_max, 1e-6), num_points)
    if r_bar <= 0.0:
        return x_grid, np.zeros_like(x_grid)

    pdf = (1.0 / r_bar) * np.exp(-x_grid / r_bar)
    return x_grid, pdf

def compute_directional_transects(
    result: SeedShadowResult,
    wind_rose: WindRose = DEFAULT_WIND_ROSE,
    half_bins: int = 30,
) -> list[TransectSeries]:
    """Splits seeds into 4 bidirectional compass transects and computes signed-distance densities."""
    r_bar = result.theoretical_mean_radius
    if result.is_empty:
        max_dist = max(r_bar * 5.0, 1.0)
    else:
        max_dist = max(float(np.percentile(result.distances, 99.5)), r_bar * 2.0, 1.0)

    # Symmetric edges around 0.0 so positive and negative directions never share a center bin
    bin_edges = np.linspace(-max_dist, max_dist, 2 * half_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])

    if not result.is_empty:
        # Travel bearing clockwise from North (+x = East, +y = North)
        sectors = bearing_to_sector_index(np.rad2deg(np.arctan2(result.xs, result.ys)))
    else:
        sectors = np.array([], dtype=int)

    x_pos = np.linspace(0.0, max_dist, 150)
    x_neg = np.linspace(-max_dist, 0.0, 150)
    total_seeds = result.total_seeds

    # Non-wind dispersal is isotropic: every travel sector is equally likely.
    travel_rose = wind_rose if result.dispersal_type == DispersalType.WIND else WindRose.uniform()

    transects: list[TransectSeries] = []
    for pos_dir, neg_dir in TRANSECT_SPECS:
        p_pos = travel_rose.downwind_probability(pos_dir)
        p_neg = travel_rose.downwind_probability(neg_dir)

        if r_bar > 0.0:
            pdf_pos = (p_pos / r_bar) * np.exp(-x_pos / r_bar)
            pdf_neg = (p_neg / r_bar) * np.exp(-np.abs(x_neg) / r_bar)
        else:
            pdf_pos = np.zeros_like(x_pos)
            pdf_neg = np.zeros_like(x_neg)

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