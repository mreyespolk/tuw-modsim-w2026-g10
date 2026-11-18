import numpy as np

from tree_propagation.models import SeedShadowResult

COMPASS_DIRECTIONS: tuple[str, ...] = ("N", "S", "E", "W")


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
    max_extent: float, scale: float = 1.1, num_sectors: int = 8
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