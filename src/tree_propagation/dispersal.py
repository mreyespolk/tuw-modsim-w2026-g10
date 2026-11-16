import numpy as np

from src.tree_propagation.environment import EnvironmentState
from src.tree_propagation.tree_config import TreeConfig
from src.tree_propagation.types import DispersalType
from src.tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose


def empty_dispersal_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )


def compute_mean_dispersal_distance(
    config: TreeConfig,
    height_class: int,
    environment: EnvironmentState,
) -> float:
    """Computes theoretical mean dispersal distance (r_bar) in meters."""
    if height_class < 0:
        raise ValueError("height_class must be non-negative.")

    match config.dispersal_type:
        case DispersalType.WIND:
            if config.wind_spreading_factor is None:
                raise ValueError(f"Species '{config.species_name}' is missing wind_spreading_factor.")
            return environment.wind_strength * config.wind_spreading_factor * height_class
        case DispersalType.ANIMAL | DispersalType.GRAVITY:
            if config.mean_dispersal_distance is None:
                raise ValueError(f"Species '{config.species_name}' is missing mean_dispersal_distance.")
            return config.mean_dispersal_distance
        case _:
            raise ValueError(f"Unsupported dispersal type: {config.dispersal_type}")


def sample_dispersal_offsets(
    count: int,
    mean_distance: float,
    dispersal_type: DispersalType,
    rng: np.random.Generator,
    wind_rose: WindRose = DEFAULT_WIND_ROSE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples (dx, dy, distances) offsets relative to a parent tree."""
    if count <= 0:
        return empty_dispersal_result()
    if mean_distance < 0:
        raise ValueError("mean_distance must be non-negative.")

    distances = rng.exponential(scale=mean_distance, size=count)

    if dispersal_type == DispersalType.WIND:
        bearings_rad = wind_rose.sample_downwind_radians(count=count, rng=rng)
    else:
        bearings_rad = rng.uniform(0.0, 2.0 * np.pi, size=count)

    # Bearings are measured clockwise from North (+y = North, +x = East)
    dx = distances * np.sin(bearings_rad)
    dy = distances * np.cos(bearings_rad)
    return dx, dy, distances