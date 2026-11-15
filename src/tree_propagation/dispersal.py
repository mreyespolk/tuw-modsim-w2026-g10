from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.tree_propagation.tree_config import TreeConfig
from src.tree_propagation.types import DispersalType
from src.tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose


class DispersalStrategy(Protocol):
    """Defines how seeds disperse spatially from a parent tree."""

    def mean_distance(
        self,
        config: TreeConfig,
        height_class: int,
        wind_strength: int | None = None,
    ) -> float:
        """Returns the theoretical mean dispersal distance (r_bar) in meters."""
        ...

    def sample(
        self,
        count: int,
        config: TreeConfig,
        height_class: int,
        rng: np.random.Generator,
        wind_strength: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples (dx, dy, distances) offsets relative to the parent tree."""
        ...


def _empty_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.array([], dtype=float)
    return empty, empty, empty


@dataclass(frozen=True)
class WindDispersal:
    """Anisotropic exponential dispersal driven by a directional wind rose."""

    wind_rose: WindRose = DEFAULT_WIND_ROSE
    _base_angles: np.ndarray = field(init=False, repr=False)
    _probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Precompute arrays once so repeated sample() calls avoid dict iteration
        angles = np.array([d.value for d in self.wind_rose.probabilities], dtype=float)
        probs = np.array(list(self.wind_rose.probabilities.values()), dtype=float)
        object.__setattr__(self, "_base_angles", angles)
        object.__setattr__(self, "_probs", probs)

    def mean_distance(
        self,
        config: TreeConfig,
        height_class: int,
        wind_strength: int | None = None,
    ) -> float:
        if wind_strength is None:
            raise ValueError("wind_strength must be provided for wind dispersal")
        if config.wind_spreading_factor is None:
            raise ValueError(f"Species '{config.species_name}' is missing wind_spreading_factor")
        return wind_strength * config.wind_spreading_factor * height_class

    def sample(
        self,
        count: int,
        config: TreeConfig,
        height_class: int,
        rng: np.random.Generator,
        wind_strength: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if count <= 0:
            return _empty_result()

        r_bar = self.mean_distance(config, height_class, wind_strength)
        distances = rng.exponential(scale=r_bar, size=count)

        # Wind rose angles denote where wind blows *from* (0° = North, clockwise).
        # Adding 180° converts to downwind travel bearing.
        from_angles = rng.choice(self._base_angles, size=count, p=self._probs)
        jitter = rng.uniform(-22.5, 22.5, size=count)
        fly_rad = np.deg2rad((from_angles + jitter + 180.0) % 360.0)

        dx = distances * np.sin(fly_rad)
        dy = distances * np.cos(fly_rad)
        return dx, dy, distances


@dataclass(frozen=True)
class NeutralDispersal:
    """Radially symmetric exponential dispersal (e.g., gravity or generalist animal)."""

    def mean_distance(
        self,
        config: TreeConfig,
        height_class: int,
        wind_strength: int | None = None,
    ) -> float:
        if config.mean_dispersal_distance is None:
            raise ValueError(f"Species '{config.species_name}' is missing mean_dispersal_distance")
        return config.mean_dispersal_distance

    def sample(
        self,
        count: int,
        config: TreeConfig,
        height_class: int,
        rng: np.random.Generator,
        wind_strength: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if count <= 0:
            return _empty_result()

        r_bar = self.mean_distance(config, height_class, wind_strength)
        distances = rng.exponential(scale=r_bar, size=count)

        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        dx = distances * np.cos(angles)
        dy = distances * np.sin(angles)
        return dx, dy, distances


def get_dispersal_strategy(
    config: TreeConfig,
    wind_rose: WindRose = DEFAULT_WIND_ROSE,
) -> DispersalStrategy:
    """Resolves the appropriate dispersal strategy for a species configuration."""
    match config.dispersal_type:
        case DispersalType.WIND:
            return WindDispersal(wind_rose=wind_rose)
        case DispersalType.ANIMAL | DispersalType.GRAVITY:
            return NeutralDispersal()
        case _:
            raise ValueError(f"Unsupported dispersal type: {config.dispersal_type}")