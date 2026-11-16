from dataclasses import dataclass

import numpy as np

from src.tree_propagation.types import DispersalType


@dataclass(frozen=True)
class SeedShadowResult:
    """Raw spatial and count outputs from a seed-shadow Monte Carlo run."""
    species: str
    dispersal_type: DispersalType
    num_instances: int
    xs: np.ndarray
    ys: np.ndarray
    distances: np.ndarray
    seed_counts_per_instance: tuple
    theoretical_mean_radius: float

    @property
    def total_seeds(self) -> int:
        return int(self.distances.size)

    @property
    def is_empty(self) -> bool:
        return self.total_seeds == 0

    @property
    def theoretical_r_bar(self):
        return self.theoretical_mean_radius


@dataclass(frozen=True)
class SeedShadowMetrics:
    """Goodness-of-fit and summary statistics for a SeedShadowResult."""
    total_seeds: int
    mean_seeds_per_instance: float
    theoretical_mean_radius: float
    empirical_r_bar: float
    rel_error_pct: float
    ks_statistic: float
    ks_pvalue: float