from dataclasses import dataclass, field

import numpy as np

from tree_propagation.types import DispersalType


@dataclass(frozen=True)
class DispersalRecord:
    """Single step or Monte Carlo instance of seed production and dispersal."""
    step: int
    species: str
    seeds_produced: int
    xs: np.ndarray
    ys: np.ndarray
    distances: np.ndarray

    @property
    def mean_distance(self) -> float:
        return float(np.mean(self.distances)) if self.distances.size > 0 else float("nan")


@dataclass
class DispersalHistory:
    """Opt-in telemetry collector for seed production and spatial dispersal."""
    records: list[DispersalRecord] = field(default_factory=list)

    def record(
        self,
        step: int,
        species: str,
        seeds_produced: int,
        xs: np.ndarray,
        ys: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        self.records.append(
            DispersalRecord(
                step=step,
                species=species,
                seeds_produced=seeds_produced,
                xs=xs,
                ys=ys,
                distances=distances,
            )
        )

    @property
    def seed_counts(self) -> tuple[int, ...]:
        return tuple(r.seeds_produced for r in self.records)

    @property
    def total_seeds(self) -> int:
        return sum(r.seeds_produced for r in self.records)

    @property
    def mean_seeds_per_step(self) -> float:
        return self.total_seeds / len(self.records) if self.records else float("nan")

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        non_empty = [r for r in self.records if r.distances.size > 0]
        if not non_empty:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        return (
            np.concatenate([r.xs for r in non_empty]),
            np.concatenate([r.ys for r in non_empty]),
            np.concatenate([r.distances for r in non_empty]),
        )


@dataclass(frozen=True)
class SeedShadowResult:
    """Raw spatial and count outputs from a seed-shadow Monte Carlo run."""
    species: str
    dispersal_type: DispersalType
    num_instances: int
    xs: np.ndarray
    ys: np.ndarray
    distances: np.ndarray
    seed_counts_per_instance: tuple[int, ...]
    theoretical_mean_radius: float

    @classmethod
    def from_history(
        cls,
        species: str,
        dispersal_type: DispersalType,
        theoretical_mean_radius: float,
        history: DispersalHistory,
    ) -> "SeedShadowResult":
        xs, ys, distances = history.to_arrays()
        return cls(
            species=species,
            dispersal_type=dispersal_type,
            num_instances=len(history.records),
            xs=xs,
            ys=ys,
            distances=distances,
            seed_counts_per_instance=history.seed_counts,
            theoretical_mean_radius=theoretical_mean_radius,
        )

    @property
    def total_seeds(self) -> int:
        return int(self.distances.size)

    @property
    def is_empty(self) -> bool:
        return self.total_seeds == 0


@dataclass(frozen=True)
class SeedShadowMetrics:
    """Goodness-of-fit and summary statistics for a SeedShadowResult."""
    total_seeds: int
    mean_seeds_per_instance: float
    theoretical_mean_radius: float
    empirical_mean_radius: float
    rel_error_pct: float
    ks_statistic: float
    ks_pvalue: float

@dataclass(frozen=True)
class TreeSnapshot:
    """Immutable state of an individual tree."""

    species: str
    age: int
    height_class: int
    x: float
    y: float


@dataclass(frozen=True)
class ForestSnapshot:
    """State of the forest at a particular year."""

    year: int
    trees: tuple[TreeSnapshot, ...]

    @property
    def population(self) -> int:
        return len(self.trees)

    @property
    def species_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for tree in self.trees:
            counts[tree.species] = (
                counts.get(tree.species, 0) + 1
            )

        return counts


@dataclass(frozen=True)
class ForestYearStatistics:
    """Statistics of one annual simulation step."""

    year: int
    seeds_produced: int
    births: int
    deaths: int
    population: int