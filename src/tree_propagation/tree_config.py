import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tree_propagation.type_validation as type_validation
from tree_propagation.types import DispersalType

SPECIES_TRAIT_LABELS = {
    "shade_tolerance": "Shade Tolerance τ [0–1]",
    "maturity_age": "Maturity Age A [yr]",
    "mean_life_span": "Mean Lifespan L [yr]",
    "fecundity": "Fecundity φ",
    "years_per_height_class": "Years / Height Class T [yr]",
    "maximal_height_class": "Max Height Class K",
    "mean_reproductive_span": "Reproductive Span (L − A) [yr]",
    "years_to_max_height": "Age at Max Height (K · T) [yr]",
    "peak_seed_rate": "Peak Seed Rate (φ · K) [/yr]",
    "mean_dispersal_radius": "Mean Dispersal Radius r̄ [m]",
}


@dataclass(frozen=True)
class TreeConfig:
    species_name: str
    dispersal_type: DispersalType
    wind_spreading_factor: float | None
    mean_dispersal_distance: float | None
    fecundity: float
    maturity_age: int
    years_per_height_class: int
    maximal_height_class: int
    mean_life_span: int
    shade_tolerance: float

    def __post_init__(self) -> None:
        # String & Enum checks
        if not isinstance(self.species_name, str) or not self.species_name.strip():
            raise ValueError("'species_name' must be a non-empty string.")
        if not isinstance(self.dispersal_type, DispersalType):
            raise TypeError("'dispersal_type' must be a DispersalType enum member.")

        # Dispersal-specific checks
        if self.dispersal_type == DispersalType.WIND:
            if not type_validation.is_positive_finite(self.wind_spreading_factor):
                raise ValueError("WIND dispersal requires a positive 'wind_spreading_factor'.")
        elif self.dispersal_type in (DispersalType.ANIMAL, DispersalType.GRAVITY):
            if not type_validation.is_positive_finite(self.mean_dispersal_distance):
                raise ValueError(
                    f"{self.dispersal_type.name} dispersal requires a positive 'mean_dispersal_distance'."
                )

        # Trait bounds & types
        if not type_validation.is_non_negative_finite(self.fecundity):
            raise ValueError("'fecundity' must be a non-negative finite number.")
        if not (isinstance(self.shade_tolerance, (int, float)) and 0.0 <= self.shade_tolerance <= 1.0):
            raise ValueError("'shade_tolerance' must be between 0 and 1 inclusive.")
        if not type_validation.is_strict_int(self.maturity_age) or self.maturity_age < 0:
            raise ValueError("'maturity_age' must be a non-negative integer.")
        if not type_validation.is_strict_int(self.years_per_height_class) or self.years_per_height_class < 1:
            raise ValueError("'years_per_height_class' must be >= 1.")
        if not type_validation.is_strict_int(self.maximal_height_class) or self.maximal_height_class < 1:
            raise ValueError("'maximal_height_class' must be >= 1.")
        if not type_validation.is_strict_int(self.mean_life_span) or self.mean_life_span < 1:
            raise ValueError("'mean_life_span' must be >= 1.")

        # Cross-field relationships
        if self.mean_life_span <= self.maturity_age:
            raise ValueError(
                f"'mean_life_span' ({self.mean_life_span}) must be greater than "
                f"'maturity_age' ({self.maturity_age})."
            )

    @property
    def mean_reproductive_span(self) -> int:
        """Mean duration in years of the species reproductive phase."""
        return self.mean_life_span - self.maturity_age
    
    @property
    def mean_reproductive_span_ratio(self) -> float:
        """Proportion of the species mean life span which it reproduces."""
        return self.mean_reproductive_span / self.mean_life_span

    @property
    def years_to_max_height(self) -> int:
        """Time in years it takes for a species to reach its maximum heightclass."""
        return self.maximal_height_class * self.years_per_height_class

    @property
    def peak_seed_rate(self) -> float:
        """Peak seed production for the species."""
        return self.fecundity * self.maximal_height_class

    def height_class_at_age(self, age: int) -> int:
        """Height class reached at a given age: k = min(K, ⌊age / T⌋)."""
        return min(age // self.years_per_height_class, self.maximal_height_class)

    def height_classes_at_ages(self, ages: np.ndarray) -> np.ndarray:
        return self.height_classes_from_params(
            ages, self.years_per_height_class, self.maximal_height_class
        )

    @staticmethod
    def height_classes_from_params(
            ages: np.ndarray, years_per_height_class: int, maximal_height_class: int
    ) -> np.ndarray:
        return np.minimum(ages // years_per_height_class, maximal_height_class)

    @classmethod
    def from_dict(cls, config_dict: dict[str, str]) -> "TreeConfig":
        """Constructs a TreeConfig from a dictionary."""
        return cls(
            species_name=config_dict["species"].strip(),
            dispersal_type=DispersalType(config_dict["dispersal"].strip()),
            wind_spreading_factor=float(config_dict["f"]) if config_dict["f"] else None,
            mean_dispersal_distance=float(config_dict["d_m"]) if config_dict["d_m"] else None,
            fecundity=float(config_dict["phi"]),
            maturity_age=int(config_dict["A_yr"]),
            years_per_height_class=int(config_dict["T_yr"]),
            maximal_height_class=int(config_dict["K"]),
            mean_life_span=int(config_dict["L_yr"]),
            shade_tolerance=float(config_dict["tau"]),
        )


def load_tree_configs_from_csv(csv_path: str | Path) -> dict[str, TreeConfig]:
    """Loads tree species configurations from a CSV file, keyed by species name."""
    configs: dict[str, TreeConfig] = {}

    with Path(csv_path).open(mode="r", encoding="utf-8", newline="") as f:
        for line_num, row in enumerate(csv.DictReader(f), start=2):
            try:
                config = TreeConfig.from_dict(config_dict=row)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid config in {csv_path} at line {line_num}: {exc}") from exc

            if config.species_name in configs:
                raise ValueError(f"Duplicate species '{config.species_name}' at line {line_num}")

            configs[config.species_name] = config

    return configs