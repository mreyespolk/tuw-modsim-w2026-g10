import csv
from dataclasses import dataclass
from pathlib import Path

from src.tree_propagation.types import DispersalType

SPECIES_TRAIT_LABELS = {
    "shade_tolerance": "Shade Tolerance τ [0–1]",
    "maturity_age": "Maturity Age A [yr]",
    "mean_life_span": "Mean Lifespan L [yr]",
    "fecundity": "Fecundity φ",
    "years_per_height_class": "Years / Height Class T [yr]",
    "maximal_height_class": "Max Height Class K",
    "reproductive_span": "Reproductive Span (L − A) [yr]",
    "time_to_max_height": "Age at Max Height (K · T) [yr]",
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

    def __post_init__(self):
        if (self.dispersal_type == DispersalType.WIND and 
            (
                self.wind_spreading_factor is None or
                self.wind_spreading_factor <= 0
            )
        ):
            raise ValueError("WIND dispersal requires positive 'wind_spreading_factor'.")
        if (
            (
                self.dispersal_type == DispersalType.ANIMAL or
                 self.dispersal_type == DispersalType.GRAVITY
            ) and
            (
                self.mean_dispersal_distance is None or
                self.mean_dispersal_distance <= 0
            )
        ):
            raise ValueError("ANIMAL and GRAVITY dispersal require positive 'mean_dispersal_distance'.")

    @property
    def reproductive_span(self) -> int:
        return max(0, self.mean_life_span - self.maturity_age)

    @property
    def time_to_max_height(self) -> int:
        return self.maximal_height_class * self.years_per_height_class

    @property
    def peak_seed_rate(self) -> float:
        return self.fecundity * self.maximal_height_class



def load_tree_configs(csv_path: str | Path) -> dict[str, TreeConfig]:
    """Loads tree species configurations from a CSV file, keyed by species name."""
    configs: dict[str, TreeConfig] = {}

    with Path(csv_path).open(mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row["species"]
            configs[species] = TreeConfig(
                species_name=species,
                dispersal_type=DispersalType(row["dispersal"]),
                wind_spreading_factor=float(row["f"]) if row["f"] else None,
                mean_dispersal_distance=float(row["d_m"]) if row["d_m"] else None,
                fecundity=float(row["phi"]),
                maturity_age=int(row["A_yr"]),
                years_per_height_class=int(row["T_yr"]),
                maximal_height_class=int(row["K"]),
                mean_life_span=int(row["L_yr"]),
                shade_tolerance=float(row["tau"]),
            )

    return configs