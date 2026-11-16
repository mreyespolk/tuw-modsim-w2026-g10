import csv
from dataclasses import dataclass
from pathlib import Path

from src.tree_propagation.types import DispersalType


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
                DispersalType.GRAVITY
            ) and
            (
                self.mean_dispersal_distance is None
            )
        ):
            raise ValueError("ANIMAL and GRAVITY dispersal require positive 'mean_dispersal_distance'.")



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