from dataclasses import dataclass


@dataclass
class TreeConfig:
    species_name: str
    dispersal_type: str
    wind_spreading_factor: float | None
    mean_dispersal_distance: float | None
    fecundity: float
    maturity_age: int
    years_per_height_class: int
    maximal_height_class: int
    mean_life_span: int
    shade_tolerance: float