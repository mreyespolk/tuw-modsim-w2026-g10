import math
from dataclasses import dataclass
from enum import Enum


class WindDirection(Enum):
    N = 0
    NE = 45
    E = 90
    SE = 135
    S = 180
    SW = 225
    W = 270
    NW = 315

@dataclass(frozen=True)
class WindRose:
    probabilities: dict[WindDirection, float]

    def __post_init__(self) -> None:
        if any(p < 0.0 for p in self.probabilities.values()):
            raise ValueError("Probabilities must always be greater than or equal to 0.")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"WindRose probabilities must sum to 1.0, got {total}")
        if len(self.probabilities) != 8:
            raise ValueError(f"Wind Rose must have probabilities for 8 cardinal directions. {WindDirection}")
        


DEFAULT_WIND_ROSE = WindRose(probabilities={
    WindDirection.N: 0.06,
    WindDirection.NE: 0.05,
    WindDirection.E: 0.06,
    WindDirection.SE: 0.18,
    WindDirection.S: 0.07,
    WindDirection.SW: 0.06,
    WindDirection.W: 0.30,
    WindDirection.NW: 0.22,
})
