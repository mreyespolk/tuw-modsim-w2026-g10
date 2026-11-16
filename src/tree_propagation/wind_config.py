import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


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

    def sample_downwind_radians(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Samples downwind travel bearings (radians clockwise from North)."""
        directions = list(self.probabilities.keys())
        base_angles = np.array([d.value for d in directions], dtype=float)
        probs = np.array([self.probabilities[d] for d in directions], dtype=float)
        half_sector = 180.0 / len(directions)  # 22.5° for an 8-sector compass

        # Wind rose angles denote where wind blows *from*; add 180° for downwind travel
        from_angles = rng.choice(base_angles, size=count, p=probs)
        jitter = rng.uniform(-half_sector, half_sector, size=count)
        return np.deg2rad((from_angles + jitter + 180.0) % 360.0)
        


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
