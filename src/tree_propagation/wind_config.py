"""Compass sectors and the wind rose that drives wind-dispersed seed bearings.

Conventions:
- Bearings are degrees clockwise from North (+y = North, +x = East).
- `WindRose` probabilities denote where wind blows *from*; seeds travel downwind, i.e. towards
  the opposite sector.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np

import tree_propagation.type_validation as type_validation


class WindDirection(Enum):
    """The 8 compass sectors; the value is the sector's center bearing in degrees."""

    N = 0
    NE = 45
    E = 90
    SE = 135
    S = 180
    SW = 225
    W = 270
    NW = 315

    @property
    def opposite(self) -> "WindDirection":
        """The direction 180° away (e.g. N -> S)."""
        return WindDirection((self.value + 180) % 360)

    @property
    def sector_index(self) -> int:
        """Index 0..7 of the sector, clockwise from N (matches `bearing_to_sector_index`)."""
        return round(self.value / SECTOR_WIDTH_DEG)


NUM_SECTORS = len(WindDirection)
SECTOR_WIDTH_DEG = 360.0 / NUM_SECTORS  # 45° for an 8-sector compass


def bearing_to_sector_index(bearings_deg: np.ndarray) -> np.ndarray:
    """Maps bearings (degrees clockwise from North, any real value) to sector indices 0..7.

    Each sector spans ±SECTOR_WIDTH_DEG / 2 around its center, e.g. N covers [-22.5°, 22.5°).
    """
    shifted = (np.asarray(bearings_deg, dtype=float) + SECTOR_WIDTH_DEG / 2.0) % 360.0
    return np.floor(shifted / SECTOR_WIDTH_DEG).astype(int) % NUM_SECTORS


@dataclass(frozen=True)
class WindRose:
    """Probability that the wind blows *from* each of the 8 compass sectors.

    The probabilities are copied, stored in compass order (N, NE, ..., NW) and exposed read-only,
    so a rose cannot be mutated after validation and sampling does not depend on the order in
    which the input mapping was written.
    """

    probabilities: Mapping[WindDirection, float]

    def __post_init__(self) -> None:
        if set(self.probabilities.keys()) != set(WindDirection):
            missing = sorted(d.name for d in set(WindDirection) - set(self.probabilities))
            raise ValueError(f"WindRose needs a probability for each of the 8 directions; missing: {missing}.")
        for direction, prob in self.probabilities.items():
            if not type_validation.is_non_negative_finite(prob):
                raise ValueError(f"Probability for {direction.name} must be a finite number >= 0, got {prob!r}.")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"WindRose probabilities must sum to 1.0, got {total}.")

        ordered = {d: float(self.probabilities[d]) for d in WindDirection}
        object.__setattr__(self, "probabilities", MappingProxyType(ordered))

    def __reduce__(self):
        # MappingProxyType cannot be pickled (needed e.g. by st.cache_data); rebuild from a plain dict.
        return (type(self), (dict(self.probabilities),))

    @classmethod
    def uniform(cls) -> "WindRose":
        """Wind equally likely from every sector (equivalent to isotropic dispersal)."""
        return cls(probabilities={d: 1.0 / NUM_SECTORS for d in WindDirection})

    def downwind_probability(self, direction: WindDirection) -> float:
        """Probability that a wind-dispersed seed travels *towards* `direction`."""
        return self.probabilities[direction.opposite]

    def sample_downwind_radians(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Samples downwind travel bearings (radians clockwise from North).

        A blowing-from sector is drawn by its probability, jittered uniformly within the sector,
        then turned 180° to give the downwind travel bearing.
        """
        base_angles = np.array([d.value for d in self.probabilities], dtype=float)
        probs = np.array(list(self.probabilities.values()), dtype=float)
        half_sector = SECTOR_WIDTH_DEG / 2.0

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
