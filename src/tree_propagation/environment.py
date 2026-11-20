from dataclasses import dataclass

import tree_propagation.type_validation as type_validation
from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose

DEFAULT_WIND_STRENGTH = 10


@dataclass(frozen=True)
class EnvironmentState:
    wind_rose: WindRose = DEFAULT_WIND_ROSE
    wind_strength: int = DEFAULT_WIND_STRENGTH

    def __post_init__(self):
        if not isinstance(self.wind_rose, WindRose):
            raise TypeError("'wind_rose' must be a WindRose instance.")
        if not (type_validation.is_strict_int(self.wind_strength) and self.wind_strength >= 1):
            raise ValueError("'wind_strength' must be a positive integer.")