from dataclasses import dataclass

import tree_propagation.type_validation as type_validation
from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose

DEFAULT_WIND_STRENGTH = 10
DEFAULT_ESTABLISHMENT_PROBABILITY = 0.5
DEFAULT_SHADE_MORTALITY = 0.3

@dataclass(frozen=True)
class EnvironmentState:
    wind_rose: WindRose = DEFAULT_WIND_ROSE
    wind_strength: int = DEFAULT_WIND_STRENGTH

    establishment_probability: float = DEFAULT_ESTABLISHMENT_PROBABILITY
    shade_mortality: float = DEFAULT_SHADE_MORTALITY

    def __post_init__(self):
        if not isinstance(self.wind_rose, WindRose):
            raise TypeError("'wind_rose' must be a WindRose instance.")
        if not (type_validation.is_strict_int(self.wind_strength) and self.wind_strength >= 1):
            raise ValueError("'wind_strength' must be a positive integer.")

        for name, value in (
                ("establishment_probability", self.establishment_probability),
                ("shade_mortality", self.shade_mortality),
        ):
            if not (
                    type_validation.is_finite_real(value)
                    and 0 <= value <= 1
            ):
                raise ValueError(f"'{name}' must be between 0 and 1.")


DEFAULT_ENVIRONMENT_STATE = EnvironmentState()