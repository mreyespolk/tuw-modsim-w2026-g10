from dataclasses import dataclass

from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose


@dataclass
class EnvironmentState:
    wind_rose: WindRose = DEFAULT_WIND_ROSE
    wind_strength: int = 1