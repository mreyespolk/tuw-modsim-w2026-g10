import numpy as np

from src.tree_propagation.dispersal import DispersalStrategy, get_dispersal_strategy
from src.tree_propagation.tree_config import TreeConfig
from src.tree_propagation.types import DispersalType


class TreeAgent:
    def __init__(
            self,
            species_config: TreeConfig,
            age: int = 0,
            height_class: int = 0,
            x: int = 0,
            y: int = 0,
            
    ) -> None:
        self.species_config = species_config
        self.age = age
        self.height_class = height_class
        self.x = x
        self.y = y

        self.seeds = 0
        self.seed_production_history = []

    def __post_init__(self):
        if self.species_config.dispersal_type == DispersalType.WIND:
            assert self.species_config.wind_spreading_factor is not None
            assert self.species_config.wind_spreading_factor > 0

    def __repr__(self):
        return f"TreeAgent(species={self.species_config.species_name}, age={self.age}, height_class={self.height_class}, seeds={self.seeds})"

    @property
    def is_mature(self):
        return self.age >= self.species_config.maturity_age

    def grow(self):
        raise NotImplementedError

    def die(self, rng: np.random.Generator):
        raise NotImplementedError

    def produce_seeds(self, rng: np.random.Generator) -> int:
        if self.age >= self.species_config.maturity_age:
            self.seeds = int(
                rng.poisson(self.species_config.fecundity * self.height_class)
            )
        else:
            self.seeds = 0

        self.seed_production_history.append(self.seeds)
        return self.seeds

    def disperse_seeds(
        self,
        rng: np.random.Generator,
        strategy: DispersalStrategy | None = None,
        wind_strength: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Disperses all seeds currently held by this tree and resets its seed stock."""
        if self.seeds <= 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty

        active_strategy = strategy or get_dispersal_strategy(self.species_config)
        dx, dy, distances = active_strategy.sample(
            count=self.seeds,
            config=self.species_config,
            height_class=self.height_class,
            rng=rng,
            wind_strength=wind_strength,
        )
        self.seeds = 0
        return self.x + dx, self.y + dy, distances