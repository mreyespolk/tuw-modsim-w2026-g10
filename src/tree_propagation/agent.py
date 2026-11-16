import numpy as np

from src.tree_propagation.dispersal import DispersalStrategy, get_dispersal_strategy
from src.tree_propagation.tree_config import TreeConfig


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

        self._validate_init_params()

    def _validate_init_params(self):
        if (
            (self.age is None or self.age < 0) or 
            (self.height_class is None or self.height_class < 0)
        ):
            raise ValueError("Age and Height class must be non negative.")
        

    def __repr__(self):
        return f"TreeAgent(species={self.species_config.species_name}, age={self.age}, height_class={self.height_class}, seeds={self.seeds})"

    @classmethod
    def from_height_class(cls, species_config: TreeConfig, height_class: int, x: int = 0, y: int = 0) -> "TreeAgent":
        estimated_age = height_class * species_config.years_per_height_class
        return cls(
            species_config=species_config,
            age=estimated_age,
            height_class=height_class,
            x=x,
            y=y,
        )

    @classmethod
    def newborn(cls, species_config: TreeConfig, x: int = 0, y: int = 0) -> "TreeAgent":
        return cls(
            species_config=species_config,
            age=0,
            height_class=0,
            x=x,
            y=y,
        )

    @property
    def is_mature(self):
        return self.age >= self.species_config.maturity_age

    def grow(self):
        raise NotImplementedError

    def die(self, rng: np.random.Generator):
        raise NotImplementedError

    def produce_seeds(self, rng: np.random.Generator) -> int:
        if self.is_mature:
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
        wind_strength: int | None = None,
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