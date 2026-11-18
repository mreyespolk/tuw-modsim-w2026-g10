import numpy as np

from tree_propagation.dispersal import (
    compute_mean_dispersal_distance,
    empty_dispersal_result,
    sample_dispersal_offsets,
)
from tree_propagation.environment import EnvironmentState
from tree_propagation.models import DispersalHistory
from tree_propagation.tree_config import TreeConfig
from tree_propagation.types import AgentStatus


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

        self.state = AgentStatus.ALIVE

        self._validate_init_params()

    def _validate_init_params(self) -> None:
        if not isinstance(self.species_config, TreeConfig):
            raise TypeError("'species_config' must be an instance of TreeConfig.")

        for name, val in (
            ("age", self.age),
            ("height_class", self.height_class),
            ("x", self.x),
            ("y", self.y),
        ):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(f"'{name}' must be a non-negative integer, got {val!r}.")

        if self.height_class > self.species_config.maximal_height_class:
            raise ValueError(
                f"'height_class' ({self.height_class}) cannot exceed "
                f"maximal_height_class ({self.species_config.maximal_height_class}) "
                f"for species '{self.species_config.species_name}'."
            )

        expected_k = min(
            self.species_config.maximal_height_class,
            self.age // self.species_config.years_per_height_class,
        )
        if self.height_class != expected_k:
            raise ValueError("Height class of agent does not match the expected height class by it's age.")

    def __repr__(self):
        return (f"TreeAgent(species={self.species_config.species_name}, "
                f"age={self.age}, height_class={self.height_class}, seeds={self.seeds}, "
                f"x={self.x}, y={self.y})")

    @classmethod
    def from_height_class(cls, species_config: TreeConfig, height_class: int, x: int = 0, y: int = 0) -> "TreeAgent":
        """Instantiate an new agent when the age is not known, but the height class is."""
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
        """Instantiate a new agent as a newborn."""
        return cls(
            species_config=species_config,
            age=0,
            height_class=0,
            x=x,
            y=y,
        )

    @property
    def is_mature(self):
        """Whether the agent is at or over it's species maturity age."""
        return self.age >= self.species_config.maturity_age

    @property
    def height_adjusted_fecundity(self):
        """The fecundity of the tree species modified by the agent's height class."""
        return self.species_config.fecundity * self.height_class

    def mean_dispersal_distance(self, environment_state: EnvironmentState) -> float:
        return compute_mean_dispersal_distance(
            config=self.species_config,
            height_class=self.height_class,
            environment=environment_state,
        )

    def grow(self):
        """Not implemented: Ages the agent one time-step."""
        raise NotImplementedError

    def _evaluate_death_outcome(self):
        """Not implemented: Obtains the outcome of whether the agent dies on this timestep or not."""
        raise NotImplementedError

    def die(self, rng: np.random.Generator):
        """Not implemented: Performs the action of killing the agent."""
        raise NotImplementedError

    def produce_seeds(self, rng: np.random.Generator) -> int:
        """Generate a set amount of seeds backed by a random distribution. (Currently only Poisson)"""
        if self.is_mature and self.height_class > 0:
            self.seeds = int(
                rng.poisson(self.height_adjusted_fecundity)
            )
        else:
            self.seeds = 0
        return self.seeds

    def disperse_seeds(
        self,
        rng: np.random.Generator,
        environment_state: EnvironmentState,
        history: DispersalHistory | None = None,
        step: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Disperses all seeds currently held by this tree and resets its seed stock."""
        produced = self.seeds
        self.seeds = 0

        if produced <= 0:
            x_land, y_land, distances = empty_dispersal_result()
        else:
            mean_dispersal = self.mean_dispersal_distance(environment_state)
            dx, dy, distances = sample_dispersal_offsets(
                count=produced,
                mean_distance=mean_dispersal,
                dispersal_type=self.species_config.dispersal_type,
                rng=rng,
                wind_rose=environment_state.wind_rose,
            )
            x_land = self.x + dx
            y_land = self.y + dy

        if history is not None:
            history.record(
                step=step,
                species=self.species_config.species_name,
                seeds_produced=produced,
                xs=x_land,
                ys=y_land,
                distances=distances,
            )

        return x_land, y_land, distances