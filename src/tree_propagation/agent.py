from typing import Any, Mapping

import numpy as np

import tree_propagation.type_validation as type_validation
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
    """A single tree at position (x, y) in meters (+x = East, +y = North).

    Coordinates may be any finite real number (negative values are valid, e.g. seeds landing
    west/south of an origin parent). `age` and `height_class` are non-negative integers.

    Spatial requirements (see README "Spatial model conventions"):
    - `(x, y)` is the tree's exact position in meters; it is never a grid-cell index. Grid cells
      are derived from it by a cell-size mapping.
    - Seeds are dispersed from this exact position, never from a cell center.
    - Trees established from seeds are created at the exact seed landing location.
    """

    def __init__(
            self,
            species_config: TreeConfig,
            age: int = 0,
            height_class: int = 0,
            x: float = 0.0,
            y: float = 0.0,
    ) -> None:
        self.species_config = species_config
        self.age = age
        self.height_class = height_class
        self.x = float(x) if type_validation.is_finite_real(x) else x
        self.y = float(y) if type_validation.is_finite_real(y) else y

        self.seeds = 0

        self.state = AgentStatus.ALIVE

        self._validate_init_params()

    def _validate_init_params(self) -> None:
        if not isinstance(self.species_config, TreeConfig):
            raise TypeError("'species_config' must be an instance of TreeConfig.")

        for name, val in (("age", self.age), ("height_class", self.height_class)):
            if not type_validation.is_strict_int(val) or val < 0:
                raise ValueError(f"'{name}' must be a non-negative integer, got {val!r}.")

        for name, val in (("x", self.x), ("y", self.y)):
            if not type_validation.is_finite_real(val):
                raise ValueError(f"'{name}' must be a finite real number, got {val!r}.")

        if self.height_class > self.species_config.maximal_height_class:
            raise ValueError(
                f"'height_class' ({self.height_class}) cannot exceed "
                f"maximal_height_class ({self.species_config.maximal_height_class}) "
                f"for species '{self.species_config.species_name}'."
            )

        expected_k = self.species_config.height_class_at_age(self.age)
        if self.height_class != expected_k:
            raise ValueError("Height class of agent does not match the expected height class by it's age.")

    def __repr__(self):
        return (f"TreeAgent(species={self.species_config.species_name}, "
                f"age={self.age}, height_class={self.height_class}, seeds={self.seeds}, "
                f"x={self.x}, y={self.y})")

    @classmethod
    def from_height_class(
        cls, species_config: TreeConfig, height_class: int, x: float = 0.0, y: float = 0.0
    ) -> "TreeAgent":
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
    def from_age(
        cls, species_config: TreeConfig, age: int, x: float = 0.0, y: float = 0.0
    ) -> "TreeAgent":
        """Instantiate an new agent when the age is known, but not the height class."""
        height_class = species_config.height_class_at_age(age)
        return cls(
            species_config=species_config,
            age=age,
            height_class=height_class,
            x=x,
            y=y,
        )

    @classmethod
    def newborn(cls, species_config: TreeConfig, x: float = 0.0, y: float = 0.0) -> "TreeAgent":
        """Instantiate a new agent as a newborn.

        For a tree established from a seed, pass the exact seed landing location `(x_land, y_land)`
        returned by `disperse_seeds`; do not snap it to a grid-cell center.
        """
        return cls(
            species_config=species_config,
            age=0,
            height_class=0,
            x=x,
            y=y,
        )

    @classmethod
    def from_dict(
        cls, input_dict: Mapping[str, Any], species_config: TreeConfig, col_override: dict | None = None
    ) -> "TreeAgent":
        if col_override is None:
            col_override = {}

        age_col = col_override.get("age", "age")
        height_class_col = col_override.get("height_class", "height_class")
        x_col = col_override.get("x", "x")
        y_col = col_override.get("y", "y")

        has_age = input_dict.get(age_col) is not None
        has_height_class = input_dict.get(height_class_col) is not None

        x_val = float(input_dict.get(x_col, 0.0))
        y_val = float(input_dict.get(y_col, 0.0))

        if has_age and has_height_class:
                return cls(
                    species_config=species_config,
                    age=int(input_dict[age_col]),
                    height_class=int(input_dict[height_class_col]),
                    x=x_val,
                    y=y_val,
                )
        if not has_age and has_height_class:
            return cls.from_height_class(
                species_config=species_config,
                height_class=int(input_dict[height_class_col]),
                x=x_val,
                y=y_val,
            )
        elif not has_height_class:
            return cls.from_age(
                species_config=species_config,
                age=int(input_dict[age_col]),
                x=x_val,
                y=y_val,
            )
        else:
            return cls.newborn(
                species_config=species_config,
                x=x_val,
                y=y_val,
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
        """Disperses all seeds currently held by this tree and resets its seed stock.

        Seeds leave from the tree's exact `(x, y)` (never a grid-cell center), so the landing
        positions are `x + dx`, `y + dy` in meters. These exact landing positions are where
        seedlings are created; grid cells for occupancy/out-of-grid checks are derived from them.

        Returns:
            `(x_land, y_land, distances)` arrays in meters, one entry per dispersed seed.
        """
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