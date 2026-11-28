from dataclasses import dataclass

import numpy as np

from tree_propagation.agent import TreeAgent
from tree_propagation.consts import DEFAULT_HEIGHT_CLASS, DEFAULT_RANDOM_SEED
from tree_propagation.environment import DEFAULT_ENVIRONMENT_STATE, EnvironmentState
from tree_propagation.models import DispersalHistory, SeedShadowResult
from tree_propagation.species_config import SpeciesConfig
import tree_propagation.type_validation as type_validation
from tree_propagation.establishment import Seed, establish_seeds
from tree_propagation.forest import ForestGrid
from tree_propagation.models import (
    ForestSnapshot,
    ForestYearStatistics,
    TreeSnapshot,
)

DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES = 1000
DEFAULT_FOREST_SIMULATION_YEARS = 250


@dataclass(frozen=True)
class SeedDispersalSimulationParams:
    config: SpeciesConfig
    environment_state: EnvironmentState = DEFAULT_ENVIRONMENT_STATE
    height_class: int = DEFAULT_HEIGHT_CLASS
    num_instances: int = DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES
    seed: int = DEFAULT_RANDOM_SEED


class SeedDispersalSimulation:
    def __init__(self, params: SeedDispersalSimulationParams, rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(params.seed)
        self.agent = TreeAgent.from_height_class(
            species_config=params.config,
            height_class=params.height_class,
            x=0.0,
            y=0.0,
        )
        self.history = DispersalHistory()
        self.current_step = 0

    def step(self) -> None:
        self.agent.produce_seeds(rng=self.rng)
        self.agent.disperse_seeds(
            rng=self.rng,
            environment_state=self.params.environment_state,
            history=self.history,
            step=self.current_step,
        )
        self.current_step += 1

    def run(self) -> SeedShadowResult:
        while self.current_step < self.params.num_instances:
            self.step()

        return SeedShadowResult.from_history(
            species=self.params.config.species_name,
            dispersal_type=self.params.config.dispersal_type,
            theoretical_mean_radius=self.agent.mean_dispersal_distance(self.params.environment_state),
            history=self.history,
        )

### ===== Forest Simulation ===== ###

@dataclass(frozen=True)
class ForestSimulationParams:
    environment_state: EnvironmentState = DEFAULT_ENVIRONMENT_STATE
    num_years: int = DEFAULT_FOREST_SIMULATION_YEARS
    seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self):
        if not (
            type_validation.is_strict_int(self.num_years)
            and self.num_years >= 0
        ):
            raise ValueError(
                "'num_years' must be a non-negative integer."
            )

class ForestSimulation:
    """Annual agent-based simulation of a forest."""

    def __init__(
        self,
        forest: ForestGrid,
        params: ForestSimulationParams,
        rng: np.random.Generator | None = None,
    ):
        self.forest = forest
        self.params = params

        self.rng = (
            rng if rng is not None
            else np.random.default_rng(params.seed)
        )

        self.current_year = 0

        self.history: list[ForestSnapshot] = []
        self.statistics: list[ForestYearStatistics] = []

        self._record_snapshot()

    def _record_snapshot(self) -> None:
        """Record the current forest without mutable agent references."""

        trees = tuple(
            TreeSnapshot(
                species=tree.species_config.species_name,
                age=tree.age,
                height_class=tree.height_class,
                x=tree.x,
                y=tree.y,
            )
            for tree in self.forest.trees()
        )

        self.history.append(
            ForestSnapshot(
                year=self.current_year,
                trees=trees,
            )
        )

    def _produce_and_disperse_seeds(self) -> list[Seed]:
        """Produce and disperse seeds from the initial population."""

        trees = self.forest.trees()
        seeds: list[Seed] = []

        # Stage 1: Seed production.
        for tree in trees:
            tree.produce_seeds(rng=self.rng)

        # Stage 2: Seed dispersal.
        for tree in trees:
            xs, ys, _ = tree.disperse_seeds(
                rng=self.rng,
                environment_state=self.params.environment_state,
            )

            for x, y in zip(xs, ys, strict=True):
                seeds.append(
                    Seed(
                        species_config=tree.species_config,
                        x=float(x),
                        y=float(y),
                    )
                )

        return seeds

    def _establish_seeds(
        self,
        seeds: list[Seed],
    ) -> tuple[TreeAgent, ...]:
        """Establish seeds on empty forest cells."""

        return establish_seeds(
            forest=self.forest,
            seeds=seeds,
            environment_state=self.params.environment_state,
            rng=self.rng,
        )

    def _grow_trees(self) -> None:
        """Grow every tree by one year."""

        for tree in self.forest.trees():
            tree.grow()

    def _apply_mortality(self) -> int:
        """Evaluate mortality simultaneously and remove dead trees."""

        trees = self.forest.trees()

        # Phase 1: Calculate shade for every tree.
        shade_values = [
            self.forest.calculate_shade(
                self.forest.position_to_cell(tree.x, tree.y)
            )
            for tree in trees
        ]

        # Phase 2: Evaluate mortality.
        dead_cells = []

        for tree, shade in zip(trees, shade_values, strict=True):
            died = tree.die(
                rng=self.rng,
                shade=shade,
                environment_state=self.params.environment_state,
            )

            if died:
                cell = self.forest.position_to_cell(
                    tree.x, tree.y
                )

                dead_cells.append(cell)

        # Phase 3: Remove dead trees.
        for cell in dead_cells:
            self.forest.remove_tree(cell)

        return len(dead_cells)

    def step(self) -> ForestYearStatistics:
        """Simulate one complete year."""

        seeds = self._produce_and_disperse_seeds()

        newborns = self._establish_seeds(seeds)

        self._grow_trees()

        deaths = self._apply_mortality()

        self.current_year += 1

        statistics = ForestYearStatistics(
            year=self.current_year,
            seeds_produced=len(seeds),
            births=len(newborns),
            deaths=deaths,
            population=len(self.forest),
        )

        self.statistics.append(statistics)
        self._record_snapshot()

        return statistics

    def run(self) -> tuple[ForestSnapshot, ...]:
        """Run the simulation until the configured final year."""

        while self.current_year < self.params.num_years:
            self.step()

        return tuple(self.history)