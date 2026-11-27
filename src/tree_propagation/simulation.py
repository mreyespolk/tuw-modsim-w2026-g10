from dataclasses import dataclass

import numpy as np

from tree_propagation.agent import TreeAgent
from tree_propagation.consts import DEFAULT_HEIGHT_CLASS, DEFAULT_RANDOM_SEED
from tree_propagation.environment import DEFAULT_ENVIRONMENT_STATE, EnvironmentState
from tree_propagation.models import DispersalHistory, SeedShadowResult
from tree_propagation.species_config import SpeciesConfig

DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES = 1000


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
