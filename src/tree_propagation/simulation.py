import numpy as np

from src.tree_propagation.agent import TreeAgent
from src.tree_propagation.consts import DEFAULT_RANDOM_SEED
from src.tree_propagation.environment import EnvironmentState
from src.tree_propagation.models import DispersalHistory, SeedShadowResult
from src.tree_propagation.tree_config import TreeConfig


def simulate_seed_shadow(
    config: TreeConfig,
    environment_state: EnvironmentState,
    height_class: int = 4,
    num_instances: int = 100,
    rng: np.random.Generator | None = None,
) -> SeedShadowResult:
    """Simulates seed production and spatial dispersal for a single parent tree."""
    rng = rng if rng is not None else np.random.default_rng(DEFAULT_RANDOM_SEED)

    agent = TreeAgent(
        species_config=config,
        age=config.maturity_age,
        height_class=height_class,
        x=0,
        y=0,
    )
    history = DispersalHistory()
    theoretical_mean_radius = agent.mean_dispersal_distance(environment_state)

    for step in range(num_instances):
        agent.produce_seeds(rng=rng)
        agent.disperse_seeds(
            rng=rng,
            environment_state=environment_state,
            history=history,
            step=step,
        )

    return SeedShadowResult.from_history(
        species=config.species_name,
        dispersal_type=config.dispersal_type,
        theoretical_mean_radius=theoretical_mean_radius,
        history=history,
    )