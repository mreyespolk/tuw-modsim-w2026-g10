import numpy as np

from src.tree_propagation.agent import TreeAgent
from src.tree_propagation.dispersal import get_dispersal_strategy
from src.tree_propagation.environment import EnvironmentState
from src.tree_propagation.models import SeedShadowResult
from src.tree_propagation.tree_config import TreeConfig


def simulate_seed_shadow(
    config: TreeConfig,
    environment_state: EnvironmentState,
    height_class: int = 4,
    num_instances: int = 100,
    rng: np.random.Generator | None = None,
) -> SeedShadowResult:
    """Simulates seed production and spatial dispersal for a single parent tree."""
    rng = rng or np.random.default_rng(10)
    strategy = get_dispersal_strategy(config, wind_rose=environment_state.wind_rose)

    theoretical_mean_radius = strategy.mean_distance(
        config=config,
        height_class=height_class,
        wind_strength=environment_state.wind_strength,
    )

    agent = TreeAgent(
        species_config=config,
        age=config.maturity_age,
        height_class=height_class,
        x=0,
        y=0,
    )

    xs_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    distances_list: list[np.ndarray] = []

    for _ in range(num_instances):
        agent.produce_seeds(rng=rng)
        xs, ys, distances = agent.disperse_seeds(
            rng=rng,
            strategy=strategy,
            wind_strength=environment_state.wind_strength,
        )
        if distances.size > 0:
            xs_list.append(xs)
            ys_list.append(ys)
            distances_list.append(distances)

    if distances_list:
        all_xs = np.concatenate(xs_list)
        all_ys = np.concatenate(ys_list)
        all_distances = np.concatenate(distances_list)
    else:
        all_xs = np.array([], dtype=float)
        all_ys = np.array([], dtype=float)
        all_distances = np.array([], dtype=float)

    return SeedShadowResult(
        species=config.species_name,
        dispersal_type=config.dispersal_type,
        num_instances=num_instances,
        xs=all_xs,
        ys=all_ys,
        distances=all_distances,
        seed_counts_per_instance=tuple(agent.seed_production_history),
        theoretical_mean_radius=theoretical_mean_radius
    )