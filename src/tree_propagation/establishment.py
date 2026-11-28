from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

import tree_propagation.type_validation as type_validation
from tree_propagation.agent import TreeAgent
from tree_propagation.environment import EnvironmentState
from tree_propagation.forest import Cell, ForestGrid
from tree_propagation.species_config import SpeciesConfig


@dataclass(frozen=True)
class Seed:
    """A dispersed seed with a species and landing position."""

    species_config: SpeciesConfig
    x: float
    y: float

def establishment_probability(
    species_config: SpeciesConfig,
    shade: float,
    environment_state: EnvironmentState,
) -> float:
    """Calculate establishment probability under the given shade."""

    if not (
        type_validation.is_finite_real(shade)
        and 0 <= shade <= 1
    ):
        raise ValueError("'shade' must be between 0 and 1.")

    return (
        environment_state.establishment_probability
        * (1 - (1 - species_config.shade_tolerance) * shade)
    )

def establish_seeds(
    forest: ForestGrid,
    seeds: Iterable[Seed],
    environment_state: EnvironmentState,
    rng: np.random.Generator,
) -> tuple[TreeAgent, ...]:
    """Establish seeds simultaneously on empty forest cells."""

    candidates: dict[Cell, list[Seed]] = defaultdict(list)

    # Phase 1: Evaluate all seeds.
    for seed in seeds:
        cell = forest.position_to_cell(seed.x, seed.y)

        if not forest.in_bounds(cell):
            continue

        if forest.is_occupied(cell):
            continue

        shade = forest.calculate_shade(cell)

        probability = establishment_probability(
            species_config=seed.species_config,
            shade=shade,
            environment_state=environment_state,
        )

        if rng.random() < probability:
            candidates[cell].append(seed)

    # Phase 2: Resolve competition and create newborn trees.
    newborns: list[TreeAgent] = []

    for cell, successful_seeds in candidates.items():
        winner = successful_seeds[
            int(rng.integers(len(successful_seeds)))
        ]

        tree = TreeAgent.newborn(
            species_config=winner.species_config,
            x=winner.x,
            y=winner.y,
        )

        forest.add_tree(tree)
        newborns.append(tree)

    return tuple(newborns)