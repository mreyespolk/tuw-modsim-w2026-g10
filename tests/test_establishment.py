from unittest.mock import Mock

import numpy as np
import pytest

from tree_propagation.agent import TreeAgent
from tree_propagation.environment import EnvironmentState
from tree_propagation.establishment import (
    Seed,
    establishment_probability,
    establish_seeds,
)
from tree_propagation.forest import ForestGrid
from tree_propagation.species_config import SpeciesConfig

def test_establishment_in_full_light(
    ahorn_species_config: SpeciesConfig,
):
    environment = EnvironmentState(
        establishment_probability=0.5
    )

    probability = establishment_probability(
        species_config=ahorn_species_config,
        shade=0.0,
        environment_state=environment,
    )

    assert probability == pytest.approx(0.5)

def test_establishment_in_full_shade(
    ahorn_species_config: SpeciesConfig,
):
    environment = EnvironmentState(
        establishment_probability=0.5
    )

    probability = establishment_probability(
        species_config=ahorn_species_config,
        shade=1.0,
        environment_state=environment,
    )

    expected = 0.5 * ahorn_species_config.shade_tolerance

    assert probability == pytest.approx(expected)

def test_successful_establishment(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    seed = Seed(
        species_config=ahorn_species_config,
        x=12.4,
        y=7.8,
    )

    newborns = establish_seeds(
        forest=forest,
        seeds=[seed],
        environment_state=EnvironmentState(
            establishment_probability=1.0
        ),
        rng=np.random.default_rng(42),
    )

    assert len(newborns) == 1
    assert len(forest) == 1

    tree = forest.get_tree((1, 2))

    assert tree is newborns[0]
    assert tree.age == 0
    assert tree.height_class == 0
    assert tree.x == 12.4
    assert tree.y == 7.8

def test_failed_establishment(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    seed = Seed(ahorn_species_config, x=2, y=2)

    newborns = establish_seeds(
        forest=forest,
        seeds=[seed],
        environment_state=EnvironmentState(
            establishment_probability=0.0
        ),
        rng=np.random.default_rng(42),
    )

    assert len(newborns) == 0
    assert len(forest) == 0

@pytest.mark.parametrize(
    ("x", "y"),
    [
        (-1, 2),
        (2, -1),
        (50, 2),
        (2, 50),
    ],
)
def test_seeds_outside_forest(
    ahorn_species_config: SpeciesConfig,
    x: float,
    y: float,
):
    forest = ForestGrid(10, 10)

    seed = Seed(ahorn_species_config, x=x, y=y)

    newborns = establish_seeds(
        forest=forest,
        seeds=[seed],
        environment_state=EnvironmentState(
            establishment_probability=1.0
        ),
        rng=np.random.default_rng(42),
    )

    assert len(newborns) == 0
    assert len(forest) == 0

def test_seed_on_occupied_cell(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    existing_tree = TreeAgent.newborn(
        ahorn_species_config,
        x=2,
        y=2,
    )

    forest.add_tree(existing_tree)

    seed = Seed(ahorn_species_config, x=3, y=3)

    newborns = establish_seeds(
        forest=forest,
        seeds=[seed],
        environment_state=EnvironmentState(
            establishment_probability=1.0
        ),
        rng=np.random.default_rng(42),
    )

    assert len(newborns) == 0
    assert len(forest) == 1
    assert forest.get_tree((0, 0)) is existing_tree

def test_seed_competition(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    seeds = [
        Seed(ahorn_species_config, x=1, y=1),
        Seed(ahorn_species_config, x=2, y=2),
        Seed(ahorn_species_config, x=3, y=3),
    ]

    rng = Mock()
    rng.random.return_value = 0.0
    rng.integers.return_value = 2

    newborns = establish_seeds(
        forest=forest,
        seeds=seeds,
        environment_state=EnvironmentState(
            establishment_probability=1.0
        ),
        rng=rng,
    )

    assert len(newborns) == 1
    assert len(forest) == 1

    assert newborns[0].x == 3
    assert newborns[0].y == 3

    assert rng.random.call_count == 3
    rng.integers.assert_called_once_with(3)

def test_establishment_in_multiple_cells(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    seeds = [
        Seed(ahorn_species_config, x=1, y=1),
        Seed(ahorn_species_config, x=2, y=2),
        Seed(ahorn_species_config, x=6, y=6),
        Seed(ahorn_species_config, x=12, y=12),
    ]

    newborns = establish_seeds(
        forest=forest,
        seeds=seeds,
        environment_state=EnvironmentState(
            establishment_probability=1.0
        ),
        rng=np.random.default_rng(42),
    )

    assert len(newborns) == 3
    assert len(forest) == 3

    assert forest.is_occupied((0, 0))
    assert forest.is_occupied((1, 1))
    assert forest.is_occupied((2, 2))