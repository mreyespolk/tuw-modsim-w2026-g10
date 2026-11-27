import numpy as np
import pytest

from tree_propagation.agent import TreeAgent
from tree_propagation.forest import ForestGrid
from tree_propagation.species_config import SpeciesConfig
from tree_propagation.types import DispersalType


@pytest.fixture
def config():
    return SpeciesConfig(
        species_name="Ahorn",
        dispersal_type=DispersalType.WIND,
        wind_spreading_factor=0.9,
        mean_dispersal_distance=None,
        fecundity=2.0,
        maturity_age=20,
        years_per_height_class=12,
        maximal_height_class=5,
        mean_life_span=120,
        shade_tolerance=0.8,
    )


def test_empty_forest():
    forest = ForestGrid(10, 10)

    assert len(forest) == 0


def test_position_to_cell():
    forest = ForestGrid(10, 10)

    assert forest.position_to_cell(12.4, 7.8) == (1, 2)


def test_boundaries():
    forest = ForestGrid(10, 10)

    assert forest.in_bounds((0, 0))
    assert forest.in_bounds((9, 9))

    assert not forest.in_bounds((-1, 0))
    assert not forest.in_bounds((10, 0))

    assert not forest.in_bounds(
        forest.position_to_cell(-0.1, 0)
    )

    assert not forest.in_bounds(
        forest.position_to_cell(50, 0)
    )


def test_add_tree(config):
    forest = ForestGrid(10, 10)

    tree = TreeAgent.from_age(
        config, age=25, x=12.4, y=7.8
    )

    forest.add_tree(tree)

    assert len(forest) == 1
    assert forest.is_occupied((1, 2))
    assert forest.get_tree((1, 2)) is tree

    assert tree.x == 12.4
    assert tree.y == 7.8


def test_duplicate_cell(config):
    forest = ForestGrid(10, 10)

    tree1 = TreeAgent.newborn(config, x=1, y=1)
    tree2 = TreeAgent.newborn(config, x=2, y=2)

    forest.add_tree(tree1)

    with pytest.raises(ValueError):
        forest.add_tree(tree2)


def test_remove_tree(config):
    forest = ForestGrid(10, 10)

    tree = TreeAgent.newborn(config, x=1, y=1)

    forest.add_tree(tree)
    forest.remove_tree((0, 0))

    assert len(forest) == 0
    assert not forest.is_occupied((0, 0))


def test_random_forest(config):
    forest = ForestGrid.random(
        rows=10,
        cols=10,
        num_trees=20,
        configs={"Ahorn": config},
        rng=np.random.default_rng(42),
    )

    assert len(forest) == 20

    cells = {
        forest.position_to_cell(tree.x, tree.y)
        for tree in forest.trees()
    }

    assert len(cells) == 20
    assert all(forest.in_bounds(cell) for cell in cells)


def test_excessive_population(config):
    with pytest.raises(ValueError):
        ForestGrid.random(
            rows=2,
            cols=2,
            num_trees=5,
            configs={"Ahorn": config},
            rng=np.random.default_rng(42),
        )