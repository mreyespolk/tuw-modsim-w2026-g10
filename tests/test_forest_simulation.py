from unittest.mock import Mock, patch

import numpy as np

from tree_propagation.agent import TreeAgent
from tree_propagation.environment import EnvironmentState
from tree_propagation.establishment import Seed
from tree_propagation.forest import ForestGrid
from tree_propagation.simulation import (
    ForestSimulation,
    ForestSimulationParams,
)
from tree_propagation.species_config import SpeciesConfig

def test_initial_snapshot(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    forest.add_tree(
        TreeAgent.from_age(
            ahorn_species_config,
            age=30,
            x=2,
            y=2,
        )
    )

    simulation = ForestSimulation(
        forest,
        ForestSimulationParams(num_years=10),
    )

    assert simulation.current_year == 0
    assert len(simulation.history) == 1
    assert simulation.history[0].year == 0
    assert simulation.history[0].population == 1
    assert simulation.history[0].trees[0].age == 30

def test_empty_forest():
    simulation = ForestSimulation(
        forest=ForestGrid(10, 10),
        params=ForestSimulationParams(num_years=10),
    )

    history = simulation.run()

    assert simulation.current_year == 10
    assert len(history) == 11
    assert len(simulation.statistics) == 10

    assert all(
        snapshot.population == 0
        for snapshot in history
    )

def test_growth_and_snapshot_independence(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    tree = TreeAgent.newborn(
        ahorn_species_config,
        x=2,
        y=2,
    )

    forest.add_tree(tree)

    rng = Mock()
    rng.random.return_value = 0.99

    simulation = ForestSimulation(
        forest=forest,
        params=ForestSimulationParams(num_years=5),
        rng=rng,
    )

    history = simulation.run()

    assert tree.age == 5

    assert [
        snapshot.trees[0].age
        for snapshot in history
    ] == [0, 1, 2, 3, 4, 5]

def test_mortality_removes_trees(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    forest.add_tree(
        TreeAgent.newborn(
            ahorn_species_config,
            x=2,
            y=2,
        )
    )

    rng = Mock()
    rng.random.return_value = 0.0

    simulation = ForestSimulation(
        forest=forest,
        params=ForestSimulationParams(num_years=1),
        rng=rng,
    )

    statistics = simulation.step()

    assert statistics.deaths == 1
    assert statistics.population == 0

    assert len(forest) == 0
    assert not forest.is_occupied((0, 0))

def test_newborn_growth(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    rng = Mock()
    rng.random.side_effect = [0.0, 0.99]
    rng.integers.return_value = 0

    simulation = ForestSimulation(
        forest=forest,
        params=ForestSimulationParams(
            num_years=1,
            environment_state=EnvironmentState(
                establishment_probability=1.0,
            ),
        ),
        rng=rng,
    )

    seeds = [
        Seed(ahorn_species_config, x=2, y=2)
    ]

    with patch.object(
        simulation,
        "_produce_and_disperse_seeds",
        return_value=seeds,
    ):
        statistics = simulation.step()

    assert statistics.births == 1
    assert statistics.deaths == 0

    tree = forest.get_tree((0, 0))

    assert tree is not None
    assert tree.age == 1
    assert tree.height_class == 0

def test_mortality_is_simultaneous(
    ahorn_species_config: SpeciesConfig,
):
    forest = ForestGrid(10, 10)

    tall_tree = TreeAgent.from_height_class(
        ahorn_species_config,
        height_class=4,
        x=2,
        y=2,
    )

    small_tree = TreeAgent.newborn(
        ahorn_species_config,
        x=7,
        y=2,
    )

    forest.add_tree(tall_tree)
    forest.add_tree(small_tree)

    simulation = ForestSimulation(
        forest=forest,
        params=ForestSimulationParams(num_years=1),
    )

    with patch.object(
        TreeAgent,
        "die",
        autospec=True,
        side_effect=[True, False],
    ) as mock_die:

        deaths = simulation._apply_mortality()

    assert deaths == 1

    assert mock_die.call_args_list[0].kwargs["shade"] == 0
    assert mock_die.call_args_list[1].kwargs["shade"] == 1 / 8

    assert len(forest) == 1

def test_population_conservation(
    ahorn_species_config: SpeciesConfig,
):
    rng = np.random.default_rng(42)

    forest = ForestGrid.random(
        rows=10,
        cols=10,
        num_trees=10,
        configs={"Ahorn": ahorn_species_config},
        rng=rng,
    )

    simulation = ForestSimulation(
        forest=forest,
        params=ForestSimulationParams(num_years=20),
        rng=rng,
    )

    simulation.run()

    previous_population = 10

    for statistics in simulation.statistics:
        assert statistics.population == (
            previous_population
            + statistics.births
            - statistics.deaths
        )

        assert 0 <= statistics.population <= 100

        previous_population = statistics.population

def test_simulation_reproducibility(
    ahorn_species_config: SpeciesConfig,
):
    def run(seed: int):
        rng = np.random.default_rng(seed)

        forest = ForestGrid.random(
            rows=10,
            cols=10,
            num_trees=10,
            configs={"Ahorn": ahorn_species_config},
            rng=rng,
        )

        simulation = ForestSimulation(
            forest=forest,
            params=ForestSimulationParams(num_years=20),
            rng=rng,
        )

        return simulation.run()

    assert run(42) == run(42)