import numpy as np
import pytest

from tree_propagation.agent import TreeAgent
from tree_propagation.tree_config import TreeConfig, load_tree_configs

# python -m pytest -v tests/test_tree_agent.py 

# region Fixtures
@pytest.fixture
def dummy_rng() -> np.random.Generator:
    return np.random.default_rng(seed=10)

@pytest.fixture
def tree_species_config() -> dict[str, TreeConfig]:
    return load_tree_configs("./data/tree_species_configs.csv")

@pytest.fixture
def ahorn_tree_config(tree_species_config) -> TreeConfig:
    return tree_species_config["Ahorn"]

@pytest.fixture
def dummy_ahorn_immature_agent(ahorn_tree_config, dummy_rng) -> TreeAgent:
    return TreeAgent(species_config=ahorn_tree_config, age=2, height_class=2, rng=dummy_rng)

@pytest.fixture
def dummy_ahorn_mature_agent(ahorn_tree_config, dummy_rng) -> TreeAgent:
    return TreeAgent(species_config=ahorn_tree_config, age=25, height_class=2, rng=dummy_rng)
# endregion

# region Geometry
def test_wind_offsets_blowing_from_north():
    dx, dy = TreeAgent._wind_offsets(
        distances=np.array([10.0]),
        from_degrees=np.array([0.0]),
    )
    assert dx[0] == pytest.approx(0.0)
    assert dy[0] == pytest.approx(-10.0)
# endregion

# region Seed Production
@pytest.mark.parametrize("age", [20, 25, 100])
def test_ahorn_tree_produces_seeds(ahorn_tree_config: TreeConfig, dummy_rng: np.random.Generator, age: int):
    tree = TreeAgent(species_config=ahorn_tree_config, age=age, height_class=2, rng=dummy_rng)
    tree.produce_seeds(rng=dummy_rng)
    assert tree.seeds > 0, "Mature tree should produce seeds."

@pytest.mark.parametrize("age", [1, 5, 19])
def test_ahorn_tree_does_not_produce_seeds(ahorn_tree_config: TreeConfig, dummy_rng: np.random.Generator, age: int):
    tree = TreeAgent(species_config=ahorn_tree_config, age=age, height_class=2, rng=dummy_rng)
    tree.produce_seeds(rng=dummy_rng)
    assert tree.seeds == 0, "Immature tree should not produce seeds."
# endregion
