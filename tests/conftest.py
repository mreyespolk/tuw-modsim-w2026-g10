from pathlib import Path

import numpy as np
import pytest

from tree_propagation.tree_config import TreeConfig, load_tree_configs
from tree_propagation.wind_config import WindDirection, WindRose

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tree_species_configs.csv"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=10)


@pytest.fixture(scope="session")
def tree_species_config() -> dict[str, TreeConfig]:
    return load_tree_configs(DATA_PATH)


@pytest.fixture
def ahorn_tree_config(tree_species_config: dict[str, TreeConfig]) -> TreeConfig:
    """Wind-dispersed species (A=20 yr, K=5)."""
    return tree_species_config["Ahorn"]


@pytest.fixture
def north_only_wind_rose() -> WindRose:
    """Wind always blows *from* the North, so seeds travel South."""
    return WindRose(probabilities={d: (1.0 if d == WindDirection.N else 0.0) for d in WindDirection})
