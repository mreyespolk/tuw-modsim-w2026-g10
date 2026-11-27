"""Unit tests for TreeAgent construction, seed production and seed dispersal.

Run with: python -m pytest -v tests/test_tree_agent.py
Shared fixtures (rng, ahorn_species_config, north_only_wind_rose) live in conftest.py.
"""

import numpy as np
import pytest

from tree_propagation.agent import TreeAgent
from tree_propagation.environment import DEFAULT_ENVIRONMENT_STATE, EnvironmentState
from tree_propagation.models import DispersalHistory
from tree_propagation.species_config import SpeciesConfig
from tree_propagation.types import AgentStatus

# Number of repeated Poisson draws used to make seed-production assertions robust.
SEED_DRAWS = 50


def _agent_at_age(config: SpeciesConfig, age: int) -> TreeAgent:
    """Builds an agent whose height class is consistent with its age."""
    return TreeAgent(species_config=config, age=age, height_class=config.height_class_at_age(age))


# region Construction
def test_newborn_has_zero_age_and_height(ahorn_species_config: SpeciesConfig):
    agent = TreeAgent.newborn(ahorn_species_config, x=3, y=4)
    assert (agent.age, agent.height_class, agent.x, agent.y) == (0, 0, 3, 4)
    assert agent.seeds == 0
    assert agent.status == AgentStatus.ALIVE


@pytest.mark.parametrize("height_class", [0, 1, 3, 5])
def test_from_height_class_estimates_age(ahorn_species_config: SpeciesConfig, height_class: int):
    agent = TreeAgent.from_height_class(ahorn_species_config, height_class=height_class)
    assert agent.height_class == height_class
    assert agent.age == height_class * ahorn_species_config.years_per_height_class


def test_height_class_capped_at_species_maximum(ahorn_species_config: SpeciesConfig):
    agent = _agent_at_age(ahorn_species_config, age=500)
    assert agent.height_class == ahorn_species_config.maximal_height_class


def test_rejects_height_class_inconsistent_with_age(ahorn_species_config: SpeciesConfig):
    with pytest.raises(ValueError, match="expected height class"):
        TreeAgent(species_config=ahorn_species_config, age=0, height_class=2)


def test_rejects_height_class_above_species_maximum(ahorn_species_config: SpeciesConfig):
    too_tall = ahorn_species_config.maximal_height_class + 1
    with pytest.raises(ValueError, match="cannot exceed"):
        TreeAgent.from_height_class(ahorn_species_config, height_class=too_tall)


@pytest.mark.parametrize(
    "kwargs",
    [{"age": -1}, {"age": 1.5}, {"age": True}, {"height_class": -1}],
)
def test_rejects_invalid_integer_params(ahorn_species_config: SpeciesConfig, kwargs: dict):
    with pytest.raises(ValueError, match="non-negative integer"):
        TreeAgent(species_config=ahorn_species_config, **kwargs)


@pytest.mark.parametrize(
    ("x", "y"),
    [(0, 0), (-12.5, 3.75), (6585.4, 6956.4), (np.float64(-1.5), np.int64(2)), (-1e9, 1e9)],
)
def test_accepts_arbitrary_finite_coordinates(ahorn_species_config: SpeciesConfig, x, y):
    agent = TreeAgent.newborn(ahorn_species_config, x=x, y=y)
    assert (agent.x, agent.y) == (float(x), float(y))
    assert isinstance(agent.x, float) and isinstance(agent.y, float)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "1.0", None, True])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_rejects_non_finite_or_non_numeric_coordinates(ahorn_species_config: SpeciesConfig, axis: str, bad):
    with pytest.raises(ValueError, match="finite real number"):
        TreeAgent.newborn(ahorn_species_config, **{axis: bad})


def test_rejects_non_species_config():
    with pytest.raises(TypeError):
        TreeAgent(species_config="Ahorn")  # type: ignore[arg-type]


# Ahorn: T = 12 yr, K = 8.
@pytest.mark.parametrize(
    ("row", "expected_age", "expected_height_class"),
    [
        ({"age": 30, "height_class": 2}, 30, 2),  # both given: used as-is
        ({"height_class": 3}, 36, 3),  # height class only: age = k * T
        ({"age": 50}, 50, 4),  # age only: k = min(K, age // T)
        ({"age": 500}, 500, 8),  # age only: capped at K
        ({}, 0, 0),  # neither: newborn
        ({"age": None, "height_class": None}, 0, 0),  # explicit None counts as missing
    ],
)
def test_from_dict_age_height_class_combinations(
    ahorn_species_config: SpeciesConfig, row: dict, expected_age: int, expected_height_class: int
):
    agent = TreeAgent.from_dict({**row, "x": 1.5, "y": -2.5}, species_config=ahorn_species_config)
    assert (agent.age, agent.height_class) == (expected_age, expected_height_class)
    assert (agent.x, agent.y) == (1.5, -2.5)


def test_from_dict_defaults_missing_coordinates_to_origin(ahorn_species_config: SpeciesConfig):
    agent = TreeAgent.from_dict({"height_class": 1}, species_config=ahorn_species_config)
    assert (agent.x, agent.y) == (0.0, 0.0)


def test_from_dict_parses_string_values(ahorn_species_config: SpeciesConfig):
    """Rows from csv.DictReader hold strings."""
    row = {"age": "24", "height_class": "2", "x": "6585.4", "y": "6956.4"}
    agent = TreeAgent.from_dict(row, species_config=ahorn_species_config)
    assert (agent.age, agent.height_class, agent.x, agent.y) == (24, 2, 6585.4, 6956.4)


def test_from_dict_col_override_maps_vienna_columns(ahorn_species_config: SpeciesConfig):
    """Matches the trees_vienna.csv schema: x_m, y_m, height_class (no age)."""
    row = {"district": "13", "x_m": "6585.4", "y_m": "6956.4", "species": "Ahorn", "height_class": "2"}
    agent = TreeAgent.from_dict(row, species_config=ahorn_species_config, col_override={"x": "x_m", "y": "y_m"})
    assert (agent.age, agent.height_class, agent.x, agent.y) == (24, 2, 6585.4, 6956.4)


def test_from_dict_rejects_inconsistent_age_and_height_class(ahorn_species_config: SpeciesConfig):
    with pytest.raises(ValueError):
        TreeAgent.from_dict({"age": 30, "height_class": 5}, species_config=ahorn_species_config)
# endregion


# region Seed Production
@pytest.mark.parametrize("age", [20, 25, 100])
def test_mature_tree_produces_seeds(ahorn_species_config: SpeciesConfig, rng: np.random.Generator, age: int):
    tree = _agent_at_age(ahorn_species_config, age)
    assert tree.is_mature
    total = sum(tree.produce_seeds(rng=rng) for _ in range(SEED_DRAWS))
    assert total > 0, "Mature tree should produce seeds."


@pytest.mark.parametrize("age", [0, 1, 5, 19])
def test_immature_tree_does_not_produce_seeds(ahorn_species_config: SpeciesConfig, rng: np.random.Generator, age: int):
    tree = _agent_at_age(ahorn_species_config, age)
    assert not tree.is_mature
    assert all(tree.produce_seeds(rng=rng) == 0 for _ in range(SEED_DRAWS))
    assert tree.seeds == 0


def test_mean_seed_count_matches_height_adjusted_fecundity(
        ahorn_species_config: SpeciesConfig, rng: np.random.Generator
):
    tree = _agent_at_age(ahorn_species_config, age=60)  # k = 5
    draws = [tree.produce_seeds(rng=rng) for _ in range(5_000)]
    expected = ahorn_species_config.fecundity * tree.height_class
    assert np.mean(draws) == pytest.approx(expected, rel=0.05)
# endregion


# region Seed Dispersal
def test_disperse_seeds_resets_stock_and_returns_one_point_per_seed(
    ahorn_species_config: SpeciesConfig, rng: np.random.Generator
):
    tree = _agent_at_age(ahorn_species_config, age=60)
    produced = tree.produce_seeds(rng=rng)
    assert produced > 0

    xs, ys, distances = tree.disperse_seeds(rng=rng, environment_state=DEFAULT_ENVIRONMENT_STATE)

    assert tree.seeds == 0
    assert xs.shape == ys.shape == distances.shape == (produced,)
    np.testing.assert_allclose(np.hypot(xs - tree.x, ys - tree.y), distances)


def test_seeds_land_relative_to_off_origin_parent(ahorn_species_config: SpeciesConfig):
    """Same RNG stream from two parents differs only by the parent offset."""
    def landings(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        tree = TreeAgent.from_height_class(ahorn_species_config, height_class=5, x=x, y=y)
        tree.produce_seeds(rng=rng)
        xs, ys, _ = tree.disperse_seeds(rng=rng, environment_state=DEFAULT_ENVIRONMENT_STATE)
        return xs, ys

    xs0, ys0 = landings(0.0, 0.0)
    xs1, ys1 = landings(-250.5, 1234.25)
    np.testing.assert_allclose(xs1 - xs0, -250.5)
    np.testing.assert_allclose(ys1 - ys0, 1234.25)


def test_disperse_without_seeds_returns_empty_arrays(ahorn_species_config: SpeciesConfig, rng: np.random.Generator):
    tree = TreeAgent.newborn(ahorn_species_config)
    xs, ys, distances = tree.disperse_seeds(rng=rng, environment_state=DEFAULT_ENVIRONMENT_STATE)
    assert xs.size == ys.size == distances.size == 0


def test_wind_from_north_carries_seeds_south(
    ahorn_species_config: SpeciesConfig, rng: np.random.Generator, north_only_wind_rose
):
    tree = _agent_at_age(ahorn_species_config, age=60)
    tree.produce_seeds(rng=rng)
    env = EnvironmentState(wind_rose=north_only_wind_rose, wind_strength=5)

    xs, ys, _ = tree.disperse_seeds(rng=rng, environment_state=env)

    # Downwind bearings lie within 180° ± 22.5°, so every seed lands south of the parent
    # and |dx| <= tan(22.5°) * |dy|.
    assert np.all(ys < 0)
    assert np.all(np.abs(xs) <= np.tan(np.deg2rad(22.5)) * np.abs(ys) + 1e-9)


def test_disperse_records_history(ahorn_species_config: SpeciesConfig, rng: np.random.Generator):
    tree = _agent_at_age(ahorn_species_config, age=60)
    history = DispersalHistory()

    produced = tree.produce_seeds(rng=rng)
    tree.disperse_seeds(rng=rng, environment_state=DEFAULT_ENVIRONMENT_STATE, history=history, step=7)

    assert len(history.records) == 1
    record = history.records[0]
    assert record.step == 7
    assert record.species == ahorn_species_config.species_name
    assert record.seeds_produced == produced == record.distances.size


def test_same_seed_gives_identical_dispersal(ahorn_species_config: SpeciesConfig):
    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        tree = _agent_at_age(ahorn_species_config, age=60)
        tree.produce_seeds(rng=rng)
        return np.stack(tree.disperse_seeds(rng=rng, environment_state=DEFAULT_ENVIRONMENT_STATE))

    np.testing.assert_array_equal(run(123), run(123))
# endregion
