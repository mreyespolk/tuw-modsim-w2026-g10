"""Unit tests for compass sectors and WindRose validation/sampling."""

import pickle

import numpy as np
import pytest

from tree_propagation.wind_config import (
    DEFAULT_WIND_ROSE,
    NUM_SECTORS,
    WindDirection,
    WindRose,
    bearing_to_sector_index,
)


def _probs(**overrides: float) -> dict[WindDirection, float]:
    """Uniform probabilities with selected directions overridden by name."""
    probs = {d: 1.0 / NUM_SECTORS for d in WindDirection}
    probs.update({WindDirection[name]: p for name, p in overrides.items()})
    return probs


# region WindDirection
@pytest.mark.parametrize(
    ("direction", "opposite"),
    [(WindDirection.N, WindDirection.S), (WindDirection.NE, WindDirection.SW),
     (WindDirection.W, WindDirection.E), (WindDirection.NW, WindDirection.SE)],
)
def test_opposite(direction: WindDirection, opposite: WindDirection):
    assert direction.opposite == opposite
    assert opposite.opposite == direction


def test_sector_index_is_clockwise_from_north():
    assert [d.sector_index for d in WindDirection] == list(range(NUM_SECTORS))


@pytest.mark.parametrize(
    ("bearing", "expected"),
    [(0.0, WindDirection.N), (22.4, WindDirection.N), (-22.5, WindDirection.N), (359.9, WindDirection.N),
     (22.5, WindDirection.NE), (90.0, WindDirection.E), (200.0, WindDirection.S), (315.0, WindDirection.NW),
     (720.0 + 90.0, WindDirection.E)],
)
def test_bearing_to_sector_index(bearing: float, expected: WindDirection):
    assert bearing_to_sector_index(np.array([bearing]))[0] == expected.sector_index


def test_sector_centers_map_to_their_own_sector():
    centers = np.array([d.value for d in WindDirection], dtype=float)
    np.testing.assert_array_equal(bearing_to_sector_index(centers), [d.sector_index for d in WindDirection])
# endregion


# region WindRose validation
def test_rejects_missing_direction_with_clear_message():
    probs = _probs()
    del probs[WindDirection.SW]
    with pytest.raises(ValueError, match="missing.*SW"):
        WindRose(probabilities=probs)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_rejects_invalid_probability_values(bad):
    with pytest.raises(ValueError):
        WindRose(probabilities=_probs(N=bad))


def test_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        WindRose(probabilities=_probs(N=0.5))


def test_probabilities_are_read_only_and_decoupled_from_input():
    source = _probs()
    rose = WindRose(probabilities=source)
    source[WindDirection.N] = 0.9
    assert rose.probabilities[WindDirection.N] == pytest.approx(1.0 / NUM_SECTORS)
    with pytest.raises(TypeError):
        rose.probabilities[WindDirection.N] = 0.5  # type: ignore[index]


def test_probabilities_are_stored_in_compass_order():
    rose = WindRose(probabilities=dict(reversed(_probs().items())))
    assert list(rose.probabilities) == list(WindDirection)


def test_pickle_round_trip():
    assert pickle.loads(pickle.dumps(DEFAULT_WIND_ROSE)) == DEFAULT_WIND_ROSE
# endregion


# region WindRose behaviour
def test_uniform_rose():
    rose = WindRose.uniform()
    assert all(p == pytest.approx(1.0 / NUM_SECTORS) for p in rose.probabilities.values())


def test_downwind_probability_uses_opposite_sector():
    for direction in WindDirection:
        assert DEFAULT_WIND_ROSE.downwind_probability(direction) == DEFAULT_WIND_ROSE.probabilities[direction.opposite]


def test_sampling_ignores_input_key_order():
    forward = WindRose(probabilities=dict(DEFAULT_WIND_ROSE.probabilities))
    backward = WindRose(probabilities=dict(reversed(DEFAULT_WIND_ROSE.probabilities.items())))
    a = forward.sample_downwind_radians(500, np.random.default_rng(3))
    b = backward.sample_downwind_radians(500, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_single_sector_wind_stays_within_downwind_sector(north_only_wind_rose: WindRose, rng: np.random.Generator):
    bearings_deg = np.rad2deg(north_only_wind_rose.sample_downwind_radians(1_000, rng))
    assert np.all(bearing_to_sector_index(bearings_deg) == WindDirection.S.sector_index)


def test_sampled_sector_frequencies_match_downwind_probabilities(rng: np.random.Generator):
    n = 200_000
    bearings_deg = np.rad2deg(DEFAULT_WIND_ROSE.sample_downwind_radians(n, rng))
    freqs = np.bincount(bearing_to_sector_index(bearings_deg), minlength=NUM_SECTORS) / n
    expected = [DEFAULT_WIND_ROSE.downwind_probability(d) for d in WindDirection]
    np.testing.assert_allclose(freqs, expected, atol=0.005)
# endregion
