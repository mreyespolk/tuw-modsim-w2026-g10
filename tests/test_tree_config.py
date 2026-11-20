"""Tests for TreeConfig derived quantities.

Run with: python -m pytest -v tests/test_tree_config.py
"""

import dataclasses

import numpy as np
import pytest

from tree_propagation.tree_config import TreeConfig


@pytest.fixture
def t12_k5_config(ahorn_tree_config: TreeConfig) -> TreeConfig:
    """Ahorn with fixed T = 12 yr, K = 5 so expected height classes don't track CSV edits."""
    return dataclasses.replace(ahorn_tree_config, years_per_height_class=12, maximal_height_class=5)


# region Height class at age
@pytest.mark.parametrize(
    ("age", "expected"),
    # t12_k5_config: T = 12 yr, K = 5
    [(0, 0), (11, 0), (12, 1), (23, 1), (24, 2), (59, 4), (60, 5), (500, 5)],
)
def test_height_class_at_age_scalar(t12_k5_config: TreeConfig, age: int, expected: int):
    k = t12_k5_config.height_class_at_age(age)
    assert k == expected
    assert isinstance(k, int)


def test_height_class_at_age_array_matches_scalar(ahorn_tree_config: TreeConfig):
    ages = np.arange(0, ahorn_tree_config.mean_life_span + 1)
    ks = ahorn_tree_config.height_classes_at_ages(ages)
    assert isinstance(ks, np.ndarray)
    assert ks.shape == ages.shape
    assert ks.tolist() == [ahorn_tree_config.height_class_at_age(int(a)) for a in ages]


def test_height_class_at_age_is_monotone_and_capped(tree_species_config: dict[str, TreeConfig]):
    for cfg in tree_species_config.values():
        ks = cfg.height_classes_at_ages(np.arange(0, 2 * cfg.mean_life_span))
        assert np.all(np.diff(ks) >= 0), cfg.species_name
        assert ks.max() == cfg.maximal_height_class, cfg.species_name


def test_module_function_matches_method(ahorn_tree_config: TreeConfig):
    cfg = ahorn_tree_config
    assert (
        TreeConfig.height_classes_from_params(np.array(37),
                                              cfg.years_per_height_class,
                                              cfg.maximal_height_class
        ) == cfg.height_class_at_age(37)
    )
# endregion
