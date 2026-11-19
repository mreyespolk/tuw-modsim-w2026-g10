"""Tests for TreeConfig derived quantities.

Run with: python -m pytest -v tests/test_tree_config.py
"""

import numpy as np
import pytest

from tree_propagation.tree_config import TreeConfig, height_class_at_age


# region Height class at age
@pytest.mark.parametrize(
    ("age", "expected"),
    # Ahorn: T = 12 yr, K = 5
    [(0, 0), (11, 0), (12, 1), (23, 1), (24, 2), (59, 4), (60, 5), (500, 5)],
)
def test_height_class_at_age_scalar(ahorn_tree_config: TreeConfig, age: int, expected: int):
    k = ahorn_tree_config.height_class_at_age(age)
    assert k == expected
    assert isinstance(k, int)


def test_height_class_at_age_array_matches_scalar(ahorn_tree_config: TreeConfig):
    ages = np.arange(0, ahorn_tree_config.mean_life_span + 1)
    ks = ahorn_tree_config.height_class_at_age(ages)
    assert isinstance(ks, np.ndarray)
    assert ks.shape == ages.shape
    assert ks.tolist() == [ahorn_tree_config.height_class_at_age(int(a)) for a in ages]


def test_height_class_at_age_is_monotone_and_capped(tree_species_config: dict[str, TreeConfig]):
    for cfg in tree_species_config.values():
        ks = cfg.height_class_at_age(np.arange(0, 2 * cfg.mean_life_span))
        assert np.all(np.diff(ks) >= 0), cfg.species_name
        assert ks.max() == cfg.maximal_height_class, cfg.species_name


def test_module_function_matches_method(ahorn_tree_config: TreeConfig):
    cfg = ahorn_tree_config
    assert height_class_at_age(37, cfg.years_per_height_class, cfg.maximal_height_class) == cfg.height_class_at_age(37)
# endregion
