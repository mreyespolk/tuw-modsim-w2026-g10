"""Tests for TreeConfig derived quantities and the species CSV loader.

Run with: python -m pytest -v tests/test_tree_config.py
"""

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tree_propagation.tree_config import CSV_COLUMNS, TreeConfig, load_tree_configs_from_csv
from tree_propagation.types import DispersalType


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


# region CSV schema
def test_csv_columns_cover_every_tree_config_field():
    assert set(CSV_COLUMNS) == {f.name for f in dataclasses.fields(TreeConfig)}


def test_species_csv_header_matches_csv_columns(species_csv_path: Path):
    with species_csv_path.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert set(header) == set(CSV_COLUMNS.values())


def test_blank_optional_dispersal_traits_parse_to_none(tree_species_config: dict[str, TreeConfig]):
    for cfg in tree_species_config.values():
        if cfg.dispersal_type == DispersalType.WIND:
            assert cfg.mean_dispersal_distance is None, cfg.species_name
        else:
            assert cfg.wind_spreading_factor is None, cfg.species_name


def test_loader_reports_missing_columns(tmp_path: Path):
    columns = [c for c in CSV_COLUMNS.values() if c != "tau"]
    csv_file = tmp_path / "species.csv"
    csv_file.write_text(",".join(columns) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required columns: \['tau'\]"):
        load_tree_configs_from_csv(csv_file)


def test_loader_reports_invalid_row_with_line_number(tmp_path: Path):
    csv_file = tmp_path / "species.csv"
    csv_file.write_text(
        ",".join(CSV_COLUMNS.values()) + "\n"
        "Ahorn,wind,0.9,,2.0,20,12,8,300,0.8\n"
        "Bad,wind,,,2.0,20,12,8,300,0.8\n",  # wind species without f
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        load_tree_configs_from_csv(csv_file)
# endregion
