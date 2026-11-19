"""Tests for the public value validators.

Run with: python -m pytest -v tests/test_type_validation.py
"""

import numpy as np
import pytest

from tree_propagation.type_validation import (
    is_finite_real,
    is_non_negative_finite,
    is_positive_finite,
    is_strict_int,
)

NAN, INF = float("nan"), float("inf")
NON_NUMERIC = ["1", None, True, False, np.bool_(True), [1]]


@pytest.mark.parametrize(("val", "expected"), [(0, True), (-3, True), (1.0, False), *[(v, False) for v in NON_NUMERIC]])
def test_is_strict_int(val, expected: bool):
    assert is_strict_int(val) is expected


@pytest.mark.parametrize(
    ("val", "expected"),
    [(0, True), (-2.5, True), (np.float64(1.5), True), (np.int64(-4), True),
     (NAN, False), (INF, False), (-INF, False), *[(v, False) for v in NON_NUMERIC]],
)
def test_is_finite_real(val, expected: bool):
    assert is_finite_real(val) is expected


@pytest.mark.parametrize(
    ("val", "positive", "non_negative"),
    [(1, True, True), (0.5, True, True), (0, False, True), (0.0, False, True), (-0.1, False, False),
     (INF, False, False), (NAN, False, False), (True, False, False), (None, False, False)],
)
def test_sign_predicates(val, positive: bool, non_negative: bool):
    assert is_positive_finite(val) is positive
    assert is_non_negative_finite(val) is non_negative
