"""Reusable value validators for domain dataclasses and agents.

All predicates return a bool (they never raise) and reject `bool` values, which Python otherwise
treats as integers.
"""

import math
import numbers

import numpy as np


def is_strict_int(val: object) -> bool:
    """True for built-in ints, excluding bools."""
    return isinstance(val, int) and not isinstance(val, bool)


def is_finite_real(val: object) -> bool:
    """True for finite real numbers, including numpy scalars (e.g. np.float64 from pandas); excludes bools."""
    return isinstance(val, numbers.Real) and not isinstance(val, (bool, np.bool_)) and math.isfinite(val)


def is_positive_finite(val: object) -> bool:
    """True for finite real numbers strictly greater than 0."""
    return is_finite_real(val) and val > 0  # type: ignore[operator]


def is_non_negative_finite(val: object) -> bool:
    """True for finite real numbers greater than or equal to 0."""
    return is_finite_real(val) and val >= 0  # type: ignore[operator]
