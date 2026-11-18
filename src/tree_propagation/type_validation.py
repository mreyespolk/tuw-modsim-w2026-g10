import math


def _is_strict_int(val: object) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_positive_finite(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val) and val > 0


def _is_non_negative_finite(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val) and val >= 0