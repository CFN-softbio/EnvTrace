from __future__ import annotations
from typing import Protocol, Any, Dict
import math

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

class ValueComparator(Protocol):
    def compare(self, a: Any, b: Any, channel: str) -> bool: ...

class ExactEqualityComparator:
    def compare(self, a: Any, b: Any, channel: str) -> bool:
        if _is_number(a) and _is_number(b):
            return float(a) == float(b)
        return str(a) == str(b)

class NumericToleranceComparator:
    """
    RELATIVE numeric tolerance: numbers match if math.isclose(a, b, rel_tol=tol,
    abs_tol=0). Only int and float values are compared as numbers; anything else
    falls back to string equality.

    This is not the package default (since the fix-paper-defaults change, the
    default is AbsoluteToleranceComparator(1e-3), as in the paper). Note that a
    relative tolerance never matches a value against exactly 0.0 unless both are 0
    (0.0 against 3.6e-16 fails), and that it widens with the magnitude (313.5 against
    313.6 matches at 1e-3). Also available as RelativeToleranceComparator.
    """
    def __init__(self, tol: float = 1e-3) -> None:
        self.tol = tol

    def compare(self, a: Any, b: Any, channel: str) -> bool:
        if _is_number(a) and _is_number(b):
            # e.g., 0.1 vs 0.1005 is within 1e-2 but not within 1e-3.
            return math.isclose(float(a), float(b), rel_tol=self.tol, abs_tol=0.0)
        return str(a) == str(b)


# Explicit name for the relative-tolerance comparator.
RelativeToleranceComparator = NumericToleranceComparator

class AbsoluteToleranceComparator:
    """
    Absolute numeric tolerance with float coercion, as in the paper's original scorer
    and the package default: values match if abs(float(a) - float(b)) < tol (strict).
    Values that cannot be converted to float (for example None or non-numeric
    strings) fall back to string equality. Numeric strings such as "1" are compared
    as numbers, and so are booleans (True == 1.0).
    """
    def __init__(self, tol: float = 1e-3) -> None:
        self.tol = tol

    def compare(self, a: Any, b: Any, channel: str) -> bool:
        try:
            return abs(float(a) - float(b)) < self.tol
        except (ValueError, TypeError):
            return str(a) == str(b)

class AlwaysTrueComparator:
    def compare(self, a: Any, b: Any, channel: str) -> bool:
        return True

def get_comparator_for_channel(
    default: ValueComparator,
    comparator_map: Dict[str, ValueComparator] | None,
    channel: str
) -> ValueComparator:
    """
    Resolve which comparator to use for a given channel.
    Falls back to the default comparator if no per-channel override exists.
    """
    if comparator_map and channel in comparator_map:
        return comparator_map[channel]
    return default
