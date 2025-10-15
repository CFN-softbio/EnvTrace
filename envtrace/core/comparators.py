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
    def __init__(self, tol: float = 1e-3) -> None:
        self.tol = tol

    def compare(self, a: Any, b: Any, channel: str) -> bool:
        if _is_number(a) and _is_number(b):
            # Use relative tolerance to match expected behavior:
            # e.g., 0.1 vs 0.1005 is within 1e-2 but not within 1e-3.
            return math.isclose(float(a), float(b), rel_tol=self.tol, abs_tol=0.0)
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
