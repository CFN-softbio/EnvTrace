from envtrace.core.comparators import (
    ExactEqualityComparator,
    NumericToleranceComparator,
    AlwaysTrueComparator,
    get_comparator_for_channel,
)

def test_exact_equality_comparator():
    comp = ExactEqualityComparator()
    assert comp.compare(1, 1, "ch")
    assert comp.compare("x", "x", "ch")
    assert not comp.compare(1, 2, "ch")

def test_numeric_tolerance_comparator():
    comp = NumericToleranceComparator(tol=1e-3)
    assert comp.compare(0.1, 0.1005, "ch") is False
    comp2 = NumericToleranceComparator(tol=1e-2)
    assert comp2.compare(0.1, 0.1005, "ch") is True

def test_get_comparator_for_channel():
    default = ExactEqualityComparator()
    overrides = {"special": NumericToleranceComparator(tol=1e-2)}
    assert get_comparator_for_channel(default, overrides, "special") is overrides["special"]
    assert get_comparator_for_channel(default, overrides, "other") is default
