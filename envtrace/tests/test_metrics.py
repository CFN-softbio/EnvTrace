from envtrace.core.event import Event
from envtrace.core.metrics import DiscreteMatchMetric, TimingMetric
from envtrace.core.comparators import NumericToleranceComparator

def test_discrete_metric_basic():
    alignment = [
        (Event("a", 0.0, 1, {}), Event("a", 0.0, 1, {})),
        (Event("b", 0.1, 2, {}), Event("b", 0.1, 2, {})),
        (Event("c", 0.2, 3, {}), Event("c", 0.2, 3, {})),
    ]
    metric = DiscreteMatchMetric()
    res = metric.evaluate(alignment)
    assert res.name == "discrete"
    assert res.score == 1.0
    assert res.binary_pass is True
    assert res.details["value_matches"] == 3
    assert res.details["total_pairs"] == 3

def test_timing_metric_perfect():
    alignment = [
        (Event("a", 0.0, 1, {}), Event("a", 0.0, 1, {})),
        (Event("b", 1.0, 2, {}), Event("b", 1.0, 2, {})),
        (Event("c", 2.0, 3, {}), Event("c", 2.0, 3, {})),
    ]
    comp = NumericToleranceComparator()
    metric = TimingMetric()
    res = metric.evaluate(alignment, comparator=comp)
    assert res.name == "timing"
    assert res.binary_pass is True
    assert res.score > 0.95
