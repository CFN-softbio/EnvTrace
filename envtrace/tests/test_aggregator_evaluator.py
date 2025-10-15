from envtrace.core import (
    Event, Trace, Evaluator, EvaluateRequest,
    ExactEqualityComparator, NumericToleranceComparator,
    WeightedAggregator, BinaryDecision,
)
from envtrace.core.metrics import MetricResult

def test_weighted_aggregator_custom_formula():
    # Prepare dummy metrics
    metrics = {
        "discrete": MetricResult(name="discrete", score=0.8, binary_pass=True, details={}),
        "timing": MetricResult(name="timing", score=0.9, binary_pass=True, details={}),
        "continuous": MetricResult(name="continuous", score=0.7, binary_pass=True, details={}),
        "structure": MetricResult(name="structure", score=0.95, binary_pass=True, details={}),
    }
    def formula(m):
        return 0.5 * m["discrete"].score + 0.2 * m["timing"].score + 0.2 * m["continuous"].score + 0.1 * m["structure"].score
    agg = WeightedAggregator(formula=formula)
    score = agg.aggregate(metrics)
    assert abs(score - (0.5*0.8 + 0.2*0.9 + 0.2*0.7 + 0.1*0.95)) < 1e-9

def test_weighted_aggregator_defaults_no_continuous():
    metrics = {
        "discrete": MetricResult(name="discrete", score=0.5, binary_pass=True, details={}),
        "timing": MetricResult(name="timing", score=0.75, binary_pass=True, details={}),
    }
    agg = WeightedAggregator()
    score = agg.aggregate(metrics)
    assert abs(score - (0.8*0.5 + 0.2*0.75)) < 1e-9

def test_weighted_aggregator_defaults_with_continuous():
    metrics = {
        "discrete": MetricResult(name="discrete", score=0.5, binary_pass=True, details={}),
        "timing": MetricResult(name="timing", score=0.75, binary_pass=True, details={}),
        "continuous": MetricResult(name="continuous", score=0.9, binary_pass=True, details={}),
        # Additional metrics not referenced by default weights should be ignored:
        "structure": MetricResult(name="structure", score=0.1, binary_pass=True, details={}),
    }
    agg = WeightedAggregator()
    score = agg.aggregate(metrics)
    assert abs(score - (0.6*0.5 + 0.2*0.75 + 0.2*0.9)) < 1e-9

def test_binary_decision_logic():
    bd = BinaryDecision()
    metrics = {
        "discrete": MetricResult(name="discrete", score=0.3, binary_pass=True, details={}),
        "timing": MetricResult(name="timing", score=0.3, binary_pass=True, details={}),
    }
    assert bd.decide(metrics) is True
    # Fail timing
    metrics["timing"].binary_pass = False
    assert bd.decide(metrics) is False
    # Fail discrete
    metrics["discrete"].binary_pass = False
    metrics["timing"].binary_pass = True
    assert bd.decide(metrics) is False

def test_evaluator_best_of():
    # GT variant 1: mismatched last value
    gt1 = Trace([
        Event("motor:x", 0.00, 0.0),
        Event("det:Acquire", 0.10, 1),
        Event("det:Acquire", 1.10, 1),  # wrong final value
    ])
    # GT variant 2: perfect
    gt2 = Trace([
        Event("motor:x", 0.00, 0.0),
        Event("det:Acquire", 0.10, 1),
        Event("det:Acquire", 1.10, 0),
    ])
    pred = Trace([
        Event("motor:x", 0.00, 0.0),
        Event("det:Acquire", 0.10, 1),
        Event("det:Acquire", 1.10, 0),
    ])
    ev = Evaluator.default()
    result, idx = ev.evaluate_best_of([gt1, gt2], pred)
    assert idx == 1
    assert result.accuracy is True
    assert result.metrics["discrete"].binary_pass is True

def test_evaluator_with_continuous_and_structure():
    # Traces include discrete actions + temperature series
    gt = Trace([
        Event("motor:x", 0.00, 0.0),
        Event("det:AcquireTime", 0.05, 1.0),
        Event("det:Acquire", 0.10, 1),
        Event("det:Acquire", 1.10, 0),
        Event("stage:temp", 0.20, 20.0),
        Event("stage:temp", 0.40, 40.0),
        Event("motor:y", 1.20, -0.2),
    ])
    pred = Trace([
        Event("motor:x", 0.00, 0.0),
        Event("det:AcquireTime", 0.05, 1.0),
        Event("det:Acquire", 0.12, 1),
        Event("det:Acquire", 1.08, 0),
        Event("stage:temp", 0.20, 20.0),
        Event("stage:temp", 0.40, 39.5),
        Event("motor:y", 1.21, -0.2),
    ])

    comp_map = {
        "det:Acquire": ExactEqualityComparator(),
        "motor:x": NumericToleranceComparator(1e-2),
        "motor:y": NumericToleranceComparator(1e-2),
    }
    continuous_cfg = {
        "stage:temp": {"mae_scale": 15.0, "final_scale": 15.0, "mae_thresh": 5.0, "final_thresh": 5.0, "weight": 1.0}
    }
    ignore_channels = {"det:AcquireTime"}

    ev = Evaluator()
    res = ev.evaluate(EvaluateRequest(
        gt=gt,
        pred=pred,
        comparator=NumericToleranceComparator(1e-3),
        comparators_by_channel=comp_map,
        ignore_channels=ignore_channels,
        continuous_channels=continuous_cfg,
        include_structure=True,
    ))

    # Basic assertions
    assert "discrete" in res.metrics
    assert "timing" in res.metrics
    assert "continuous" in res.metrics
    assert "structure" in res.metrics

    # With small differences, everything should pass
    assert res.metrics["discrete"].binary_pass is True
    assert res.metrics["timing"].binary_pass is True
    assert res.metrics["continuous"].binary_pass is True

    # Full score should be reasonable
    assert 0.7 <= res.full_score <= 1.0
