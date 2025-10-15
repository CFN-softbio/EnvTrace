from envtrace.core.metrics import ContinuousProfileMetric, StructureMetric, MetricResult

def test_continuous_metric_basic_pass():
    gt_series = [(0.0, 20.0), (0.4, 40.0), (0.8, 60.0)]
    pred_series = [(0.0, 20.0), (0.4, 39.5), (0.8, 60.5)]
    metric = ContinuousProfileMetric()
    res = metric.evaluate(gt_series, pred_series, mae_scale=15.0, final_scale=15.0, mae_thresh=5.0, final_thresh=5.0)
    assert res.name == "continuous"
    assert res.binary_pass is True
    assert 0.9 <= res.score <= 1.0

def test_continuous_metric_fail_large_diff():
    gt_series = [(0.0, 20.0), (0.4, 40.0), (0.8, 60.0)]
    pred_series = [(0.0, 50.0), (0.4, 80.0), (0.8, 110.0)]
    metric = ContinuousProfileMetric()
    res = metric.evaluate(gt_series, pred_series, mae_scale=15.0, final_scale=15.0, mae_thresh=5.0, final_thresh=5.0)
    assert res.binary_pass is False
    assert res.score < 0.5

def test_structure_metric_gaps():
    # Alignment with gaps: total 5 pairs, 2 gaps
    alignment = [
        (None, None),  # gap
        (None, None),  # gap
        (None, None),  # gap (we'll make exactly 2 counted gaps below)
        (None, None),
        (None, None),
    ]
    # Build alignment with precisely 2 gaps and 3 matched pairs (gt,pred not None)
    alignment = [
        (None, None),
        (None, None),
        (object(), object()),
        (object(), object()),
        (object(), object()),
    ]
    metric = StructureMetric()
    res = metric.evaluate(alignment)
    assert res.name == "structure"
    assert "gap_rate" in res.details
    expected_gap_rate = 2 / 5
    assert abs(res.details["gap_rate"] - expected_gap_rate) < 1e-9
    assert res.score == 1.0 - expected_gap_rate
