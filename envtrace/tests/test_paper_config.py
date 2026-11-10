"""Tests for the options that reproduce the paper's original scorer (examples/paper)."""
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from envtrace.core import (
    AbsoluteToleranceComparator,
    DifflibAligner,
    DiscreteMatchMetric,
    EvaluateRequest,
    Evaluator,
    Event,
    NumericToleranceComparator,
    TimingMetric,
    Trace,
)
from envtrace.io import build_from_config, load_config
from envtrace.reporting import build_json_report

PAPER_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "paper" / "paper_config.json"
TEMP = "XF:11BM-ES:{LINKAM}:TEMP"


def _pairs(gt, pred):
    return DifflibAligner.align(gt, pred, lambda e: e.channel)


def test_absolute_tolerance_comparator():
    c = AbsoluteToleranceComparator(tol=1e-3)
    assert c.compare(1.0, 1.0009, "x")
    assert not c.compare(1.0, 1.001 + 1e-12, "x")
    assert not c.compare(0.0, 1e-3, "x")          # strict <
    assert c.compare("1", 1.0, "x")               # numeric strings are numbers
    assert c.compare(True, 1, "x")                # float(True) == 1.0
    assert c.compare("", "", "x")                 # float("") fails: string equality
    assert not c.compare("a", "b", "x")
    assert c.compare(None, None, "x")
    # relative tolerance (the package default) differs for large values
    assert NumericToleranceComparator(tol=1e-3).compare(1000.0, 1000.5, "x")
    assert not c.compare(1000.0, 1000.5, "x")


def test_require_channel_match_in_replace_block():
    gt = [Event("A", 0.0, 0)]
    pred = [Event("B", 0.0, 0)]
    pairs = _pairs(gt, pred)
    assert pairs == [(gt[0], pred[0])]            # a 'replace' pair with different channels
    legacy = DiscreteMatchMetric().evaluate(pairs)
    assert legacy.score == 1.0                    # earlier behaviour, kept as default
    strict = DiscreteMatchMetric(require_channel_match=True).evaluate(pairs)
    assert strict.score == 0.0 and strict.binary_pass is False
    t = TimingMetric().evaluate(pairs, comparator=NumericToleranceComparator(), require_channel_match=True)
    assert t.score == 0.0 and t.binary_pass is False


def test_empty_reference_fails():
    assert DiscreteMatchMetric().evaluate([]).score == 1.0
    res = DiscreteMatchMetric(empty_reference_fails=True).evaluate([])
    assert res.score == 0.0 and res.binary_pass is False
    assert res.details["mismatch_rate"] == 1.0
    pred_only = _pairs([], [Event("A", 0.0, 1)])
    assert DiscreteMatchMetric(empty_reference_fails=True).evaluate(pred_only).score == 0.0


def test_pearson_regression_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.sort(rng.uniform(0, 100, 6))
        y = x * rng.uniform(0.7, 1.3) + rng.normal(0, 2, 6)
        x, y = x - x[0], y - y[0]
        slope, _, r2 = TimingMetric._linregress_pearson(x, y)
        ref = stats.linregress(x, y)
        assert abs(slope - ref.slope) < 1e-12
        assert abs(r2 - ref.rvalue ** 2) < 1e-12


def test_pearson_regression_degenerate():
    # constant prediction: polyfit reports R^2 = 1, Pearson r is undefined (NaN)
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    assert TimingMetric()._linregress_np(x, y)[2] == 1.0
    slope, _, r2 = TimingMetric._linregress_pearson(x, y)
    assert slope == 0.0 and math.isnan(r2)
    with pytest.raises(ValueError):
        TimingMetric(regression="ransac")


def test_sort_by_time_false_keeps_recorded_order():
    gt = Trace([Event("A", 0.0, 1), Event("B", 1.0, 2)])
    pred = Trace([Event("A", 1.0, 1), Event("B", 0.9, 2)])   # recorded A then B, B earlier
    ev = Evaluator()
    by_time = ev.evaluate(EvaluateRequest(gt=gt, pred=pred))
    recorded = ev.evaluate(EvaluateRequest(gt=gt, pred=pred, sort_by_time=False))
    assert [p.channel for _, p in by_time.alignment if p] == ["B", "A"]
    assert by_time.metrics["discrete"].score < 1.0
    assert [p.channel for _, p in recorded.alignment if p] == ["A", "B"]
    assert recorded.metrics["discrete"].score == 1.0


def test_continuous_weighting_if_present():
    cc = {TEMP: {}}
    gt = Trace([Event("A", 0.0, 1), Event("B", 1.0, 2)])
    pred = Trace([Event("A", 0.0, 1), Event("B", 1.0, 3)])
    ev = Evaluator()
    always = ev.evaluate(EvaluateRequest(gt=gt, pred=pred, continuous_channels=cc))
    present = ev.evaluate(EvaluateRequest(gt=gt, pred=pred, continuous_channels=cc,
                                          continuous_weighting="if_present"))
    # no temperature event: continuous scores 1.0 in both, but only 'always' weights it
    assert always.metrics["continuous"].score == 1.0
    assert math.isclose(always.full_score, 0.6 * 0.5 + 0.2 * 1.0 + 0.2 * 1.0)
    assert math.isclose(present.full_score, 0.8 * 0.5 + 0.2 * 1.0)
    rep = build_json_report(present, evaluator=ev)
    assert rep["aggregation"]["weights"] == {"discrete": 0.8, "timing": 0.2}
    # with a temperature event in either trace the continuous weights apply
    pred_t = Trace(pred.events + [Event(TEMP, 0.5, 25.0)])
    res = ev.evaluate(EvaluateRequest(gt=gt, pred=pred_t, continuous_channels=cc,
                                      continuous_weighting="if_present"))
    assert res.metrics["continuous"].score == 0.0
    assert math.isclose(res.full_score, 0.6 * 0.5 + 0.2 * 1.0)
    with pytest.raises(ValueError):
        ev.evaluate(EvaluateRequest(gt=gt, pred=pred, continuous_weighting="sometimes"))


def test_best_of_with_template():
    pred = Trace([Event("A", 0.0, 1), Event("B", 1.0, 2)])
    refs = [Trace([Event("A", 0.0, 1)]),
            Trace([Event("A", 0.0, 1), Event("B", 1.0, 2)]),
            Trace([Event("A", 0.0, 1), Event("B", 1.0, 2)])]
    tmpl = EvaluateRequest(gt=Trace([]), pred=Trace([]), require_channel_match=True,
                           timing_metric=TimingMetric(regression="pearson"))
    res, idx = Evaluator().evaluate_best_of(refs, pred, template=tmpl)
    assert idx == 1                               # ties keep the earlier reference
    assert res.full_score == 1.0


def test_paper_config_loads_and_scores():
    cfg = load_config(PAPER_CONFIG)
    ev, req = build_from_config(Trace([]), Trace([]), cfg)
    assert req.sort_by_time is False and req.require_channel_match is True
    assert req.empty_reference_fails is True and req.continuous_weighting == "if_present"
    assert isinstance(req.comparator, AbsoluteToleranceComparator)
    assert req.timing_metric.regression == "pearson"
    assert ev.aggregator.weights is None
    assert ev.aggregator.weights_no_cont == {"discrete": 0.8, "timing": 0.2}
    assert ev.aggregator.weights_with_cont == {"discrete": 0.6, "timing": 0.2, "continuous": 0.2}
    # sam.setOrigin parameters are not compared; motor values are
    gt = Trace([Event("sam.setOrigin", 0.0, "[0], None"), Event("motor:x", 1.0, 2.0),
                Event("motor:x", 2.0, 3.0)])
    pred = Trace([Event("sam.setOrigin", 0.0, "[1], None"), Event("motor:x", 1.0, 2.0),
                  Event("motor:x", 2.0, 3.0)])
    res = ev.evaluate(replace(req, gt=gt, pred=pred))
    assert res.accuracy is True and res.full_score == 1.0
    # empty reference: score 0 and fail, as in the original scorer
    res = ev.evaluate(replace(req, gt=Trace([]), pred=Trace([])))
    assert res.full_score == 0.0 and res.accuracy is False
    assert res.metrics["continuous"].score == 1.0
