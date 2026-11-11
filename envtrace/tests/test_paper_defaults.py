"""Tests for the fix-paper-defaults change: new defaults, the options that restore the
earlier defaults, the degenerate-timing guard, evaluate_best_of(**kwargs), the two
corrections of the paper's scorer (the continuous facet honours the ignore list; one
matched pair leaves timing undefined) and the opt-in harness rules (no-discrete-reference
rule, exact code match)."""

import json
import math
from dataclasses import replace

import pytest

from envtrace.core import (
    AbsoluteToleranceComparator,
    BinaryDecision,
    EvaluateRequest,
    Evaluator,
    Event,
    NumericToleranceComparator,
    RelativeToleranceComparator,
    TimingMetric,
    Trace,
    exact_code_match,
    exact_code_match_result,
    strip_comments,
)
from envtrace.io import build_from_config
from envtrace.reporting import build_json_report, format_text_report

TEMP = "XF:11BM-ES:{LINKAM}:TEMP"
CC = {TEMP: {}}


def tr(x):
    return Trace([Event(c, float(t), v) for c, t, v in x])


def ramp(t0, v0, step, n):
    return [(TEMP, t0 + i, v0 + i * step) for i in range(n)]


def legacy_request(gt, pred, **kw):
    """Every earlier default, requested explicitly."""
    return EvaluateRequest(
        gt=gt,
        pred=pred,
        comparator=NumericToleranceComparator(1e-3),
        sort_by_time=True,
        require_channel_match=False,
        empty_reference_fails=False,
        continuous_weighting="always",
        **kw,
    )


def legacy_evaluator():
    return Evaluator(decision=BinaryDecision(require_continuous=False))


# --- the earlier defaults stay reachable -------------------------------------------
def test_legacy_options_reproduce_earlier_defaults():
    # different channels, equal values: matched only without the channel check
    r = legacy_evaluator().evaluate(legacy_request(tr([("X", 0, 0.5)]), tr([("Y", 0, 0.5)])))
    assert r.metrics["discrete"].score == 1.0
    # two empty traces with temperature configured: 0.6 * 1 + 0.2 * 0 (no timing pair)
    # + 0.2 * 1 = 0.8 under the old defaults (the new defaults give 0)
    r = legacy_evaluator().evaluate(legacy_request(tr([]), tr([]), continuous_channels=CC))
    assert r.full_score == pytest.approx(0.8) and r.accuracy is False
    assert (
        Evaluator()
        .evaluate(EvaluateRequest(gt=tr([]), pred=tr([]), continuous_channels=CC))
        .full_score
        == 0.0
    )
    # old decision ignored the continuous check
    gt = [("SP", 0.0, 50.0)] + ramp(0.5, 25.0, 1.0, 26)
    pred = [("SP", 0.0, 50.0)] + ramp(0.5, 25.0, 3.0, 26)
    assert (
        legacy_evaluator()
        .evaluate(legacy_request(tr(gt), tr(pred), continuous_channels=CC))
        .accuracy
        is True
    )
    assert (
        Evaluator()
        .evaluate(EvaluateRequest(gt=tr(gt), pred=tr(pred), continuous_channels=CC))
        .accuracy
        is False
    )


def test_relative_tolerance_alias():
    assert RelativeToleranceComparator is NumericToleranceComparator
    assert isinstance(
        EvaluateRequest(gt=Trace(), pred=Trace()).comparator, AbsoluteToleranceComparator
    )


def test_config_defaults_and_unknown_comparator():
    ev, req = build_from_config(Trace(), Trace(), {})
    assert req.sort_by_time is False and req.require_channel_match is True
    assert req.empty_reference_fails is True and req.continuous_weighting == "if_present"
    assert (
        req.ignore_applies_to_continuous is True and req.no_discrete_reference_scores_zero is False
    )
    assert req.timing_metric.single_matched_pair == "undefined"
    _, req = build_from_config(
        Trace(), Trace(),
        {"ignore_applies_to_continuous": False, "timing": {"single_matched_pair": "full_credit"}},
    )
    assert req.ignore_applies_to_continuous is False
    assert req.timing_metric.single_matched_pair == "full_credit"
    with pytest.raises(ValueError):
        build_from_config(Trace(), Trace(), {"timing": {"single_matched_pair": "half"}})
    assert isinstance(req.comparator, AbsoluteToleranceComparator)
    assert ev.decision.require_continuous is True
    _, req = build_from_config(
        Trace(), Trace(), {"comparators": {"default": {"type": "relative_tolerance"}}}
    )
    assert isinstance(req.comparator, NumericToleranceComparator)
    ev, _ = build_from_config(Trace(), Trace(), {"decision": {"require_continuous": False}})
    assert ev.decision.require_continuous is False
    with pytest.raises(ValueError):
        build_from_config(Trace(), Trace(), {"comparators": {"default": {"type": "fuzzy"}}})


def test_decision_report_lists_continuous_when_involved():
    gt = [("SP", 0.0, 50.0)] + ramp(0.5, 25.0, 1.0, 5)
    ev = Evaluator()
    req = EvaluateRequest(gt=tr(gt), pred=tr(gt), continuous_channels=CC)
    rep = build_json_report(ev.evaluate(req), evaluator=ev, req=req)
    assert rep["decision"]["required_metrics"] == ["continuous"]
    assert rep["decision"]["passes"]["continuous"] is True
    req = EvaluateRequest(
        gt=tr([("SP", 0.0, 1.0)]), pred=tr([("SP", 0.0, 1.0)]), continuous_channels=CC
    )
    rep = build_json_report(ev.evaluate(req), evaluator=ev, req=req)
    assert rep["decision"]["required_metrics"] == []


# --- N8: degenerate timing ---------------------------------------------------------
@pytest.mark.parametrize("regression", ["polyfit", "pearson"])
def test_degenerate_reference_timestamps(regression):
    tm = TimingMetric(regression=regression)
    comp = AbsoluteToleranceComparator()
    gt = [Event("A", 5.0, 1), Event("B", 5.0, 2), Event("C", 5.0, 3)]
    same = [Event("A", 7.0, 1), Event("B", 7.0, 2), Event("C", 7.0, 3)]
    spread = [Event("A", 7.0, 1), Event("B", 8.0, 2), Event("C", 9.0, 3)]
    both = tm.evaluate(list(zip(gt, same)), comparator=comp)
    assert both.score == 1.0 and both.binary_pass is True
    ref = tm.evaluate(list(zip(gt, spread)), comparator=comp)
    assert ref.score == 0.0 and ref.binary_pass is False
    # constant prediction against a spread reference: no NaN, no R^2 = 1 credit
    pred_const = tm.evaluate(list(zip(spread, same)), comparator=comp)
    assert not math.isnan(pred_const.score)
    assert pred_const.details["r2"] == 0.0 and pred_const.score == 0.0
    assert pred_const.binary_pass is False


# --- D9/N9: evaluate_best_of passes every field ------------------------------------
def test_best_of_kwargs_build_full_request():
    gt = tr([("SP", 0.0, 50.0)] + ramp(0.5, 25.0, 1.0, 5))
    res, idx = Evaluator().evaluate_best_of([gt], gt, continuous_channels=CC)
    assert idx == 0 and res.full_score == pytest.approx(1.0) and res.accuracy is True
    # one discrete pair (SP), the TEMP events are not aligned as discrete events
    assert len(res.alignment) == 1
    # use_continuous (honoured only with continuous_weighting="always") no longer gives
    # a perfect match 0.8 by default
    res, _ = Evaluator().evaluate_best_of([gt], gt, use_continuous=True)
    assert res.full_score == pytest.approx(1.0)
    with pytest.raises(TypeError):
        Evaluator().evaluate_best_of([gt], gt, not_a_field=1)


def test_best_of_kwargs_override_template():
    tmpl = EvaluateRequest(gt=Trace(), pred=Trace(), require_channel_match=False)
    gt, pred = tr([("X", 0, 0.5)]), tr([("Y", 0, 0.5)])
    res, _ = Evaluator().evaluate_best_of([gt], pred, template=tmpl)
    assert res.metrics["discrete"].score == 1.0
    res, _ = Evaluator().evaluate_best_of([gt], pred, template=tmpl, require_channel_match=True)
    assert res.metrics["discrete"].score == 0.0


# --- R6: continuous facet honours the ignore list (default) --------------------------
def test_ignore_applies_to_continuous():
    gt = [("SP", 0.0, 50.0), ("SP", 1.0, 60.0)]
    pred = gt + ramp(1.5, 25.0, 1.0, 5)  # temperature only in the prediction
    base = EvaluateRequest(gt=tr(gt), pred=tr(pred), continuous_channels=CC, ignore_channels={TEMP})
    ev = Evaluator()
    default = ev.evaluate(base)
    assert base.ignore_applies_to_continuous is True
    assert default.metrics["continuous"].details["involved"] is False
    assert default.metrics["continuous"].score == 1.0
    # no continuous event left: weights 0.8/0.2, strict decision without the continuous check
    assert default.full_score == pytest.approx(1.0) and default.accuracy is True
    # the paper's scorer and the earlier default: the ignored channel is still scored
    paper = ev.evaluate(replace(base, ignore_applies_to_continuous=False))
    assert paper.metrics["continuous"].score == 0.0
    assert paper.full_score == pytest.approx(0.8) and paper.accuracy is False
    # a continuous channel that is not ignored is scored as before
    other = ev.evaluate(replace(base, ignore_channels={"OTHER"}))
    assert other.full_score == pytest.approx(0.8) and other.accuracy is False
    # TEMP stays out of the discrete alignment either way
    assert len(default.alignment) == len(paper.alignment) == 2


def test_ignore_applies_to_continuous_partial():
    # two continuous channels, one ignored: only the other one is scored
    cc = {TEMP: {}, "T2": {}}
    gt = [("SP", 0.0, 1.0), ("SP", 1.0, 2.0), ("T2", 1.5, 30.0)]
    pred = [("SP", 0.0, 1.0), ("SP", 1.0, 2.0), ("T2", 1.5, 30.0)] + ramp(2.0, 25.0, 1.0, 3)
    req = EvaluateRequest(gt=tr(gt), pred=tr(pred), continuous_channels=cc, ignore_channels={TEMP})
    res = Evaluator().evaluate(req)
    assert set(res.metrics["continuous"].details["per_channel"]) == {"T2"}
    assert res.full_score == pytest.approx(1.0) and res.accuracy is True
    old = Evaluator().evaluate(replace(req, ignore_applies_to_continuous=False))
    assert set(old.metrics["continuous"].details["per_channel"]) == {TEMP, "T2"}
    assert old.metrics["continuous"].score == pytest.approx(0.5)
    assert old.full_score == pytest.approx(0.9) and old.accuracy is False


# --- R2/N4: one matched pair leaves timing undefined (default) ----------------------
ONE_PAIR = ([("X", 0.0, 1.0), ("Y", 1.0, 2.0)], [("X", 0.0, 1.0), ("Y", 1.0, 3.0)])


def test_single_matched_pair_timing_undefined():
    gt, pred = ONE_PAIR
    res = Evaluator().evaluate(EvaluateRequest(gt=tr(gt), pred=tr(pred)))
    t = res.metrics["timing"]
    assert math.isnan(t.score) and t.details["undefined"] is True and t.details["matched_pairs"] == 1
    assert t.binary_pass is True                 # does not apply, so it does not fail the decision
    assert res.metrics["discrete"].score == 0.5
    assert res.full_score == pytest.approx(0.5)  # 0.8/0.2 renormalised: the PV match rate
    assert res.accuracy is False                 # not PV-exact
    # the paper's scorer and the earlier default: timing 1.0
    old = Evaluator().evaluate(EvaluateRequest(gt=tr(gt), pred=tr(pred),
                                               timing_metric=TimingMetric(single_matched_pair="full_credit")))
    assert old.metrics["timing"].score == 1.0
    assert old.full_score == pytest.approx(0.6) and old.accuracy is False
    with pytest.raises(ValueError):
        TimingMetric(single_matched_pair="half")


def test_single_matched_pair_with_continuous_renormalises():
    # 0.6/0.2/0.2 with timing undefined: 0.75 x PV + 0.25 x continuous
    gt = ONE_PAIR[0] + ramp(1.5, 25.0, 1.0, 5)
    pred = ONE_PAIR[1] + ramp(1.5, 25.0, 1.0, 5)
    ev = Evaluator()
    req = EvaluateRequest(gt=tr(gt), pred=tr(pred), continuous_channels=CC)
    res = ev.evaluate(req)
    assert res.metrics["continuous"].score == pytest.approx(1.0)
    assert res.full_score == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    assert ev.aggregator.effective_weights(res.metrics) == pytest.approx({"discrete": 0.75, "continuous": 0.25})
    old = ev.evaluate(replace(req, timing_metric=TimingMetric(single_matched_pair="full_credit")))
    assert old.full_score == pytest.approx(0.6 * 0.5 + 0.2 + 0.2)


def test_single_matched_pair_exact_still_passes():
    # one event in each trace, equal: PV-exact, nothing to time; the decision is unchanged
    one = tr([("X", 0.0, 1.0)])
    res = Evaluator().evaluate(EvaluateRequest(gt=one, pred=tr([("X", 5.0, 1.0)])))
    assert math.isnan(res.metrics["timing"].score)
    assert res.full_score == 1.0 and res.accuracy is True


def test_zero_and_two_matched_pairs_unchanged():
    tm, comp = TimingMetric(), AbsoluteToleranceComparator()
    zero = tm.evaluate([(Event("X", 0.0, 1.0), Event("X", 0.0, 2.0))], comparator=comp)
    assert zero.score == 0.0 and zero.binary_pass is False
    pairs = [(Event("X", 0.0, 1.0), Event("X", 0.0, 1.0)), (Event("Y", 1.0, 2.0), Event("Y", 1.0, 2.0))]
    for mode in ("undefined", "full_credit"):
        two = TimingMetric(single_matched_pair=mode).evaluate(pairs, comparator=comp)
        assert two.score == pytest.approx(1.0) and two.binary_pass is True


def test_undefined_timing_in_reports():
    gt, pred = ONE_PAIR
    ev = Evaluator()
    req = EvaluateRequest(gt=tr(gt), pred=tr(pred))
    res = ev.evaluate(req)
    rep = build_json_report(res, evaluator=ev, req=req)
    assert rep["metrics"]["timing"]["score"] is None          # NaN written as null
    assert rep["metrics"]["timing"]["details"]["score"] is None
    assert rep["aggregation"]["weights"] == pytest.approx({"discrete": 1.0})
    assert rep["aggregation"]["contributions"]["_sum"] == pytest.approx(res.full_score)
    json.dumps(rep, allow_nan=False)                           # valid JSON
    text = format_text_report(res, evaluator=ev, req=req)
    assert "timing: score=undefined" in text


def test_best_of_with_undefined_timing():
    # selection by the corrected full score: a one-pair reference scores its PV rate only
    pred = tr([("X", 0.0, 1.0), ("Y", 1.0, 3.0)])
    refs = [tr([("X", 0.0, 1.0), ("Y", 1.0, 2.0)]),                   # one matched pair: 0.5
            tr([("X", 0.0, 1.0), ("Y", 1.0, 3.0), ("Z", 2.0, 1.0)])]  # two: 0.8 x 2/3 + 0.2
    res, idx = Evaluator().evaluate_best_of(refs, pred)
    assert idx == 1 and res.full_score == pytest.approx(0.8 * 2 / 3 + 0.2)


def test_all_weighted_facets_undefined_gives_nan():
    from envtrace.core import WeightedAggregator
    gt, pred = ONE_PAIR
    ev = Evaluator(aggregator=WeightedAggregator(weights={"timing": 1.0}))
    res = ev.evaluate(EvaluateRequest(gt=tr(gt), pred=tr(pred)))
    assert math.isnan(res.full_score)


# --- D10: harness rules (opt-in) ---------------------------------------------------
def test_no_discrete_reference_rule():
    temp_only = tr(ramp(0.0, 25.0, 1.0, 5))
    ev = Evaluator()
    req = EvaluateRequest(gt=temp_only, pred=temp_only, continuous_channels=CC)
    assert ev.evaluate(req).full_score == pytest.approx(0.2)  # empty-reference rule, temp facet 1.0
    zero = ev.evaluate(replace(req, no_discrete_reference_scores_zero=True))
    assert zero.full_score == 0.0 and zero.accuracy is False
    assert zero.metrics["continuous"].score == 0.0
    # best-of: applies only if NO reference has a discrete event
    res, idx = ev.evaluate_best_of(
        [temp_only, temp_only], temp_only, template=req, no_discrete_reference_scores_zero=True
    )
    assert idx == -1 and res.full_score == 0.0
    disc = tr([("SP", 0.0, 1.0)])
    res, idx = ev.evaluate_best_of(
        [disc, temp_only], temp_only, template=req, no_discrete_reference_scores_zero=True
    )
    assert idx == 1 and res.full_score == pytest.approx(
        0.2
    )  # the temperature-only one is still scored


def test_exact_code_match():
    ref = "RE(bp.count([det]))  # count once\n\n"
    assert strip_comments(ref) == "RE(bp.count([det]))"
    assert (
        exact_code_match("  RE(bp.count([det]))\n# done", ["x = 1", ref]) is True
    )  # outer whitespace
    assert (
        exact_code_match("for i in r:\n    f(i)", ["for i in r:\n  f(i)"]) is False
    )  # inner indentation counts
    assert exact_code_match("RE(bp.count([det, det2]))", [ref]) is False
    res = exact_code_match_result()
    assert res.full_score == 1.0 and res.accuracy is True
    assert all(m.score == 1.0 for m in res.metrics.values())
