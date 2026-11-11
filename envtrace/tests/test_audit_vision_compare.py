"""Audit of the package against the paper's original scorer (VISION tests/test_op_cog.py).

Written for the audit in analysis/envtrace_vs_vision/ of the paper repository
(2026-11-11). Expected values are those of the original scorer, which produced every
number in the paper (checked with its verbatim copy in analysis/A4_leak/orig_scoring.py,
see analysis/envtrace_vs_vision/out/synthetic.txt).

Three groups:
  1. examples/paper/paper_config.json reproduces the original scorer (must pass).
  2. KNOWN DEVIATIONS of the package defaults from the original scorer and the paper,
     marked xfail(strict=True): each assertion states the paper's behaviour and fails
     today. A fix of the default makes the test pass, and strict xfail then reports it.
  3. Rules shared by both implementations that the paper does not state (pass; they
     document behaviour, they do not endorse it).
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from envtrace.core.event import Event, Trace
from envtrace.core.evaluator import EvaluateRequest, Evaluator
from envtrace.io.config_loader import build_from_config

TEMP = "XF:11BM-ES:{LINKAM}:TEMP"
X = "XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr"
Y = "XF:11BMB-ES{Chm:Smpl-Ax:Y}Mtr"
Z = "XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr"
SP = "XF:11BM-ES:{LINKAM}:SETPOINT:SET"
CFG = json.loads((Path(__file__).resolve().parents[1] / "examples" / "paper" / "paper_config.json").read_text())


def tr(x):
    return Trace([Event(c, float(t), v) for c, t, v in x])


def ramp(t0, v0, step, n):
    return [(TEMP, t0 + i, v0 + i * step) for i in range(n)]


def paper(gt, pred, **over):
    ev, tpl = build_from_config(Trace([]), Trace([]), dict(CFG, **over))
    return ev.evaluate(replace(tpl, gt=tr(gt), pred=tr(pred)))


def default(gt, pred):
    # package defaults, temperature declared continuous (as the README suggests)
    return Evaluator().evaluate(EvaluateRequest(gt=tr(gt), pred=tr(pred), continuous_channels={TEMP: {}}))


S1 = ([(X, 0.0, 0.5)], [(Y, 0.0, 0.5)])
S2 = ([(X, 0.0, 0.0), (X, 1.0, 0.15)], [(X, 0.0, 3.608224830031759e-16), (X, 1.0, 0.15)])
S2B = ([(SP, 0.0, 313.5)], [(SP, 0.0, 313.6)])
S3 = ([], [])
S4 = ([(X, 0.0, 1.0), (Y, 1.0, 2.0)], [(X, 0.0, 1.0), (Y, 1.0, 3.0)])
S5 = ([(SP, 0.0, 50.0)] + ramp(0.5, 25.0, 1.0, 26), [(SP, 0.0, 50.0)] + ramp(0.5, 25.0, 3.0, 26))
S6 = ([(Y, 0.0, 1.0), (X, 0.1, 2.0)], [(Y, 1.0, 1.0), (X, 0.9, 2.0)])


# 1. paper configuration = original scorer -----------------------------------------
@pytest.mark.parametrize("case,full,passed", [
    (S1, 0.0, False), (S2, 1.0, True), (S2B, 0.0, False), (S4, 0.6, False),
    (S5, 0.8285830, False), (S6, 0.88, False),
])
def test_paper_config_matches_original(case, full, passed):
    res = paper(*case)
    assert res.full_score == pytest.approx(full, abs=1e-6)
    assert res.accuracy is passed


def test_paper_config_empty_reference_scores_zero():
    res = paper(*S3)
    assert res.full_score == 0.0 and res.accuracy is False


def test_paper_config_channel_check_with_value_free_function_events():
    # without the channel check, a wsam reference event (value ignored by rule) matches
    # any predicted event in a 'replace' block: here sam.setOrigin.
    gt = [("sam.align", 0.0, 0), ("wsam", 1e-5, "")]
    pred = [("sam.align", 0.0, 0), ("sam.setOrigin", 1e-5, "['x'], None")]
    assert paper(gt, pred).full_score == pytest.approx(0.6)
    assert paper(gt, pred, require_channel_match=False).full_score == pytest.approx(1.0)


# 2. known deviations of the defaults ------------------------------------------------
@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: no channel check on aligned pairs")
def test_default_requires_equal_channel():
    res = default(*S1)
    assert res.metrics["discrete"].score == 0.0 and res.accuracy is False


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: relative tolerance fails float noise at 0")
def test_default_tolerance_absolute_near_zero():
    assert default(*S2).metrics["discrete"].score == 1.0


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: relative tolerance accepts 313.5 vs 313.6")
def test_default_tolerance_not_relative_for_large_values():
    assert default(*S2B).metrics["discrete"].score == 0.0


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: two empty traces score 0.8 (paper: 0)")
def test_default_empty_reference():
    assert default(*S3).full_score == 0.0


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: continuous weights used without temperature events")
def test_default_weights_without_temperature_events():
    assert default(*S4).full_score == pytest.approx(0.6)


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: strict decision ignores the continuous check")
def test_default_decision_requires_continuous_check():
    assert default(*S5).accuracy is False


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: sort by time reorders recorded events")
def test_default_keeps_recorded_order():
    assert default(*S6).metrics["discrete"].score == 1.0


@pytest.mark.xfail(strict=True, reason="KNOWN DEVIATION: evaluate_best_of(**kwargs) drops continuous_channels")
def test_best_of_kwargs_keep_continuous_channels():
    gt = tr(S5[0])
    pred = tr(S5[1])
    res, _ = Evaluator().evaluate_best_of([gt], pred, continuous_channels={TEMP: {}})
    assert "continuous" in res.metrics


# 3. shared rules the paper does not state ------------------------------------------
def test_shared_rule_alignment_by_channel_only():
    # a scan that skips its first step: the channel-only alignment pairs every event
    # with the wrong step, so no value matches (both implementations).
    gt = [(X, 0.0, 1.0), (X, 1.0, 2.0), (X, 2.0, 3.0)]
    pred = [(X, 0.0, 2.0), (X, 1.0, 3.0)]
    assert paper(gt, pred).metrics["discrete"].score == 0.0


def test_shared_rule_single_matched_pair_full_timing_credit():
    gt = [(X, 0.0, 1.0), (Y, 1.0, 2.0), (Z, 2.0, 3.0)]
    res = paper(gt, [(X, 0.0, 1.0)])
    assert res.metrics["timing"].score == 1.0 and res.metrics["timing"].binary_pass is True


def test_shared_rule_two_matched_pairs_r2_is_one():
    res = paper([(X, 0.0, 1.0), (Y, 1e-4, 2.0)], [(X, 0.0, 1.0), (Y, 5e-4, 2.0)])
    assert res.metrics["timing"].details["r2"] == pytest.approx(1.0)
