from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol
from envtrace.core.event import Event
from envtrace.core.comparators import ValueComparator, NumericToleranceComparator, get_comparator_for_channel
import numpy as np

@dataclass
class MetricResult:
    name: str
    score: float
    binary_pass: Optional[bool] = None
    details: Dict[str, Any] = None

class Metric(Protocol):
    def evaluate(self, *args, **kwargs) -> MetricResult: ...

Pair = Tuple[Optional[Event], Optional[Event]]

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom))

class DiscreteMatchMetric:
    def __init__(self,
                 comparator: ValueComparator | None = None,
                 ignore_channels: Optional[set[str]] = None,
                 comparator_map: Optional[Dict[str, ValueComparator]] = None) -> None:
        self.comparator = comparator or NumericToleranceComparator()
        self.ignore_channels = ignore_channels or set()
        self.comparator_map = comparator_map or {}

    def evaluate(self, alignment: List[Pair]) -> MetricResult:
        value_matches = 0
        total_pairs = 0
        gt_count = 0
        pred_count = 0
        aligned_gt: List[Optional[Event]] = []
        aligned_pr: List[Optional[Event]] = []

        for g, p in alignment:
            if g and g.channel in self.ignore_channels:
                continue
            if p and p.channel in self.ignore_channels:
                continue

            aligned_gt.append(g)
            aligned_pr.append(p)
            total_pairs += 1
            if g is not None:
                gt_count += 1
            if p is not None:
                pred_count += 1
            if g is not None and p is not None and get_comparator_for_channel(self.comparator, self.comparator_map, g.channel).compare(g.value, p.value, g.channel):
                value_matches += 1

        rate = (value_matches / total_pairs) if total_pairs else 1.0
        mismatch_rate = (total_pairs - value_matches) / total_pairs if total_pairs else 0.0
        exact = (value_matches == gt_count) and (value_matches == pred_count)

        return MetricResult(
            name="discrete",
            score=rate,
            binary_pass=exact,
            details={
                "mismatch_rate": mismatch_rate,
                "exact": exact,
                "value_matches": value_matches,
                "total_pairs": total_pairs,
                "gt_event_count": gt_count,
                "pred_event_count": pred_count,
                "aligned_gt": [(e.channel, e.timestamp, e.value) if e else None for e in aligned_gt],
                "aligned_pred": [(e.channel, e.timestamp, e.value) if e else None for e in aligned_pr],
            },
        )

class TimingMetric:
    def __init__(self,
                 r2_thresh: float = 0.90,
                 slope_lo: float = 0.8,
                 slope_hi: float = 1.2,
                 dur_tol: float = 0.25,
                 mape_tol: float = 1.0) -> None:
        self.r2_thresh = r2_thresh
        self.slope_lo = slope_lo
        self.slope_hi = slope_hi
        self.dur_tol = dur_tol
        self.mape_tol = mape_tol

    def _linregress_np(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        # Returns slope, intercept, r2 using numpy (no SciPy dependency)
        x = x.astype(float)
        y = y.astype(float)
        if x.size < 2:
            return 0.0, float(y.mean() if y.size else 0.0), 1.0
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(y.mean())) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        return float(slope), float(intercept), float(r2)

    def evaluate(self, aligned_pairs: List[Pair], comparator: ValueComparator, comparator_map: Optional[Dict[str, ValueComparator]] = None) -> MetricResult:
        matched_gt_ts: List[float] = []
        matched_pr_ts: List[float] = []

        for g, p in aligned_pairs:
            if g is None or p is None:
                continue
            if not get_comparator_for_channel(comparator, comparator_map, g.channel).compare(g.value, p.value, g.channel):
                continue
            matched_gt_ts.append(float(g.timestamp))
            matched_pr_ts.append(float(p.timestamp))

        n = len(matched_gt_ts)
        if n == 0:
            return MetricResult(
                name="timing",
                score=0.0,
                binary_pass=False,
                details={"reason": "no matching PV/value pairs for timing evaluation", "score": 0.0},
            )
        if n < 2:
            return MetricResult(name="timing", score=1.0, binary_pass=True, details={"score": 1.0})

        gt = np.asarray(matched_gt_ts) - matched_gt_ts[0]
        pr = np.asarray(matched_pr_ts) - matched_pr_ts[0]

        slope, intercept, r2 = self._linregress_np(gt, pr)

        r2_score = r2
        slope_score = 1.0 - min(abs(slope - 1.0), 1.0)

        dur_gt, dur_pr = float(gt[-1]), float(pr[-1])
        if dur_gt == 0.0:
            duration_score = 0.0
        else:
            duration_diff = abs(dur_pr - dur_gt) / dur_gt
            duration_score = max(0.0, 1.0 - duration_diff / self.dur_tol)

        gt_intervals = np.diff(gt)
        pr_intervals = np.diff(pr)
        mape = calculate_mape(gt_intervals, pr_intervals)
        mape_score = max(0.0, 1.0 - mape / self.mape_tol)

        timing_score = (0.4 * r2_score + 0.2 * slope_score + 0.2 * duration_score + 0.2 * mape_score)

        binary_match = (r2 >= self.r2_thresh and
                        self.slope_lo <= slope <= self.slope_hi and
                        (dur_gt == 0.0 or abs(dur_pr - dur_gt) / dur_gt <= self.dur_tol) and
                        mape <= self.mape_tol)

        return MetricResult(
            name="timing",
            score=timing_score,
            binary_pass=binary_match,
            details={
                "r2": r2,
                "slope": slope,
                "mape": mape,
                "duration_ratio": (dur_pr / (dur_gt or 1.0)),
                "score": timing_score,
                "r2_score": r2_score,
                "slope_score": slope_score,
                "duration_score": duration_score,
                "mape_score": mape_score,
            },
        )

class ContinuousProfileMetric:
    """
    Generic continuous-series fidelity metric (e.g., temperature ramp).
    Compares GT and Pred series of (timestamp, value) tuples.

    Score = 0.7*exp(-MAE/mae_scale) + 0.3*exp(-final_diff/final_scale)
    Binary pass if mae <= mae_thresh and final_diff <= final_thresh.
    """
    def evaluate(
        self,
        gt_series: list[tuple[float, float]],
        pred_series: list[tuple[float, float]],
        *,
        mae_scale: float = 15.0,
        final_scale: float = 15.0,
        mae_thresh: float = 5.0,
        final_thresh: float = 5.0,
    ) -> MetricResult:
        # Handle empty cases
        if not gt_series and not pred_series:
            return MetricResult(
                name="continuous",
                score=1.0,
                binary_pass=True,
                details={"reason": "both empty", "score": 1.0},
            )
        if not gt_series or not pred_series:
            return MetricResult(
                name="continuous",
                score=0.0,
                binary_pass=False,
                details={"reason": "one empty", "score": 0.0},
            )

        # Extract values only (ignore timestamps for MAE/final diff)
        gt_vals = np.array([float(v) for _, v in gt_series], dtype=float)
        pr_vals = np.array([float(v) for _, v in pred_series], dtype=float)

        # Pad shorter sequence with its last value
        max_len = max(len(gt_vals), len(pr_vals))
        if len(gt_vals) < max_len:
            gt_vals = np.pad(gt_vals, (0, max_len - len(gt_vals)), mode="edge")
        elif len(pr_vals) < max_len:
            pr_vals = np.pad(pr_vals, (0, max_len - len(pr_vals)), mode="edge")

        mae = float(np.mean(np.abs(gt_vals - pr_vals)))
        final_diff = float(abs(gt_vals[-1] - pr_vals[-1]))

        mae_score = float(np.exp(-mae / mae_scale))
        final_score = float(np.exp(-final_diff / final_scale))
        score = 0.7 * mae_score + 0.3 * final_score

        binary = (mae <= mae_thresh and final_diff <= final_thresh)

        return MetricResult(
            name="continuous",
            score=score,
            binary_pass=binary,
            details={
                "mae": mae,
                "final_diff": final_diff,
                "mae_score": mae_score,
                "final_score": final_score,
                "score": score,
            },
        )

class StructureMetric:
    """
    Simple structural metric based on alignment gaps (insert/delete).
    gap_rate = gaps / total_pairs; score = 1 - min(1, gap_rate)
    binary pass if gap_rate == 0 (no gaps).
    """
    def evaluate(self, alignment: List[Pair]) -> MetricResult:
        total_pairs = len(alignment)
        gaps = sum(1 for g, p in alignment if g is None or p is None)
        gap_rate = (gaps / total_pairs) if total_pairs else 0.0
        score = 1.0 - min(1.0, gap_rate)
        binary = (gap_rate == 0.0)
        return MetricResult(
            name="structure",
            score=score,
            binary_pass=binary,
            details={"gap_rate": gap_rate, "gaps": gaps, "total_pairs": total_pairs},
        )
