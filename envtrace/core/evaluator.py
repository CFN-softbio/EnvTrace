from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List
from envtrace.core.event import Trace, Event
from envtrace.core.alignment import SequenceAligner, DifflibAligner, Pair
from envtrace.core.comparators import ValueComparator, NumericToleranceComparator
from envtrace.core.metrics import DiscreteMatchMetric, TimingMetric, MetricResult, ContinuousProfileMetric, StructureMetric, Metric
from envtrace.core.scoring import WeightedAggregator, BinaryDecision

@dataclass
class EvaluateRequest:
    gt: Trace
    pred: Trace
    key_fn: Callable[[Event], str] = lambda e: e.channel
    aligner: SequenceAligner = DifflibAligner()
    comparator: ValueComparator = NumericToleranceComparator()
    comparators_by_channel: Optional[Dict[str, ValueComparator]] = None
    ignore_channels: Optional[set[str]] = None
    alias_map: Optional[Dict[str, str]] = None
    use_continuous: bool = False
    continuous_channels: Optional[Dict[str, Dict[str, float]]] = None  # e.g., {"stage:temp": {"mae_scale": 15.0, "final_scale": 15.0, "mae_thresh": 5.0, "final_thresh": 5.0, "weight": 1.0}}
    include_structure: bool = True
    timing_metric: Optional[TimingMetric] = None
    # Optional: override the per-channel continuous metric used inside continuous aggregation.
    # Custom metric should implement evaluate(gt_series, pred_series, channel=None, config=None, **kwargs) -> MetricResult
    continuous_metric: Optional[Metric] = None
    # Optional: user-supplied extra metrics to compute after alignment.
    # Each Metric will be called as metric.evaluate(alignment=..., gt=..., pred=...) (with a positional fallback).
    custom_metrics: Optional[Dict[str, Metric]] = None

@dataclass
class EvaluateResult:
    metrics: Dict[str, MetricResult]
    full_score: float
    accuracy: bool
    alignment: List[Pair]

class Evaluator:
    def __init__(self,
                 aggregator: Optional[WeightedAggregator] = None,
                 decision: Optional[BinaryDecision] = None) -> None:
        self.aggregator = aggregator or WeightedAggregator()
        self.decision = decision or BinaryDecision()

    def evaluate(self, req: EvaluateRequest) -> EvaluateResult:
        gt = req.gt
        pred = req.pred

        if req.alias_map:
            gt = gt.canonicalize(req.alias_map)
            pred = pred.canonicalize(req.alias_map)

        gt_sorted = gt.sort_by_time(in_place=False)
        pred_sorted = pred.sort_by_time(in_place=False)

        # Build combined ignore set BEFORE alignment - includes both regular ignores and continuous channels
        ignore = set(req.ignore_channels or set())
        continuous_channels_set = set(req.continuous_channels.keys()) if req.continuous_channels else set()
        
        # Filter out ignored AND continuous channels from traces BEFORE alignment
        # Continuous channels will be evaluated separately as time-series
        channels_to_exclude = ignore | continuous_channels_set
        gt_events_for_alignment = [e for e in gt_sorted.events if e.channel not in channels_to_exclude]
        pred_events_for_alignment = [e for e in pred_sorted.events if e.channel not in channels_to_exclude]

        # Align only the discrete (non-ignored, non-continuous) events
        alignment = req.aligner.align(gt_events_for_alignment, pred_events_for_alignment, req.key_fn)

        # Discrete metric evaluates the already-filtered alignment
        discrete_metric = DiscreteMatchMetric(
            comparator=req.comparator,
            ignore_channels=ignore,  # Pass for defensive programming, but already filtered
            comparator_map=req.comparators_by_channel
        ).evaluate(alignment)

        tm = req.timing_metric or TimingMetric()
        timing_metric = tm.evaluate(alignment, comparator=req.comparator, comparator_map=req.comparators_by_channel)

        metrics: Dict[str, MetricResult] = {
            "discrete": discrete_metric,
            "timing": timing_metric,
        }

        # Optional: structural metric based on alignment gaps (now only considers discrete events)
        if req.include_structure:
            metrics["structure"] = StructureMetric().evaluate(alignment)

        # Optional: continuous channels (e.g., temperature) - evaluated separately from the full traces
        if req.continuous_channels:
            cont_metric = self._evaluate_continuous(gt_sorted, pred_sorted, req.continuous_channels, req.continuous_metric)
            metrics["continuous"] = cont_metric

        # Add any user-provided custom metrics
        if req.custom_metrics:
            for name, metric in req.custom_metrics.items():
                try:
                    mres = metric.evaluate(alignment=alignment, gt=gt_sorted, pred=pred_sorted)
                except TypeError:
                    # Fallback to positional arguments if the metric doesn't accept keywords
                    mres = metric.evaluate(alignment, gt_sorted, pred_sorted)
                metrics[name] = mres

        aggregator = self.aggregator
        aggregator.use_continuous = req.use_continuous or bool(req.continuous_channels)
        full_score = aggregator.aggregate(metrics)
        accuracy = self.decision.decide(metrics)

        return EvaluateResult(metrics=metrics, full_score=full_score, accuracy=accuracy, alignment=alignment)

    def evaluate_best_of(self, gt_list: List[Trace], pred: Trace, **kwargs) -> tuple[EvaluateResult, int]:
        """
        Evaluate multiple ground-truth traces against a single predicted trace,
        returning the best EvaluateResult and its index.

        kwargs may include: key_fn, aligner, comparator, comparators_by_channel,
        ignore_channels, alias_map, use_continuous.
        """
        best_result: Optional[EvaluateResult] = None
        best_idx: int = -1
        best_score: float = -1.0

        for i, gt in enumerate(gt_list):
            req = EvaluateRequest(
                gt=gt,
                pred=pred,
                key_fn=kwargs.get("key_fn", (lambda e: e.channel)),
                aligner=kwargs.get("aligner", DifflibAligner()),
                comparator=kwargs.get("comparator", NumericToleranceComparator()),
                comparators_by_channel=kwargs.get("comparators_by_channel"),
                ignore_channels=kwargs.get("ignore_channels"),
                alias_map=kwargs.get("alias_map"),
                use_continuous=kwargs.get("use_continuous", False),
            )
            res = self.evaluate(req)
            if res.full_score > best_score:
                best_result = res
                best_idx = i
                best_score = res.full_score

        if best_result is None:
            empty = Trace([])
            best_result = self.evaluate(EvaluateRequest(gt=empty, pred=pred))
            best_idx = -1

        return best_result, best_idx

    def _evaluate_continuous(self, gt: Trace, pred: Trace, cfg: Dict[str, Dict[str, float]], per_channel_metric: Optional[Metric] = None) -> MetricResult:
        """
        Aggregate continuous scores over channels defined in cfg.
        cfg[channel] may include: mae_scale, final_scale, mae_thresh, final_thresh, weight.
        If per_channel_metric is provided, it will be used for each channel as:
            per_channel_metric.evaluate(gt_series, pred_series, channel=<name>, config=<dict>)
        Otherwise, ContinuousProfileMetric is used with mae/final args from cfg.
        """
        default_comp = ContinuousProfileMetric()
        per_channel: Dict[str, Dict] = {}
        total_weight = 0.0
        total_score = 0.0
        binary_all = True

        for ch, ch_cfg in cfg.items():
            mae_scale = float(ch_cfg.get("mae_scale", 15.0))
            final_scale = float(ch_cfg.get("final_scale", 15.0))
            mae_thresh = float(ch_cfg.get("mae_thresh", 5.0))
            final_thresh = float(ch_cfg.get("final_thresh", 5.0))
            weight = float(ch_cfg.get("weight", 1.0))

            gt_series = [(e.timestamp, float(e.value)) for e in gt.events if e.channel == ch]
            pr_series = [(e.timestamp, float(e.value)) for e in pred.events if e.channel == ch]

            if per_channel_metric is not None:
                try:
                    res = per_channel_metric.evaluate(gt_series, pr_series, channel=ch, config=ch_cfg)
                except TypeError:
                    # Fallback to positional if custom metric doesn't accept keywords
                    res = per_channel_metric.evaluate(gt_series, pr_series)
            else:
                res = default_comp.evaluate(
                    gt_series,
                    pr_series,
                    mae_scale=mae_scale,
                    final_scale=final_scale,
                    mae_thresh=mae_thresh,
                    final_thresh=final_thresh,
                )

            per_channel[ch] = {
                "score": res.score,
                "binary_pass": res.binary_pass,
                "details": res.details,
                "weight": weight,
            }
            total_weight += weight
            total_score += weight * float(res.score)
            binary_all = binary_all and bool(res.binary_pass)

        agg_score = (total_score / total_weight) if total_weight > 0 else 1.0

        return MetricResult(
            name="continuous",
            score=agg_score,
            binary_pass=binary_all,
            details={"per_channel": per_channel, "weights_sum": total_weight},
        )

    @staticmethod
    def default() -> "Evaluator":
        return Evaluator()
