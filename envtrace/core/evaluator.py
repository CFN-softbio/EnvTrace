from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Optional, Callable, List
from envtrace.core.event import Trace, Event
from envtrace.core.alignment import SequenceAligner, DifflibAligner, Pair
from envtrace.core.comparators import ValueComparator, AbsoluteToleranceComparator
from envtrace.core.metrics import DiscreteMatchMetric, TimingMetric, MetricResult, ContinuousProfileMetric, StructureMetric, Metric
from envtrace.core.scoring import WeightedAggregator, BinaryDecision

CONTINUOUS_WEIGHTING_MODES = ("if_present", "always")


@dataclass
class EvaluateRequest:
    """One evaluation: a reference trace (gt), a predicted trace and the scoring options.

    The defaults implement the method of the EnvTrace paper (see the README section
    "Defaults and the paper"); every earlier default is reachable through an explicit
    option, noted per field.
    """
    gt: Trace
    pred: Trace
    key_fn: Callable[[Event], str] = lambda e: e.channel
    aligner: SequenceAligner = DifflibAligner()
    # Default comparator: absolute tolerance, abs(a - b) < 1e-3, as in the paper.
    # Before: NumericToleranceComparator(1e-3), a RELATIVE tolerance.
    comparator: ValueComparator = AbsoluteToleranceComparator()
    comparators_by_channel: Optional[Dict[str, ValueComparator]] = None
    ignore_channels: Optional[set[str]] = None
    alias_map: Optional[Dict[str, str]] = None
    # Only used with continuous_weighting="always" (forces the continuous weights).
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
    # False (default): keep the recorded order of the events, as the paper's scorer did.
    # True: sort both traces by timestamp before alignment (the earlier default).
    sort_by_time: bool = False
    # True (default): an aligned pair matches only if the channels are equal (discrete
    # and timing). False: the earlier default (equal values on different channels match).
    require_channel_match: bool = True
    # True (default): a reference with no discrete events scores 0 and is never exact.
    # False: the earlier default (two empty traces score 1.0 and exact).
    empty_reference_fails: bool = True
    # "if_present" (default): the continuous weights are used only if a configured
    # continuous channel has an event in gt or pred, otherwise the no-continuous weights
    # (as in the paper). "always": the earlier default, the continuous weights are used
    # whenever continuous_channels is configured (or use_continuous is True).
    continuous_weighting: str = "if_present"
    # True (default, a correction of the paper's scorer): a continuous channel listed in
    # ignore_channels is left out of the continuous facet as well (and so of the weight
    # switch and the strict decision). False: the paper's original scorer and the earlier
    # default: ignore_channels removes events from the discrete and timing facets only,
    # and the continuous facet reads every event.
    ignore_applies_to_continuous: bool = True
    # False (default): every reference is scored. True: the harness rule of the paper's
    # scorer: if the reference (evaluate) or every reference (evaluate_best_of) has no
    # discrete event, the result is 0 on every facet and fails, without scoring.
    no_discrete_reference_scores_zero: bool = False

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

    @staticmethod
    def _prepare(req: EvaluateRequest):
        """Canonicalize, order and split the traces. Returns
        (gt, pred, gt_discrete, pred_discrete, ignore, continuous_cfg)."""
        if req.continuous_weighting not in CONTINUOUS_WEIGHTING_MODES:
            raise ValueError(f"unknown continuous_weighting: {req.continuous_weighting!r}")
        gt = req.gt
        pred = req.pred
        if req.alias_map:
            gt = gt.canonicalize(req.alias_map)
            pred = pred.canonicalize(req.alias_map)

        if req.sort_by_time:
            gt = gt.sort_by_time(in_place=False)
            pred = pred.sort_by_time(in_place=False)
        else:
            gt = Trace(list(gt.events))
            pred = Trace(list(pred.events))

        ignore = set(req.ignore_channels or set())
        continuous_all = dict(req.continuous_channels or {})
        if req.ignore_applies_to_continuous:
            continuous_cfg = {ch: c for ch, c in continuous_all.items() if ch not in ignore}
        else:
            continuous_cfg = continuous_all

        # Ignored AND continuous channels are removed BEFORE alignment; continuous
        # channels are evaluated separately as series from the full traces.
        exclude = ignore | set(continuous_all)
        gt_disc = [e for e in gt.events if e.channel not in exclude]
        pred_disc = [e for e in pred.events if e.channel not in exclude]
        return gt, pred, gt_disc, pred_disc, ignore, continuous_cfg

    @staticmethod
    def zero_result(req: EvaluateRequest, reason: str) -> EvaluateResult:
        """Every facet 0 and fail (the no-discrete-reference rule of the paper's harness)."""
        names = ["discrete", "timing"] + (["continuous"] if req.continuous_channels else [])
        metrics = {n: MetricResult(name=n, score=0.0, binary_pass=False, details={"reason": reason})
                   for n in names}
        return EvaluateResult(metrics=metrics, full_score=0.0, accuracy=False, alignment=[])

    def evaluate(self, req: EvaluateRequest) -> EvaluateResult:
        gt_sorted, pred_sorted, gt_events_for_alignment, pred_events_for_alignment, ignore, continuous_cfg = \
            self._prepare(req)

        if req.no_discrete_reference_scores_zero and not gt_events_for_alignment:
            return self.zero_result(req, "no discrete event in the reference")

        # Align only the discrete (non-ignored, non-continuous) events
        alignment = req.aligner.align(gt_events_for_alignment, pred_events_for_alignment, req.key_fn)

        # Discrete metric evaluates the already-filtered alignment
        discrete_metric = DiscreteMatchMetric(
            comparator=req.comparator,
            ignore_channels=ignore,  # Pass for defensive programming, but already filtered
            comparator_map=req.comparators_by_channel,
            require_channel_match=req.require_channel_match,
            empty_reference_fails=req.empty_reference_fails,
        ).evaluate(alignment)

        tm = req.timing_metric or TimingMetric()
        timing_metric = tm.evaluate(alignment, comparator=req.comparator, comparator_map=req.comparators_by_channel,
                                    require_channel_match=req.require_channel_match)

        metrics: Dict[str, MetricResult] = {
            "discrete": discrete_metric,
            "timing": timing_metric,
        }

        # Optional: structural metric based on alignment gaps (now only considers discrete events)
        if req.include_structure:
            metrics["structure"] = StructureMetric().evaluate(alignment)

        # Optional: continuous channels (e.g., temperature) - evaluated separately from the full traces
        if req.continuous_channels:
            cont_metric = self._evaluate_continuous(gt_sorted, pred_sorted, continuous_cfg, req.continuous_metric)
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
        if req.continuous_weighting == "if_present":
            aggregator.use_continuous = any(
                e.channel in continuous_cfg for e in list(gt_sorted.events) + list(pred_sorted.events)
            )
        else:  # "always"
            aggregator.use_continuous = req.use_continuous or bool(req.continuous_channels)
        full_score = aggregator.aggregate(metrics)
        accuracy = self.decision.decide(metrics)

        return EvaluateResult(metrics=metrics, full_score=full_score, accuracy=accuracy, alignment=alignment)

    def evaluate_best_of(self, gt_list: List[Trace], pred: Trace, template: Optional[EvaluateRequest] = None, **kwargs) -> tuple[EvaluateResult, int]:
        """
        Evaluate several reference traces against one predicted trace and return the
        best EvaluateResult and its index. The first reference with the strictly
        highest full_score wins (ties keep the earlier reference), as in the paper.

        template: an EvaluateRequest (for example from build_from_config) whose
            settings are used for every reference; only gt and pred are replaced.
        kwargs: any EvaluateRequest field except gt and pred. Without a template they
            build the request (unknown names raise TypeError); with a template they
            override its fields.

        Returns index -1 when gt_list is empty, or when no_discrete_reference_scores_zero
        is set and no reference has a discrete event (the result is then all zero).
        """
        if template is not None:
            base = replace(template, gt=Trace([]), pred=pred, **kwargs)
        else:
            base = EvaluateRequest(gt=Trace([]), pred=pred, **kwargs)

        if base.no_discrete_reference_scores_zero and gt_list:
            if not any(self._prepare(replace(base, gt=gt))[2] for gt in gt_list):
                return self.zero_result(base, "no discrete event in any reference"), -1
            base = replace(base, no_discrete_reference_scores_zero=False)

        best_result: Optional[EvaluateResult] = None
        best_idx: int = -1
        for i, gt in enumerate(gt_list):
            res = self.evaluate(replace(base, gt=gt))
            if best_result is None or res.full_score > best_result.full_score:
                best_result = res
                best_idx = i

        if best_result is None:
            best_result = self.evaluate(base)
            best_idx = -1

        return best_result, best_idx

    def _evaluate_continuous(self, gt: Trace, pred: Trace, cfg: Dict[str, Dict[str, float]], per_channel_metric: Optional[Metric] = None) -> MetricResult:
        """
        Aggregate continuous scores over channels defined in cfg.
        cfg[channel] may include: mae_scale, final_scale, mae_thresh, final_thresh, weight.
        If per_channel_metric is provided, it will be used for each channel as:
            per_channel_metric.evaluate(gt_series, pred_series, channel=<name>, config=<dict>)
        Otherwise, ContinuousProfileMetric is used with mae/final args from cfg.
        details["involved"] is True if any channel in cfg has an event in gt or pred.
        """
        default_comp = ContinuousProfileMetric()
        per_channel: Dict[str, Dict] = {}
        total_weight = 0.0
        total_score = 0.0
        binary_all = True
        involved = False

        for ch, ch_cfg in cfg.items():
            mae_scale = float(ch_cfg.get("mae_scale", 15.0))
            final_scale = float(ch_cfg.get("final_scale", 15.0))
            mae_thresh = float(ch_cfg.get("mae_thresh", 5.0))
            final_thresh = float(ch_cfg.get("final_thresh", 5.0))
            weight = float(ch_cfg.get("weight", 1.0))

            gt_series = [(e.timestamp, float(e.value)) for e in gt.events if e.channel == ch]
            pr_series = [(e.timestamp, float(e.value)) for e in pred.events if e.channel == ch]
            involved = involved or bool(gt_series) or bool(pr_series)

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
            details={"per_channel": per_channel, "weights_sum": total_weight, "involved": involved},
        )

    @staticmethod
    def default() -> "Evaluator":
        return Evaluator()
