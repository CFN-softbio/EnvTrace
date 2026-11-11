from __future__ import annotations
from typing import Any, Dict, Tuple
import json
from pathlib import Path

from envtrace.core.event import Trace
from envtrace.core.evaluator import EvaluateRequest, Evaluator
from envtrace.core.alignment import DifflibAligner, SequenceAligner
from envtrace.core.comparators import (
    ValueComparator,
    ExactEqualityComparator,
    NumericToleranceComparator,
    AbsoluteToleranceComparator,
    AlwaysTrueComparator,
)
from envtrace.core.metrics import TimingMetric, Metric, MetricResult
from envtrace.core.scoring import WeightedAggregator, BinaryDecision

def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        text = f.read()
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError("PyYAML is required to load YAML config files. Install with `pip install pyyaml`.") from e
        return yaml.safe_load(text) or {}
    # default to JSON
    return json.loads(text or "{}")

def _make_comparator(spec: Dict[str, Any] | None) -> ValueComparator:
    """Comparator from a config spec. No spec: absolute tolerance 1e-3 (the default).

    Types: "absolute_tolerance" (aliases "abs_tolerance", "abs_tol"); "numeric_tolerance"
    (aliases "tolerance", "tol", "relative_tolerance", "rel_tol"), a RELATIVE tolerance;
    "exact"; "always_true" (aliases "any", "ignore"). An unknown type raises ValueError.
    """
    if not spec:
        return AbsoluteToleranceComparator(tol=1e-3)
    t = str(spec.get("type", "absolute_tolerance")).lower()
    if t in ("exact", "equals", "eq"):
        return ExactEqualityComparator()
    if t in ("numeric_tolerance", "tolerance", "tol", "relative_tolerance", "rel_tol"):
        tol = float(spec.get("tol", 1e-3))
        return NumericToleranceComparator(tol=tol)
    if t in ("absolute_tolerance", "abs_tolerance", "abs_tol"):
        return AbsoluteToleranceComparator(tol=float(spec.get("tol", 1e-3)))
    if t in ("always_true", "any", "ignore"):
        return AlwaysTrueComparator()
    raise ValueError(f"unknown comparator type: {t!r}")

def _make_aligner(spec: Dict[str, Any] | None) -> SequenceAligner:
    # For now only DifflibAligner is supported; this hook allows future expansion
    return DifflibAligner()

def _make_timing_metric(spec: Dict[str, Any] | None) -> TimingMetric:
    if not spec:
        return TimingMetric()
    return TimingMetric(
        r2_thresh=float(spec.get("r2_thresh", 0.90)),
        slope_lo=float(spec.get("slope_lo", 0.8)),
        slope_hi=float(spec.get("slope_hi", 1.2)),
        dur_tol=float(spec.get("dur_tol", 0.25)),
        mape_tol=float(spec.get("mape_tol", 1.0)),
        regression=str(spec.get("regression", "polyfit")),
        single_matched_pair=str(spec.get("single_matched_pair", "undefined")),
    )

def _make_aggregator(spec: Dict[str, Any] | None) -> WeightedAggregator:
    if not spec:
        return WeightedAggregator()
    weights = spec.get("weights")
    if isinstance(weights, dict):
        # ensure float conversion
        w = {str(k): float(v) for k, v in weights.items()}
        return WeightedAggregator(weights=w)
    # Separate weight sets for traces without and with continuous channels.
    agg = WeightedAggregator()
    w_no = spec.get("weights_without_continuous")
    w_with = spec.get("weights_with_continuous")
    if isinstance(w_no, dict):
        agg.weights_no_cont = {str(k): float(v) for k, v in w_no.items()}
    if isinstance(w_with, dict):
        agg.weights_with_cont = {str(k): float(v) for k, v in w_with.items()}
    return agg

def _make_decision(spec: Dict[str, Any] | None) -> BinaryDecision:
    if not spec:
        return BinaryDecision()
    require_discrete_exact = bool(spec.get("require_discrete_exact", True))
    timing_metric_name = str(spec.get("timing_metric_name", "timing"))
    required_metrics = spec.get("required_metrics") or []
    # ensure list[str]
    required_metrics = [str(x) for x in required_metrics if isinstance(x, (str, bytes))]
    return BinaryDecision(
        require_discrete_exact=require_discrete_exact,
        timing_metric_name=timing_metric_name,
        required_metric_names=required_metrics,
        require_continuous=bool(spec.get("require_continuous", True)),
    )

def build_from_config(gt: Trace, pred: Trace, cfg: Dict[str, Any]) -> Tuple[Evaluator, EvaluateRequest]:
    aliases = cfg.get("aliases") or {}
    ignore_channels = set(cfg.get("ignore_channels") or [])
    aligner = _make_aligner(cfg.get("aligner"))
    # Comparators
    comp_default = _make_comparator((cfg.get("comparators") or {}).get("default"))
    comp_map_cfg = ((cfg.get("comparators") or {}).get("by_channel")) or {}
    comp_map: Dict[str, ValueComparator] = {}
    for ch, spec in comp_map_cfg.items():
        comp_map[str(ch)] = _make_comparator(spec)

    # Timing metric thresholds
    timing_metric = _make_timing_metric(cfg.get("timing"))

    # Continuous channels configuration
    continuous_channels = cfg.get("continuous_channels") or None
    use_continuous = bool(continuous_channels)

    include_structure = bool(cfg.get("include_structure", True))

    # Aggregation / scoring
    aggregator = _make_aggregator(cfg.get("aggregation"))
    decision = _make_decision(cfg.get("decision"))
    evaluator = Evaluator(aggregator=aggregator, decision=decision)

    req = EvaluateRequest(
        gt=gt,
        pred=pred,
        key_fn=lambda e: e.channel,
        aligner=aligner,
        comparator=comp_default,
        comparators_by_channel=comp_map or None,
        ignore_channels=ignore_channels or None,
        alias_map=aliases or None,
        use_continuous=use_continuous,
        continuous_channels=continuous_channels,
        include_structure=include_structure,
        timing_metric=timing_metric,
        sort_by_time=bool(cfg.get("sort_by_time", False)),
        require_channel_match=bool(cfg.get("require_channel_match", True)),
        empty_reference_fails=bool(cfg.get("empty_reference_fails", True)),
        continuous_weighting=str(cfg.get("continuous_weighting", "if_present")),
        ignore_applies_to_continuous=bool(cfg.get("ignore_applies_to_continuous", True)),
        no_discrete_reference_scores_zero=bool(cfg.get("no_discrete_reference_scores_zero", False)),
    )
    return evaluator, req
