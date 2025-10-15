from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from envtrace.core.alignment import Pair
from envtrace.core.comparators import ValueComparator, get_comparator_for_channel
from envtrace.core.evaluator import EvaluateRequest, EvaluateResult, Evaluator
from envtrace.core.metrics import MetricResult
from envtrace.core.scoring import WeightedAggregator, BinaryDecision


def _event_to_dict(e) -> Dict[str, Any]:
    if e is None:
        return None  # type: ignore[return-value]
    return {"channel": e.channel, "timestamp": e.timestamp, "value": e.value}


def build_alignment_rows(
    alignment: List[Pair],
    *,
    comparator: ValueComparator,
    comparator_map: Optional[Dict[str, ValueComparator]] = None,
    ignore_channels: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Turn an alignment into display-friendly rows, including match flags.
    
    Note: Continuous channels (e.g., temperature) should already be filtered out
    before alignment, so they will not appear in these rows. This function is only
    for discrete events.

    match is:
      - True if both events exist, same channel, and comparator says values match;
      - False if both exist but channel differs or comparator says mismatch;
      - None for gaps or ignored channels.
    """
    ignore = ignore_channels or set()
    rows: List[Dict[str, Any]] = []
    for idx, (g, p) in enumerate(alignment, start=1):
        ignored = (g is not None and g.channel in ignore) or (p is not None and p.channel in ignore)
        if g is not None and p is not None and not ignored:
            if g.channel == p.channel:
                comp = get_comparator_for_channel(comparator, comparator_map, g.channel)
                is_match = bool(comp.compare(g.value, p.value, g.channel))
            else:
                is_match = False
        else:
            is_match = None  # gap or ignored
        rows.append(
            {
                "index": idx,
                "gt": _event_to_dict(g),
                "pred": _event_to_dict(p),
                "match": is_match,
                "ignored": ignored,
            }
        )
    return rows


def _trim(s: str, width: int) -> str:
    return s if len(s) <= width else (s[: max(0, width - 1)] + "…")


def format_alignment_table(
    rows: List[Dict[str, Any]],
    *,
    max_rows: int = 50,
    col_widths: Optional[Dict[str, int]] = None,
) -> str:
    """
    Pretty-print a text table of alignment rows with a match column.

    Columns:
      IDX | GT.channel | GT.t | GT.value | PRED.channel | PRED.t | PRED.value | MATCH
    """
    # Defaults
    widths = {
        "idx": 4,
        "gt_ch": 16,
        "gt_t": 8,
        "gt_v": 16,
        "pr_ch": 16,
        "pr_t": 8,
        "pr_v": 16,
        "match": 7,
    }
    if col_widths:
        widths.update(col_widths)

    # Header
    header = (
        f"{'IDX':>{widths['idx']}} | "
        f"{'GT.channel':<{widths['gt_ch']}} | {'GT.t':>{widths['gt_t']}} | {'GT.value':<{widths['gt_v']}} | "
        f"{'PR.channel':<{widths['pr_ch']}} | {'PR.t':>{widths['pr_t']}} | {'PR.value':<{widths['pr_v']}} | "
        f"{'MATCH':^{widths['match']}}"
    )
    sep = "-" * len(header)

    def fmt_e(d: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        if not d:
            return ("—", "—", "—")
        ch = _trim(str(d.get("channel", "")), widths["gt_ch"])
        ts = d.get("timestamp", None)
        ts_s = "—" if ts is None else f"{float(ts):.3f}"
        val = _trim(str(d.get("value", "")), widths["gt_v"])
        return ch, ts_s, val

    out_lines = [header, sep]
    shown = 0
    for r in rows:
        if shown >= max_rows:
            break
        gt_ch, gt_t, gt_v = fmt_e(r["gt"])
        pr_ch, pr_t, pr_v = fmt_e(r["pred"])
        match = r.get("match", None)
        match_s = "✓" if match is True else ("✗" if match is False else "·")
        line = (
            f"{r['index']:>{widths['idx']}} | "
            f"{gt_ch:<{widths['gt_ch']}} | {gt_t:>{widths['gt_t']}} | {gt_v:<{widths['gt_v']}} | "
            f"{_trim(pr_ch, widths['pr_ch']):<{widths['pr_ch']}} | {_trim(pr_t, widths['pr_t']):>{widths['pr_t']}} | {_trim(pr_v, widths['pr_v']):<{widths['pr_v']}} | "
            f"{match_s:^{widths['match']}}"
        )
        out_lines.append(line)
        shown += 1

    if shown < len(rows):
        out_lines.append(f"... ({len(rows) - shown} more rows)")
    return "\n".join(out_lines)


def _compute_contributions(
    metrics: Dict[str, MetricResult], aggregator: WeightedAggregator
) -> Optional[Dict[str, float]]:
    # Only compute contributions if using weights (no custom formula)
    if getattr(aggregator, "formula", None) is not None:
        return None
    # Decide which weights apply
    if getattr(aggregator, "weights", None):
        weights = aggregator.weights
    else:
        weights = aggregator.weights_with_cont if "continuous" in metrics else aggregator.weights_no_cont
    contribs: Dict[str, float] = {}
    total = 0.0
    for name, w in weights.items():
        m = metrics.get(name)
        if m is None:
            continue
        part = float(w) * float(m.score)
        contribs[name] = part
        total += part
    contribs["_sum"] = total
    return contribs


def format_metrics_summary(
    metrics: Dict[str, MetricResult],
    *,
    aggregator: Optional[WeightedAggregator] = None,
) -> str:
    lines: List[str] = []
    # Per-metric summaries
    for name in ["discrete", "timing", "continuous", "structure"]:
        m = metrics.get(name)
        if not m:
            continue
        lines.append(f"- {name}: score={m.score:.3f}, pass={m.binary_pass}")
        d = m.details or {}
        if name == "discrete":
            vm = d.get("value_matches")
            tp = d.get("total_pairs")
            mmr = d.get("mismatch_rate")
            if vm is not None and tp is not None:
                lines.append(f"  · value_matches={vm}/{tp}, mismatch_rate={mmr:.3f}" if mmr is not None else f"  · value_matches={vm}/{tp}")
        elif name == "timing":
            for k in ("r2", "slope", "duration_ratio", "mape"):
                if k in d:
                    val = d[k]
                    if isinstance(val, (int, float)):
                        lines.append(f"  · {k}={val:.3f}")
                    else:
                        lines.append(f"  · {k}={val}")
        elif name == "continuous":
            if "per_channel" in d:
                per = d["per_channel"]
                for ch, info in per.items():
                    lines.append(f"  · {ch}: score={info.get('score', 0.0):.3f}, pass={info.get('binary_pass')}")
            else:
                for k in ("mae", "final_diff", "mae_score", "final_score"):
                    if k in d:
                        val = d[k]
                        lines.append(f"  · {k}={val:.3f}" if isinstance(val, (int, float)) else f"  · {k}={val}")
        elif name == "structure":
            if "gap_rate" in d:
                lines.append(f"  · gap_rate={float(d['gap_rate']):.3f}")
    # Contributions if available
    if aggregator is not None:
        contribs = _compute_contributions(metrics, aggregator)
        if contribs is None:
            lines.append("- aggregation: custom formula (per-metric contributions not shown)")
        else:
            lines.append("- aggregation contributions:")
            for k, v in contribs.items():
                if k == "_sum":
                    continue
                lines.append(f"  · {k}: {v:.3f}")
            lines.append(f"  · total: {contribs.get('_sum', 0.0):.3f}")
    return "\n".join(lines)


def _decision_breakdown(decision: BinaryDecision, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "require_discrete_exact": decision.require_discrete_exact,
        "timing_metric_name": decision.timing_metric_name,
        "required_metrics": list(decision.required_metric_names),
        "passes": {},  # per-metric pass/fail considered by decision
    }
    # Discrete
    if decision.require_discrete_exact:
        m = metrics.get("discrete")
        out["passes"]["discrete"] = bool(m and m.binary_pass)
    # Timing
    t = metrics.get(decision.timing_metric_name)
    out["passes"][decision.timing_metric_name] = bool(t and t.binary_pass)
    # Additional required metrics
    for name in decision.required_metric_names:
        m = metrics.get(name)
        out["passes"][name] = bool(m and m.binary_pass)
    return out


def build_json_report(
    result: EvaluateResult,
    *,
    evaluator: Optional[Evaluator] = None,
    req: Optional[EvaluateRequest] = None,
) -> Dict[str, Any]:
    """
    Create a rich JSON-serializable report including alignment rows (with match flags),
    per-metric details, and aggregation/decision breakdown.
    
    Note: The alignment rows will only include discrete events. Continuous channels
    (e.g., temperature) are filtered before alignment and evaluated separately.
    """
    report: Dict[str, Any] = {
        "full_score": result.full_score,
        "accuracy": result.accuracy,
        "metrics": {
            k: {"score": v.score, "binary_pass": v.binary_pass, "details": v.details}
            for k, v in result.metrics.items()
        },
        "alignment": [
            [
                (g.channel, g.timestamp, g.value) if g else None,
                (p.channel, p.timestamp, p.value) if p else None,
            ]
            for g, p in result.alignment
        ],
    }

    # Alignment rows with match flags (if we have request info)
    # Note: Continuous channels already filtered before alignment, so won't appear here
    if req is not None:
        rows = build_alignment_rows(
            result.alignment,
            comparator=req.comparator,
            comparator_map=req.comparators_by_channel,
            ignore_channels=req.ignore_channels,  # Only regular ignores, continuous already filtered
        )
        report["alignment_rows"] = rows

    # Aggregation contributions if available
    if evaluator is not None:
        contribs = _compute_contributions(result.metrics, evaluator.aggregator)
        if contribs is not None:
            report["aggregation"] = {
                "weights": evaluator.aggregator.weights
                or (
                    evaluator.aggregator.weights_with_cont
                    if "continuous" in result.metrics
                    else evaluator.aggregator.weights_no_cont
                ),
                "contributions": contribs,
            }
        else:
            report["aggregation"] = {"info": "custom_formula"}

        # Decision breakdown
        report["decision"] = _decision_breakdown(evaluator.decision, result.metrics)

    return report


def format_text_report(
    result: EvaluateResult,
    evaluator: Optional[Evaluator] = None,
    req: Optional[EvaluateRequest] = None,
    *,
    max_rows: int = 50,
    title: Optional[str] = None,
) -> str:
    """
    Build a human-friendly text report with:
      - header + full score + accuracy,
      - per-metric summary with details (including continuous scores),
      - alignment table (first max_rows) - only discrete events, continuous filtered out.
    """
    lines: List[str] = []
    hdr = title or "EnvTrace Report"
    lines.append("=" * 80)
    lines.append(hdr)
    lines.append("=" * 80)
    lines.append(f"Full score: {result.full_score:.3f}")
    lines.append(f"Accuracy:   {result.accuracy}")

    if evaluator is not None:
        lines.append("")
        lines.append("Metrics:")
        lines.append(
            format_metrics_summary(
                result.metrics, aggregator=getattr(evaluator, "aggregator", None)
            )
        )
        # Decision
        lines.append("")
        lines.append("Decision breakdown:")
        dec = _decision_breakdown(evaluator.decision, result.metrics)
        lines.append(f"  require_discrete_exact: {dec['require_discrete_exact']}")
        lines.append(f"  timing_metric_name:     {dec['timing_metric_name']}")
        lines.append(f"  required_metrics:       {', '.join(dec['required_metrics']) or '(none)'}")
        passes = dec.get("passes", {})
        for k, v in passes.items():
            lines.append(f"  pass[{k}]: {v}")

    # Alignment table (only discrete events - continuous channels already filtered)
    if req is not None:
        rows = build_alignment_rows(
            result.alignment,
            comparator=req.comparator,
            comparator_map=req.comparators_by_channel,
            ignore_channels=req.ignore_channels,  # Only regular ignores, continuous already filtered
        )
        lines.append("")
        lines.append("Alignment (first rows, discrete events only):")
        lines.append(format_alignment_table(rows, max_rows=max_rows))
    else:
        # If no request provided, still show a basic alignment table without match flags
        basic_rows = []
        for idx, (g, p) in enumerate(result.alignment, start=1):
            basic_rows.append(
                {
                    "index": idx,
                    "gt": _event_to_dict(g),
                    "pred": _event_to_dict(p),
                    "match": None,
                    "ignored": False,
                }
            )
        lines.append("")
        lines.append("Alignment (first rows, discrete events only):")
        lines.append(format_alignment_table(basic_rows, max_rows=max_rows))

    lines.append("=" * 80)
    return "\n".join(lines)
