from __future__ import annotations
from typing import Dict, Callable, List
from envtrace.core.metrics import MetricResult, is_undefined

class WeightedAggregator:
    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        use_continuous: bool | None = None,
        formula: Callable[[Dict[str, MetricResult]], float] | None = None,
    ) -> None:
        """
        If 'formula' is provided, it will be used to compute the full score
        from the metrics dict (it receives undefined facets as NaN scores and must
        handle them). Otherwise, weights-based aggregation is used.

        Undefined facets: a weighted metric whose score is NaN (the timing facet with
        one matched pair, see TimingMetric(single_matched_pair=...)) is left out and the
        weights of the other facets are scaled up proportionally, so that they keep
        their original sum (see effective_weights).
        """
        self.weights_no_cont = {"discrete": 0.8, "timing": 0.2}
        self.weights_with_cont = {"discrete": 0.6, "timing": 0.2, "continuous": 0.2}
        self.weights = weights
        self.use_continuous = use_continuous
        self.formula = formula

    def base_weights(self, metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """The weight set that applies (explicit weights, or the set with or without continuous)."""
        use_cont = self.use_continuous
        if use_cont is None:
            use_cont = "continuous" in metrics
        return self.weights or (self.weights_with_cont if use_cont else self.weights_no_cont)

    def effective_weights(self, metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """Weights after leaving out undefined (NaN) facets, renormalised proportionally.

        Example: 0.6/0.2/0.2 with timing undefined gives discrete 0.75, continuous 0.25;
        0.8/0.2 gives discrete 1.0. A weighted metric that is missing from metrics keeps
        its weight (it contributes 0, as before). If every weighted facet is undefined,
        the result is empty and the full score is NaN.
        """
        weights = self.base_weights(metrics)
        undefined = {n for n in weights if is_undefined(metrics.get(n))}
        if not undefined:
            return dict(weights)
        total = sum(weights.values())
        kept = {n: w for n, w in weights.items() if n not in undefined}
        kept_sum = sum(kept.values())
        if kept_sum == 0:
            return {}
        return {n: w * total / kept_sum for n, w in kept.items()}

    def aggregate(self, metrics: Dict[str, MetricResult]) -> float:
        # Custom formula takes precedence
        if self.formula is not None:
            return float(self.formula(metrics))

        weights = self.effective_weights(metrics)
        if not weights:
            return float("nan")

        total = 0.0
        for name, w in weights.items():
            m = metrics.get(name)
            if m is None:
                continue
            total += w * float(m.score)
        return total

def continuous_involved(metrics: Dict[str, MetricResult]) -> bool:
    """True if a continuous metric was computed and a continuous channel has an event.

    The evaluator records this as details["involved"]; a continuous metric built
    elsewhere without that key counts as involved.
    """
    m = metrics.get("continuous")
    if m is None:
        return False
    return bool((m.details or {}).get("involved", True))


class BinaryDecision:
    def __init__(
        self,
        require_discrete_exact: bool = True,
        timing_metric_name: str = "timing",
        required_metric_names: List[str] | None = None,
        require_continuous: bool = True,
    ) -> None:
        """
        Strict pass/fail ("EnvTrace accuracy"): discrete exact AND timing pass AND
        every metric in required_metric_names passes.

        require_continuous (default True): also require the continuous check whenever
        a continuous channel is involved (an event of a configured continuous channel
        in the reference or the prediction), as in the paper. False restores the
        behaviour before the fix-paper-defaults change (the continuous check counts
        only if "continuous" is listed in required_metric_names).
        """
        self.require_discrete_exact = require_discrete_exact
        self.timing_metric_name = timing_metric_name
        self.required_metric_names = required_metric_names or []
        self.require_continuous = require_continuous

    def required_names(self, metrics: Dict[str, MetricResult]) -> List[str]:
        names = list(self.required_metric_names)
        if self.require_continuous and continuous_involved(metrics) and "continuous" not in names:
            names.append("continuous")
        return names

    def decide(self, metrics: Dict[str, MetricResult]) -> bool:
        if self.require_discrete_exact:
            disc = metrics.get("discrete")
            if not disc or not bool(disc.binary_pass):
                return False
        timing = metrics.get(self.timing_metric_name)
        if not timing or not bool(timing.binary_pass):
            return False
        for name in self.required_names(metrics):
            m = metrics.get(name)
            if not m or not bool(m.binary_pass):
                return False
        return True
