from __future__ import annotations
from typing import Dict, Callable, List
from envtrace.core.metrics import MetricResult

class WeightedAggregator:
    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        use_continuous: bool | None = None,
        formula: Callable[[Dict[str, MetricResult]], float] | None = None,
    ) -> None:
        """
        If 'formula' is provided, it will be used to compute the full score
        from the metrics dict. Otherwise, weights-based aggregation is used.
        """
        self.weights_no_cont = {"discrete": 0.8, "timing": 0.2}
        self.weights_with_cont = {"discrete": 0.6, "timing": 0.2, "continuous": 0.2}
        self.weights = weights
        self.use_continuous = use_continuous
        self.formula = formula

    def aggregate(self, metrics: Dict[str, MetricResult]) -> float:
        # Custom formula takes precedence
        if self.formula is not None:
            return float(self.formula(metrics))

        use_cont = self.use_continuous
        if use_cont is None:
            use_cont = "continuous" in metrics
        weights = self.weights or (self.weights_with_cont if use_cont else self.weights_no_cont)

        total = 0.0
        for name, w in weights.items():
            m = metrics.get(name)
            if m is None:
                continue
            total += w * float(m.score)
        return total

class BinaryDecision:
    def __init__(
        self,
        require_discrete_exact: bool = True,
        timing_metric_name: str = "timing",
        required_metric_names: List[str] | None = None,
    ) -> None:
        self.require_discrete_exact = require_discrete_exact
        self.timing_metric_name = timing_metric_name
        self.required_metric_names = required_metric_names or []

    def decide(self, metrics: Dict[str, MetricResult]) -> bool:
        if self.require_discrete_exact:
            disc = metrics.get("discrete")
            if not disc or not bool(disc.binary_pass):
                return False
        timing = metrics.get(self.timing_metric_name)
        if not timing or not bool(timing.binary_pass):
            return False
        for name in self.required_metric_names:
            m = metrics.get(name)
            if not m or not bool(m.binary_pass):
                return False
        return True
