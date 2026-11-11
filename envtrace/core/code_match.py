"""Opt-in exact-code-match rule of the paper's original harness.

The EnvTrace paper's harness (VISION backend/tests/test_op_cog.py) did not execute a
generated program whose text equals one of the reference programs after comments and
surrounding whitespace are removed: such a row scores 1.0 on every facet and passes.
The core evaluator works on traces only and does not apply this rule; call these
helpers before evaluating if you want it:

    if exact_code_match(generated_code, reference_codes):
        result = exact_code_match_result()
    else:
        result, idx = evaluator.evaluate_best_of(reference_traces, pred_trace, template=req)

strip_comments is a verbatim port of the harness function and has its limitation: it
cuts every line at the first '#', also inside a string literal, so two programs that
differ only after a '#' in a string count as equal.
"""

from __future__ import annotations

from io import StringIO
from typing import Iterable

from envtrace.core.evaluator import EvaluateResult
from envtrace.core.metrics import MetricResult


def strip_comments(code: str) -> str:
    """Remove '#' comments and trailing whitespace; drop lines that become empty."""
    result = []
    for line in StringIO(code).readlines():
        code_line = line.split("#")[0].rstrip()
        if code_line.strip():
            result.append(code_line)
    return "\n".join(result)


def exact_code_match(generated_code: str, reference_codes: Iterable[str]) -> bool:
    """True if the generated code equals any reference after strip_comments and strip()."""
    gen = strip_comments(generated_code).strip()
    return any(gen == strip_comments(ref).strip() for ref in reference_codes)


def exact_code_match_result() -> EvaluateResult:
    """The harness result for an exact code match: every facet 1.0, full 1.0, passes."""
    metrics = {
        name: MetricResult(
            name=name,
            score=1.0,
            binary_pass=True,
            details={"reason": "exact code match (not executed)"},
        )
        for name in ("discrete", "timing", "continuous")
    }
    return EvaluateResult(metrics=metrics, full_score=1.0, accuracy=True, alignment=[])
