"""
Simple EnvTrace example that loads JSON traces and prints a pretty report.
"""
from pathlib import Path
import envtrace
from envtrace.io.json_io import load_trace
from envtrace.core import Evaluator, EvaluateRequest, NumericToleranceComparator
from envtrace.reporting import format_text_report


def main() -> None:
    pkg_dir = Path(envtrace.__file__).resolve().parent
    traces_dir = pkg_dir / "examples" / "traces"
    gt = load_trace(traces_dir / "gt.json")
    pred = load_trace(traces_dir / "pred.json")

    req = EvaluateRequest(
        gt=gt,
        pred=pred,
        comparator=NumericToleranceComparator(1e-3),
        ignore_channels={"det:AcquireTime"},
        include_structure=True,
    )
    ev = Evaluator.default()
    res = ev.evaluate(req)

    print(format_text_report(res, evaluator=ev, req=req, max_rows=30, title="EnvTrace Simple Example"))


if __name__ == "__main__":
    main()
