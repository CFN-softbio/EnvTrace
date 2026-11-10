from __future__ import annotations
import argparse
from dataclasses import replace
from typing import Any, Dict
from envtrace.io.json_io import load_trace, save_json
from envtrace.io import load_config, build_from_config
from envtrace.core.evaluator import Evaluator, EvaluateRequest
from envtrace.reporting import format_text_report, build_json_report
from envtrace import __version__

def main() -> None:
    parser = argparse.ArgumentParser(prog="envtrace", description="EnvTrace CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_align = sub.add_parser("align", help="Align two traces and compute scores")
    p_align.add_argument("--gt", required=True, help="Path to ground-truth trace JSON")
    p_align.add_argument("--pred", required=True, help="Path to predicted trace JSON")
    p_align.add_argument("--out", required=False, help="Path to write evaluation JSON report")
    p_ver = sub.add_parser("version", help="Show EnvTrace version and exit")

    p_eval = sub.add_parser("evaluate", help="Evaluate using a config file (JSON or YAML)")
    p_eval.add_argument("--gt", required=True, nargs="+",
                        help="Path(s) to ground-truth trace JSON; with several, the reference "
                             "with the highest full score is used")
    p_eval.add_argument("--pred", required=True, help="Path to predicted trace JSON")
    p_eval.add_argument("--config", required=True, help="Path to configuration file (JSON/YAML)")
    p_eval.add_argument("--out", required=False, help="Path to write evaluation JSON report")

    args = parser.parse_args()

    if args.cmd == "version":
        print(__version__)
        return

    if args.cmd == "align":
        gt = load_trace(args.gt)
        pred = load_trace(args.pred)
        ev = Evaluator.default()
        req = EvaluateRequest(gt=gt, pred=pred)
        result = ev.evaluate(req)

        report: Dict[str, Any] = build_json_report(result, evaluator=ev, req=req)

        if args.out:
            save_json(args.out, report)
        else:
            print(format_text_report(result, evaluator=ev, req=req))

    if args.cmd == "evaluate":
        cfg = load_config(args.config)
        gts = [load_trace(p) for p in args.gt]
        pred = load_trace(args.pred)
        evaluator, req = build_from_config(gts[0], pred, cfg)
        if len(gts) == 1:
            result = evaluator.evaluate(req)
            best = 0
        else:
            result, best = evaluator.evaluate_best_of(gts, pred, template=req)
            req = replace(req, gt=gts[best])

        report: Dict[str, Any] = build_json_report(result, evaluator=evaluator, req=req)
        report["selected_reference"] = {"index": best, "path": args.gt[best]}

        if args.out:
            save_json(args.out, report)
        else:
            print(format_text_report(result, evaluator=evaluator, req=req))
            if len(gts) > 1:
                print(f"Selected reference: {best} ({args.gt[best]})")

if __name__ == "__main__":
    main()
