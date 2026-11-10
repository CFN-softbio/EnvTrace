"""Acceptance test: the envtrace package with paper_config.json reproduces the paper's scores.

Reads the main-batch run CSVs (saved_named_runs/new_prompt_new_dataset_12_8, one
results_op_agent.csv per run) of the EnvTrace paper working folder and scores every row
with the package. Three checks:

  A  stored     every row: package score of (stored selected reference trace, stored
                predicted trace) against the stored pv_match_rate, timing_score,
                temp_score, full_score, and the strict pass flag against the stored
                full_match column (where present) and the original scorer.
  B  best-of    every non-exact row: all references of the command (stored trace for the
                selected one, VISION execution_cache.json for the others), package
                Evaluator.evaluate_best_of versus the original selection loop: selected
                reference, scores and pass flag.
  C  corrected  rows whose traces carry a leaked sam.align event (analysis/A4_leak, rule G
                and the proposed predicted-trace rule): package versus the corr_* and
                prop_* columns of analysis/A4_leak/out/corrected_rows_main.csv.gz.

Two rules of the original harness are not trace scoring and live here, not in the
package: an exact code match scores 1/1/1/1 and passes without running anything, and a
command whose references have no discrete event at all scores 0/0/0/0 and fails.

B and C, and the pass-flag check on runs without a full_match column, need the original
scorer copy analysis/A4_leak/orig_scoring.py (and SciPy for it).

Usage, from the paper working folder (the parent of EnvTrace/):
    PYTHONPATH=EnvTrace python EnvTrace/envtrace/examples/paper/check_against_paper.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the EnvTrace repository root

from envtrace.core import Event, Trace  # noqa: E402
from envtrace.io import build_from_config, load_config  # noqa: E402

FIELDS = ["pv_match_rate", "timing_score", "temp_score", "full_score"]
TEMP = "XF:11BM-ES:{LINKAM}:TEMP"
MAIN = "new_prompt_new_dataset_12_8"
TOL = 1e-9


def jl(s):
    return json.loads(s) if s else []


def as_bool(x):
    return x if isinstance(x, bool) else str(x).strip() == "True"


def to_trace(logs):
    return Trace([Event(str(pv), float(ts), val) for pv, ts, val in logs])


def same(a, b, tol=TOL):
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def dev(a, b):
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return math.inf
    return abs(a - b)


class Paper:
    """The package, configured with paper_config.json."""

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.evaluator, self.template = build_from_config(Trace([]), Trace([]), self.cfg)
        self.base_ignore = set(self.template.ignore_channels or set())

    def _template(self, ignore_pvs):
        from dataclasses import replace
        return replace(self.template, ignore_channels=self.base_ignore | set(ignore_pvs))

    @staticmethod
    def _fields(res):
        m = res.metrics
        return dict(pv_match_rate=m["discrete"].score, timing_score=m["timing"].score,
                    temp_score=m["continuous"].score, full_score=res.full_score,
                    full_match=bool(res.accuracy))

    def score(self, gt_logs, pred_logs, ignore_pvs):
        from dataclasses import replace
        req = replace(self._template(ignore_pvs), gt=to_trace(gt_logs), pred=to_trace(pred_logs))
        return self._fields(self.evaluator.evaluate(req))

    def best_of(self, ref_logs, pred_logs, ignore_pvs):
        res, idx = self.evaluator.evaluate_best_of(
            [to_trace(g) for g in ref_logs], to_trace(pred_logs), template=self._template(ignore_pvs))
        out = self._fields(res)
        out["selected"] = idx
        return out


def harness_nopv(ref_logs, ignore_pvs):
    """Original harness rule (test_op_cog.py test_entry): no reference has a discrete event."""
    ign = set(ignore_pvs)
    return not any(e[0] not in ign and e[0] != TEMP for logs in ref_logs for e in logs)


NOPV = dict(pv_match_rate=0.0, timing_score=0.0, temp_score=0.0, full_score=0.0, full_match=False)


def load_orig(root):
    a4 = root / "analysis" / "A4_leak"
    if not (a4 / "orig_scoring.py").exists():
        return None
    sys.path.insert(0, str(a4))
    try:
        import common
        import gt_leak
        import orig_scoring  # noqa: F401
    except ImportError as exc:  # SciPy missing
        print(f"original scorer not available ({exc}); checks B and C skipped")
        return None
    return common, gt_leak


def orig_best(common, refs, logs_by_ref, pred, ign, is_complex):
    with contextlib.redirect_stdout(io.StringIO()):
        c = common.select_best(refs, logs_by_ref, pred, ign, is_complex)
    out = {f: float(c[f]) for f in FIELDS}
    out["full_match"] = bool(c["full_match"])
    out["selected"] = refs.index(c["snippet"])
    return out


class Tally:
    def __init__(self):
        self.n = 0
        self.maxdev = defaultdict(float)
        self.bad = defaultdict(int)
        self.flags = defaultdict(int)
        self.examples = []

    def add(self, mine, ref, where, fields=FIELDS, flag_keys=("full_match",)):
        self.n += 1
        for f in fields:
            d = dev(mine[f], ref[f])
            self.maxdev[f] = max(self.maxdev[f], d)
            if not same(mine[f], ref[f]):
                self.bad[f] += 1
                if len(self.examples) < 10:
                    self.examples.append((where, f, mine[f], ref[f]))
        for k in flag_keys:
            if k in ref and ref[k] is not None:
                self.flags[k] += 1
                if bool(mine[k]) != bool(ref[k]):
                    self.bad[k] += 1
                    if len(self.examples) < 10:
                        self.examples.append((where, k, mine[k], ref[k]))

    def lines(self, label):
        out = [f"{label}: {self.n} rows"]
        out.append("  max |deviation|: " + ", ".join(f"{f}={self.maxdev[f]:.3g}" for f in FIELDS
                                                      if f in self.maxdev))
        out.append("  pass flags compared: " + (", ".join(f"{k}={v}" for k, v in self.flags.items()) or "0"))
        out.append("  rows differing (> 1e-9, or flag differs): "
                   + (", ".join(f"{k}={v}" for k, v in sorted(self.bad.items())) or "none"))
        for e in self.examples:
            out.append(f"    example: {e}")
        return out


def run_dirs(batch_dir):
    for mdir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        if mdir.name == "archived":
            continue
        for csv in sorted(mdir.glob("*/run_*/results_op_agent.csv")):
            yield mdir.name, csv


def group_of(model):
    if model == "debug":
        return "debug (GT vs GT)"
    if model in ("bs_1", "bs_2"):
        return "scientists (bs_1, bs_2)"
    return "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]),
                    help="paper working folder (holds saved_named_runs/ and analysis/)")
    ap.add_argument("--config", default=str(HERE / "paper_config.json"))
    ap.add_argument("--batch", default=MAIN,
                    help="batch folder under saved_named_runs (check C only for the main batch)")
    ap.add_argument("--out", default=None,
                    help="summary file (default: check_against_paper_summary.txt here for the "
                         "main batch, none for other batches)")
    args = ap.parse_args()
    root = Path(args.root)
    paper = Paper(args.config)
    orig = load_orig(root)

    cache = None
    cache_path = root / "VISION_dev_be" / "tests" / "results" / "op_cog" / "execution_cache.json"
    if orig is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    tA = defaultdict(Tally)           # per group, against the stored values
    tA_orig = defaultdict(Tally)      # pass flag against the original scorer
    tB = Tally()
    counts = defaultdict(int)
    nan_rows = 0
    key_rows = {}                     # (model, timestamp, run, row) -> row, for check C

    for model, csv in run_dirs(root / "saved_named_runs" / args.batch):
        g = group_of(model)
        df = pd.read_csv(csv, keep_default_na=False, dtype=str)
        ts, run = csv.parts[-3], csv.parts[-2]
        for i, r in enumerate(df.itertuples(index=False)):
            if g == "models":
                key_rows[(model, ts, run, i)] = r
            stored = {f: float(getattr(r, f)) for f in FIELDS}
            stored["full_match"] = as_bool(r.full_match) if "full_match" in df.columns and r.full_match != "" else None
            if as_bool(r.exact_match):
                counts[(g, "exact code match (harness rule, 1/1/1/1)")] += 1
                mine = dict(pv_match_rate=1.0, timing_score=1.0, temp_score=1.0, full_score=1.0, full_match=True)
                tA[g].add(mine, stored, (model, ts, run, i))
                continue
            gt, pred, ign = jl(r.gt_pv_logs), jl(r.pred_pv_logs), jl(r.ignore_pvs)
            is_complex = as_bool(r.is_complex)
            counts[(g, "scored by the package")] += 1
            mine = paper.score(gt, pred, ign)
            if any(math.isnan(mine[f]) for f in FIELDS):
                nan_rows += 1
            tA[g].add(mine, stored, (model, ts, run, i))
            if orig is not None:
                common, _ = orig
                o = orig_best(common, [r.ground_truth_snippet], {r.ground_truth_snippet: gt},
                              pred, ign, is_complex)
                tA_orig[g].add(mine, o, (model, ts, run, i))

            # B: best of all references
            if cache is not None and g == "models":
                common, _ = orig
                refs = json.loads(r.expected_codes)
                logs = {}
                for s in refs:
                    if s == r.ground_truth_snippet:
                        logs[s] = gt
                    else:
                        c = cache.get(common.cache_key(s))
                        if c is not None:
                            logs[s] = c
                used = [s for s in refs if s in logs]
                if harness_nopv([logs[s] for s in used], ign):
                    counts[("best-of", "no discrete event in any reference (harness rule)")] += 1
                    continue
                o = orig_best(common, used, logs, pred, ign, is_complex)
                mine_b = paper.best_of([logs[s] for s in used], pred, ign)
                counts[("best-of", "rows with more than one reference")] += len(used) > 1
                counts[("best-of", "selected reference differs")] += mine_b["selected"] != o["selected"]
                tB.add(mine_b, o, (model, ts, run, i))

    # C: corrected traces (rule G, and rule G plus the proposed predicted-trace rule)
    tC = {"corr": Tally(), "prop": Tally()}
    corr_path = root / "analysis" / "A4_leak" / "out" / "corrected_rows_main.csv.gz"
    if args.batch == MAIN and orig is not None and cache is not None and corr_path.exists():
        common, gt_leak = orig
        cr = pd.read_csv(corr_path)
        for c in cr.itertuples(index=False):
            if c.exact_match:
                continue
            r = key_rows[(c.model, c.timestamp, c.run, int(c.row))]
            variants = []
            if c.gt_leaked:
                variants.append(("corr", True, False))
            if c.gt_leaked or c.pred_leaked:
                variants.append(("prop", bool(c.gt_leaked or c.any_ref_cache_leaked), True))
            for name, all_refs, strip_pred in variants:
                refs = json.loads(r.expected_codes)
                pred = jl(r.pred_pv_logs)
                if strip_pred:
                    pred, _ = gt_leak.strip_stray(pred, r.generated_code)
                logs = {}
                for s in refs:
                    if s == r.ground_truth_snippet:
                        logs[s], _ = gt_leak.strip_stray(jl(r.gt_pv_logs), s)
                    elif all_refs:
                        e = cache.get(common.cache_key(s))
                        if e is not None:
                            logs[s], _ = gt_leak.strip_stray(e, s)
                used = [s for s in refs if s in logs]
                ign = jl(r.ignore_pvs)
                if harness_nopv([logs[s] for s in used], ign):
                    mine = dict(NOPV)
                else:
                    mine = paper.best_of([logs[s] for s in used], pred, ign)
                ref = {f: getattr(c, f"{name}_{f}") for f in FIELDS}
                ref["full_match"] = bool(getattr(c, f"{name}_full_match"))
                tC[name].add(mine, ref, (c.model, c.timestamp, c.run, int(c.row)))

    lines = [f"check_against_paper.py: envtrace with {Path(args.config).name} versus the paper's scores",
             f"batch: saved_named_runs/{args.batch} (all model folders except archived); tolerance {TOL}", ""]
    for (grp, what), n in sorted(counts.items()):
        lines.append(f"{grp}: {what}: {n}")
    lines.append(f"rows where the package returns NaN in a score: {nan_rows}")
    lines.append("")
    lines.append("A. Stored values (pass flag against the stored full_match column, where present)")
    for g, t in tA.items():
        lines += ["  " + x for x in t.lines(g)]
    if tA_orig:
        lines.append("A'. Same rows, against the original scorer (orig_scoring.py), incl. pass flag")
        for g, t in tA_orig.items():
            lines += ["  " + x for x in t.lines(g)]
    if tB.n:
        lines.append("B. Best of all references (package evaluate_best_of vs original selection loop)")
        lines += ["  " + x for x in tB.lines("models")]
    for name, label in (("corr", "C. Corrected, rule G (corr_* columns)"),
                        ("prop", "C. Corrected, rule G plus proposed predicted-trace rule (prop_* columns)")):
        if tC[name].n:
            lines.append(label)
            lines += ["  " + x for x in tC[name].lines("models")]
    text = "\n".join(lines)
    print(text)
    out = args.out or (str(HERE / "check_against_paper_summary.txt") if args.batch == MAIN else None)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
