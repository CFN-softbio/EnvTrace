# Changelog

## Unreleased: defaults implement the paper's method

The package defaults now follow the method of the EnvTrace paper ("Semantic Evaluation of
Language Models for Physical System Control") and its original scorer (VISION
`backend/tests/test_op_cog.py`). An audit (2026-11-11) found that the earlier defaults
differed from that scorer in eight ways; each one is fixed below and covered by a test in
`envtrace/tests/test_audit_vision_compare.py` (formerly strict xfail). **Scores computed with
the earlier defaults change.** Every earlier default is still available through an explicit
option, listed per item.

The same audit found two rules of the original scorer that are defects for the method's
purpose; the defaults correct them (next section), and `paper_config.json` keeps the original
rules so that it still reproduces the paper.

### Changed defaults: corrections of the paper's scorer

- **One matched pair: timing undefined.** With exactly one aligned pair that matches, no pacing
  can be measured, so the timing facet is undefined: score `NaN`, `details["undefined"] = True`,
  `binary_pass` True (the check does not apply, so the strict decision is unchanged). The
  `WeightedAggregator` leaves an undefined facet out and scales the other weights up
  proportionally: the full score is the PV match rate without a continuous channel, and
  0.75 x PV + 0.25 x continuous with one (weights 0.6/0.2/0.2). 0 matched pairs (timing 0) and
  2 or more are unchanged. The original scorer gives 1.0 and pass, which adds 0.2 to a row that
  has one correct event among several (PV match rate below 1). The text report prints
  `score=undefined`; the JSON report writes `null` for the score, and its `aggregation.weights`
  are the weights applied. A custom `formula` receives the `NaN`. If every weighted facet is
  undefined (custom weights on timing only), the full score is `NaN`. Old behaviour:
  `TimingMetric(single_matched_pair="full_credit")`, config `timing.single_matched_pair:
  "full_credit"`.
- **Ignored continuous channels are left out of the continuous facet.**
  `ignore_applies_to_continuous=True` is now the default (it was opt-in): a continuous channel
  in `ignore_channels` is also left out of the continuous facet, the weight switch and the
  strict decision. The original scorer read the temperature facet from the unfiltered traces,
  so a command whose ignore list contains the temperature readback could still lose 0.2. Old
  behaviour: `ignore_applies_to_continuous=False` (also in a config).

### Changed defaults

- **Channel check on matched pairs.** An aligned pair matches only if both events have the
  same channel and their values match (`require_channel_match=True` in `EvaluateRequest`,
  `DiscreteMatchMetric`, `TimingMetric.evaluate` and configs). Before, a pair from a difflib
  `replace` block with different channels but equal values counted as a match (X = 0.2 against
  Z = 0.2 scored 1.0). Old behaviour: `require_channel_match=False`.
- **Absolute tolerance.** The default comparator is `AbsoluteToleranceComparator(1e-3)`
  (`abs(float(a) - float(b)) < 1e-3`, string fallback), also for a config without a comparator
  spec. Before, it was the relative `NumericToleranceComparator(1e-3)`, which failed float noise
  at zero (0.0 against 7.2e-16) and accepted 313.5 against 313.6. Old behaviour:
  `comparator=NumericToleranceComparator(1e-3)`, now also exported as
  `RelativeToleranceComparator`; config type `numeric_tolerance` (new alias `relative_tolerance`).
- **Recorded order.** Traces keep their recorded order (`sort_by_time=False`). The recorded order
  is what the paper's scorer aligned, and the field and `Trace.sort_by_time()` remain, so no API
  is lost; sort explicitly if a trace is merged from several sources. Old behaviour:
  `sort_by_time=True`.
- **Empty reference.** A reference without discrete events gives PV match rate 0 and is never
  exact (`empty_reference_fails=True`). Before, two empty traces scored 1.0 and exact (full score
  0.8 with temperature configured). Old behaviour: `empty_reference_fails=False`.
- **Continuous weights only when a continuous channel has an event.** 0.6/0.2/0.2 applies only
  if a configured continuous channel has an event in the reference or the prediction, else
  0.8/0.2 (`continuous_weighting="if_present"`). Before, the continuous weights applied whenever
  `continuous_channels` was configured, so code that changed nothing earned 0.2. Old behaviour:
  `continuous_weighting="always"`.
- **Strict decision includes the continuous check.** `BinaryDecision(require_continuous=True)`
  requires the continuous metric to pass whenever a continuous channel has an event (recorded in
  the continuous metric as `details["involved"]`). Before, the check counted only if
  `"continuous"` was listed in `required_metric_names`. Old behaviour:
  `BinaryDecision(require_continuous=False)`, config `decision.require_continuous: false`.
- **Degenerate timing (new guard).** With two or more matched pairs: if all matched reference
  timestamps are equal, timing is 1.0 and passes when the predicted ones are all equal too, else
  0.0 and fails (polyfit raised `LinAlgError`, the Pearson formulas gave NaN). If only the
  predicted timestamps are all equal, R² is 0 (polyfit reported 1.0, which gave 0.4 of timing
  credit; Pearson gave NaN). No stored row of the paper is affected.

### Fixed

- `Evaluator.evaluate_best_of(gt_list, pred, **kwargs)` builds the request from every
  `EvaluateRequest` field; before, it kept only seven of them, so for example
  `continuous_channels` was dropped (temperature events were aligned as discrete events) and
  `use_continuous=True` gave a perfect match a full score of 0.8. Unknown keyword names now raise
  `TypeError`. With `template=...`, keyword arguments override the template's fields (before they
  were ignored). An empty `gt_list` is evaluated with the template's settings.
- The text and JSON reports list `"continuous"` among the required metrics when the decision
  requires it.

### Added (opt-in, default off)

- `no_discrete_reference_scores_zero=True`: the paper harness's rule that a command whose
  references have no discrete event scores 0 on every facet and fails; `evaluate` applies it to
  its reference, `evaluate_best_of` only when no reference has a discrete event (returning index
  -1).
- `envtrace.core.exact_code_match(generated_code, reference_codes)`,
  `exact_code_match_result()` and `strip_comments(code)`: the harness's exact-code rule (equal
  after removing `#` comments and surrounding whitespace scores 1.0 everywhere and passes, without
  execution). It agrees with the stored `exact_match` column on all 12,688 model and scientist
  rows of the paper's main batch.
- `envtrace/examples/paper/check_against_paper.py --defaults` scores the main batch with no
  configuration file; `--rows-out FILE` writes every differing (row, field) as a CSV.
- `WeightedAggregator.base_weights(metrics)` and `effective_weights(metrics)`: the weight set
  that applies, before and after leaving out undefined facets.

### Other API changes a user may notice

- An unknown comparator `type` in a config raises `ValueError` (before: silently the relative
  tolerance).
- `use_continuous` is honoured only with `continuous_weighting="always"`.
- Name-only matching of function-call events (the paper's `sam.setOrigin`, `wsam`, `sam.pos`) is
  not a default: the package cannot know which channels carry argument text. Configure it with
  `AlwaysTrueComparator` per channel (`paper_config.json` does).

### Reproduction (main batch, `saved_named_runs/new_prompt_new_dataset_12_8`)

- `paper_config.json` (with `timing.single_matched_pair: "full_credit"` and
  `ignore_applies_to_continuous: false`): all 13,096 rows equal the stored scores (max deviation
  7.8e-16, no pass flag differs; `check_against_paper_summary.txt`).
- Defaults, no configuration file (only the temperature channel declared continuous and the
  per-command ignore lists): 5,644 of the 6,016 scored model rows have the paper's full score;
  372 differ (`check_against_paper_defaults_summary.txt`):
  - 51 `sam.setOrigin` pairs whose argument text differs, which the paper's scorer matches by
    name only (all lower, 15 strict passes lost against the original scorer, the only strict
    differences); 2 of them are also lowered by the one-pair correction;
  - 302 rows with one matched pair and a PV match rate below 1 (lower by 0.10 to 0.19), and 19
    rows with the temperature readback on the ignore list and a temperature event (higher by
    0.13 to 0.20, no strict flag changes). On the corrected trace set of the paper's analysis
    these are exactly the 285 and 19 rows of corrections V2 and V3 of the pre-registered
    robustness check, with the same scores (1e-9); the other 17 rows have one matched pair only
    in the stored, uncorrected traces.
  The timing score is undefined (`NaN`) on the 648 rows with one matched pair. Adding the
  name-only comparators for the three function-call channels leaves only the 321 corrected rows.
  Debug (GT against GT, 408 rows): 9 one-pair rows lower and 1 row higher (run_2 row 98, "Heat
  up the thin film sample to 102.7", now passes: it failed only the temperature check, on an
  ignored readback). 2 of the 40 scientist rows differ (the `sam.setOrigin` reason).
