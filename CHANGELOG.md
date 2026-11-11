# Changelog

## Unreleased: defaults implement the paper's method

The package defaults now follow the method of the EnvTrace paper ("Semantic Evaluation of
Language Models for Physical System Control") and its original scorer (VISION
`backend/tests/test_op_cog.py`). An audit (2026-11-11) found that the earlier defaults
differed from that scorer in eight ways; each one is fixed below and covered by a test in
`envtrace/tests/test_audit_vision_compare.py` (formerly strict xfail). **Scores computed with
the earlier defaults change.** Every earlier default is still available through an explicit
option, listed per item.

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

- `ignore_applies_to_continuous=True`: a continuous channel in `ignore_channels` is also left out
  of the continuous facet, the weight switch and the decision. The paper's scorer reads the
  continuous facet from the unfiltered traces, so the default keeps that.
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
  configuration file.

### Other API changes a user may notice

- An unknown comparator `type` in a config raises `ValueError` (before: silently the relative
  tolerance).
- `use_continuous` is honoured only with `continuous_weighting="always"`.
- Name-only matching of function-call events (the paper's `sam.setOrigin`, `wsam`, `sam.pos`) is
  not a default: the package cannot know which channels carry argument text. Configure it with
  `AlwaysTrueComparator` per channel (`paper_config.json` does).

### Reproduction (main batch, `saved_named_runs/new_prompt_new_dataset_12_8`)

- `paper_config.json`: unchanged, all 13,096 rows equal the stored scores (max deviation 7.8e-16,
  no pass flag differs; `check_against_paper_summary.txt`).
- Defaults, no configuration file (only the temperature channel declared continuous and the
  per-command ignore lists): 5,965 of the 6,016 scored model rows equal the paper's scores; 51
  differ (all lower, 15 strict passes lost), and every one is a `sam.setOrigin` pair whose
  argument text differs, which the paper's scorer matches by name only. Adding the name-only
  comparators for the three function-call channels leaves 0 differing rows. The 408 debug rows
  match; 2 of the 40 scientist rows differ for the same reason
  (`check_against_paper_defaults_summary.txt`).
