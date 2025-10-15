# EnvTrace TODO and Roadmap

A domain-agnostic Python package for trace alignment and semantic evaluation of code-driven environments.

This document tracks implementation steps toward a pip-installable, well-documented, and extensible EnvTrace package. Items are grouped by milestone versions to keep scope manageable.

---

## Milestone v0.1.0 — Minimal, dependency-light core (trace-in, metrics-out)

- [x] Project scaffolding
  - [x] Create `pyproject.toml` (name: `envtrace`, version, metadata; minimal deps)
  - [x] Choose license (Apache-2.0), still has to be approved by BNL
  - [x] Set up basic tooling (ruff and black) and CI (lint)
- [x] Core data model
  - [x] `Event` dataclass: `channel: str`, `timestamp: float`, `value: Any`, `meta: dict[str, Any] = {}`
  - [x] `Trace` abstraction: ordered collection + helpers
    - [x] sort by time, set t0 baseline, filter by channels/time window
    - [x] canonicalize channels via alias map
    - [x] deduplicate and coalesce events (optional)
  - [x] JSON I/O: load/save `Trace` (define simple JSON schema + version)
- [x] Alignment
  - [x] `SequenceAligner` protocol (plug-in interface)
  - [x] `DifflibAligner` (default), aligning by channel key
  - [x] Unit tests: alignment edge cases (insert/delete/replace)
- [x] Comparators
  - [x] `ValueComparator` protocol (pluggable)
  - [x] Built-ins: exact equality, numeric tolerance, “ignore-params” (markers like function calls)
  - [x] Registry and per-channel comparator selection
- [x] Metrics (discrete + timing)
  - [x] `Metric` protocol and `MetricResult` structure
  - [x] `DiscreteMatchMetric`:
    - [x] Computes match_rate, mismatch_rate, exact flag
    - [x] Supports ignore lists and channel aliasing
    - [x] Returns aligned tables and counts
  - [x] `TimingMetric`:
    - [x] Uses timestamps of value-matched aligned pairs
    - [x] Outputs R², slope, duration ratio, MAPE, and a normalized timing score
    - [x] Thresholds configurable (e.g., R²≥0.9, slope ∈ [0.8, 1.2], MAPE ≤ 1.0)
- [x] Scoring and decisions
  - [x] `WeightedAggregator` for full_score
    - [x] Default: no continuous channels → `0.8 * pv_match_rate + 0.2 * timing_score`
    - [x] Configurable weights and formula
  - [x] `BinaryDecision` (“EnvTrace accuracy”)
    - [x] Encapsulate strict pass/fail thresholds for discrete + timing metrics
- [x] Evaluator orchestration
  - [x] `EvaluateRequest`: traces + config (aligner, comparators, metrics, weights)
  - [x] `EvaluateResult`: aligned pairs, per-metric outputs, `full_score`, `accuracy`, diagnostics
  - [x] Multi-reference support: pick GT with max `full_score`
- [x] CLI (initial)
  - [x] `envtrace align --gt gt.json --pred pred.json --out result.json`
  - [x] Text summary to stdout; JSON report to file
- [x] Tests
  - [x] Unit tests for data model, alignment, comparators, metrics, aggregator, evaluator
  - [x] Golden tests with tiny example traces
- [x] Docs (initial)
  - [x] README quickstart (trace-in, score-out), JSON schema, API overview

---

## Milestone v0.2.0 — Continuous profiles, config files, richer reports

- [x] Continuous metrics
  - [x] `ContinuousProfileMetric` (generic): MAE + final-diff with exponential scoring
  - [x] Configurable characteristic scales (e.g., degrees, % of range)
  - [x] Aggregate multiple continuous channels into a composite continuous score
- [x] Custom metrics injection and aggregator overrides
  - [x] EvaluateRequest.custom_metrics support
  - [x] EvaluateRequest.continuous_metric override
  - [x] WeightedAggregator(formula=...) support
- [x] Configuration
  - [x] YAML/JSON config for:
    - [x] Channel aliases and ignore lists
    - [x] Channel types (discrete/continuous), comparators per channel
    - [x] Metric thresholds and weights
    - [x] Aligner selection and params (current: difflib)
    - [x] Binary decision config (require discrete exact, timing metric name, required metrics)
  - [x] Loader to build `EvaluateRequest` from config file
- [x] Reporting
  - [x] Text and JSON reports with:
    - [x] Aligned rows (gt/pred/None), match flags (basically pretty printing alignment)
    - [x] Per-metric details (R², slope, MAPE, MAE/final-diff)
    - [x] Full score and decision breakdown
- [ ] CLI (expanded)
  - [ ] `envtrace evaluate --gt-dir ... --pred-dir ... --config config.yaml --out result.json`
  - [ ] Option to select best among multiple GTs

---

## Milestone v0.3.0 — Caching, notebooks, and optional extras (no simulator integration)
- [ ] Pandas/Notebook helpers
  - [ ] DataFrame exporters for aligned tables and metrics
  - [ ] Example notebooks for visualization and analysis
- [ ] Examples and documentation
  - [x] Expand README examples and cookbook for common domains
  - [x] Add guidance on custom comparators/metrics/aggregators/decisions

---

## Milestone v1.0 — API stability, docs site, and release

- [ ] API stabilization and semantic versioning
- [ ] Docs site (Sphinx/MkDocs) with:
  - [ ] Architecture overview and design rationale
  - [ ] Tutorials (basic → advanced)
  - [ ] Plugin authoring guide (metrics/aligners/comparators)
- [ ] Examples gallery
  - [ ] Scientific instrumentation JSON trace examples (no simulator dependency)
  - [ ] Robotics and process control JSON trace examples
- [ ] Governance and community
  - [ ] CONTRIBUTING.md and Code of Conduct
  - [ ] Issue/PR templates
- [ ] Packaging and citation
  - [ ] Publish to PyPI
  - [ ] `CITATION.cff` and Zenodo DOI (actually I'd rather have them cite the paper)
  - [ ] “How to cite” section mirroring the paper

---

## Mapping from current code (for migration)

- [ ] `is_temp_match` → `ContinuousProfileMetric` (generic)
- [ ] `_compute_pv_match_metrics` → `DiscreteMatchMetric` + `SequenceAligner` + `ValueComparator`
- [ ] `_timing_match` → `TimingMetric` (R², slope, duration ratio, MAPE)
- [ ] PV canonicalization and ignore sets → transform/config + comparators
- [ ] Full score policy:
  - [ ] with continuous channels: `0.6 * pv_rate + 0.2 * timing + 0.2 * continuous`
  - [ ] without continuous: `0.8 * pv_rate + 0.2 * timing`
- [ ] Strict accuracy → `BinaryDecision` thresholds matching paper SI

---

## Open questions

- [ ] Provide Needleman–Wunsch aligner and DTW for continuous channels?
- [ ] Robust timing (RANSAC) for outlier resistance?
- [ ] Visualization helpers (HTML reports) as optional extra?
- [ ] Default comparator policies for categories (bools/enums/floats)?
- [ ] Multi-stream synchronization (if multiple clocks)?

---

## Nice-to-haves

- [ ] Pluggable penalty models in aligners (gap/substitution costs)
- [ ] Event coalescing strategies (by channel/value/time window)
- [ ] CLI subcommands for threshold calibration from labeled datasets
- [ ] Built-in small synthetic datasets for demos and tests

---

## Acceptance criteria per milestone

- v0.1.0:
  - [x] Run `envtrace align` on two JSON traces and get a deterministic full score and decision
  - [ ] 90%+ unit test coverage on core (events, aligner, metrics, scoring)
- v0.2.0:
  - [x] Evaluate continuous channels with config-driven scoring
  - [ ] Reproduce paper’s scoring policies via config
- v0.3.0:
  - [ ] Demonstrate evaluation using provided JSON traces and custom metrics (no simulator/adapters required)
- v1.0:
  - [ ] Stable API, docs site, PyPI package, and citation artifacts

---
