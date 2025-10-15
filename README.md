# EnvTrace

**Behavioral evaluation of code through execution trace alignment**

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

EnvTrace is a domain-agnostic Python toolkit for evaluating code that interacts with physical or simulated environments. Instead of comparing code syntax, EnvTrace compares **what the code actually does** by analyzing execution traces—time-ordered sequences of state changes in the environment.

**Key Features:**
- 🎯 **Semantic evaluation**: Compare code by behavior, not syntax
- 🔄 **Trace alignment**: Intelligent sequence matching with insertions/deletions
- 📊 **Multi-faceted scoring**: Discrete state matching, timing analysis, and continuous profile tracking
- 🔌 **Extensible**: Custom comparators, metrics, and scoring formulas
- 🛠️ **Production-ready**: CLI tools, configuration files, and comprehensive reporting

**Use Cases:**
- Evaluating LLM-generated control code for scientific instruments
- Validating robotic control sequences
- Testing industrial automation scripts
- Comparing alternative implementations of cyber-physical systems

---

## Table of Contents
- [Citation](#citation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [Configuration Files](#configuration-files)
- [CLI Reference](#cli-reference)
- [Advanced Topics](#advanced-topics)

## Citation
If you decide to use this library, please consider citing:
```bibtex
arXiv submission forthcoming:

EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines
```
---

## Installation

**Requirements:** Python 3.10 or higher

### From PyPI (when available)
```bash
pip install envtrace
```

### From Source
```bash
git clone https://github.com/CFN-softbio/EnvTrace.git
cd EnvTrace/envtrace
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Python API

```python
from envtrace.core import Event, Trace, Evaluator, EvaluateRequest
from envtrace.core import NumericToleranceComparator

# Create execution traces (ground truth and predicted)
gt = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.10, 1),
    Event("det:Acquire", 1.10, 0),
])

pred = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.12, 1),  # Slightly different timing
    Event("det:Acquire", 1.08, 0),
])

# Evaluate with default metrics
evaluator = Evaluator.default()
request = EvaluateRequest(
    gt=gt, 
    pred=pred, 
    comparator=NumericToleranceComparator(tol=1e-3)
)
result = evaluator.evaluate(request)

# View results
print(f"Full score: {result.full_score:.3f}")
print(f"Binary accuracy: {result.accuracy}")
print(f"Discrete match: {result.metrics['discrete'].score:.3f}")
print(f"Timing score: {result.metrics['timing'].score:.3f}")
```

### Command Line Interface

```bash
# Basic alignment and scoring
envtrace align --gt traces/ground_truth.json --pred traces/predicted.json

# With configuration file
envtrace evaluate --gt traces/gt.json --pred traces/pred.json \
                  --config config.json --out results/report.json

# View version
envtrace version
```

### Example Output

```
================================================================================
EnvTrace Report
================================================================================
Full score: 0.991
Accuracy:   True

Metrics:
- discrete: score=1.000, pass=True
  · value_matches=4/4, mismatch_rate=0.000
- timing: score=0.955, pass=True
  · r2=0.999
  · slope=0.989
  · duration_ratio=1.008
  · mape=0.180
- structure: score=1.000, pass=True
  · gap_rate=0.000

Alignment (first rows, discrete events only):
 IDX | GT.channel       |     GT.t | GT.value         | PR.channel       |     PR.t | PR.value         |  MATCH 
----------------------------------------------------------------------------------------------------------------
   1 | motor:x          |    0.000 | 0.0              | motor:x          |    0.000 | 0.0              |    ✓   
   2 | det:Acquire      |    0.100 | 1                | det:Acquire      |    0.120 | 1                |    ✓   
   3 | det:Acquire      |    1.100 | 0                | det:Acquire      |    1.080 | 0                |    ✓   
================================================================================
```

---

## Core Concepts

### Event and Trace
- Event: `(channel: str, timestamp: float, value: Any, meta: dict)` represents a single environment change.
- Trace: an ordered list of Events with helpers for sorting, canonicalization, filtering, and JSON I/O.

### Aligners
- Sequence alignment is performed over channel tokens (values are checked later).
- Default: DifflibAligner (stable, fast).
- You may introduce other aligners (Needleman–Wunsch, DTW for continuous), by implementing the `SequenceAligner` protocol.

### Comparators
- A ValueComparator decides whether two values match for a given channel. Built-ins:
  - ExactEqualityComparator: numeric equality or string equality.
  - NumericToleranceComparator: numeric match within a relative tolerance.
  - AlwaysTrueComparator: always match (useful for function-call markers where value does not matter).
- You can override comparators per channel:
```python
from envtrace.core import ExactEqualityComparator, NumericToleranceComparator

comparator_map = {
    "det:Acquire": ExactEqualityComparator(),       # exact match for detector toggles
    "motor:x":     NumericToleranceComparator(1e-2) # fuzzy match for motor values (1% rel tol)
}
req = EvaluateRequest(
    gt=gt, pred=pred,
    comparator=NumericToleranceComparator(1e-3),    # default comparator
    comparators_by_channel=comparator_map,          # per-channel overrides
)
```

### 4. Evaluation Metrics

EnvTrace computes multiple metrics to assess different aspects of code behavior:

#### Discrete Match Metric
- **What it measures**: Correctness of state sequence (which actions, in what order, with what values)
- **Output**: Match rate (0.0–1.0), exact match flag, mismatch details
- **Key for**: Verifying the code performs the right operations

#### Timing Metric
- **What it measures**: Temporal dynamics (pacing, duration, interval consistency)
- **Components**: 
  - R² (linearity of timing relationship)
  - Slope (overall rate: 1.0 = perfect match)
  - Duration ratio
  - MAPE (Mean Absolute Percentage Error of intervals)
- **Key for**: Real-time systems, synchronized operations

#### Continuous Profile Metric
- **What it measures**: Fidelity of continuous processes (e.g., temperature ramps)
- **Components**:
  - MAE (Mean Absolute Error) across entire profile
  - Final value difference
- **Scoring**: Exponential decay: `0.7 * exp(-MAE/scale) + 0.3 * exp(-final_diff/scale)`
- **Key for**: Process control, gradual state changes

#### Structure Metric
- **What it measures**: Alignment quality (gaps = insertions/deletions)
- **Output**: Gap rate, structural similarity score
- **Key for**: Detecting missing or extra operations

### 5. Scoring and Decisions

**Full Score** (continuous, 0.0–1.0):
```
Without continuous channels: 0.8 * discrete + 0.2 * timing
With continuous channels:    0.6 * discrete + 0.2 * timing + 0.2 * continuous
```

**Binary Accuracy** (strict pass/fail):
- Requires exact discrete match
- Requires timing thresholds met (R² ≥ 0.9, slope ∈ [0.8, 1.2], etc.)
- Requires continuous thresholds met (if applicable)

**Custom scoring:**
```python
def custom_formula(metrics):
    return 0.5 * metrics["discrete"].score + \
           0.3 * metrics["timing"].score + \
           0.2 * metrics["structure"].score

evaluator = Evaluator(aggregator=WeightedAggregator(formula=custom_formula))
```

---

## Usage Examples

### Example 1: Basic Evaluation

```python
from envtrace.core import Event, Trace, Evaluator, EvaluateRequest
from envtrace.core import NumericToleranceComparator

# Ground truth: move motor, take two measurements
gt = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.10, 1),
    Event("det:Acquire", 1.10, 0),
])

# Predicted: same operations, slightly different timing
pred = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.12, 1),
    Event("det:Acquire", 1.08, 0),
])

evaluator = Evaluator.default()
result = evaluator.evaluate(EvaluateRequest(
    gt=gt,
    pred=pred,
    comparator=NumericToleranceComparator(tol=1e-3)
))

print(f"Full score: {result.full_score:.3f}")  # ~0.99
print(f"Accuracy: {result.accuracy}")          # True
```

### Example 2: Ignoring Channels

Some channels (e.g., parameter settings) shouldn't affect correctness:

```python
request = EvaluateRequest(
    gt=gt,
    pred=pred,
    ignore_channels={"det:AcquireTime"},  # Ignore exposure time settings
    comparator=NumericToleranceComparator(tol=1e-3)
)
```

### Example 3: Mixed Exact/Fuzzy Matching

```python
from envtrace.core import ExactEqualityComparator, NumericToleranceComparator

comparators = {
    "det:Acquire": ExactEqualityComparator(),      # Binary: must be exact
    "motor:x": NumericToleranceComparator(1e-2),   # Continuous: 1% tolerance
    "motor:y": NumericToleranceComparator(1e-2),
}

request = EvaluateRequest(
    gt=gt,
    pred=pred,
    comparator=NumericToleranceComparator(1e-3),  # Default
    comparators_by_channel=comparators,            # Per-channel overrides
)
```

### Example 4: Continuous Channels (Temperature Ramp)

```python
gt = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.10, 1),
    Event("stage:temp", 0.20, 20.0),
    Event("stage:temp", 0.40, 40.0),
    Event("stage:temp", 0.60, 60.0),
])

pred = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:Acquire", 0.12, 1),
    Event("stage:temp", 0.20, 20.0),
    Event("stage:temp", 0.40, 39.5),  # Slight deviation
    Event("stage:temp", 0.60, 59.8),
])

continuous_config = {
    "stage:temp": {
        "mae_scale": 15.0,      # Characteristic scale for MAE scoring
        "final_scale": 15.0,    # Characteristic scale for final value
        "mae_thresh": 5.0,      # Binary pass threshold for MAE
        "final_thresh": 5.0,    # Binary pass threshold for final value
        "weight": 1.0,
    }
}

request = EvaluateRequest(
    gt=gt,
    pred=pred,
    continuous_channels=continuous_config,
    ignore_channels={"stage:temp"},  # Exclude from discrete matching
)

result = evaluator.evaluate(request)
print(f"Continuous score: {result.metrics['continuous'].score:.3f}")
```

### Example 5: Multiple Ground Truths

When multiple valid implementations exist:

```python
gt_variants = [
    Trace([...]),  # Implementation 1
    Trace([...]),  # Implementation 2
    Trace([...]),  # Implementation 3
]

result, best_idx = evaluator.evaluate_best_of(
    gt_list=gt_variants,
    pred=pred,
    comparator=NumericToleranceComparator(tol=1e-3)
)

print(f"Best match: variant {best_idx} with score {result.full_score:.3f}")
```

---

## Configuration Files

For complex evaluations, use JSON or YAML configuration files:

### Example Configuration (JSON)

```json
{
  "aliases": {
    "XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr": "motor:x"
  },
  "ignore_channels": ["det:AcquireTime"],
  "aligner": {
    "name": "difflib"
  },
  "comparators": {
    "default": {
      "type": "numeric_tolerance",
      "tol": 0.001
    },
    "by_channel": {
      "det:Acquire": {"type": "exact"},
      "motor:x": {"type": "numeric_tolerance", "tol": 0.01}
    }
  },
  "timing": {
    "r2_thresh": 0.9,
    "slope_lo": 0.8,
    "slope_hi": 1.2,
    "dur_tol": 0.25,
    "mape_tol": 1.0
  },
  "continuous_channels": {
    "stage:temp": {
      "mae_scale": 15.0,
      "final_scale": 15.0,
      "mae_thresh": 5.0,
      "final_thresh": 5.0,
      "weight": 1.0
    }
  },
  "aggregation": {
    "weights": {
      "discrete": 0.6,
      "timing": 0.2,
      "continuous": 0.2
    }
  },
  "decision": {
    "require_discrete_exact": true,
    "timing_metric_name": "timing",
    "required_metrics": ["continuous"]
  },
  "include_structure": true
}
```

### Using Configuration Files

**Python:**
```python
from envtrace.io import load_config, build_from_config
from envtrace.io import load_trace

config = load_config("config.json")
gt = load_trace("traces/gt.json")
pred = load_trace("traces/pred.json")

evaluator, request = build_from_config(gt, pred, config)
result = evaluator.evaluate(request)
```

**CLI:**
```bash
envtrace evaluate --gt traces/gt.json --pred traces/pred.json \
                  --config config.json --out results/report.json
```

---

## CLI Reference

### `envtrace align`

Basic alignment and scoring without configuration file.

```bash
envtrace align --gt <ground_truth.json> --pred <predicted.json> [--out <report.json>]
```

**Options:**
- `--gt`: Path to ground truth trace (required)
- `--pred`: Path to predicted trace (required)
- `--out`: Path to save JSON report (optional; prints to stdout if omitted)

### `envtrace evaluate`

Full evaluation with configuration file.

```bash
envtrace evaluate --gt <gt.json> --pred <pred.json> --config <config.json> [--out <report.json>]
```

**Options:**
- `--gt`: Path to ground truth trace (required)
- `--pred`: Path to predicted trace (required)
- `--config`: Path to configuration file (JSON/YAML) (required)
- `--out`: Path to save JSON report (optional)

### `envtrace version`

Display EnvTrace version.

```bash
envtrace version
```

---

## Advanced Topics

### Custom Metrics

Implement your own metrics by following the `Metric` protocol:

```python
from envtrace.core.metrics import Metric, MetricResult

class CustomSafetyMetric:
    def evaluate(self, *, alignment, gt, pred):
        # Your custom logic here
        violations = 0
        for g, p in alignment:
            if p and p.channel == "emergency_stop" and p.value == 1:
                violations += 1
        
        score = 1.0 if violations == 0 else 0.0
        return MetricResult(
            name="safety",
            score=score,
            binary_pass=(violations == 0),
            details={"violations": violations}
        )

# Use in evaluation
request = EvaluateRequest(
    gt=gt,
    pred=pred,
    custom_metrics={"safety": CustomSafetyMetric()}
)
```

### Custom Continuous Metrics

Override the default continuous metric:

```python
class RMSEContinuousMetric:
    def evaluate(self, gt_series, pred_series, *, channel=None, config=None):
        import numpy as np
        
        gt_vals = np.array([v for _, v in gt_series])
        pr_vals = np.array([v for _, v in pred_series])
        
        # Pad to same length
        max_len = max(len(gt_vals), len(pr_vals))
        if len(gt_vals) < max_len:
            gt_vals = np.pad(gt_vals, (0, max_len - len(gt_vals)), mode="edge")
        if len(pr_vals) < max_len:
            pr_vals = np.pad(pr_vals, (0, max_len - len(pr_vals)), mode="edge")
        
        rmse = float(np.sqrt(np.mean((gt_vals - pr_vals) ** 2)))
        scale = config.get("rmse_scale", 10.0) if config else 10.0
        score = float(np.exp(-rmse / scale))
        
        return MetricResult(
            name="rmse_continuous",
            score=score,
            binary_pass=(rmse <= config.get("rmse_thresh", 5.0) if config else 5.0),
            details={"rmse": rmse}
        )

# Use in evaluation
request = EvaluateRequest(
    gt=gt,
    pred=pred,
    continuous_channels={"stage:temp": {"rmse_scale": 10.0, "rmse_thresh": 5.0}},
    continuous_metric=RMSEContinuousMetric()
)
```

### Custom Aggregation Formulas

```python
def custom_formula(metrics):
    d = metrics.get("discrete", MetricResult("discrete", 0.0)).score
    t = metrics.get("timing", MetricResult("timing", 0.0)).score
    s = metrics.get("safety", MetricResult("safety", 0.0)).score
    
    # Safety is critical: if it fails, overall score is 0
    if s < 1.0:
        return 0.0
    
    return 0.7 * d + 0.3 * t

evaluator = Evaluator(aggregator=WeightedAggregator(formula=custom_formula))
```

### Channel Aliasing

Map verbose channel names to shorter aliases:

```python
aliases = {
    "XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr": "motor:x",
    "XF:11BMB-ES{Det:PIL2M}:cam1:Acquire": "det:Acquire",
}

request = EvaluateRequest(
    gt=gt,
    pred=pred,
    alias_map=aliases
)
```

### Trace Preprocessing

```python
# Sort by time
trace = trace.sort_by_time()

# Zero timestamps (start from t=0)
trace = trace.zero_time()

# Filter by time window
trace = trace.filter_by_time_window(start=1.0, end=5.0)

# Filter by channels
trace = trace.filter_by_channels(["motor:x", "motor:y"])

# Deduplicate (coalesce rapid updates)
trace = trace.deduplicate(coalesce_window=0.01)  # 10ms window
```

---

## Worked Example: Mixed Exact/Fuzzy Matching + Continuous Channel + Custom Full Score

Scenario:
- You have a trace with many PV changes.
- For `det:Acquire`, you want exact value matching.
- For motor values (`motor:x`, `motor:y`), you allow fuzzy numeric tolerance.
- You want to exclude the temperature channel from discrete alignment, but still compute a continuous fidelity score for it.
- Finally, you want a custom full score that includes a structural dimension (gaps in alignment).

```python
from envtrace.core import (
    Event, Trace, Evaluator, EvaluateRequest,
    ExactEqualityComparator, NumericToleranceComparator,
    WeightedAggregator
)

# Build traces (abridged)
gt = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:AcquireTime", 0.05, 1.0),
    Event("det:Acquire", 0.10, 1),
    Event("det:Acquire", 1.10, 0),
    Event("stage:temp", 0.20, 20.0),
    Event("stage:temp", 0.40, 40.0),
    Event("motor:y", 1.20, -0.2),
])
pred = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:AcquireTime", 0.05, 1.0),
    Event("det:Acquire", 0.12, 1),
    Event("det:Acquire", 1.08, 0),
    Event("stage:temp", 0.20, 20.0),
    Event("stage:temp", 0.40, 39.5),
    Event("motor:y", 1.21, -0.2),
])

# Per-channel comparators: exact for Acquire, fuzzy for motors
comp_map = {
    "det:Acquire": ExactEqualityComparator(),
    "motor:x": NumericToleranceComparator(1e-2),
    "motor:y": NumericToleranceComparator(1e-2),
}

# Continuous channels: evaluate temperature outside discrete alignment
continuous_cfg = {
    "stage:temp": {
        "mae_scale": 15.0,      # characteristic scale for MAE decay
        "final_scale": 15.0,    # characteristic scale for final diff decay
        "mae_thresh": 5.0,      # binary pass threshold for MAE
        "final_thresh": 5.0,    # binary pass threshold for final value
        "weight": 1.0,          # weight in aggregate continuous scoring
    }
}

# Ignore AcquireTime in discrete matching (parameter channel, not a discrete action)
ignore_channels = {"det:AcquireTime"}

# Custom full score formula including structure dimension
def custom_full_score(metrics):
    disc = metrics.get("discrete")
    tim = metrics.get("timing")
    cont = metrics.get("continuous")
    struct = metrics.get("structure")
    # Fallbacks if absent
    disc_s = disc.score if disc else 0.0
    tim_s = tim.score if tim else 0.0
    cont_s = cont.score if cont else 0.0
    struct_s = struct.score if struct else 0.0
    # Custom policy:
    # 50% discrete + 20% timing + 20% continuous + 10% structure
    return 0.5 * disc_s + 0.2 * tim_s + 0.2 * cont_s + 0.1 * struct_s

# Build Evaluator with custom aggregator formula
ev = Evaluator(aggregator=WeightedAggregator(formula=custom_full_score))

req = EvaluateRequest(
    gt=gt,
    pred=pred,
    comparator=NumericToleranceComparator(1e-3),   # default comparator
    comparators_by_channel=comp_map,               # per-channel overrides
    ignore_channels=ignore_channels,               # excluded channels
    continuous_channels=continuous_cfg,            # series-based evaluation
    include_structure=True,                        # adds structural metric
)

res = ev.evaluate(req)

print("Custom full score:", res.full_score)
print("Binary accuracy:", res.accuracy)
print("Discrete score:", res.metrics["discrete"].score)
print("Timing score:", res.metrics["timing"].score)
print("Continuous score:", res.metrics["continuous"].score)
print("Structure score:", res.metrics["structure"].score)
```

Notes:
- Temperature channel is excluded from discrete alignment via `ignore_channels` and handled by the `continuous_channels` config.
- AcquireTime is excluded from discrete matching because it’s a parameter-setting PV rather than a discrete action.
- Per-channel comparators give fine-grained control over numeric tolerated matching versus exact toggles.

---

## Advanced Topics

- Aliasing:
  - Use `alias_map` in `EvaluateRequest` to canonicalize channel names.
- Multi-reference evaluation:
  - `Evaluator.evaluate_best_of(gt_list, pred, ...)` picks the ground-truth variant that yields the best score.
- Configuration (JSON/YAML):
  - Define aliases, ignore lists, comparators, timing thresholds, continuous channels, and aggregation weights in a single file.
  - Example JSON:
    ```json
    {
      "aliases": {"XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr": "motor:x"},
      "ignore_channels": ["det:AcquireTime"],
      "aligner": {"name": "difflib"},
      "comparators": {
        "default": {"type": "numeric_tolerance", "tol": 0.001},
        "by_channel": {
          "det:Acquire": {"type": "exact"},
          "motor:x": {"type": "numeric_tolerance", "tol": 0.01}
        }
      },
      "timing": {"r2_thresh": 0.9, "slope_lo": 0.8, "slope_hi": 1.2, "dur_tol": 0.25, "mape_tol": 1.0},
      "continuous_channels": {
        "stage:temp": {"mae_scale": 15.0, "final_scale": 15.0, "mae_thresh": 5.0, "final_thresh": 5.0, "weight": 1.0}
      },
      "aggregation": {"weights": {"discrete": 0.6, "timing": 0.2, "continuous": 0.2}},
      "decision": {
        "require_discrete_exact": true,
        "timing_metric_name": "timing",
        "required_metrics": ["continuous"]
      },
      "include_structure": true
    }
    ```
  - CLI:
    ```bash
    envtrace evaluate --gt envtrace/examples/traces/gt.json --pred envtrace/examples/traces/pred.json --config config.json --out results/report.json
    ```
  - Python:
    ```python
    from envtrace.io import load_config, build_from_config
    from envtrace.io import load_trace
    cfg = load_config("config.json")
    gt = load_trace("envtrace/examples/traces/gt.json")
    pred = load_trace("envtrace/examples/traces/pred.json")
    evaluator, req = build_from_config(gt, pred, cfg)
    result = evaluator.evaluate(req)
    print(result.full_score, result.accuracy)
    ```
- Custom metrics:
  - Implement your own class with `evaluate(...) -> MetricResult` and combine via a custom aggregator formula.
- Ignoring function-call parameters:
  - Use the `AlwaysTrueComparator` for channels that are simple markers (e.g., `sam.align()` logged as a call marker).

---

## API Summary

- Event, Trace: core data types.
- SequenceAligner (protocol), DifflibAligner (default).
- Comparators:
  - ValueComparator (protocol),
  - ExactEqualityComparator,
  - NumericToleranceComparator,
  - AlwaysTrueComparator,
  - get_comparator_for_channel(default, comparator_map, channel).
- Metrics:
  - MetricResult, Metric (protocol),
  - DiscreteMatchMetric,
  - TimingMetric,
  - ContinuousProfileMetric,
  - StructureMetric.
- Scoring:
  - WeightedAggregator(weights=None, use_continuous=None, formula=None),
  - BinaryDecision(require_discrete_exact=True, timing_metric_name="timing").
- Evaluator:
  - EvaluateRequest(gt, pred, key_fn, aligner, comparator, comparators_by_channel, ignore_channels, alias_map, use_continuous, continuous_channels, include_structure),
  - evaluate(req) -> EvaluateResult,
  - evaluate_best_of(gt_list, pred, **kwargs) -> (EvaluateResult, best_index).

If you have questions or want to contribute adapters (EPICS, Bluesky, etc.) or additional metrics, please open an issue or PR.

---

## Using Custom Metrics and Overriding the Continuous Metric

EnvTrace lets you inject your own metrics without modifying the core. There are two common ways:
1) Add extra metrics computed after alignment.
2) Override the per-channel continuous metric used for continuous channels (e.g., temperature).

A Metric must implement an `evaluate(...) -> MetricResult` method. You can accept keyword args like `alignment`, `gt`, `pred`, or, for continuous channels, `(gt_series, pred_series, channel=None, config=None)`.

Example: define a custom continuous metric (RMSE-based) and a safety metric that checks ordering constraints.

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from envtrace.core import (
    Event, Trace, Evaluator, EvaluateRequest,
    NumericToleranceComparator, WeightedAggregator,
)
from envtrace.core.metrics import Metric, MetricResult

# 1) Custom continuous metric (RMSE-based)
class RMSEContinuousMetric:
    def evaluate(
        self,
        gt_series: list[tuple[float, float]],
        pred_series: list[tuple[float, float]],
        *,
        channel: str | None = None,
        config: Dict[str, Any] | None = None,
    ) -> MetricResult:
        # Handle empty series
        if not gt_series and not pred_series:
            return MetricResult(name="rmse_cont", score=1.0, binary_pass=True, details={"reason": "both empty"})
        if not gt_series or not pred_series:
            return MetricResult(name="rmse_cont", score=0.0, binary_pass=False, details={"reason": "one empty"})

        gt_vals = np.array([float(v) for _, v in gt_series], dtype=float)
        pr_vals = np.array([float(v) for _, v in pred_series], dtype=float)
        # pad
        max_len = max(len(gt_vals), len(pr_vals))
        if len(gt_vals) < max_len:
            gt_vals = np.pad(gt_vals, (0, max_len - len(gt_vals)), mode="edge")
        elif len(pr_vals) < max_len:
            pr_vals = np.pad(pr_vals, (0, max_len - len(pr_vals)), mode="edge")

        rmse = float(np.sqrt(np.mean((gt_vals - pr_vals) ** 2)))
        # simple scoring: exp(-rmse / scale)
        scale = float((config or {}).get("rmse_scale", 10.0))
        score = float(np.exp(-rmse / scale))
        # binary pass threshold
        thresh = float((config or {}).get("rmse_thresh", 5.0))
        binary = rmse <= thresh
        return MetricResult(name="rmse_cont", score=score, binary_pass=binary, details={"rmse": rmse, "scale": scale})

# 2) Extra metric computed from alignment: penalize if any acquisition happens before first motor:x move
class OrderingSafetyMetric:
    def evaluate(self, *, alignment: List[tuple[Optional[Event], Optional[Event]]], gt: Trace, pred: Trace) -> MetricResult:
        # Build a simple "first occurrence" index map for each trace separately, then compare order
        def first_index(trace: Trace, channel: str) -> Optional[int]:
            for i, e in enumerate(trace.events):
                if e.channel == channel:
                    return i
            return None

        # if det:Acquire happens before motor:x in pred trace, penalize
        idx_acq = first_index(pred, "det:Acquire")
        idx_motor = first_index(pred, "motor:x")
        if idx_acq is not None and idx_motor is not None and idx_acq < idx_motor:
            # violation
            return MetricResult(name="ordering_safety", score=0.0, binary_pass=False, details={"violation": True})
        return MetricResult(name="ordering_safety", score=1.0, binary_pass=True, details={"violation": False})

# Build traces
gt = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:AcquireTime", 0.05, 1.0),
    Event("det:Acquire", 0.10, 1),
    Event("det:Acquire", 1.10, 0),
    Event("stage:temp", 0.20, 20.0),
    Event("stage:temp", 0.40, 40.0),
])
pred = Trace([
    Event("motor:x", 0.00, 0.0),
    Event("det:AcquireTime", 0.05, 1.0),
    Event("det:Acquire", 0.12, 1),
    Event("det:Acquire", 1.08, 0),
    Event("stage:temp", 0.20, 21.0),
    Event("stage:temp", 0.40, 39.0),
])

# Use custom continuous metric for stage:temp, and add the safety metric as an extra metric
continuous_cfg = {
    "stage:temp": {"rmse_scale": 10.0, "rmse_thresh": 5.0, "weight": 1.0}
}

def custom_agg(metrics: Dict[str, MetricResult]) -> float:
    # Combine discrete + timing + custom continuous + safety in a custom way
    d = metrics.get("discrete").score if "discrete" in metrics else 0.0
    t = metrics.get("timing").score if "timing" in metrics else 0.0
    c = metrics.get("continuous").score if "continuous" in metrics else 0.0
    s = metrics.get("ordering_safety").score if "ordering_safety" in metrics else 0.0
    return 0.5*d + 0.2*t + 0.2*c + 0.1*s

ev = Evaluator(aggregator=WeightedAggregator(formula=custom_agg))
req = EvaluateRequest(
    gt=gt,
    pred=pred,
    comparator=NumericToleranceComparator(1e-3),
    ignore_channels={"det:AcquireTime"},
    # 1) Override continuous metric per channel:
    continuous_channels=continuous_cfg,
    continuous_metric=RMSEContinuousMetric(),
    # 2) Add extra custom metrics:
    custom_metrics={"ordering_safety": OrderingSafetyMetric()},
    include_structure=False,   # optional
)
res = ev.evaluate(req)
print("Full score (custom agg):", res.full_score)
print("Custom continuous score:", res.metrics["continuous"].score)
print("Ordering safety score:", res.metrics["ordering_safety"].score)
```

Guidelines for custom metrics:
- For continuous channels override:
  - Implement `evaluate(gt_series, pred_series, channel=None, config=None, **kwargs)` and return `MetricResult`.
  - Supply it via `EvaluateRequest(continuous_metric=<your_metric>, continuous_channels={...})`.
- For extra metrics after alignment:
  - Implement `evaluate(alignment=..., gt=..., pred=...)` and add to `EvaluateRequest(custom_metrics={"name": YourMetric()})`.
- Use a custom aggregator formula to incorporate your metrics into the full score.
