import json
from pathlib import Path
from envtrace.io import load_config, build_from_config
from envtrace.io.json_io import save_trace
from envtrace.core import Event, Trace

def test_build_from_json_config(tmp_path: Path):
    # Prepare traces
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
        Event("stage:temp", 0.20, 20.0),
        Event("stage:temp", 0.40, 39.5),
    ])

    # Save traces to files (CLI-like flow)
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    save_trace(gt_path, gt)
    save_trace(pred_path, pred)

    cfg = {
        "aliases": {"XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr": "motor:x"},
        "ignore_channels": ["det:AcquireTime"],
        "aligner": {"name": "difflib"},
        "comparators": {
            "default": {"type": "numeric_tolerance", "tol": 0.001},
            "by_channel": {"det:Acquire": {"type": "exact"}}
        },
        "timing": {"r2_thresh": 0.9, "slope_lo": 0.8, "slope_hi": 1.2, "dur_tol": 0.25, "mape_tol": 1.0},
        "continuous_channels": {
            "stage:temp": {"mae_scale": 15.0, "final_scale": 15.0, "mae_thresh": 5.0, "final_thresh": 5.0, "weight": 1.0}
        },
        "aggregation": {"weights": {"discrete": 0.6, "timing": 0.2, "continuous": 0.2}},
        "decision": {
            "require_discrete_exact": True,
            "timing_metric_name": "timing",
            "required_metrics": ["continuous"]
        },
        "include_structure": True
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)
    evaluator, req = build_from_config(gt, pred, loaded)
    assert req.ignore_channels == set(["det:AcquireTime"])
    assert req.continuous_channels is not None and "stage:temp" in req.continuous_channels
    res = evaluator.evaluate(req)
    # Expect presence of continuous metric and overall healthy score
    assert "continuous" in res.metrics
    assert res.metrics["discrete"].binary_pass is True
    assert res.metrics["timing"].binary_pass is True
    assert res.metrics["continuous"].binary_pass is True
