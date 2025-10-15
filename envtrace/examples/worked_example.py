from envtrace.core import (
    Event, Trace, Evaluator, EvaluateRequest,
    ExactEqualityComparator, NumericToleranceComparator,
    WeightedAggregator,
)
from envtrace.reporting import format_text_report

def main():
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
            "mae_scale": 15.0,
            "final_scale": 15.0,
            "mae_thresh": 5.0,
            "final_thresh": 5.0,
            "weight": 1.0,
        }
    }

    # Ignore AcquireTime in discrete matching
    ignore_channels = {"det:AcquireTime"}

    # Custom full score formula including structure dimension
    def custom_full_score(metrics):
        disc = metrics.get("discrete")
        tim = metrics.get("timing")
        cont = metrics.get("continuous")
        struct = metrics.get("structure")
        disc_s = disc.score if disc else 0.0
        tim_s = tim.score if tim else 0.0
        cont_s = cont.score if cont else 0.0
        struct_s = struct.score if struct else 0.0
        return 0.5 * disc_s + 0.2 * tim_s + 0.2 * cont_s + 0.1 * struct_s

    # Build Evaluator with custom aggregator formula
    ev = Evaluator(aggregator=WeightedAggregator(formula=custom_full_score))

    req = EvaluateRequest(
        gt=gt,
        pred=pred,
        comparator=NumericToleranceComparator(1e-3),
        comparators_by_channel=comp_map,
        ignore_channels=ignore_channels,
        continuous_channels=continuous_cfg,
        include_structure=True,
    )

    res = ev.evaluate(req)

    # Pretty text report with alignment and metric breakdown
    print(format_text_report(res, evaluator=ev, req=req, max_rows=25, title="EnvTrace Worked Example"))

    # Component scores (still printed for reference)
    print("Custom full score:", res.full_score)
    print("Binary accuracy:", res.accuracy)
    print("Discrete score:", res.metrics["discrete"].score)
    print("Timing score:", res.metrics["timing"].score)
    print("Continuous score:", res.metrics["continuous"].score)
    print("Structure score:", res.metrics["structure"].score)

if __name__ == "__main__":
    main()
