from envtrace.core.event import Event, Trace
from envtrace.core.evaluator import Evaluator, EvaluateRequest

def test_evaluator_simple_full_match():
    gt = Trace([
        Event("motor:x", 0.00, 0.0, {}),
        Event("det:Acquire", 0.10, 1, {}),
        Event("det:Acquire", 1.10, 0, {}),
    ])
    pred = Trace([
        Event("motor:x", 0.00, 0.0, {}),
        Event("det:Acquire", 0.10, 1, {}),
        Event("det:Acquire", 1.10, 0, {}),
    ])
    ev = Evaluator.default()
    res = ev.evaluate(EvaluateRequest(gt=gt, pred=pred))
    assert res.accuracy is True
    assert abs(res.full_score - 1.0) < 1e-9
    assert res.metrics["discrete"].score == 1.0
    assert res.metrics["timing"].binary_pass is True
