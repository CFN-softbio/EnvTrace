import json
from pathlib import Path
from envtrace.core.event import Event, Trace
from envtrace.io.json_io import save_trace, load_trace

def test_trace_json_roundtrip(tmp_path: Path):
    events = [
        Event("motor:x", 0.00, 0.0, {}),
        Event("det:Acquire", 0.10, 1, {}),
        Event("det:Acquire", 1.10, 0, {}),
    ]
    t = Trace(events)
    out = tmp_path / "trace.json"
    save_trace(out, t)

    loaded = load_trace(out)
    assert len(loaded.events) == len(events)
    for e_loaded, e_orig in zip(loaded.events, events):
        assert e_loaded.channel == e_orig.channel
        assert e_loaded.timestamp == e_orig.timestamp
        assert e_loaded.value == e_orig.value
        assert e_loaded.meta == e_orig.meta
