from __future__ import annotations
import json
from typing import Any, Dict
from pathlib import Path
from envtrace.core.event import Trace

def save_trace(path: str | Path, trace: Trace) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

def load_trace(path: str | Path) -> Trace:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return Trace.from_dict(data)
    if isinstance(data, list):
        return Trace.from_dict({"events": data, "schema_version": "1"})
    raise ValueError("Unrecognized trace JSON format")

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
