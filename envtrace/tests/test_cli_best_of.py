import json
import sys
from pathlib import Path

from envtrace import cli
from envtrace.core import Event, Trace
from envtrace.io.json_io import save_trace

PAPER_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "paper" / "paper_config.json"


def test_cli_evaluate_selects_best_reference(tmp_path: Path, monkeypatch):
    pred = Trace([Event("motor:x", 0.0, 1.0), Event("motor:x", 1.0, 2.0)])
    save_trace(tmp_path / "pred.json", pred)
    save_trace(tmp_path / "gt0.json", Trace([Event("motor:x", 0.0, 5.0)]))
    save_trace(tmp_path / "gt1.json", pred)
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "envtrace", "evaluate", "--gt", str(tmp_path / "gt0.json"), str(tmp_path / "gt1.json"),
        "--pred", str(tmp_path / "pred.json"), "--config", str(PAPER_CONFIG), "--out", str(out)])
    cli.main()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["selected_reference"]["index"] == 1
    assert rep["accuracy"] is True
    assert abs(rep["full_score"] - 1.0) < 1e-12
