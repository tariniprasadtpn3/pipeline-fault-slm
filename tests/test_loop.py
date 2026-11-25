import json
from datetime import datetime, timezone

import pytest

from src.agent import AgentLoop, ToyTable
from src.inference import ActionModel, MockModel

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEDUPE = {"action": "deduplicate", "params": {"key": "event_id", "table": "orders", "keep": "latest"}}
QUAR = {"action": "quarantine_records", "params": {"column": "amount", "condition": "is_null", "table": "orders"}}


def _table():
    rows = ({"event_id": 1, "amount": 5}, {"event_id": 1, "amount": 5}, {"event_id": 2, "amount": None})
    return ToyTable("orders", rows=rows)


def _loop(replies, **kw):
    model = ActionModel(MockModel([json.dumps(r) if isinstance(r, dict) else r for r in replies]), max_retries=0)
    return AgentLoop(model, clock=lambda: NOW, **kw)


def test_single_step_applies_and_audits(tmp_path):
    path = tmp_path / "audit.jsonl"
    res = _loop([DEDUPE], audit_path=str(path)).run(_table(), "dups")
    assert res.stopped == "done" and len(res.table.rows) == 2
    assert res.audit[0]["status"] == "applied" and res.audit[0]["report"]["changed"] == 1
    assert json.loads(path.read_text().splitlines()[0]) == res.audit[0]
    assert res.audit[0]["ts"] == "2026-01-01T00:00:00Z"


def test_dry_run_leaves_table_untouched():
    t = _table()
    res = _loop([DEDUPE], dry_run=True).run(t, "dups")
    assert res.table is t and res.audit[0]["status"] == "dry_run"
    assert res.audit[0]["report"]["changed"] == 1


def test_allow_list_blocks_action():
    res = _loop([DEDUPE], allowed_actions=["quarantine_records"]).run(_table(), "dups")
    assert res.stopped == "blocked" and len(res.table.rows) == 3
    assert res.audit[0]["status"] == "blocked"


def test_invalid_output_stops_without_touching_table():
    res = _loop(['{"action": "drop_table", "params": {}}']).run(_table(), "x")
    assert res.stopped == "invalid_output" and len(res.table.rows) == 3
    assert res.audit[0]["errors"]


def test_tool_error_is_logged_not_raised():
    wrong = {"action": "deduplicate", "params": {"key": "event_id", "table": "other", "keep": "latest"}}
    res = _loop([wrong]).run(_table(), "x")
    assert res.stopped == "tool_error" and "targets" in res.audit[0]["error"]


def test_multi_step_until_healthy_then_done():
    logs = iter(["nulls", None])
    res = _loop([DEDUPE, QUAR]).run(_table(), "dups", next_log=lambda t: next(logs))
    assert res.stopped == "done" and res.steps == 2
    assert [r["event_id"] for r in res.table.rows] == [1] and len(res.table.quarantine) == 1


def test_max_steps_caps_model_calls():
    res = _loop([DEDUPE, QUAR, DEDUPE], max_steps=2).run(_table(), "a", next_log=lambda t: "again")
    assert res.stopped == "max_steps" and res.steps == 2


def test_repeat_guard_stops_identical_consecutive_action():
    res = _loop([DEDUPE, DEDUPE], max_steps=5).run(_table(), "a", next_log=lambda t: "still broken")
    assert res.stopped == "repeat" and res.audit[-1]["status"] == "repeat"
    assert sum(e["status"] == "applied" for e in res.audit) == 1


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        _loop([], allowed_actions=["drop_table"])
    with pytest.raises(ValueError):
        _loop([], max_steps=0)
