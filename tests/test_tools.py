import pytest

from src.agent import ToolError, ToyTable, apply_action
from src.agent import tools as T


def act(name, **params):
    return {"action": name, "params": params}


def test_add_column_mapping_renames_and_does_not_mutate():
    t = ToyTable("orders", rows=({"id": 1, "cust": "a"},))
    out, rep = apply_action(t, act("add_column_mapping", source="cust", target="customer_id", table="orders"))
    assert out.rows == ({"id": 1, "customer_id": "a"},)
    assert rep.changed == 1 and t.rows == ({"id": 1, "cust": "a"},)


def test_add_column_mapping_refuses_clobber_and_missing_source():
    t = ToyTable("orders", rows=({"cust": 1, "customer_id": 2},))
    with pytest.raises(ToolError):
        T.add_column_mapping(t, "cust", "customer_id")
    with pytest.raises(ToolError):
        T.add_column_mapping(t, "nope", "x")


def test_quarantine_moves_null_rows():
    t = ToyTable("orders", rows=({"id": 1, "amount": None}, {"id": 2, "amount": 5}))
    out, rep = apply_action(t, act("quarantine_records", column="amount", condition="is_null", table="orders"))
    assert [r["id"] for r in out.rows] == [2] and [r["id"] for r in out.quarantine] == [1]
    assert rep.changed == 1


def test_deduplicate_latest_and_earliest():
    rows = ({"k": 1, "v": "old"}, {"k": 2, "v": "x"}, {"k": 1, "v": "new"})
    t = ToyTable("events", rows=rows)
    latest, rep = T.deduplicate(t, "k", "latest")
    assert [r["v"] for r in latest.rows] == ["x", "new"] and rep.changed == 1
    earliest, _ = T.deduplicate(t, "k", "earliest")
    assert [r["v"] for r in earliest.rows] == ["old", "x"]


def test_replay_window_is_idempotent_and_bounded():
    src = tuple({"id": i, "event_ts": f"2026-03-01T10:{i:02d}:00"} for i in range(10))
    t = ToyTable("events", rows=(src[0],), source=src)
    out, rep = T.replay_window(t, "2026-03-01T10:00", "2026-03-01T10:05")
    assert [r["id"] for r in out.rows] == [0, 1, 2, 3, 4] and rep.changed == 4
    again, rep2 = T.replay_window(out, "2026-03-01T10:00", "2026-03-01T10:05")
    assert rep2.changed == 0 and again.rows == out.rows


def test_normalize_timestamps_mixed_formats_and_quarantine():
    rows = (
        {"id": 1, "ts": "2026-03-01T10:00:00+02:00"},
        {"id": 2, "ts": 1772359200},
        {"id": 3, "ts": "03/01/2026 10:00"},
        {"id": 4, "ts": "garbage"},
        {"id": 5, "ts": "2026-03-01T08:00:00Z"},
    )
    out, rep = T.normalize_timestamps(ToyTable("orders", rows=rows), "ts", "UTC")
    got = {r["id"]: r["ts"] for r in out.rows}
    assert got[1] == "2026-03-01T08:00:00Z" and got[3] == "2026-03-01T10:00:00Z"
    assert got[5] == "2026-03-01T08:00:00Z" and 4 not in got
    assert [r["id"] for r in out.quarantine] == [4] and rep.extra["quarantined"] == 1


def test_repartition_is_deterministic_and_spreads():
    rows = tuple({"user": f"u{i}"} for i in range(200))
    t = ToyTable("events", rows=rows)
    a, rep = T.repartition(t, "user", 4)
    b, _ = T.repartition(t, "user", 4)
    assert a.rows == b.rows and a.partitions == 4
    assert sum(rep.extra["sizes"]) == 200 and min(rep.extra["sizes"]) > 20
    assert "_partition" not in a.columns


def test_apply_action_rejects_invalid_and_wrong_table():
    t = ToyTable("orders", rows=({"a": 1},))
    with pytest.raises(ToolError, match="invalid action"):
        apply_action(t, {"action": "drop_table", "params": {}})
    with pytest.raises(ToolError, match="targets"):
        apply_action(t, act("repartition", key="a", table="payments", partitions=2))
