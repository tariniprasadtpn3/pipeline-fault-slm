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


@pytest.mark.parametrize("n", [0, 10**12])
def test_repartition_refuses_out_of_range_counts_when_called_directly(n):
    with pytest.raises(ToolError, match="between 1 and"):
        T.repartition(ToyTable("events", rows=({"user": "u1"},)), "user", n)


WINDOW = ("2026-03-01T10:00", "2026-03-01T11:00")


def event(ts="2026-03-01T10:15:00", **cols):
    return {"id": 1, "event_ts": ts, **cols}


@pytest.mark.parametrize(
    "step",
    [
        lambda t: T.repartition(t, "id", 4),
        lambda t: T.normalize_timestamps(t, "event_ts", "UTC"),
        lambda t: T.quarantine_records(t, "amount", "is_null"),
    ],
    ids=["repartition", "normalize_timestamps", "quarantine_records"],
)
def test_replay_does_not_re_add_rows_after_another_tool(step):
    row = event(amount=None)
    after, _ = step(ToyTable("events", rows=(row,), source=(row,)))
    out, rep = T.replay_window(after, *WINDOW)
    assert rep.changed == 0 and out.rows == after.rows and out.quarantine == after.quarantine


def test_replay_adds_a_repeated_source_row_once():
    out, rep = T.replay_window(ToyTable("events", source=(event(), event())), *WINDOW)
    assert rep.changed == 1 and out.rows == (event(),)


def test_replay_keeps_rows_that_differ_only_in_fractional_seconds():
    first, second = event("2026-03-01T10:15:00.100000"), event("2026-03-01T10:15:00.900000")
    t = ToyTable("events", rows=(first,), source=(first, second))
    out, rep = T.replay_window(t, *WINDOW)
    assert rep.changed == 1 and out.rows == (first, second)


@pytest.mark.parametrize(
    "ts, added",
    [
        ("2026-03-01 10:00:00", 1),  # space separator at the inclusive start
        ("2026-03-01T11:00:00", 0),  # exclusive end
        ("2026-03-01T15:45:00+05:30", 1),  # 10:15 UTC
        ("2026-03-01T10:15:00+05:30", 0),  # 04:45 UTC
        ("03/01/2026 10:15", 1),
        ("garbage", 0),
        (None, 0),
    ],
)
def test_replay_window_is_compared_in_utc(ts, added):
    _, rep = T.replay_window(ToyTable("events", source=(event(ts),)), *WINDOW)
    assert rep.changed == added


def test_normalize_quarantines_a_timestamp_that_overflows_in_utc():
    t = ToyTable("events", rows=(event("0001-01-01T00:00:00+05:30"),))
    out, rep = T.normalize_timestamps(t, "event_ts", "UTC")
    assert out.rows == () and rep.extra["quarantined"] == 1


def test_normalize_zero_pads_early_years():
    t = ToyTable("events", rows=(event("0999-01-01 00:00:00"),))
    out, _ = T.normalize_timestamps(t, "event_ts", "UTC")
    assert out.rows[0]["event_ts"] == "0999-01-01T00:00:00Z"
