import json

import pytest

from src.agent import ACTION_PARAMS, MAX_PARTITIONS, parse_action, validate_action
from src.faults import ACTIONS, generate_faults


def test_schema_covers_exactly_the_action_vocabulary():
    assert set(ACTION_PARAMS) == set(ACTIONS)


def test_every_generated_ground_truth_action_is_valid():
    for s in generate_faults(600, seed=5):
        r = validate_action(s.action)
        assert r.ok, (s.sample_id, r.errors)
        assert r.action == s.action


def test_unknown_action_rejected():
    r = validate_action({"action": "drop_table", "params": {"table": "orders"}})
    assert not r.ok and "unknown action" in r.errors[0]


def test_missing_and_extra_params_rejected():
    r = validate_action({"action": "deduplicate", "params": {"key": "event_id", "table": "events", "rm": "x"}})
    assert not r.ok
    assert "missing param: keep" in r.errors and "unexpected param: rm" in r.errors


def test_bad_values_rejected():
    base = {"action": "repartition", "params": {"key": "region", "table": "orders", "partitions": 128}}
    assert validate_action(base).ok
    for bad in (0, -4, "128", True, 1.5):
        a = json.loads(json.dumps(base))
        a["params"]["partitions"] = bad
        assert not validate_action(a).ok, bad


def test_identifier_injection_rejected():
    a = {"action": "quarantine_records", "params": {"column": "x; DROP TABLE t", "condition": "is_null", "table": "orders"}}
    assert not validate_action(a).ok


def test_replay_window_must_be_ordered_and_well_formed():
    ok = {"action": "replay_window", "params": {"table": "orders", "start": "2026-01-01T00:00", "end": "2026-01-01T03:00"}}
    assert validate_action(ok).ok
    rev = {"action": "replay_window", "params": {"table": "orders", "start": "2026-01-01T03:00", "end": "2026-01-01T00:00"}}
    assert not validate_action(rev).ok
    bad = {"action": "replay_window", "params": {"table": "orders", "start": "yesterday", "end": "now"}}
    assert not validate_action(bad).ok


def test_non_object_and_extra_top_level_keys():
    assert not validate_action([1, 2]).ok
    assert not validate_action({"action": "deduplicate", "params": {}, "note": "hi"}).ok
    assert not validate_action({"action": "deduplicate", "params": "nope"}).ok


def test_parse_action_handles_json_fence_and_garbage():
    s = generate_faults(1, seed=1)[0]
    assert parse_action(s.action_json()).ok
    assert parse_action("```json\n" + s.action_json() + "\n```").ok
    r = parse_action("I think you should dedupe")
    assert not r.ok and r.errors[0].startswith("invalid JSON")
    assert not parse_action(None).ok


@pytest.mark.parametrize("text", ["", "null", "42", '"deduplicate"'])
def test_parse_action_rejects_non_action_json(text):
    assert not parse_action(text).ok


@pytest.mark.parametrize(
    "text",
    [
        '{"action": [], "params": {}}',
        '{"action": {"name": "deduplicate"}, "params": {}}',
        "[" * 100_000,
        '{"action": "repartition", "params": ' + "9" * 10_000 + "}",
    ],
    ids=["list-action", "dict-action", "deep-nesting", "oversized-int"],
)
def test_malformed_model_output_is_rejected_not_raised(text):
    r = parse_action(text)
    assert not r.ok and r.errors


@pytest.mark.parametrize("key", ["event_id\n", "event_id ", "évent_id", ""])
def test_identifiers_must_match_in_full(key):
    a = {"action": "deduplicate", "params": {"key": key, "table": "events", "keep": "latest"}}
    assert validate_action(a).errors == [f"invalid value for key: {key!r}"]


@pytest.mark.parametrize(
    "start",
    [
        "2026-02-30T00:00",
        "2026-13-01T00:00",
        "2026-01-01T24:00",
        "2026-01-01T00:60",
        "2026-01-01T00:00\n",
        "\u0662\u0660\u0662\u0666-01-01T00:00",  # non-ASCII digits
    ],
)
def test_replay_window_rejects_times_that_do_not_exist(start):
    a = {"action": "replay_window", "params": {"table": "orders", "start": start, "end": "2026-12-31T00:00"}}
    assert validate_action(a).errors == [f"invalid value for start: {start!r}"]


def test_partition_count_is_bounded():
    def repartition(n):
        return {"action": "repartition", "params": {"key": "region", "table": "orders", "partitions": n}}

    assert validate_action(repartition(1)).ok
    assert validate_action(repartition(MAX_PARTITIONS)).ok
    assert not validate_action(repartition(MAX_PARTITIONS + 1)).ok
    assert not validate_action(repartition(10**12)).ok
