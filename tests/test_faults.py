import json

import pytest

from src.faults import ACTIONS, CORRECT_ACTION, FAULT_TYPES, generate_faults


def test_same_seed_is_reproducible():
    assert generate_faults(60, seed=1) == generate_faults(60, seed=1)


def test_different_seed_differs():
    assert generate_faults(60, seed=1) != generate_faults(60, seed=2)


def test_all_fault_types_are_covered_evenly():
    samples = generate_faults(60)
    counts = {f: 0 for f in FAULT_TYPES}
    for s in samples:
        counts[s.fault_type] += 1
    assert set(counts.values()) == {10}


def test_sample_ids_are_unique():
    samples = generate_faults(120)
    assert len({s.sample_id for s in samples}) == len(samples)


def test_action_matches_fault_type_and_vocabulary():
    for s in generate_faults(120):
        assert s.action["action"] in ACTIONS
        assert s.action["action"] == CORRECT_ACTION[s.fault_type]
        assert s.action["params"]["table"] == s.table


def test_action_json_round_trips_and_is_stable():
    for s in generate_faults(30):
        assert json.loads(s.action_json()) == s.action
        assert s.action_json() == json.dumps(s.action, sort_keys=True)


def test_log_mentions_pipeline_and_table_and_has_error_line():
    for s in generate_faults(60):
        assert f"[{s.pipeline}]" in s.log
        assert s.table in s.log
        assert "ERROR" in s.log or "WARN" in s.log


def test_each_fault_log_contains_its_signal():
    signals = {
        "schema_drift": "schema validation failed",
        "null_spike": "null ratio",
        "duplicate_events": "duplicate values",
        "late_data": "after watermark",
        "bad_timestamp": "cannot parse timestamp",
        "partition_skew": "partition skew",
    }
    for s in generate_faults(120):
        assert signals[s.fault_type] in s.log


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        generate_faults(0)
