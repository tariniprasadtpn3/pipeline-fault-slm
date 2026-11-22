import json

import pytest

from src.dataset import build_dataset, build_prompt, split_samples, to_record, write_splits
from src.faults import ACTIONS, FAULT_TYPES, generate_faults


def test_splits_are_disjoint_and_cover_everything():
    ds = build_dataset(300, seed=1)
    ids = [r["id"] for name in ds for r in ds[name]]
    assert len(ids) == len(set(ids)) == 300


def test_split_sizes_follow_ratios():
    ds = build_dataset(600, seed=1)
    assert (len(ds["train"]), len(ds["val"]), len(ds["test"])) == (480, 60, 60)


def test_every_fault_type_in_every_split():
    ds = build_dataset(300, seed=3)
    for name in ds:
        assert {r["fault_type"] for r in ds[name]} == set(FAULT_TYPES)


def test_build_is_deterministic():
    assert build_dataset(120, seed=9) == build_dataset(120, seed=9)


def test_record_completion_is_valid_json_with_allowed_action():
    for r in build_dataset(60, seed=2)["train"]:
        action = json.loads(r["completion"])
        assert action["action"] in ACTIONS
        assert isinstance(action["params"], dict)


def test_record_messages_match_prompt_and_completion():
    sample = generate_faults(1, seed=5)[0]
    r = to_record(sample)
    assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
    assert r["messages"][1]["content"] == r["prompt"] == build_prompt(sample.log)
    assert r["messages"][2]["content"] == r["completion"]
    assert sample.log in r["prompt"]


def test_bad_ratios_rejected():
    samples = generate_faults(30)
    with pytest.raises(ValueError):
        split_samples(samples, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        split_samples(samples, (1.0, 0.0, 0.0))


def test_write_splits_roundtrip(tmp_path):
    paths = write_splits(tmp_path / "out", n=60, seed=4)
    assert set(paths) == {"train", "val", "test"}
    rows = [json.loads(line) for line in paths["val"].read_text().splitlines()]
    assert rows and all("prompt" in r and "completion" in r for r in rows)
    assert len(rows) == len(build_dataset(60, seed=4)["val"])
