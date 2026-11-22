"""Dataset builder: synthetic faults -> train/val/test instruction-tuning JSONL.

Each record carries the prompt, the target completion (a JSON action) and a
chat-style `messages` list, so it can feed both prompt/completion and chat
trainers (for example trl's SFTTrainer). Splits are disjoint by sample id and
stratified by fault type, so every fault type appears in every split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.faults import ACTIONS, FAULT_TYPES, FaultSample, generate_faults

SYSTEM_PROMPT = (
    "You are a data-pipeline reliability assistant. Read the failure log and "
    "reply with exactly one JSON object of the form "
    '{"action": <name>, "params": {...}}. '
    f"Allowed actions: {', '.join(ACTIONS)}."
)

DEFAULT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "val", "test")


def build_prompt(log: str) -> str:
    """The user-turn text shown to the model for one failure log."""
    return f"Failure log:\n{log}\n\nReturn the remediation action as JSON."


def to_record(sample: FaultSample) -> Dict[str, object]:
    """One JSONL record for a sample."""
    prompt = build_prompt(sample.log)
    completion = sample.action_json()
    return {
        "id": sample.sample_id,
        "fault_type": sample.fault_type,
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
    }


def split_samples(
    samples: Sequence[FaultSample], ratios: Tuple[float, float, float] = DEFAULT_RATIOS
) -> Dict[str, List[FaultSample]]:
    """Stratified, deterministic split of samples into train/val/test."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be three positive numbers summing to 1")
    out: Dict[str, List[FaultSample]] = {name: [] for name in SPLIT_NAMES}
    for fault in FAULT_TYPES:
        group = sorted((s for s in samples if s.fault_type == fault), key=lambda s: s.sample_id)
        n = len(group)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        n_train = n - n_val - n_test
        out["train"] += group[:n_train]
        out["val"] += group[n_train : n_train + n_val]
        out["test"] += group[n_train + n_val :]
    for name in SPLIT_NAMES:
        out[name].sort(key=lambda s: s.sample_id)
    return out


def build_dataset(
    n: int = 600, seed: int = 42, ratios: Tuple[float, float, float] = DEFAULT_RATIOS
) -> Dict[str, List[Dict[str, object]]]:
    """Generate `n` faults and return {split: [record, ...]}."""
    splits = split_samples(generate_faults(n, seed=seed), ratios)
    return {name: [to_record(s) for s in items] for name, items in splits.items()}


def write_jsonl(records: Sequence[Dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_splits(out_dir: Path, n: int = 600, seed: int = 42) -> Dict[str, Path]:
    """Write train.jsonl, val.jsonl and test.jsonl into `out_dir`."""
    out_dir = Path(out_dir)
    paths = {}
    for name, records in build_dataset(n, seed).items():
        paths[name] = out_dir / f"{name}.jsonl"
        write_jsonl(records, paths[name])
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name, p in write_splits(Path(args.out), args.n, args.seed).items():
        print(f"wrote {p}")
