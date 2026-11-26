"""Demo logic: sample fault -> toy table -> diagnosis -> guarded agent step.

The toy table is built from the sample's ground-truth fault, never from the
prediction, so a wrong prediction cannot make up a table that suits it.
"""

from __future__ import annotations

from typing import Any, List

from src.agent import AgentLoop, RunResult, ToyTable, validate_action
from src.baseline import diagnose
from src.faults import FAULT_TYPES, FaultSample, generate_faults
from src.faults import generator as faults
from src.inference import InferenceResult


class RuleModel:
    """The rule baseline behind the same `predict(log)` interface as a model."""

    def predict(self, log: str) -> InferenceResult:
        action = diagnose(log)
        if action is None:
            return InferenceResult(ok=False, errors=["no rule matched this log"], attempts=1)
        res = validate_action(action)
        return InferenceResult(ok=res.ok, action=res.action, errors=res.errors, attempts=1)


def sample_for(fault_type: str, seed: int = 42) -> FaultSample:
    """One reproducible sample of the given fault type."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unknown fault type: {fault_type}")
    samples = generate_faults(len(FAULT_TYPES), seed=seed)
    return next(s for s in samples if s.fault_type == fault_type)


def toy_table(sample: FaultSample) -> ToyTable:
    """A few invented rows that exhibit the sample's fault."""
    p: Any = sample.action["params"]
    fault = sample.fault_type
    source: List[dict] = []
    if fault == faults.SCHEMA_DRIFT:
        rows = [{"id": i, p["source"]: f"value-{i}"} for i in range(1, 5)]
    elif fault == faults.NULL_SPIKE:
        rows = [{"id": i, p["column"]: None if i % 2 else f"value-{i}"} for i in range(1, 7)]
    elif fault == faults.DUPLICATE_EVENTS:
        rows = [{p["key"]: k, "version": v} for k, v in ((1, 1), (2, 1), (1, 2), (3, 1), (2, 2))]
    elif fault == faults.LATE_DATA:
        # One on-time row; two rows inside the window exist only upstream.
        on_time = {"id": 1, "event_ts": "2025-12-31T23:00:00"}
        rows = [on_time]
        source = [on_time, {"id": 2, "event_ts": f"{p['start']}:00"}, {"id": 3, "event_ts": f"{p['start']}:30"}]
    elif fault == faults.BAD_TIMESTAMP:
        values = ("2026-03-01 10:00:00", "03/01/2026 11:15", "13/45/2026")
        rows = [{"id": i, p["column"]: v} for i, v in enumerate(values, 1)]
    else:  # partition skew: one key value dominates
        keys = ["hot"] * 8 + ["cold-a", "cold-b"]
        rows = [{"id": i, p["key"]: k} for i, k in enumerate(keys, 1)]
    return ToyTable(sample.table, rows=tuple(rows), source=tuple(source))


def run_sample(sample: FaultSample, model: Any, apply: bool = False) -> RunResult:
    """One guarded agent step on the sample's toy table (dry-run unless `apply`)."""
    loop = AgentLoop(model, dry_run=not apply, max_steps=1)
    return loop.run(toy_table(sample), sample.log)


def diagnose_log(log: str, model: Any) -> InferenceResult:
    """Diagnose a pasted log. Nothing is executed: there is no table for it."""
    return model.predict(log)
