"""Synthetic pipeline-fault generator.

Each sample is a realistic-looking failure log from a made-up data pipeline,
paired with the correct remediation as a structured action. Everything is
synthetic: pipeline, table and column names are invented, and nothing comes
from any employer or client.

The same seed always produces the same samples, so train/eval splits are
reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List

SCHEMA_DRIFT = "schema_drift"
NULL_SPIKE = "null_spike"
DUPLICATE_EVENTS = "duplicate_events"
LATE_DATA = "late_data"
BAD_TIMESTAMP = "bad_timestamp"
PARTITION_SKEW = "partition_skew"
FAULT_TYPES = (SCHEMA_DRIFT, NULL_SPIKE, DUPLICATE_EVENTS, LATE_DATA, BAD_TIMESTAMP, PARTITION_SKEW)

# The closed action vocabulary the agent is allowed to emit.
ADD_COLUMN_MAPPING = "add_column_mapping"
QUARANTINE_RECORDS = "quarantine_records"
DEDUPLICATE = "deduplicate"
REPLAY_WINDOW = "replay_window"
NORMALIZE_TIMESTAMPS = "normalize_timestamps"
REPARTITION = "repartition"
ACTIONS = (
    ADD_COLUMN_MAPPING,
    QUARANTINE_RECORDS,
    DEDUPLICATE,
    REPLAY_WINDOW,
    NORMALIZE_TIMESTAMPS,
    REPARTITION,
)

# Each fault type has exactly one correct action.
CORRECT_ACTION = {
    SCHEMA_DRIFT: ADD_COLUMN_MAPPING,
    NULL_SPIKE: QUARANTINE_RECORDS,
    DUPLICATE_EVENTS: DEDUPLICATE,
    LATE_DATA: REPLAY_WINDOW,
    BAD_TIMESTAMP: NORMALIZE_TIMESTAMPS,
    PARTITION_SKEW: REPARTITION,
}

PIPELINES = ("orders_ingest", "payments_stream", "clickstream_etl", "inventory_sync", "sensor_batch")
TABLES = ("orders", "payments", "events", "stock_levels", "readings")
COLUMNS = ("customer_id", "amount", "status", "region", "device_type", "sku", "channel")
BASE_TIME = datetime(2026, 1, 1)


@dataclass(frozen=True)
class FaultSample:
    """One failure log and its ground-truth remediation."""

    sample_id: str
    fault_type: str
    pipeline: str
    table: str
    log: str
    action: Dict[str, object] = field(default_factory=dict)

    def action_json(self) -> str:
        """The target output the model is trained to produce."""
        return json.dumps(self.action, sort_keys=True)


def _ts(rng: random.Random) -> datetime:
    return BASE_TIME + timedelta(seconds=rng.randint(0, 300 * 86400))


def _header(rng: random.Random, pipeline: str, table: str) -> List[str]:
    t = _ts(rng)
    run_id = f"run-{rng.randint(10000, 99999)}"
    return [
        f"{t:%Y-%m-%d %H:%M:%S} INFO  [{pipeline}] starting {run_id} target={table}",
        f"{t + timedelta(seconds=2):%Y-%m-%d %H:%M:%S} INFO  [{pipeline}] reading batch rows={rng.randint(5_000, 900_000)}",
    ]


def _schema_drift(rng, pipeline, table):
    old = rng.choice(COLUMNS)
    new = f"{old}_v{rng.randint(2, 4)}" if rng.random() < 0.5 else f"{old.upper()}"
    lines = [
        f"ERROR [{pipeline}] schema validation failed for {table}",
        f"ERROR [{pipeline}] expected column '{old}' not found; unexpected column '{new}' present",
        f"WARN  [{pipeline}] 100% of rows failed contract check",
    ]
    return lines, {"action": ADD_COLUMN_MAPPING, "params": {"source": new, "target": old, "table": table}}


def _null_spike(rng, pipeline, table):
    col = rng.choice(COLUMNS)
    pct = rng.randint(31, 96)
    lines = [
        f"WARN  [{pipeline}] null ratio for column '{col}' is {pct}% (threshold 5%)",
        f"ERROR [{pipeline}] data quality check not_null({col}) failed on {table}",
    ]
    return lines, {"action": QUARANTINE_RECORDS, "params": {"column": col, "condition": "is_null", "table": table}}


def _duplicate_events(rng, pipeline, table):
    dups = rng.randint(200, 50_000)
    key = rng.choice(("event_id", "order_id", "txn_id", "reading_id"))
    lines = [
        f"ERROR [{pipeline}] primary key violation on {table}: {dups} duplicate values for '{key}'",
        f"WARN  [{pipeline}] upstream retry detected, at-least-once delivery produced repeated records",
    ]
    return lines, {"action": DEDUPLICATE, "params": {"key": key, "table": table, "keep": "latest"}}


def _late_data(rng, pipeline, table):
    hours = rng.randint(3, 48)
    start = _ts(rng).replace(minute=0, second=0)
    end = start + timedelta(hours=rng.randint(1, 6))
    lines = [
        f"WARN  [{pipeline}] {rng.randint(1000, 90_000)} records arrived {hours}h after watermark",
        f"ERROR [{pipeline}] late events dropped for window {start:%Y-%m-%dT%H:%M} to {end:%Y-%m-%dT%H:%M}",
    ]
    return lines, {
        "action": REPLAY_WINDOW,
        "params": {"table": table, "start": f"{start:%Y-%m-%dT%H:%M}", "end": f"{end:%Y-%m-%dT%H:%M}"},
    }


def _bad_timestamp(rng, pipeline, table):
    col = rng.choice(("created_at", "event_time", "updated_at"))
    bad = rng.choice(("1970-01-01 00:00:00", "9999-12-31 23:59:59", "13/45/2026", "0000-00-00"))
    lines = [
        f"ERROR [{pipeline}] cannot parse timestamp '{bad}' in column '{col}'",
        f"WARN  [{pipeline}] mixed timestamp formats detected in {table}.{col}",
    ]
    return lines, {"action": NORMALIZE_TIMESTAMPS, "params": {"column": col, "table": table, "timezone": "UTC"}}


def _partition_skew(rng, pipeline, table):
    pct = rng.randint(70, 98)
    key = rng.choice(("region", "customer_id", "channel"))
    partitions = rng.choice((64, 128, 256))
    lines = [
        f"WARN  [{pipeline}] partition skew: one partition holds {pct}% of rows for key '{key}'",
        f"ERROR [{pipeline}] task timeout after 3600s on straggler partition in {table}",
        f"INFO  [{pipeline}] cluster advisor recommends {partitions} partitions for {table}",
    ]
    return lines, {"action": REPARTITION, "params": {"key": key, "table": table, "partitions": partitions}}


_BUILDERS: Dict[str, Callable] = {
    SCHEMA_DRIFT: _schema_drift,
    NULL_SPIKE: _null_spike,
    DUPLICATE_EVENTS: _duplicate_events,
    LATE_DATA: _late_data,
    BAD_TIMESTAMP: _bad_timestamp,
    PARTITION_SKEW: _partition_skew,
}


def generate_faults(n: int = 600, seed: int = 42) -> List[FaultSample]:
    """Return `n` reproducible fault samples, cycling evenly through fault types."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    samples: List[FaultSample] = []
    for i in range(n):
        fault = FAULT_TYPES[i % len(FAULT_TYPES)]
        pipeline = rng.choice(PIPELINES)
        table = rng.choice(TABLES)
        body, action = _BUILDERS[fault](rng, pipeline, table)
        log = "\n".join(_header(rng, pipeline, table) + body)
        samples.append(
            FaultSample(
                sample_id=f"FLT{i + 1:05d}",
                fault_type=fault,
                pipeline=pipeline,
                table=table,
                log=log,
                action=action,
            )
        )
    rng.shuffle(samples)
    return samples
