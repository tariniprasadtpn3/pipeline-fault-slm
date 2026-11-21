"""Rule-based baseline diagnoser.

Reads a pipeline failure log with keyword and regex rules and returns a
structured action, with no model involved. It is the bar a fine-tuned small
language model has to clear: if regexes already solve the task, a model adds
nothing.

`diagnose` returns None when no rule recognises the log, so callers can count
"no answer" separately from "wrong answer".
"""

from __future__ import annotations

import re
from typing import Dict, Optional

Action = Dict[str, object]

_TARGET = re.compile(r"target=(\w+)")

_SCHEMA = re.compile(
    r"expected column '(?P<target>\w+)' not found; unexpected column '(?P<source>\w+)' present"
)
_NULL = re.compile(r"null ratio for column '(?P<column>\w+)'")
_DUPLICATE = re.compile(r"duplicate values for '(?P<key>\w+)'")
_LATE = re.compile(
    r"late events dropped for window (?P<start>\d{4}-\d\d-\d\dT\d\d:\d\d) to (?P<end>\d{4}-\d\d-\d\dT\d\d:\d\d)"
)
_TIMESTAMP = re.compile(r"cannot parse timestamp '[^']*' in column '(?P<column>\w+)'")
_SKEW = re.compile(r"partition skew: .* for key '(?P<key>\w+)'")
_PARTITIONS = re.compile(r"recommends (?P<n>\d+) partitions")


def extract_table(log: str) -> Optional[str]:
    """The target table named in the run header, if present."""
    m = _TARGET.search(log)
    return m.group(1) if m else None


def diagnose(log: str) -> Optional[Action]:
    """Return the action for a failure log, or None if no rule matches."""
    table = extract_table(log)
    if table is None:
        return None

    m = _SCHEMA.search(log)
    if m:
        return {
            "action": "add_column_mapping",
            "params": {"source": m["source"], "target": m["target"], "table": table},
        }

    m = _NULL.search(log)
    if m:
        return {
            "action": "quarantine_records",
            "params": {"column": m["column"], "condition": "is_null", "table": table},
        }

    m = _DUPLICATE.search(log)
    if m:
        return {
            "action": "deduplicate",
            "params": {"key": m["key"], "table": table, "keep": "latest"},
        }

    m = _LATE.search(log)
    if m:
        return {
            "action": "replay_window",
            "params": {"table": table, "start": m["start"], "end": m["end"]},
        }

    m = _TIMESTAMP.search(log)
    if m:
        return {
            "action": "normalize_timestamps",
            "params": {"column": m["column"], "table": table, "timezone": "UTC"},
        }

    m = _SKEW.search(log)
    if m:
        n = _PARTITIONS.search(log)
        return {
            "action": "repartition",
            "params": {
                "key": m["key"],
                "table": table,
                "partitions": int(n["n"]) if n else 128,
            },
        }

    return None
