"""Safe tools on a toy in-memory table.

Each of the six allowed actions maps to one tool. Tools never mutate the table
they are given: they return a new ``ToyTable`` plus a ``ToolReport`` describing
what changed, so a caller can dry-run, diff or audit any step. Actions are
validated with the action schema before dispatch, and an action aimed at a
different table than the one supplied is refused.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schema import validate_action

Row = Dict[str, Any]
TS_COLUMN = "event_ts"  # replay_window filters on this column (minute-resolution ISO)


class ToolError(ValueError):
    """Raised when an action cannot be applied safely."""


@dataclass(frozen=True)
class ToyTable:
    name: str
    rows: Tuple[Row, ...] = ()
    # Upstream rows available for replay (a stand-in for a source system/log).
    source: Tuple[Row, ...] = ()
    quarantine: Tuple[Row, ...] = ()
    partitions: int = 1

    @property
    def columns(self) -> List[str]:
        seen: List[str] = []
        for r in self.rows:
            for c in r:
                if c not in seen and c != "_partition":
                    seen.append(c)
        return seen


@dataclass(frozen=True)
class ToolReport:
    action: str
    changed: int
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _copy(rows) -> List[Row]:
    return [dict(r) for r in rows]


def add_column_mapping(t: ToyTable, source: str, target: str) -> Tuple[ToyTable, ToolReport]:
    if any(target in r for r in t.rows):
        raise ToolError(f"target column already exists: {target}")
    if not any(source in r for r in t.rows):
        raise ToolError(f"source column not found: {source}")
    rows = _copy(t.rows)
    n = 0
    for r in rows:
        if source in r:
            r[target] = r.pop(source)
            n += 1
    return replace(t, rows=tuple(rows)), ToolReport("add_column_mapping", n, f"{source} -> {target}")


def quarantine_records(t: ToyTable, column: str, condition: str) -> Tuple[ToyTable, ToolReport]:
    if condition != "is_null":
        raise ToolError(f"unsupported condition: {condition}")
    keep, moved = [], []
    for r in t.rows:
        (moved if r.get(column) is None else keep).append(dict(r))
    return (
        replace(t, rows=tuple(keep), quarantine=t.quarantine + tuple(moved)),
        ToolReport("quarantine_records", len(moved), f"{column} is null"),
    )


def deduplicate(t: ToyTable, key: str, keep: str) -> Tuple[ToyTable, ToolReport]:
    if keep not in ("latest", "earliest"):
        raise ToolError(f"unsupported keep: {keep}")
    order = list(t.rows) if keep == "earliest" else list(reversed(t.rows))
    seen, kept = set(), []
    for r in order:
        k = r.get(key)
        if k is not None and k in seen:
            continue
        seen.add(k)
        kept.append(dict(r))
    if keep == "latest":
        kept.reverse()
    return replace(t, rows=tuple(kept)), ToolReport("deduplicate", len(t.rows) - len(kept), f"key={key} keep={keep}")


def replay_window(t: ToyTable, start: str, end: str) -> Tuple[ToyTable, ToolReport]:
    """Re-ingest source rows with start <= event_ts < end that are not already present."""
    if start >= end:
        raise ToolError("replay window start must be before end")
    present = {tuple(sorted(r.items())) for r in t.rows}
    added = []
    for r in t.source:
        ts = r.get(TS_COLUMN)
        if isinstance(ts, str) and start <= ts[:16] < end and tuple(sorted(r.items())) not in present:
            added.append(dict(r))
    return (
        replace(t, rows=t.rows + tuple(added)),
        ToolReport("replay_window", len(added), f"{start}..{end}"),
    )


_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%d-%b-%Y %H:%M:%S")


def _to_utc(v: Any) -> Optional[str]:
    """Best-effort parse of a timestamp to ISO-8601 UTC; None when unparseable."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        dt = None
        for fmt in _FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_timestamps(t: ToyTable, column: str, timezone_name: str) -> Tuple[ToyTable, ToolReport]:
    if timezone_name != "UTC":
        raise ToolError(f"unsupported timezone: {timezone_name}")
    rows, changed, bad = [], 0, []
    for r in _copy(t.rows):
        if column in r:
            out = _to_utc(r[column])
            if out is None:  # never guess: unparseable values go to quarantine
                bad.append(r)
                continue
            if out != r[column]:
                r[column] = out
                changed += 1
        rows.append(r)
    return (
        replace(t, rows=tuple(rows), quarantine=t.quarantine + tuple(bad)),
        ToolReport("normalize_timestamps", changed, f"{column} -> UTC", {"quarantined": len(bad)}),
    )


def repartition(t: ToyTable, key: str, partitions: int) -> Tuple[ToyTable, ToolReport]:
    if partitions < 1:
        raise ToolError("partitions must be positive")
    rows = _copy(t.rows)
    sizes = [0] * partitions
    for r in rows:
        r["_partition"] = zlib.crc32(str(r.get(key)).encode()) % partitions
        sizes[r["_partition"]] += 1
    return (
        replace(t, rows=tuple(rows), partitions=partitions),
        ToolReport("repartition", len(rows), f"key={key} partitions={partitions}", {"sizes": sizes}),
    )


def apply_action(t: ToyTable, action: Dict[str, Any]) -> Tuple[ToyTable, ToolReport]:
    """Validate then dispatch an action. Raises ToolError on anything unsafe."""
    res = validate_action(action)
    if not res.ok:
        raise ToolError("invalid action: " + "; ".join(res.errors))
    p = action["params"]
    if p["table"] != t.name:
        raise ToolError(f"action targets {p['table']!r} but table is {t.name!r}")
    name = action["action"]
    if name == "add_column_mapping":
        return add_column_mapping(t, p["source"], p["target"])
    if name == "quarantine_records":
        return quarantine_records(t, p["column"], p["condition"])
    if name == "deduplicate":
        return deduplicate(t, p["key"], p["keep"])
    if name == "replay_window":
        return replay_window(t, p["start"], p["end"])
    if name == "normalize_timestamps":
        return normalize_timestamps(t, p["column"], p["timezone"])
    return repartition(t, p["key"], p["partitions"])
