"""Action schema validation.

Model output is untrusted text. Before anything is executed it must parse as
JSON and match the closed action vocabulary: a known action name, exactly the
required parameters for that action, and well-formed parameter values. Extra
or missing parameters are rejected rather than ignored, so a hallucinated
field can never reach a tool.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.faults import ACTIONS

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MINUTE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d$")


def _ident(v: Any) -> bool:
    return isinstance(v, str) and bool(_IDENT.match(v))


def _minute(v: Any) -> bool:
    return isinstance(v, str) and bool(_MINUTE.match(v))


def _pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


# action -> {param name: validator}; every listed param is required.
ACTION_PARAMS: Dict[str, Dict[str, Callable[[Any], bool]]] = {
    "add_column_mapping": {"source": _ident, "target": _ident, "table": _ident},
    "quarantine_records": {
        "column": _ident,
        "condition": lambda v: v == "is_null",
        "table": _ident,
    },
    "deduplicate": {"key": _ident, "table": _ident, "keep": lambda v: v in ("latest", "earliest")},
    "replay_window": {"table": _ident, "start": _minute, "end": _minute},
    "normalize_timestamps": {"column": _ident, "table": _ident, "timezone": lambda v: v == "UTC"},
    "repartition": {"key": _ident, "table": _ident, "partitions": _pos_int},
}

assert set(ACTION_PARAMS) == set(ACTIONS), "schema must cover exactly the action vocabulary"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    action: Optional[Dict[str, Any]] = None


def validate_action(obj: Any) -> ValidationResult:
    """Check a parsed object against the action schema."""
    if not isinstance(obj, dict):
        return ValidationResult(False, ["action must be a JSON object"])
    errors: List[str] = []
    extra_top = set(obj) - {"action", "params"}
    if extra_top:
        errors.append(f"unexpected top-level keys: {sorted(extra_top)}")
    name = obj.get("action")
    if name not in ACTION_PARAMS:
        errors.append(f"unknown action: {name!r}")
        return ValidationResult(False, errors)
    params = obj.get("params")
    if not isinstance(params, dict):
        errors.append("params must be an object")
        return ValidationResult(False, errors)
    spec = ACTION_PARAMS[name]
    for key in sorted(set(spec) - set(params)):
        errors.append(f"missing param: {key}")
    for key in sorted(set(params) - set(spec)):
        errors.append(f"unexpected param: {key}")
    for key, check in spec.items():
        if key in params and not check(params[key]):
            errors.append(f"invalid value for {key}: {params[key]!r}")
    if name == "replay_window" and not errors and params["start"] >= params["end"]:
        errors.append("replay window start must be before end")
    return ValidationResult(not errors, errors, obj if not errors else None)


def parse_action(text: str) -> ValidationResult:
    """Parse raw model output and validate it. Tolerates a ```json fence."""
    if not isinstance(text, str):
        return ValidationResult(False, ["output must be a string"])
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        return ValidationResult(False, [f"invalid JSON: {e}"])
    return validate_action(obj)
