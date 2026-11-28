"""Agent loop with an audit log and guardrails.

The loop asks a model (anything with ``predict(log) -> InferenceResult``, such
as ``ActionModel``) for an action, checks it against guardrails, applies it
with the safe tools and records every step. Guardrails:

* allow-list: only actions in ``allowed_actions`` may run (default: all six);
* dry-run: the action is computed and reported but the table is left as is;
* max steps: a hard cap on model calls per run;
* repeat guard: the same action twice in a row stops the run;
* nothing unvalidated runs: invalid model output and tool errors are logged
  and stop the run instead of being retried blindly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional

from .schema import ACTION_PARAMS
from .tools import ToolError, ToyTable, apply_action

STATUSES = ("applied", "dry_run", "invalid_output", "blocked", "tool_error", "repeat")


@dataclass
class RunResult:
    table: ToyTable
    audit: List[Dict[str, Any]] = field(default_factory=list)
    stopped: str = "done"  # done | max_steps | invalid_output | blocked | tool_error | repeat

    @property
    def steps(self) -> int:
        return len(self.audit)


class AgentLoop:
    def __init__(
        self,
        model: Any,
        allowed_actions: Optional[Iterable[str]] = None,
        dry_run: bool = False,
        max_steps: int = 3,
        audit_path: Optional[str] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        allowed: FrozenSet[str] = frozenset(ACTION_PARAMS if allowed_actions is None else allowed_actions)
        unknown = allowed - set(ACTION_PARAMS)
        if unknown:
            raise ValueError(f"unknown actions in allow-list: {sorted(unknown)}")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model, self.allowed = model, allowed
        self.dry_run, self.max_steps = dry_run, max_steps
        self.audit_path, self.clock = audit_path, clock

    def _log(self, result: RunResult, status: str, **fields: Any) -> None:
        entry = {"step": len(result.audit) + 1, "ts": self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "status": status, "dry_run": self.dry_run, **fields}
        result.audit.append(entry)
        if self.audit_path:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

    def run(
        self,
        table: ToyTable,
        log: str,
        next_log: Optional[Callable[[ToyTable], Optional[str]]] = None,
    ) -> RunResult:
        """Remediate ``table`` given ``log``.

        ``next_log`` re-inspects the table after each applied step and returns a
        new failure log, or None when the table looks healthy. Without it the
        loop performs a single step.
        """
        result = RunResult(table=table)
        last: Optional[Dict[str, Any]] = None
        current: Optional[str] = log
        while current is not None:
            if result.steps >= self.max_steps:
                result.stopped = "max_steps"
                return result
            pred = self.model.predict(current)
            if not pred.ok:
                self._log(result, "invalid_output", errors=list(pred.errors))
                result.stopped = "invalid_output"
                return result
            action = pred.action
            name = action["action"]
            if name not in self.allowed:
                self._log(result, "blocked", action=action, reason=f"{name} not in allow-list")
                result.stopped = "blocked"
                return result
            if action == last:
                self._log(result, "repeat", action=action, reason="same action twice in a row")
                result.stopped = "repeat"
                return result
            try:
                new_table, report = apply_action(result.table, action)
            except ToolError as e:
                self._log(result, "tool_error", action=action, error=str(e))
                result.stopped = "tool_error"
                return result
            last = action
            detail = {"tool": report.action, "changed": report.changed, "detail": report.detail}
            if "quarantined" in report.extra:  # rows moved aside are not counted in `changed`
                detail["quarantined"] = report.extra["quarantined"]
            if self.dry_run:
                self._log(result, "dry_run", action=action, report=detail)
            else:
                result.table = new_table
                self._log(result, "applied", action=action, report=detail)
            current = None if (next_log is None or self.dry_run) else next_log(result.table)
        return result
