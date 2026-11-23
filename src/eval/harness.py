"""Eval harness: score any predictor on held-out fault records.

A predictor is a callable `prompt -> raw model text`. That keeps the harness
model-agnostic: the rule baseline, a mock model in tests and a fine-tuned SLM
all plug in the same way. Metrics:

- valid_json_rate: output parses and passes the action schema
- action_accuracy: the chosen action name matches the label
- exact_match: action name and every param match the label
- confusion: per fault type, counts of predicted action names
  (`<invalid>` when the output was not a valid action)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from src.agent import parse_action
from src.baseline import diagnose

INVALID = "<invalid>"
Predictor = Callable[[str], str]


@dataclass
class EvalReport:
    n: int = 0
    valid: int = 0
    action_correct: int = 0
    exact: int = 0
    confusion: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    per_fault: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def valid_json_rate(self) -> float:
        return self.valid / self.n if self.n else 0.0

    @property
    def action_accuracy(self) -> float:
        return self.action_correct / self.n if self.n else 0.0

    @property
    def exact_match(self) -> float:
        return self.exact / self.n if self.n else 0.0


def load_jsonl(path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(records: Iterable[dict], predict: Predictor) -> EvalReport:
    """Run `predict` on each record's prompt and score against its completion."""
    rep = EvalReport()
    for rec in records:
        target = json.loads(rec["completion"])
        fault = rec["fault_type"]
        stats = rep.per_fault.setdefault(fault, {"n": 0, "valid": 0, "action": 0, "exact": 0})
        res = parse_action(predict(rec["prompt"]))
        rep.n += 1
        stats["n"] += 1
        if res.ok:
            rep.valid += 1
            stats["valid"] += 1
            pred_name = res.action["action"]
            if pred_name == target["action"]:
                rep.action_correct += 1
                stats["action"] += 1
                if res.action == target:
                    rep.exact += 1
                    stats["exact"] += 1
        else:
            pred_name = INVALID
        rep.confusion[fault][pred_name] += 1
    return rep


def baseline_predictor(prompt: str) -> str:
    """Adapt the rule baseline to the prompt -> text predictor interface."""
    log = prompt.split("Failure log:\n", 1)[-1].rsplit("\n\nReturn the remediation", 1)[0]
    action: Optional[dict] = diagnose(log)
    return json.dumps(action) if action is not None else "no action"


def format_report(rep: EvalReport, name: str = "model") -> str:
    lines = [
        f"{name}: n={rep.n}  valid_json={rep.valid_json_rate:.1%}  "
        f"action_acc={rep.action_accuracy:.1%}  exact_match={rep.exact_match:.1%}",
        "",
        f"{'fault_type':<18}{'n':>5}{'valid':>8}{'action':>8}{'exact':>8}  predicted actions",
    ]
    for fault in sorted(rep.per_fault):
        s = rep.per_fault[fault]
        conf = ", ".join(f"{k}={v}" for k, v in sorted(rep.confusion[fault].items()))
        lines.append(f"{fault:<18}{s['n']:>5}{s['valid']:>8}{s['action']:>8}{s['exact']:>8}  {conf}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the rule baseline on a JSONL split")
    ap.add_argument("--data", default="data/test.jsonl")
    args = ap.parse_args(argv)
    print(format_report(evaluate(load_jsonl(args.data), baseline_predictor), "rule-baseline"))


if __name__ == "__main__":
    main()
