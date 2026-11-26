"""Comparison report: several predictors scored on the same held-out split.

Systems are named `prompt -> text` predictors (rule baseline, base model, tuned
SLM). Model outputs generated elsewhere (for example on a Colab GPU) are read
from a predictions JSONL with one `{"id": ..., "output": "<raw text>"}` per line,
so the report itself needs no model, GPU or network. A record with no prediction
counts as invalid output.

CLI:
    python -m src.eval.compare --data data/test.jsonl \
        --pred base-model=preds_base.jsonl --pred slm-lora=preds_slm.jsonl \
        --out comparison.md
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, List, Optional

from .harness import EvalReport, baseline_predictor, evaluate, load_jsonl

Predictor = Callable[[str], str]


def predictions_predictor(records: Iterable[dict], preds: Dict[str, str]) -> Predictor:
    """Predictor replaying stored outputs, looked up by record id via its prompt."""
    by_prompt = {r["prompt"]: preds.get(r["id"], "") for r in records}
    return lambda prompt: by_prompt.get(prompt, "")


def load_predictions(path) -> Dict[str, str]:
    return {str(r["id"]): str(r["output"]) for r in load_jsonl(path)}


def compare(records: List[dict], systems: Dict[str, Predictor]) -> Dict[str, EvalReport]:
    """Score every system on the same records, preserving insertion order."""
    return {name: evaluate(records, pred) for name, pred in systems.items()}


def format_comparison(reports: Dict[str, EvalReport]) -> str:
    """Markdown report: overall table plus per-fault exact-match by system."""
    names = list(reports)
    lines = [
        "| system | n | valid JSON | action acc | exact match |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in reports.items():
        lines.append(
            f"| {name} | {r.n} | {r.valid_json_rate:.1%} | {r.action_accuracy:.1%} | {r.exact_match:.1%} |"
        )
    faults = sorted({f for r in reports.values() for f in r.per_fault})
    lines += ["", "Exact match by fault type:", "", "| fault type | " + " | ".join(names) + " |",
              "|---|" + "---:|" * len(names)]
    for fault in faults:
        cells = []
        for r in reports.values():
            s = r.per_fault.get(fault)
            cells.append(f"{s['exact']}/{s['n']}" if s else "-")
        lines.append(f"| {fault} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compare baseline and model predictions")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=FILE",
                    help="predictions JSONL ({id, output}) for a named system")
    ap.add_argument("--out", help="write the markdown report here")
    args = ap.parse_args(argv)
    records = load_jsonl(args.data)
    systems: Dict[str, Predictor] = {"rule-baseline": baseline_predictor}
    for spec in args.pred:
        name, _, path = spec.partition("=")
        if not name or not path:
            ap.error(f"--pred expects NAME=FILE, got {spec!r}")
        systems[name] = predictions_predictor(records, load_predictions(path))
    text = format_comparison(compare(records, systems))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
