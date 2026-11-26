"""docs/results.md must match what the code actually produces."""

from pathlib import Path

from src.dataset.builder import build_dataset
from src.eval.compare import compare, format_comparison
from src.eval.harness import baseline_predictor

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results.md"


def test_results_doc_matches_a_fresh_baseline_run():
    records = build_dataset(n=600, seed=42)["test"]
    table = format_comparison(compare(records, {"rule-baseline": baseline_predictor}))
    assert table in RESULTS.read_text(encoding="utf-8")
