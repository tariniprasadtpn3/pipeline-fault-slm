import json

from src.dataset import build_dataset
from src.eval import baseline_predictor, compare, format_comparison, predictions_predictor
from src.eval.compare import main

TEST = build_dataset(n=120, seed=5)["test"]


def test_compare_ranks_systems_on_same_records():
    reps = compare(TEST, {"baseline": baseline_predictor, "chatty": lambda p: "restart it"})
    assert list(reps) == ["baseline", "chatty"]
    assert reps["baseline"].exact_match == 1.0
    assert reps["chatty"].valid_json_rate == 0.0
    assert reps["baseline"].n == reps["chatty"].n == len(TEST)


def test_predictions_replay_and_missing_ids_are_invalid():
    preds = {r["id"]: r["completion"] for r in TEST[1:]}
    rep = compare(TEST, {"m": predictions_predictor(TEST, preds)})["m"]
    assert rep.valid == len(TEST) - 1 and rep.n == len(TEST)


def test_format_comparison_markdown():
    reps = compare(TEST, {"a": baseline_predictor, "b": lambda p: ""})
    text = format_comparison(reps)
    assert "| a | " in text and "| b | " in text
    assert "100.0%" in text and "0.0%" in text
    assert "| fault type | a | b |" in text
    assert all(r["fault_type"] in text for r in TEST)


def test_cli_writes_report(tmp_path, capsys):
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in TEST), encoding="utf-8")
    pred = tmp_path / "p.jsonl"
    pred.write_text("\n".join(json.dumps({"id": r["id"], "output": r["completion"]}) for r in TEST))
    out = tmp_path / "results.md"
    main(["--data", str(data), "--pred", f"slm={pred}", "--out", str(out)])
    body = out.read_text()
    assert "| slm | " in body and "rule-baseline" in body
    assert "rule-baseline" in capsys.readouterr().out
