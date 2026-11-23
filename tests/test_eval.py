import json

from src.dataset import build_dataset
from src.eval import baseline_predictor, evaluate, format_report, load_jsonl
from src.eval.harness import INVALID, main

TEST = build_dataset(n=120, seed=5)["test"]


def _oracle(prompt):
    by_prompt = {r["prompt"]: r["completion"] for r in TEST}
    return by_prompt[prompt]


def test_oracle_scores_perfectly():
    rep = evaluate(TEST, _oracle)
    assert rep.n == len(TEST) > 0
    assert rep.valid_json_rate == rep.action_accuracy == rep.exact_match == 1.0
    for fault, row in rep.confusion.items():
        assert len(row) == 1


def test_baseline_is_perfect_on_synthetic_test_split():
    rep = evaluate(TEST, baseline_predictor)
    assert rep.exact_match == 1.0


def test_garbage_output_counts_as_invalid():
    rep = evaluate(TEST, lambda p: "I think you should restart it")
    assert rep.valid_json_rate == 0.0 and rep.action_accuracy == 0.0
    assert all(set(c) == {INVALID} for c in rep.confusion.values())


def test_right_action_wrong_params_is_not_exact():
    def pred(prompt):
        d = json.loads(_oracle(prompt))
        if d["action"] == "deduplicate":
            d["params"]["keep"] = "earliest"
        return json.dumps(d)

    rep = evaluate(TEST, pred)
    dedupe = rep.per_fault["duplicate_events"]
    assert dedupe["action"] == dedupe["n"] and dedupe["exact"] == 0
    assert rep.action_accuracy == 1.0 and rep.exact_match < 1.0


def test_confusion_records_wrong_action_names():
    fixed = json.dumps({"action": "repartition", "params": {"key": "sku", "table": "orders", "partitions": 8}})
    rep = evaluate(TEST, lambda p: fixed)
    assert rep.confusion["null_spike"]["repartition"] == rep.per_fault["null_spike"]["n"]
    assert rep.per_fault["partition_skew"]["action"] == rep.per_fault["partition_skew"]["n"]
    assert 0 < rep.action_accuracy < 1


def test_empty_input_does_not_divide_by_zero():
    rep = evaluate([], _oracle)
    assert rep.n == 0 and rep.action_accuracy == 0.0


def test_format_report_lists_every_fault_type():
    text = format_report(evaluate(TEST, baseline_predictor), "baseline")
    assert "baseline: n=" in text and "partition_skew" in text


def test_cli_reads_jsonl(tmp_path, capsys):
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in TEST) + "\n")
    assert len(load_jsonl(path)) == len(TEST)
    main(["--data", str(path)])
    assert "rule-baseline" in capsys.readouterr().out
