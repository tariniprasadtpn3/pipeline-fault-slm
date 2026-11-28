from src.dataset.builder import SYSTEM_PROMPT, build_dataset
from src.eval import compare, load_predictions, predictions_predictor, write_predictions
from src.inference.hf import hf_predictor
from tests.hf_stubs import StubModel, StubTokenizer

TEST = build_dataset(n=60, seed=5)["test"]


def test_predictor_sends_the_training_chat_and_returns_only_new_tokens():
    model, tok = StubModel(), StubTokenizer(lambda adapter_on: '{"action": "x"}')
    assert hf_predictor(model, tok, max_new_tokens=7)(TEST[0]["prompt"]) == '{"action": "x"}'
    # same system and user turns as the training records
    assert tok.conversations == [TEST[0]["messages"][:2]]
    assert tok.conversations[0][0]["content"] == SYSTEM_PROMPT


def test_predictor_passes_the_encoding_to_generate_as_keywords():
    model = StubModel()
    hf_predictor(model, StubTokenizer(), max_new_tokens=7)("p")
    (call,) = model.calls
    assert set(call) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample"}
    assert call["max_new_tokens"] == 7 and call["do_sample"] is False


def test_written_predictions_score_like_the_live_predictor(tmp_path):
    answers = {r["prompt"]: r["completion"] for r in TEST}
    answers[TEST[0]["prompt"]] = 'not JSON,\n"two lines" é'
    path = tmp_path / "preds.jsonl"
    write_predictions(TEST, answers.__getitem__, path)
    preds = load_predictions(path)
    assert preds == {r["id"]: answers[r["prompt"]] for r in TEST}
    report = compare(TEST, {"m": predictions_predictor(TEST, preds)})["m"]
    assert report.n == len(TEST) and report.exact == len(TEST) - 1


def test_write_predictions_calls_the_model_once_per_record(tmp_path):
    seen = []
    write_predictions(TEST, lambda p: seen.append(p) or "", tmp_path / "p.jsonl")
    assert seen == [r["prompt"] for r in TEST]
