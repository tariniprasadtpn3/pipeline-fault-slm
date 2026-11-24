import json

from src.baseline import diagnose
from src.dataset import build_dataset
from src.eval import evaluate
from src.inference import ActionModel, MockModel, action_json_schema, extract_json_object

TEST = build_dataset(n=60, seed=3)["test"]
GOOD = {"action": "deduplicate", "params": {"key": "event_id", "table": "orders", "keep": "latest"}}


def _log(rec):
    return rec["prompt"].split("Failure log:\n", 1)[-1].rsplit("\n\nReturn the remediation", 1)[0]


def test_extract_json_object_ignores_prose_and_braces_in_strings():
    assert extract_json_object('sure: {"a": "}"} done') == '{"a": "}"}'
    assert extract_json_object("no json") is None
    assert extract_json_object('{"a": {"b": 1}} {"c": 2}') == '{"a": {"b": 1}}'


def test_valid_reply_with_prefill_continuation():
    full = json.dumps(GOOD)
    m = MockModel([full[len('{"action": "'):]])  # model continues after the prefill
    res = ActionModel(m).predict("whatever")
    assert res.ok and res.action == GOOD and res.attempts == 1
    assert m.calls[0][1] == '{"action": "'


def test_full_object_reply_and_fence_and_prose():
    res = ActionModel(MockModel(["Here you go:\n```json\n" + json.dumps(GOOD) + "\n```"])).predict("x")
    assert res.ok and res.action == GOOD


def test_retry_feeds_back_errors_then_succeeds():
    bad = json.dumps({"action": "drop_table", "params": {}})
    m = MockModel([bad, json.dumps(GOOD)])
    res = ActionModel(m, max_retries=2).predict("x")
    assert res.ok and res.attempts == 2
    retry_msgs = m.calls[1][0]
    assert retry_msgs[-1]["role"] == "user" and "unknown action" in retry_msgs[-1]["content"]


def test_exhausted_retries_never_ok():
    m = MockModel(["I would restart it"])
    res = ActionModel(m, max_retries=1).predict("x")
    assert not res.ok and res.action is None and res.attempts == 2 and res.errors


def test_zero_retries_single_call():
    m = MockModel(["nonsense"])
    ActionModel(m, max_retries=0).predict("x")
    assert len(m.calls) == 1


def test_harness_integration_with_oracle_mock():
    by_log = {_log(r): r["completion"] for r in TEST}

    def gen(messages, prefill):
        log = messages[1]["content"].split("Failure log:\n", 1)[-1].rsplit("\n\nReturn the remediation", 1)[0]
        return by_log[log]

    rep = evaluate(TEST, ActionModel(gen).as_predictor())
    assert rep.exact_match == 1.0


def test_schema_covers_every_action_with_required_params():
    from src.agent import ACTION_PARAMS

    schema = action_json_schema()
    names = {b["properties"]["action"]["const"] for b in schema["oneOf"]}
    assert names == set(ACTION_PARAMS)
    for b in schema["oneOf"]:
        spec = ACTION_PARAMS[b["properties"]["action"]["const"]]
        assert set(b["properties"]["params"]["required"]) == set(spec)
        assert b["properties"]["params"]["additionalProperties"] is False


def test_baseline_actions_pass_through_model():
    rec = TEST[0]
    res = ActionModel(MockModel([json.dumps(diagnose(_log(rec)))])).predict(_log(rec))
    assert res.ok
