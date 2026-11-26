import io
from urllib.parse import urlencode
from wsgiref.util import setup_testing_defaults

import pytest

from src.demo import RuleModel, run_sample, sample_for, toy_table
from src.demo.app import MAX_BODY, make_app
from src.faults import FAULT_TYPES
from src.inference import ActionModel, MockModel


def call(app, form=None, method=None, path="/", length=None):
    body = urlencode(form or {}).encode()
    environ = {}
    setup_testing_defaults(environ)
    environ.update(
        REQUEST_METHOD=method or ("POST" if form is not None else "GET"),
        PATH_INFO=path,
        CONTENT_LENGTH=str(len(body)) if length is None else length,
    )
    environ["wsgi.input"] = io.BytesIO(body)
    seen = {}
    out = b"".join(app(environ, lambda status, headers: seen.update(status=status)))
    return seen["status"], out.decode()


@pytest.mark.parametrize("fault", FAULT_TYPES)
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_rule_model_applies_the_ground_truth_action(fault, seed):
    sample = sample_for(fault, seed)
    run = run_sample(sample, RuleModel(), apply=True)
    assert sample.fault_type == fault
    assert run.audit[0]["status"] == "applied"
    assert run.audit[0]["action"] == sample.action
    assert run.audit[0]["report"]["changed"] > 0


def test_each_toy_fault_is_remediated():
    def after(fault):
        sample = sample_for(fault)
        return sample.action["params"], run_sample(sample, RuleModel(), apply=True).table

    p, t = after("schema_drift")
    assert p["target"] in t.columns and p["source"] not in t.columns
    p, t = after("null_spike")
    assert all(r[p["column"]] is not None for r in t.rows) and len(t.quarantine) == 3
    p, t = after("duplicate_events")
    assert sorted(r[p["key"]] for r in t.rows) == [1, 2, 3]
    _, t = after("late_data")
    assert sorted(r["id"] for r in t.rows) == [1, 2, 3]
    p, t = after("bad_timestamp")
    assert all(r[p["column"]].endswith("Z") for r in t.rows) and len(t.quarantine) == 1
    p, t = after("partition_skew")
    assert t.partitions == p["partitions"]


def test_dry_run_is_default_and_leaves_table_unchanged():
    sample = sample_for("duplicate_events")
    run = run_sample(sample, RuleModel())
    assert run.audit[0]["status"] == "dry_run"
    assert run.table == toy_table(sample)


def test_unknown_fault_type_is_rejected():
    with pytest.raises(ValueError):
        sample_for("disk_full")


def test_get_shows_form_with_every_fault_and_no_result():
    status, page = call(make_app())
    assert status.startswith("200")
    assert all(f'value="{f}"' in page for f in FAULT_TYPES)
    assert "Agent step" not in page


def test_post_sample_dry_run_then_apply():
    app = make_app()
    _, dry = call(app, {"fault": "null_spike", "seed": "3"})
    assert "Agent step: dry_run" in dry and "Table unchanged." in dry
    _, done = call(app, {"fault": "null_spike", "seed": "3", "apply": "1"})
    assert "Agent step: applied" in done and "Quarantine (3 rows)" in done
    assert "Prediction matches" in done


def test_pasted_log_is_diagnosed_but_not_executed():
    log = sample_for("late_data").log.replace("\n", "\r\n")
    status, page = call(make_app(), {"log": log})
    assert status.startswith("200")
    assert "not executed" in page and "replay_window" in page
    assert "Agent step" not in page


def test_unrecognised_log_reports_no_action_and_is_escaped():
    _, page = call(make_app(), {"log": "</textarea><script>alert(1)</script>"})
    assert "No valid action" in page and "no rule matched this log" in page
    assert "<script>" not in page


def test_plugged_in_model_hits_the_table_guardrail():
    reply = '{"action": "deduplicate", "params": {"key": "id", "table": "wrong_table", "keep": "latest"}}'
    app = make_app(ActionModel(MockModel([reply])), predictor="mock model")
    _, page = call(app, {"fault": "duplicate_events", "apply": "1"})
    assert "mock model" in page
    assert "Agent step: tool_error" in page and "does not match" in page
    assert "Table unchanged." in page


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"form": {"fault": "disk_full"}}, "400"),
        ({"form": {"fault": "null_spike", "seed": "abc"}}, "400"),
        ({"form": {}, "length": "-1"}, "400"),
        ({"form": {}, "length": "nope"}, "400"),
        ({"form": {}, "length": str(MAX_BODY + 1)}, "413"),
        ({"path": "/admin"}, "404"),
        ({"method": "DELETE"}, "405"),
    ],
)
def test_bad_requests(kwargs, code):
    status, _ = call(make_app(), **kwargs)
    assert status.startswith(code)
