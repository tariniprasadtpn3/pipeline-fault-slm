from src.baseline import diagnose, extract_table
from src.faults import generate_faults


def test_baseline_recovers_the_exact_action_on_generated_faults():
    samples = generate_faults(600, seed=7)
    wrong = [s.sample_id for s in samples if diagnose(s.log) != s.action]
    assert wrong == []


def test_baseline_is_correct_on_a_different_seed_too():
    assert all(diagnose(s.log) == s.action for s in generate_faults(300, seed=123))


def test_extract_table_reads_run_header():
    log = "2026-01-01 00:00:00 INFO  [orders_ingest] starting run-12345 target=orders\nERROR x"
    assert extract_table(log) == "orders"


def test_unknown_log_returns_none():
    log = "2026-01-01 00:00:00 INFO  [p] starting run-1 target=orders\nINFO all good"
    assert diagnose(log) is None


def test_log_without_a_target_table_returns_none():
    assert diagnose("ERROR null ratio for column 'amount' is 50%") is None


def test_skew_without_partition_hint_falls_back_to_128():
    log = (
        "2026-01-01 00:00:00 INFO  [p] starting run-1 target=events\n"
        "WARN  [p] partition skew: one partition holds 90% of rows for key 'region'"
    )
    assert diagnose(log)["params"]["partitions"] == 128
