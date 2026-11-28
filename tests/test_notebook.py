"""Structural checks for the (unexecuted) LoRA notebook: valid JSON, code cells
parse, and it uses the expected libraries and dataset files."""
import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from src.dataset.builder import build_dataset, write_jsonl
from src.eval.compare import load_predictions
from tests.hf_stubs import StubModel, StubTokenizer

NB = Path(__file__).resolve().parent.parent / "notebooks" / "lora_finetune.ipynb"


def _cells():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    return nb["cells"]


def _code(cells):
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _python(cells):
    """Code cells as plain Python, without IPython magics and shell lines."""
    lines = _code(cells).splitlines()
    return "\n".join(l for l in lines if not l.lstrip().startswith(("!", "%")))


def test_code_cells_are_valid_python():
    for c in _cells():
        if c["cell_type"] == "code":
            ast.parse(_python([c]))


def test_uses_expected_stack_and_data():
    src = _code(_cells())
    for needle in ("transformers", "peft", "trl", "LoraConfig", "SFTTrainer",
                   "data/train.jsonl", "data/val.jsonl", "data/test.jsonl"):
        assert needle in src


def test_repo_imports_resolve():
    for node in ast.walk(ast.parse(_python(_cells()))):
        if isinstance(node, ast.ImportFrom) and node.module.startswith("src."):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name}"


def test_evaluation_cell_saves_base_and_tuned_predictions(tmp_path, monkeypatch, capsys):
    """Run the last cell's Python on stubs: base outputs come with the adapter off."""
    records = build_dataset(n=60, seed=5)["test"]
    write_jsonl(records, tmp_path / "data" / "test.jsonl")
    monkeypatch.chdir(tmp_path)
    model = StubModel()
    tok = StubTokenizer(lambda adapter_on: "tuned" if adapter_on else "base")
    trainer = SimpleNamespace(model=model)
    exec(_python(_cells()[-1:]), {"trainer": trainer, "tok": tok})
    for path, expected in (("preds_base.jsonl", "base"), ("preds_slm.jsonl", "tuned")):
        preds = load_predictions(path)
        assert set(preds) == {r["id"] for r in records}
        assert set(preds.values()) == {expected}
    assert model.adapter_on
    assert "lora-slm" in capsys.readouterr().out


def test_evaluation_cell_runs_the_comparison_report():
    src = _code(_cells()[-1:])
    for needle in ("src.eval.compare", "base-model=preds_base.jsonl",
                   "slm-lora=preds_slm.jsonl", "--out comparison.md"):
        assert needle in src


def test_outputs_are_clean():
    for c in _cells():
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
