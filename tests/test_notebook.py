"""Structural checks for the (unexecuted) LoRA notebook: valid JSON, code cells
parse, and it uses the expected libraries and dataset files."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "lora_finetune.ipynb"


def _cells():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    return nb["cells"]


def _code(cells):
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_code_cells_are_valid_python():
    for c in _cells():
        if c["cell_type"] != "code":
            continue
        # drop IPython magics / shell lines before parsing
        lines = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lines))


def test_uses_expected_stack_and_data():
    src = _code(_cells())
    for needle in ("transformers", "peft", "trl", "LoraConfig", "SFTTrainer",
                   "data/train.jsonl", "data/val.jsonl", "data/test.jsonl"):
        assert needle in src


def test_outputs_are_clean():
    for c in _cells():
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
