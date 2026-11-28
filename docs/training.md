# LoRA fine-tuning notebook

`notebooks/lora_finetune.ipynb` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with
LoRA (r=16, attention projections) using `trl`'s `SFTTrainer` on the chat
`messages` from the dataset builder. It then runs the untuned base model (the
same weights with the adapter switched off) and the tuned adapter on
`test.jsonl` and compares both with the rule baseline.

Run it on a free Colab or Kaggle T4: open the notebook, enable the GPU and
run all cells. It clones this repo, builds the synthetic dataset, trains
(about two epochs), saves `lora-adapter/`, prints the tuned model's per-fault
report and writes three files in the cloned repo folder:

- `preds_base.jsonl` and `preds_slm.jsonl`: raw outputs, one `{"id", "output"}`
  per line
- `comparison.md`: valid-JSON rate, accuracy and exact match for the rule
  baseline, base model and tuned model

Download them before the session ends. Both models are decoded greedily with no
JSON prefill or retries, so the scores are for raw model output.

The notebook is committed unexecuted; no results are claimed until it has
been run. On CPU, `tests/test_notebook.py` checks its structure and its `src`
imports and runs the evaluation cell's Python against a stub model and
tokenizer; installing, training and real generation are not covered.
