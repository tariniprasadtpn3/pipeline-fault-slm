# LoRA fine-tuning notebook

`notebooks/lora_finetune.ipynb` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with
LoRA (r=16, attention projections) using `trl`'s `SFTTrainer` on the chat
`messages` from the dataset builder, then scores the adapter on `test.jsonl`
with `src/eval/harness.py`.

Run it on a free Colab or Kaggle T4: open the notebook, enable the GPU and
run all cells. It clones this repo, builds the synthetic dataset, trains
(about two epochs), saves `lora-adapter/` and prints valid-JSON rate,
accuracy and exact match.

The notebook is committed unexecuted; no results are claimed until it has
been run. `tests/test_notebook.py` only checks its structure on CPU.
