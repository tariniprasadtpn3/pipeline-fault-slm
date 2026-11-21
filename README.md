# pipeline-fault-slm

A small language model (SLM) that diagnoses data-pipeline failures and picks a
safe remediation, built to show that a small, narrowly fine-tuned model can do
a focused data-engineering job well, cheaply and with an audit trail.

> All data in this repository is synthetic. Pipeline, table and column names
> are invented. Nothing here comes from any employer or client.

## Idea

1. A seeded generator produces realistic pipeline failure logs (schema drift,
   null spikes, duplicate events, late data, bad timestamps, partition skew),
   each paired with the correct structured action.
2. A small open model (for example Qwen2.5-0.5B/1.5B or Llama-3.2-1B) is
   fine-tuned with LoRA to read a log and emit a JSON action.
3. An agent loop executes the chosen action through a small set of safe tools
   and writes every step to an audit log.
4. An eval harness compares the SLM with a rule-based baseline and the
   untuned base model on held-out faults.

## Planned layout

```
pipeline-fault-slm/
├── src/
│   ├── faults/      # synthetic fault generator
│   ├── baseline/    # keyword/rule-based diagnoser
│   ├── dataset/     # train/val/test builder (instruction-tuning JSONL)
│   ├── agent/       # action validation, safe tools, agent loop, audit log
│   └── eval/        # metrics and comparison harness
├── notebooks/       # Colab/Kaggle LoRA training notebook
├── tests/           # unit tests
└── docs/            # design notes and results
```

## Roadmap

1. [x] Synthetic fault generator with a fixed seed
2. [x] Rule-based baseline diagnoser, with tests
3. [ ] Dataset builder: train/val/test split to instruction-tuning JSONL
4. [ ] Action schema validation (closed action vocabulary, required params)
5. [ ] Eval harness: accuracy, valid-JSON rate and per-fault confusion
6. [ ] Colab/Kaggle LoRA fine-tuning notebook
7. [ ] Inference wrapper with constrained JSON output and a mock model for tests
8. [ ] Safe tools (quarantine, dedupe, replay, normalize, repartition) on a toy table
9. [ ] Agent loop with audit log and guardrails (allow-list, dry-run, max steps)
10. [ ] Comparison report: SLM vs baseline vs base model
11. [ ] Demo UI and docs with results

## Running the tests

```
pip install pytest
pytest
```
