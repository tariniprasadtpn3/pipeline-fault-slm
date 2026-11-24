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
3. [x] Dataset builder: train/val/test split to instruction-tuning JSONL
4. [x] Action schema validation (closed action vocabulary, required params)
5. [x] Eval harness: accuracy, valid-JSON rate and per-fault confusion
6. [x] Colab/Kaggle LoRA fine-tuning notebook (see `docs/training.md`; not yet executed)
7. [x] Inference wrapper with constrained JSON output and a mock model for tests
8. [x] Safe tools (quarantine, dedupe, replay, normalize, repartition) on a toy table
9. [ ] Agent loop with audit log and guardrails (allow-list, dry-run, max steps)
10. [ ] Comparison report: SLM vs baseline vs base model
11. [ ] Demo UI and docs with results

## Running the tests

```
pip install pytest
pytest
```

## Building the dataset

```
python -m src.dataset.builder --out data --n 600 --seed 42
```

Writes `train.jsonl`, `val.jsonl` and `test.jsonl` (80/10/10, stratified by
fault type, disjoint by sample id). Each line has `prompt`, `completion` (the
target JSON action) and a chat-style `messages` list for SFT trainers.

## Validating model output

```python
from src.agent import parse_action
result = parse_action(model_text)   # tolerates a ```json fence
result.ok, result.errors, result.action
```

`src/agent/schema.py` enforces the closed action vocabulary: a known action,
exactly its required params (missing or extra params are rejected),
identifier-safe names, a positive integer partition count and an ordered
replay window. Nothing unvalidated reaches a tool.

## Evaluating a predictor

```
python -m src.dataset.builder --out data --n 600 --seed 42
python -m src.eval.harness --data data/test.jsonl
```

`src/eval/harness.py` scores any `prompt -> text` callable (the rule baseline,
a mock model, a fine-tuned SLM) on held-out records and reports valid-JSON
rate, action accuracy, exact match (action plus all params) and a per-fault
confusion of predicted actions (`<invalid>` for unparseable or schema-failing
output).

## Inference wrapper

```python
from src.inference import ActionModel, MockModel
model = ActionModel(generator, max_retries=2)   # generator: (messages, prefill) -> text
result = model.predict(log)                     # ok, action, errors, attempts
```

`src/inference/wrapper.py` rebuilds the training prompt, prefills the reply with
`{"action": "` to keep output inside a JSON action object, extracts the first
balanced JSON object, validates it with the action schema and retries with the
validation errors fed back. Unvalidated output is never returned as `ok`.
`action_json_schema()` exports the vocabulary for grammar-guided decoders, and
`ActionModel.as_predictor()` plugs any model into the eval harness. `MockModel`
scripts replies for CPU-only tests.

## Safe tools

```python
from src.agent import ToyTable, apply_action
table, report = apply_action(table, action)   # action: a validated model output
```

`src/agent/tools.py` implements the six actions on an in-memory `ToyTable`.
Tools are pure (they return a new table and a `ToolReport`, never mutate), the
action is schema-validated and must target the supplied table, `add_column_mapping`
refuses to overwrite an existing column, `replay_window` is idempotent, and
`normalize_timestamps` quarantines values it cannot parse instead of guessing.
