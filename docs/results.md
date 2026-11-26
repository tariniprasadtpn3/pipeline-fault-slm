# Results

Measured on the synthetic held-out split (`--n 600 --seed 42`, 60 test records,
10 per fault type). Only the rule baseline has been run. The untuned base model
and the LoRA-tuned SLM have **not** been run yet (the training notebook is
committed unexecuted), so no numbers are claimed for them.

## Rule baseline

| system | n | valid JSON | action acc | exact match |
|---|---:|---:|---:|---:|
| rule-baseline | 60 | 100.0% | 100.0% | 100.0% |

Exact match by fault type:

| fault type | rule-baseline |
|---|---:|
| bad_timestamp | 10/10 |
| duplicate_events | 10/10 |
| late_data | 10/10 |
| null_spike | 10/10 |
| partition_skew | 10/10 |
| schema_drift | 10/10 |

## How to read this

A perfect score here is expected and says little about real-world quality: the
regex rules were written against the same fixed message templates the generator
uses. On this split a fine-tuned model can at best tie the baseline, so it
cannot show that a model is worth its cost. That needs held-out logs with
wording the rules have not seen (paraphrased messages, reordered or noisy
lines), which this project does not have yet.

## Reproduce

```
python -m src.dataset.builder --out data --n 600 --seed 42
python -m src.eval.compare --data data/test.jsonl
```

`tests/test_results_doc.py` fails if the tables above drift from the code.

## Adding model results

Generate predictions on a GPU as JSONL (`{"id", "output"}` per line), then:

```
python -m src.eval.compare --data data/test.jsonl \
    --pred base-model=preds_base.jsonl --pred slm-lora=preds_slm.jsonl \
    --out comparison.md
```

Copy the tables from `comparison.md` into this file.
