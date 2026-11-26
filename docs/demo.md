# Demo UI

```
python -m src.demo.app            # open http://127.0.0.1:8000
```

Standard library only; it binds to localhost by default and is meant for local
use (no authentication).

- **Sample mode:** pick a fault type and seed. The page shows the synthetic log,
  the audit entry of one agent step, whether the prediction matches the ground
  truth, and the toy table before (and, with *apply* ticked, after) the step.
- **Paste mode:** paste a log to get a diagnosed action. Nothing is executed,
  because there is no table behind a pasted log.

Dry-run is the default. The toy table is built from the sample's true fault, not
from the prediction.

The default predictor is the rule baseline. To demo a model, serve
`make_app(model, predictor="my model")` with any WSGI server, where `model` has
`predict(log)`, such as `ActionModel(generator)`.

Limitation: the `repartition` toy tool hashes the key named in the action, so
rows sharing one hot key value still land in the same partition. It shows the
action being applied, not the skew being cured (real fixes include salting the
key).
