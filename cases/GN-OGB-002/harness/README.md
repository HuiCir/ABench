# GN-OGB-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
GN-OGB-002 pipeline: node classification on ogbn-arxiv (evaluator and self-check use this same file).

Transductive setting: the whole graph, all features and the labels of the train and valid nodes are public; the labels
of the test nodes (papers from 2019 on) are held out by the evaluator.
The evaluator does not train. One stage, "predict":
  setup   load graph.npz (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                          (load time, killed at 120 s)
  finish  predict(model, graph, nodes) -> (len(nodes), 40) scores for the test nodes        (≤ 120 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
