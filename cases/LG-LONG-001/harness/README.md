# LG-LONG-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
LG-LONG-001 pipeline: multi-label classification of long European Court of Human Rights cases (LexGLUE ECtHR task A;
evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict" (the frozen legal-bert-base-uncased is available):
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model                 (load time, killed at 300 s)
  finish  predict(model, texts) -> (M, 10) 0/1 decisions for the 10 articles, on the held-out cases       (≤ 600 s)
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
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.
