# U-LOOP-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-LOOP-001 pipeline: the evaluator trains your recurrent-depth language model with a fixed recipe and measures
held-out loss at 1, 2, 4 and 8 loops (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training shards (untimed)
  timed   build(CONFIG) with the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0) using model.loss(x, y) — how many
          loops are used in training is up to the model
  finish  held-out loss with model.loss(x, y, loops=k) for k in LOOPS (windows of 1,024), and at k = 8 on windows of
          2,048 (length extrapolation); parameters; forward FLOPs per token at each k (FlopCounterMode) next to the
          4-layer baseline transformer of the same width; causality check
The evaluator runs the 4-layer transformer (baselines/fourlayer: the same parameter budget, no loops) through the same
stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/fourlayer/model.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
