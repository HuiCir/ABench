# U-BYTE-007 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-BYTE-007 pipeline: the evaluator trains your tokenizer-free byte-level model with a fixed recipe and measures
bits per byte on held-out text, with and without character-level noise (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training bytes (untimed)
  timed   build() with the seed; train 3,000 steps of 32 windows of 4,096 bytes with the fixed recipe of lm_train.py
          (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0), the loss being model.loss(x)
  finish  bits per byte on the held-out bytes (windows of 4,096) and on the same bytes with 5% character-level noise
          (byte_data.add_noise); forward FLOPs per byte on a held-out window and parameters, next to the baseline
          (static 4-byte patches); causality check
The evaluator runs the static-patch baseline (baselines/static4) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/static4/model.py`
- `byte_blocks.py`
- `byte_data.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
