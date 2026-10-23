# A-NORM-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-NORM-005 pipeline: the evaluator trains the fixed language model (norm_lm.py) with your normalisation and
residual scheme at two scales and two learning rates, then measures held-out loss (evaluator and self-check use this
same file).

For each run r = (scale, lr) in RUNS, one stage "run_<scale>_<lr>" (imports norm.py):
  setup   map the training shards (untimed)
  timed   build norm_lm.LM(SCALES[scale], SCHEME) with the seed and train it with the fixed recipe of lm_train.py
          (AdamW, warm-up and cosine decay, bf16, clipping 1.0) for STEPS[scale] steps of 32 x 1,024 tokens
  finish  held-out loss, number of loss spikes, divergence; forward FLOPs per token and scheme parameters of your
          model and of the baseline scheme at the same scale (the compute gate)
The evaluator runs the pre-RMSNorm baseline (baselines/prenorm) through the same stages on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/prenorm/norm.py`
- `interface.md`
- `lm_train.py`
- `metrics.py`
- `norm_lm.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
