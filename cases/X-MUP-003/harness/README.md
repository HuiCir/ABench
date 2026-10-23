# X-MUP-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-MUP-003 pipeline: tune the learning rate on a narrow model, transfer it to wider ones (evaluator and self-check use
this same file).

One stage, "protocol" (imports param.py), per seed:
  setup   map the training shards; the last 1,048,576 tokens of the second shard are kept out of training as the
          tuning set (untimed)
  timed   (1) width 128: train with each η in ETA_GRID (7 values, factors of 2) for STEPS steps; η* = the η with the
              lowest loss on the tuning set
          (2) width 512: train with η* · m for m in (1/4, 1/2, 1, 2, 4); tuning-set loss of each (the transfer gap)
          (3) width 1,024: train with η*
          every run: mup_lm.LM(Config(width), PARAM), AdamW (β = 0.9, 0.95, weight decay 0.1 times your wd_scale),
          per-parameter learning rate η · lr_scale, 150 warm-up steps and cosine decay to 0.1, clipping 1.0, bf16
  finish  held-out loss of the width-1,024 model and of the width-512 model trained with η*
The evaluator runs the standard parametrisation (baselines/sp) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/sp/param.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `mup_lm.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
