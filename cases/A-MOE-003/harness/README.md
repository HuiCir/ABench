# A-MOE-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-MOE-003 pipeline: the evaluator trains the fixed μ2 language model (moe_lm.py) with your feed-forward layer in
every block, then measures held-out loss, load balance and compute (evaluator and self-check use this same file).

One stage, "run" (imports moe.py), repeated for 3 seeds:
  setup   map the training shards (untimed)
  timed   build moe_lm.LM(Config(), FFN) with the seed; train 3,000 steps of 32 x 1,024 tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0); your aux["loss"] terms are added
  finish  (1) held-out loss and per-expert token counts over the held-out tokens
          (2) forward FLOPs per token on a held-out batch (FlopCounterMode) against the top-2-of-8 budget
          (3) causality and batch-independence checks: changing later tokens, or other sequences in the batch, must not
              change the outputs at earlier positions of a sequence
The evaluator runs the top-2 baseline (baselines/top2) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/top2/moe.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `moe_lm.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
