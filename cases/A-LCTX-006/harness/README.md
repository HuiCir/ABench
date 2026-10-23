# A-LCTX-006 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-LCTX-006 pipeline: the evaluator trains the fixed language model of ctx_model.py with your positions module at
1,024 tokens, extends it briefly at 4,096, and measures loss and retrieval at lengths up to 16,384 (evaluator and
self-check use this same file).

Stage "train" (imports positions.py):
  setup   map the training shards (untimed)
  timed   build ctx_model.LM with make_positions and the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with
          the fixed recipe of lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16), then 200 steps of 8 x 4,096
          tokens at lr 1e-4 (the context-extension phase)
  finish  held-out FineWeb loss at 1,024; PG-19 tail loss with 4,096, 8,192 and 16,384 tokens of context; copy-test
          loss at 8,192 and 16,384 (longctx.py); parameters of the positions module; divergence
The evaluator runs the RoPE baseline (baselines/rope) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/positions.py`
- `baselines/rope/positions.py`
- `ctx_model.py`
- `interface.md`
- `lm_train.py`
- `longctx.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
