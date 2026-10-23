# A-HYB-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-HYB-001 pipeline: the evaluator builds your model twice — as a language model and as a small associative-recall
model — trains both with fixed recipes, and measures loss, recall and long-context cost (evaluator and self-check use
this same file).

Stage "lm" (imports model.py):
  setup   map the training shards (untimed)
  timed   build(LM_CONFIG) with the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0)
  finish  held-out loss; forward FLOPs per token at T = 1,024 and 16,384 (FlopCounterMode) and parameters, each next
          to the transformer of lm_model.py at the same sizes; causality check; forward latency per token at
          T = 1,024, 4,096 and 16,384 (batch 1, inference only)
Stage "mqar" (imports model.py):
  timed   build(MQAR_CONFIG) with the seed; train 8,000 steps of batch 128 on mqar.TRAIN_GRID (A-LIN-002's recipe)
  finish  predictions at the query positions of the held-out MQAR sequences (longer sequences, more pairs)
The evaluator runs the transformer baseline (baselines/transformer) through the same stages on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/transformer/model.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `mqar.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
