# X-DATA-007 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-DATA-007 pipeline: your data pipeline selects training documents from a raw web pool; the evaluator trains a
fixed small language model on the selection and measures its loss on held-out educational text (evaluator and
self-check).

Two stages, each in a new process:
  select  (8 CPU cores, no GPU) timed: import pipeline.py and call select(pool_dir, budget_tokens, workdir) → document
          indices in training order (repeats allowed); finish: check the indices and write selection.npy
  train   (GPU; your code is not imported) setup: the token stream = the pool tokens of the selected documents,
          concatenated; timed: train norm_lm.LM (the 30M-parameter "mu2" configuration with pre-RMSNorm) with the
          fixed recipe of lm_train.py for STEPS steps of 32 × 1,024 tokens sampled from the stream; finish: held-out
          loss on educational text (target) and on general web text
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/pipeline.py`
- `baseline/src/dedup.py`
- `fixed_norm.py`
- `interface.md`
- `lm_train.py`
- `metrics.py`
- `norm_lm.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
