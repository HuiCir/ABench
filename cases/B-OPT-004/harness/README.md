# B-OPT-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
B-OPT-004 pipeline: train the fixed language model (lm_model.py, as in B-OPT-001) with your update rule at three batch
sizes for the same number of training tokens, then measure held-out loss at each batch size.

For each batch size B in PARAMS["batches"] (sequences of 1,024 tokens):
Stage "train_b<B>"  (imports optimizer.py)
  setup   map the training token shards, build the model on the GPU with the seed (untimed)
  timed   build RULE with the declared defaults; peak lr = lr * (B / 32) ** lr_batch_exponent (from hparam_space.yaml);
          train TOKENS / (B * 1,024) optimizer steps, each accumulating B / 32 micro-batches of 32 sequences
  finish  save model.pt; report the loss curve, divergence and the size of the optimizer state
Stage "eval_b<B>"   (does not import your code; receives only model.pt)
  timed   load model.pt
  finish  mean next-token loss on the held-out tokens
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/adamw/hparam_space.yaml`
- `baselines/adamw/optimizer.py`
- `interface.md`
- `lm_model.py`
- `metrics.py`
- `optim_api.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
