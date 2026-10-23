# A-KV-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-KV-004 pipeline: the evaluator takes your modified SmolLM2-135M (attention replaced by your key-value structure),
continues its training on 30M FineWeb-Edu tokens with a fixed recipe, and measures held-out loss, the key-value cache
size per token and long-context behaviour (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training tokens (untimed)
  timed   build(models_dir + "/SmolLM2-135M", device) -> smol_lm.SmolLM with your mixers; train STEPS steps of 16 x 2,048
          tokens with the fixed recipe of lm_train.py (AdamW peak lr 3e-4, warm-up and cosine decay, bf16, clipping)
  finish  held-out loss on windows of 2,048 and on windows of 8,192 tokens; cache growth in bytes per token between
          1,024 and 2,048 tokens of context (bf16 autocast, smol_lm.kv_bytes_per_token); consistency of prefill +
          step decoding with the full forward (smol_lm.decode_consistency)
The evaluator runs the multi-query baseline (baselines/mqa) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/mqa/model.py`
- `interface.md`
- `lm_train.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `smol_lm.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
