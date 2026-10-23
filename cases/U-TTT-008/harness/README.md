# U-TTT-008 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-TTT-008 pipeline: the evaluator takes your modified SmolLM2-135M (at least half of its attention layers replaced by
layers with a fixed-size state), continues its training on 50M FineWeb-Edu tokens with a fixed recipe, and measures
long-context language modelling and retrieval (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training tokens (untimed)
  timed   build(models_dir + "/SmolLM2-135M", device) -> smol_lm.SmolLM with your mixers; train STEPS steps of 12 x 4,096
          tokens with the fixed recipe of lm_train.py (AdamW peak lr 3e-4, warm-up and cosine decay, bf16, clipping)
  finish  (1) loss at every position of the first 8,192 tokens of each held-out PG19 book
          (2) passkey retrieval: a 5-digit key hidden at a random depth of 2k, 4k and 8k tokens of book text; correct if
              the greedy prediction of every key token after the question is right (teacher forcing)
          (3) held-out FineWeb-Edu loss on windows of 2,048 tokens
          (4) per-layer cache growth between 1,024 and 2,048 tokens of context (bf16 autocast): the number of layers
              whose cache does not grow; decoding consistency; decoding throughput after an 8,192-token prefill
The evaluator runs the sliding-window baseline (baselines/swa) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/swa/model.py`
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
