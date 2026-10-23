# C-KV-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
C-KV-002 pipeline: long-context retrieval with a compressed key-value cache at three memory budgets, on
Qwen2.5-1.5B-Instruct (evaluator and self-check use this same file).

The evaluator does not train. One stage, "evaluate":
  timed   import cache.py (no model is loaded by your code)                                        (killed at 120 s)
  finish  the evaluator loads the model (bf16, models_dir); for each budget b in BUDGETS (bits per cached element)
          and each held-out prompt: cache = make_cache(b, config); prefill the prompt with past_key_values=cache;
          the cache's stored bits per element = 8 x (bytes of every tensor reachable from the cache object) /
          (tokens x layers x key-value heads x head dimension x 2); then 32 greedy decoding steps with the same cache;
          the answer is scored with ruler_tasks.score. The same prompts are also run with an uncompressed cache
          (transformers DynamicCache) for reference.                                               (all ≤ 2,700 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/cache.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `ruler_tasks.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
