# K-FA-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
K-FA-003 pipeline: attention forward kernel, correctness and speed against PyTorch SDPA (evaluator and self-check).

One stage, "bench":
  setup   nothing (the CUDA context is created before the clock starts)
  timed   import kernel.py (module-level compilation or autotuning counts here, capped by the stage timeout)
  finish  for every held-out shape: random bf16 q, k, v from the shape's seed; your attention() vs an fp32 reference
          for accuracy; CUDA-event timing of your attention() and of SDPA (flash backend where available)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
