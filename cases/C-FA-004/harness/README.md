# C-FA-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
C-FA-004 pipeline: a trainable attention (forward and backward) for bf16 inputs: correctness of the output and the
three input gradients, speed against PyTorch SDPA, and peak memory (evaluator and self-check).

One stage, "bench":
  timed   import kernel.py (module-level compilation or autotuning counts here, capped by the stage timeout)
  finish  for every held-out shape: random bf16 q, k, v and output gradient from the shape's seed; out = attention(q, k,
          v, causal) and out.backward(dout); output and dq, dk, dv against an fp32 reference (autograd, one (batch,
          head) at a time); CUDA-event timing of forward + backward of yours and of SDPA (flash backend where
          available); peak extra memory of one forward + backward of each
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
