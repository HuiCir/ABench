# C-QNT-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
C-QNT-003 pipeline: score a weight-only quantized Qwen2.5-0.5B under a size budget, judged by perplexity.

The evaluator does not quantize. Your quantize.py and train.py write the quantized artifact to checkpoint/ in the
sandbox; it is the only thing the evaluator uses from training.
Stage "eval" (imports quantize.py; the original weights are NOT available)
  timed   load(checkpoint_dir, device) -> model
  finish  artifact size, perplexity on the held-out WikiText-2 test windows, prefill throughput
Stage "ref" (evaluator only; the frozen bf16 model is available, the submission is not imported)
  timed   load the bf16 model with transformers
  finish  its prefill throughput under the same conditions, for the throughput ratio
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/quantize.py`
- `baseline/train.py`
- `baselines/rtn/quantize.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
