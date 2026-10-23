# PRO-ARL-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PRO-ARL-001 pipeline: audit of the training process, evaluation of the delivered checkpoint on held-out
competition problems with the Python tool, and a sampled resume of the training run (evaluator and self-check).

Stage "audit" (evaluator code only; pro_stages.audit): chain, traceability to Qwen3-4B-Base, code identity.
Stage "eval" (evaluator code only; the checkpoint is loaded by vLLM, no submission code runs):
  the delivered checkpoint/ must have the architecture of the starting model (config.json fields in ARCH_FIELDS);
  N_SAMPLES answers per held-out problem with math_tool.generate (tools on, sampling of math_tool.SAMPLING, at most
  16,384 generated tokens); the evaluator checks the answers                                       (≤ 10,800 s)
Stage "resume" (runs the submission's registered training command; pro_stages.resume): one registered checkpoint,
  picked with the secret number resume_u, resumed for at most 1 hour; the logged reward must match.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/src/math_tool.py`
- `baseline/src/rollout.py`
- `baseline/train.py`
- `interface.md`
- `math_tool.py`
- `metrics.py`
- `pro_audit.py`
- `pro_stages.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
