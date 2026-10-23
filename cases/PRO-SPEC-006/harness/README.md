# PRO-SPEC-006 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PRO-SPEC-006 pipeline: audit of the draft's training, batch-1 decoding time of your speculative-decoding server
against the evaluator's plain decoding server, a greedy-consistency check of the answers and a sampled resume of the
draft's training (evaluator and self-check use this file).

Stage "audit" (pro_stages.audit): chain, traceability (the draft run starts from random initialisation or from parts
  of Qwen3-8B copied inside the run: its start entry records no starting weights), code identity.
Stage "vanilla" (evaluator code only): baselines/vanilla/serve.py (vLLM, no speculation, one GPU) answers every
  held-out prompt one after another (MAX_NEW tokens, greedy); time per request.
Stage "spec": your serve.py with ABENCH_DRAFT_DIR=<submission>/checkpoint answers the same prompts the same way;
  time per request; the answers of check_idx are saved for the next stage.
Stage "check" (evaluator code only): serve_bench.greedy_check of the saved answers.
Stage "resume" (pro_stages.resume): one registered checkpoint of the draft's run, resumed for at most 1 hour; the
  logged loss must match within 5%.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/serve.py`
- `baseline/train.py`
- `baselines/vanilla/serve.py`
- `interface.md`
- `metrics.py`
- `pro_audit.py`
- `pro_stages.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `serve_bench.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
