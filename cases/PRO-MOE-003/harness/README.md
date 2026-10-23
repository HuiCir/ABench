# PRO-MOE-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PRO-MOE-003 pipeline: audit of the pretraining process, held-out loss and compute per token of the delivered
mixture-of-experts model, and a sampled resume of the training run (evaluator and self-check use this file).

Stage "audit" (evaluator code only; pro_stages.audit): chain, traceability (the run starts from random
  initialisation: its start entry records no starting weights), code identity.
Stage "eval" (imports model.py of the submission, which is the code of the registered run):
  timed   build_model(); load checkpoint/model.pt (a state dict)
  finish  stored elements of model.pt; forward FLOPs per token on 8 held-out windows of 1,024 tokens
          (torch.utils.flop_counter, the routing the model actually performs); mean next-token loss on the
          10,485,760 held-out FineWeb tokens in windows of EVAL_SEQ_LEN                               (≤ 3,600 s)
Stage "resume" (pro_stages.resume): one registered checkpoint, picked with the secret number resume_u, resumed for at
  most 1 hour; the logged training loss must match within 1%.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `pro_audit.py`
- `pro_stages.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
