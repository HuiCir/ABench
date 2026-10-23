# BI-PROT-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
BI-PROT-001 pipeline: fitness prediction for single-substitution mutants of 30 held-out deep mutational scanning
assays from ProteinGym, with a frozen ESM-2 and 64 labelled mutants per assay (evaluator and self-check use this same
file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model           (load time, killed at 300 s)
  finish  for each held-out assay: predict(model, assay) -> one score per query mutant, where assay holds the
          wild-type sequence, the query mutants, 64 labelled mutants (few-shot) and the coarse selection type
                                                                                          (all 30 ≤ 1,800 s)
The labels of the query mutants never leave the evaluator.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
