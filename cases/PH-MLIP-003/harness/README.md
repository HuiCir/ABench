# PH-MLIP-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PH-MLIP-003 pipeline: energies and forces of five molecules from rMD17, and 10 ps of molecular dynamics driven by
your potential (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  setup   load the public training configurations (for the bond list of the stability check; untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                        (load time, killed at 300 s)
  finish  for each molecule: predict(model, molecule, z, pos) on the held-out configurations (energies and forces,
          timed per configuration); then md.run: Langevin dynamics at 500 K, 0.5 fs steps, for MD_STEPS steps from 4
          held-out configurations, with the forces of predict(); the stable time of each trajectory  (all ≤ 2,400 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/ff.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `md.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
