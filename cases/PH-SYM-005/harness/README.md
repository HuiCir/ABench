# PH-SYM-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PH-SYM-005 pipeline: the evaluator gives your discovery method the noisy trajectories of 20 dynamical systems it
has never seen and scores the equations it returns (evaluator and self-check use this same file).

One stage, "discover" (imports discover.py):
  before  the trajectories of the held-out systems are placed in ctx.pre_dir (held_pre.npz); loading is not timed
  timed   for each system: discover(t, x, seed) -> coefficient matrix over systems.library(d)   (all twenty ≤ 1,200 s)
  finish  the coefficient matrices are returned to the evaluator, which compares their sparsity pattern with the
          true one and integrates them from new initial conditions (metrics.py)
The evaluator runs the anchor's sequentially thresholded least squares (baselines/stlsq) through the same stage.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/stlsq/discover.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `systems.py`

The held-out data, keys and judge configuration are never in this directory.
