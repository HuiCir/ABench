# PH-SYM-005 interface

## Entry point: `discover.py`

```python
import systems                      # interface/systems.py is on sys.path

def discover(t, x, seed) -> "numpy.ndarray":
    """t: float64 (1000,) uniformly spaced times; x: float64 (8, 1000, d) noisy states of 8 trajectories.
    Return float (systems.n_terms(d), d): dx_i/dt = sum_k coef[k, i] * term_k(x) over systems.library(d).
    Absent terms must be exactly 0."""
```

- Term order of `systems.library(d)`: the constant, the d linear terms, the monomials of degree 2 and 3 (in
  `itertools.combinations_with_replacement` order), then sin(x_0..x_{d-1}) and cos(x_0..x_{d-1});
  `systems.term_names(d)` gives readable names. d is 2, 3 or 4 (14, 26 or 43 terms).
- The evaluator calls `discover` once per system inside the timed phase, with seeds `run_seed`, `run_seed + 1`, ...
- `discover.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

`systems.py` (generator and library), `stages.py` (the stage and the development set), `metrics.py` (recovery,
F1, extrapolation), `baselines/stlsq/discover.py`, `abench_metrics.py`, `abench_stage.py` and `selfcheck.py`.

## Available software

Python 3, NumPy, SciPy, scikit-learn. Nothing can be installed.
