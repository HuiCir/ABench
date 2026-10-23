# PH-SYM-005: Discovering the governing equations of 20 unseen dynamical systems from noisy trajectories

Anchor: Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems* (arXiv:1509.03580). Baseline used for comparison: the anchor's SINDy with sequentially thresholded least squares (threshold 0.1, finite-difference derivatives), run by the evaluator on the same systems (`interface/baselines/stlsq/`).

## Task

Design a method that **discovers the governing equations** of a dynamical system from a few noisy trajectories, and that gets the **structure exactly right** — which terms appear in which equation — on the largest fraction of **20 systems it has never seen**.

- Submit code only (`discover.py` with `discover()`). At evaluation the evaluator gives it, system by system, 8 trajectories of each of 20 held-out systems and receives a coefficient matrix over a fixed term library (constant, monomials up to degree 3, sin and cos of each variable).
- The held-out systems come from the generator in `interface/systems.py` with a seed you do not know: 11 come from known families (oscillators, predator-prey, chemical and neural oscillators, chaotic attractors) with random parameters, a random order of the variables and random variable scales; 9 are random sparse polynomial systems in 3 or 4 variables. You never see which family a system comes from; the generator itself is yours to use for development with any other seed.
- Measurement noise varies between systems (0 to 1% of each variable's standard deviation) and is not given to you.
- You may build on any public work (sparse regression, model selection, weak or integral formulations, ensembles, derivative estimation), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data directory: all training and development data come from `interface/systems.py`.

| Function | Returns |
|---|---|
| `systems.sample_set(seed, n)` | `n` systems drawn as the held-out set is drawn (no known family repeats, at least `n // 3` random sparse) |
| `systems.sample_system(rng, family=None)` | one system: `coef` (true coefficients), `t` (1,000 time points), `x` (8, 1,000, d) noisy trajectories, `noise`, `family`, `test_x0` |
| `systems.library(d)`, `systems.term_names(d)`, `systems.eval_library(x)` | the term library of a d-variable system, its readable names, and the term matrix Θ(x) |
| `systems.simulate(coef, x0, duration)` | integrates a coefficient matrix from an initial condition |

The evaluator's 20 systems are drawn with a secret seed; their true coefficients, held-out initial conditions and family labels are never in the sandbox. The self-check uses 20 systems from a fixed development seed.

## Interface

Implement `discover(t, x, seed) -> coef` in `discover.py`: `t` is a (1000,) float64 array of uniformly spaced times, `x` a float64 array (8, 1000, d) of noisy states of 8 trajectories, and `coef` a float array (n_terms(d), d) with `dx_i/dt = Σ_k coef[k, i] · term_k(x)` over `systems.library(d)`. Terms you consider absent must be exactly 0.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Recovery rate** (primary) | Fraction of the 20 systems for which the set of non-zero coefficients equals the true set in every equation; higher is better | — |
| Term F1 | F1 of the non-zero terms against the true terms, per system, averaged | reported |
| Extrapolation error | The discovered and the true system are integrated from 4 new initial conditions over the training duration; RMS difference divided by the true trajectory's standard deviation, averaged over variables, capped at 1 (1 if the discovered system leaves ten times the true range or cannot be integrated); mean over systems | reported |
| Coefficient error | ‖Ĉ − C‖ / ‖C‖, median over systems | reported |
| Discovery time | Seconds of the 20 `discover()` calls (parent clock) | Threshold: ≤ 1,200 s |

**Timing and decision**

- The run is a new process with empty cache directories and 8 CPU threads, timed by the evaluator's parent process (`interface/abench_stage.py`); loading the trajectories is not timed, importing `discover.py` and the 20 `discover()` calls are.
- A submission is valid only if the time threshold holds; valid submissions are ranked by recovery rate, ties broken by mean term F1, then by the lower extrapolation error.
- If the run crashes, exceeds the time limit, or returns a matrix of the wrong shape or with non-finite values, the submission is invalid.
- The final ABench score is driven mainly by the recovery rate relative to the baseline, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The evaluator also runs the anchor's STLSQ baseline in `interface/baselines/stlsq/` on the same 20 systems and reports your recovery rate, term F1 and extrapolation error relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 8 CPU cores, no GPU; the evaluation uses the same CPU allotment. Available software: Python 3, NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 32 CPU-core-hours (the whole session on the 8-core allotment); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── discover.py         # required: discover()
├── src/                # optional: other .py modules imported by discover.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total; no stored equations, coefficient tables or lists of family templates.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The submission is code only and must discover each system from its trajectories: no tables of the generator's families or their coefficient patterns, and no code that inspects the generator's source or state.
- `discover()` may import `systems` (term library, integrator) but must not call `sample_system` or `sample_set`.
- Recognising a family is acceptable only as a consequence of a general method (for example a sparsity prior); the design report must explain any prior over terms.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
