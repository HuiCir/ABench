# PH-NO-001: A neural surrogate for 1D Burgers' equation (PDEBench)

Anchor: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar, *Fourier Neural Operator for Parametric Partial Differential Equations* (arXiv:2010.08895). Baseline used for comparison: a linear per-wavenumber autoregressive operator in Fourier space (`interface/baseline/`).

## Task

Design a surrogate model for the 1D viscous Burgers' equation (ν = 0.001, periodic domain) that predicts the next 31 frames of a trajectory from its first 10 frames with the **lowest normalised RMSE**, and still works when the same trajectories are given on a twice finer grid. Develop and train it on one 24GB GPU in the sandbox, then submit the trained model.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint and rolls out 1,000 held-out trajectories on the 256-point grid and again on a 512-point grid; it never retrains.
- Only the provided trajectories may be used: no pretrained weights, no external simulation data. Calling a numerical PDE solver at prediction time is not allowed; the model must be learned.
- You may build on any public work (neural operators, convolutional and attention surrogates, autoregressive training schemes, physics-informed losses), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only), from PDEBench (Takamoto et al. 2022), file `1D_Burgers_Sols_Nu0.001.hdf5` (10,000 trajectories on 1,024 points × 201 time steps), subsampled every 4th point in space and every 5th step in time:

| Key | Shape | Type | Description |
|---|---|---|---|
| `u` | (9000, 41, 256) | float32 | Trajectories: 41 frames on a 256-point periodic grid |
| `x` | (256,) | float32 | Grid coordinates |
| `t` | (41,) | float32 | Frame times |
| `dev_idx` | (900,) | int64 | Dev trajectories for self-checking, used only by `selfcheck.py` |

The held-out set is 1,000 further trajectories of the same file (fixed random split), given as their first 10 frames on the 256-point grid and on the 512-point grid (every 2nd point of the original); it is not in the sandbox.

## Interface

Implement `load(checkpoint_dir, device)` and `predict(model, u_in)` in `model.py`. `u_in` is float32 (M, 10, X) with X = 256 or 512; `predict` returns float32 (M, 31, X), the frames that follow. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **nRMSE** (primary) | Per trajectory, ‖prediction − truth‖₂ / ‖truth‖₂ over all 31 predicted frames and 256 points; mean over the 1,000 trajectories; lower is better | — |
| nRMSE at 512 points | The same on the 512-point grid (resolution transfer) | reported |
| nRMSE of the final frame / first 10 frames | Long- and short-horizon error | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py`) | reported (killed at 120 s) |


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).
**Timing and decision**

- Loading and prediction run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`); creating the CUDA context is not timed. Both `predict()` calls run after the timed phase and must finish within 120 seconds together.
- If loading or prediction crashes, exceeds a time limit, or returns predictions of the wrong shape or non-finite values, the submission is invalid; valid submissions are ranked by nRMSE.
- The final ABench score is driven mainly by nRMSE, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `darus.uni-stuttgart.de`.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 500 MB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- No pretrained weights, external simulation data; no numerical PDE solver inside `predict()`.
- `predict()` must treat each trajectory independently.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
