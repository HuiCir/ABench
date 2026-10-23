# PH-MLIP-003: A machine-learned interatomic potential for small molecules from 1,000 DFT configurations: force accuracy and 10 ps of stable dynamics (rMD17)

Anchor: Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, Gábor Csányi, *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields* (arXiv:2206.07697). Baseline used for comparison: a random-feature force field on inverse distances fitted to energies and forces, in NumPy (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design and train a **machine-learned interatomic potential** that, from 1,000 DFT-labelled configurations per molecule, predicts the **forces of five small organic molecules most accurately** and **keeps molecular dynamics stable** for 10 ps at 500 K.

- Submit the trained checkpoint and the code that loads it. The evaluator calls `predict()` on 1,000 held-out configurations per molecule (aspirin, ethanol, malonaldehyde, naphthalene, toluene; the rMD17 test split 01), then runs Langevin dynamics with your forces from 4 of them per molecule.
- A trajectory is unstable from the first check (every 10 fs) at which some bond length deviates from its training mean by more than 0.5 Å; at least 75% of the 20 trajectories must stay stable for the whole 10 ps.
- You may use one model per molecule or one for all; forces may be the gradient of an energy or predicted directly (the dynamics uses whatever `predict` returns).
- You may build on any public work (equivariant message passing, atomic cluster expansion, kernel and gradient-domain methods, descriptors), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/<molecule>.npz` for the five molecules (read-only): the training configurations of the official rMD17 split 01 (Christensen & von Lilienfeld 2020, figshare 10.6084/m9.figshare.12672038, CC0; DFT PBE/def2-SVP).

| Key | Shape | Type | Description |
|---|---|---|---|
| `z` | (N,) | int64 | Atomic numbers (H, C, O); N = 21, 9, 9, 18, 15 |
| `pos` | (1000, N, 3) | float64 | Positions in Å |
| `energy` | (1000,) | float64 | Energies in kcal/mol |
| `forces` | (1000, N, 3) | float64 | Forces in kcal/mol/Å |

The 1,000 test configurations of split 01 (their energies and forces) are not in the sandbox. The self-check holds out 200 of each molecule's training configurations.

## Interface

Implement `load(checkpoint_dir, device)` and `predict(model, molecule, z, pos) -> (energy, forces)` in `model.py`: `pos` is a float64 (B, N, 3) array in Å (B ≤ 100), the result float arrays (B,) in kcal/mol and (B, N, 3) in kcal/mol/Å. The atom order is that of the training data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean force MAE** (primary) | Mean absolute error of the force components on the 1,000 held-out configurations, in meV/Å (1 kcal/mol = 43.364 meV), averaged over the five molecules; lower is better | — |
| Energy and force MAE per molecule | meV and meV/Å | reported |
| Stable fraction | Fraction of the 20 trajectories (4 per molecule) stable for the whole 10 ps | Threshold: ≥ 0.75 |
| Mean stable time | Mean over trajectories of the time until instability (capped at 10 ps) | reported |
| Inference time | Milliseconds per configuration in batches of 100 | reported |
| Load time | Seconds from the start signal to the return of `load()` | reported (killed at 300 s) |

**Dynamics** (`interface/md.py`): Langevin dynamics with the BAOAB splitting, time step 0.5 fs, friction 0.01 fs⁻¹, 500 K, 20,000 steps, velocities drawn from the Maxwell–Boltzmann distribution with the evaluator's seed; bonds are atom pairs closer than 1.25 × the sum of covalent radii in the first training configuration.

**Timing and decision**

- Loading, prediction and the dynamics run in a new process with empty cache directories on the reference GPU and 8 CPU threads, timed by the evaluator's parent process (`interface/abench_stage.py`); prediction and the five dynamics runs must finish within 2,400 seconds.
- A submission is valid only if the stability threshold holds; valid submissions are ranked by mean force MAE. Differences inside the 95% interval over configurations reported with the score are treated as ties.
- If loading, prediction or the dynamics crashes, exceeds a time limit or returns arrays of the wrong shape, the submission is invalid.
- The final ABench score is driven mainly by the mean force MAE, and also considers how different the model is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: reproduces the checkpoint from the public data
├── src/                # optional: other .py modules
├── checkpoint/         # required: trained parameters, at most 500 MB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; the checkpoint may hold any files.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by `train.py` from the public training configurations only; no external data, other DFT labels or pretrained potentials.
- `predict` must compute energies and forces from the positions with the trained model; it must not use the molecule's identity beyond selecting the model for it, or detect the dynamics to change its behaviour (for example by adding restraints).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
