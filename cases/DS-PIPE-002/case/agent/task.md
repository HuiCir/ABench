# DS-PIPE-002: A tabular learning pipeline across six OpenML datasets on CPU

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, Alexander Smola, *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data* (arXiv:2003.06505). Baseline used for comparison: histogram gradient-boosted trees with early stopping (`interface/baseline/`).

## Task

Design one tabular learning pipeline (preprocessing, features, models, ensembling, model selection) and apply it to each of six binary classification datasets to reach the **highest mean test accuracy** across them. Develop and train it on 8 CPU cores in the sandbox, then submit the trained models.

- Submit your model code, the trained checkpoint (one fitted model per dataset) and the training program that produced it. The evaluator loads the checkpoint and predicts each dataset's held-out rows; it never retrains.
- Only the provided training rows may be used: no pretrained models and no external data. `train.py` must apply the same procedure to every dataset; choices made automatically from the data (for example by validation) are fine, hand-picked per-dataset settings are not.
- You may build on any public work from before the cutoff (gradient-boosted trees, neural tabular models, stacking, AutoML search), but the design report must state what you borrowed and what you changed. The task tests robust design under the sandbox's compute budget.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/d<id>.npz` (read-only), one file per dataset of OpenML suite 337 ("Tabular benchmark numerical classification", Grinsztajn et al. 2022). All features are numeric (float32, NaN allowed); labels are 0/1.

| File | Dataset | Instances in OpenML | Features |
|---|---|---|---|
| `d44120.npz` | electricity | 38,474 | 7 |
| `d44122.npz` | pol | 10,082 | 26 |
| `d44123.npz` | house_16H | 13,488 | 16 |
| `d44125.npz` | MagicTelescope | 13,376 | 10 |
| `d44126.npz` | bank-marketing | 10,578 | 7 |
| `d44128.npz` | MiniBooNE | 72,998 | 50 |

Each file has `x` (N, F) float32, `y` (N,) int64 and `dev_idx` (10% per class, used only by `selfcheck.py`). Per dataset, the held-out split is a stratified min(30%, 5,000) rows and the public training split is min(rest, 10,000) rows, drawn with a fixed seed. The held-out rows are not in the sandbox.

## Interface

Implement `load(checkpoint_dir)` and `predict_proba(model, dataset, X)` in `model.py`. `dataset` is the OpenML id (an int from the table above), `X` a float32 array; `predict_proba` returns an (M, 2) array of class probabilities. Your `train.py` writes `checkpoint/`; the evaluator calls `load` inside the timed phase.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean accuracy** (primary) | Accuracy of argmax(`predict_proba`) on each dataset's held-out rows, averaged over the 6 datasets; higher is better | — |
| Mean ROC AUC | ROC AUC of p(1) − p(0) per dataset, averaged over datasets | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py`) | reported (killed at 120 s) |
| Prediction time | Seconds `predict_proba()` takes for all six datasets | reported (limit 120 s) |


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).
**Timing rules**

- Loading and prediction run in a new process with 8 CPU threads (`OMP_NUM_THREADS=8`) and empty cache directories, timed by the evaluator's parent process (`interface/abench_stage.py`).

**Decision**

- If loading or prediction crashes, exceeds a time limit, or returns probabilities of the wrong shape, negative or non-finite, the submission is invalid; valid submissions are ranked by mean accuracy.
- The final ABench score is driven mainly by mean accuracy, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 8 CPU cores, no GPU; the evaluation uses the same CPU allotment. No network and no package installation; available software: Python 3, NumPy, SciPy, scikit-learn.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 16 CPU-core-hours (two hours of the 8-core allotment).
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict_proba()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 200 MB
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
- No per-dataset hand-tuned settings: the dataset id may select which fitted model to use, but `train.py` must run the same procedure on every dataset.
- `predict_proba()` must not fit to the held-out rows.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
