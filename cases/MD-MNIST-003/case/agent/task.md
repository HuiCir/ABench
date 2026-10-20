# MD-MNIST-003: Lightweight medical image classification on BloodMNIST and OrganAMNIST

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni, *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification* (arXiv:2110.14795). Baseline used for comparison: random ReLU features with a closed-form ridge classifier (`interface/baseline/`).

## Task

Design one image classification method that reaches the **highest mean test accuracy** on two 28×28 MedMNIST datasets: blood-cell microscopy (8 classes, RGB) and abdominal CT organ slices (11 classes, grayscale). Develop and train it on one 24GB GPU in the sandbox, then submit the trained models.

- Submit your model code, the trained checkpoint (covering both datasets) and the training program that produced it. The evaluator loads the checkpoint and classifies each official test split; it never retrains.
- Only the provided images may be used: no pretrained weights and no external data. The same method must serve both datasets (dataset-specific heads or settings chosen by your code from the data are fine).
- You may build on any public work from before the cutoff, but the design report must state what you borrowed and what you changed. MedMNIST is for research on lightweight models; it is **not** intended for clinical use.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/blood.npz` and `$ABENCH_DATA_DIR/organa.npz` (read-only), the official 28×28 releases of BloodMNIST and OrganAMNIST (MedMNIST, Zenodo record 10519652, CC BY 4.0):

| Key | Shape | Type | Description |
|---|---|---|---|
| `x_train`, `x_val` | (N, 28, 28, 3) blood; (N, 28, 28) organa | uint8 | Images, official train and validation splits |
| `y_train`, `y_val` | (N,) | int64 | Classes 0–7 (blood) or 0–10 (organa) |

The held-out sets are the official test splits; they are not in the sandbox. The self-check uses the validation split as its held-out set.

## Interface

Implement `load(checkpoint_dir, device)` and `predict(model, name, x)` in `model.py`. `predict` returns (M, n_classes) class scores for the images `x` of dataset `name` (`"blood"` or `"organa"`). Your `train.py` writes `checkpoint/` from the arrays above.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean accuracy** (primary) | Test accuracy of argmax(scores), averaged over the two datasets; higher is better | — |
| Mean AUC | Macro one-vs-rest ROC AUC per dataset, averaged over datasets (the MedMNIST v2 metrics are AUC and ACC) | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py`) | reported (killed at 120 s) |


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).
**Timing and decision**

- Loading and prediction run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`); creating the CUDA context is not timed. `predict()` runs after the timed phase; both test splits must be classified within 60 seconds.
- If loading or prediction crashes, exceeds a time limit, or returns scores of the wrong shape or non-finite values, the submission is invalid; valid submissions are ranked by mean accuracy.
- The final ABench score is driven mainly by mean accuracy, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 1 GPU-hour.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
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
- No pretrained weights, external data.
- `predict()` must not update the model on the test images (no test-time training or statistics pooled across the test split).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
