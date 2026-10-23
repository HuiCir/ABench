# FN-LTSF-005: A long-horizon forecasting architecture that beats a tuned linear model: five multivariate series, horizons 96 to 720

Anchor: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, *Are Transformers Effective for Time Series Forecasting?* (arXiv:2205.13504). Baseline used for comparison: a channel-shared ridge forecaster with last-value normalisation, lookback and penalty chosen on the validation rows for each dataset and horizon (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **forecasting architecture** for multivariate series that, trained in the sandbox, gives a **lower mean squared error than a well-tuned linear model** at horizons of 96, 192, 336 and 720 steps on five standard series — two hourly and two 15-minute transformer-station series (ETT) and an hourly electricity-load panel of 321 clients — measured as the **geometric mean over the 20 dataset-horizon pairs of your MSE divided by the reference's MSE**.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator gives your model the 720 standardised steps before each test origin and scores its forecasts; it never retrains.
- The comparison is the point of the case: published architectures often fail to beat a linear map from the recent past to the future once it is tuned. The reference is such a map, refitted on all public rows with its penalty chosen on validation; a ratio below 1 means your architecture beats it.
- Only the provided rows may be used: no pretrained forecasting models (foundation models such as Chronos, TimesFM or Moirai included) and no external data. `train.py` must apply the same procedure to every dataset; choices made from the data (for example on the validation rows) are fine, hand-set per-dataset settings are not.
- You may build on any public work (linear and decomposition models, patching and channel independence, reversible normalisation, MLP mixers, state-space and attention models, frequency-domain models), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/<dataset>.npz` (read-only), one per dataset, with `series` (T, C) float32 standardised by the training rows, `train_end` and `val_end` (row borders) and the training `mean` and `std`:

| Dataset | Source | Step | Channels | Public rows (train / validation) | Test rows |
|---|---|---|---|---|---|
| `etth1`, `etth2` | ETT (Zhou et al. 2021), CC BY-ND 4.0 | 1 hour | 7 | 8,640 / 2,880 | 2,880 |
| `ettm1`, `ettm2` | the same stations, 15 minutes | 15 minutes | 7 | 34,560 / 11,520 | 11,520 |
| `electricity` | UCI ElectricityLoadDiagrams20112014 (CC BY 4.0), 321 clients, hourly version of Lai et al. 2018 | 1 hour | 321 | 18,412 / 2,632 | 5,260 |

The splits are the ones used by most long-horizon papers (Informer, Autoformer, DLinear, PatchTST). The test rows follow the validation rows and are not in the sandbox. Test origins run from the first test row every 23 steps (ETTh), 95 steps (ETTm) or 167 steps (electricity) up to the last origin whose 720-step forecast ends inside the test rows: 94, 94, 114, 114 and 28 windows. These strides do not divide a day or a week, so the origins cover all hours and weekdays. `interface/ltsf.py` defines the splits, origins and the reference.

## Interface

Implement `load(checkpoint_dir, device)` and `predict(state, dataset, history, horizon)` in `model.py`. `dataset` is one of `"etth1"`, `"etth2"`, `"ettm1"`, `"ettm2"`, `"electricity"`; `history` is float32 (B, 720, C), the standardised steps before B forecast origins; `predict` returns float32 (B, horizon, C) for `horizon` in 96, 192, 336, 720; write a `train.py` that produces the checkpoint from the public data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **MSE ratio** (primary) | Geometric mean over the 20 dataset-horizon pairs of (your MSE) / (reference MSE) on the test windows, MSE averaged over windows, steps and channels on the standardised scale; lower is better, 1 is the reference | — |
| Per pair | MSE, MAE and the ratio for each dataset and horizon; the ratio per dataset; the share of pairs below 1; the worst ratio | reported |
| Reference | The reference's MSE for each pair | reported |
| Cost | Load seconds, prediction seconds for all windows, checkpoint size | reported |

No threshold applies to any metric: a submission is valid if it loads, returns finite forecasts of the right shape within the time limits and the code follows the rules (see below).

**Measurement and decision**

- Loading runs in a new process on one GPU (killed at 300 s); `predict` is then called per dataset and horizon on batches of at most 64 windows, all within 900 s.
- The reference is `ltsf.ridge_fit` with lookback 336: penalty chosen on the validation rows after a fit on the training rows, then refitted on training and validation rows together. Its errors on the test windows are computed when the case is frozen.
- The primary metric is reported with a 95% bootstrap interval over windows (resampled per dataset, jointly over horizons); submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by the MSE ratio, and also considers how different the architecture is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, at most 500 MB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights and no external data.
- `predict` forecasts each window from its own history; it may not use other windows of the batch, keep state between calls or fit anything to the test histories.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
