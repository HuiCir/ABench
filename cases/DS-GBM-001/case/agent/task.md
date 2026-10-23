# DS-GBM-001: A tabular learner trained by the evaluator on twenty unseen datasets

Anchor: Tianqi Chen, Carlos Guestrin, *XGBoost: A Scalable Tree Boosting System* (arXiv:1603.02754). Baseline used for comparison: scikit-learn HistGradientBoosting with default settings (`interface/baselines/hgb/`), trained by the evaluator under the same conditions.

## Task

Design a **tabular learning algorithm** — preprocessing, model family, hyperparameter selection and ensembling, all decided by your code from the data — that the evaluator trains on **20 datasets it has never shown you**, each within a fixed CPU time budget, and that reaches the **highest mean test score** across them.

- Submit code only (`model.py` with `fit()` and `predict()`). At evaluation your `fit()` is called on the training split of each held-out dataset, on 8 CPU cores with about 60 seconds per dataset (1,200 seconds for all 20), then `predict()` on its test split; this is repeated for two seeds unknown to you.
- The 20 held-out datasets come from the same tabular benchmark as the 10 dev datasets you get, but are different datasets: binary classification and regression, about 700 to 10,000 training rows, 3 to several hundred numeric features (some are one-hot or ordinal codes of categorical columns), missing values possible.
- Your code must decide everything from the training data it receives; nothing may depend on recognising a particular dataset.
- You may build on any public work (gradient-boosted trees, neural tabular models, AutoML search and portfolios, stacking), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/dev_<name>.npz` (read-only): ten dev datasets from the tabular benchmark of Grinsztajn et al. 2022 (Hugging Face `inria-soda/tabular-benchmark`): default-of-credit-card-clients, heloc, eye_movements and california (classification); abalone, cpu_act, wine_quality, house_sales, elevators and Bike_Sharing_Demand (regression).

| Key | Shape | Type | Description |
|---|---|---|---|
| `x_train`, `x_test` | (n, F) | float32 | Features, NaN allowed |
| `y_train`, `y_test` | (n,) | int64 (0/1) or float64 | Target |
| `task` | (1,) | int64 | 0 classification, 1 regression |

Each dataset was shuffled with a fixed seed; the test split is min(30%, 5,000) rows (stratified for classification) and the training split min(rest, 10,000) rows. The held-out datasets are split the same way; none of them, and no part of their data, is in the sandbox. The self-check trains your learner on the dev training splits and scores the dev test splits, exactly as the evaluator does on the held-out ones.

## Interface

Implement `fit(x, y, task, budget_seconds, seed) -> model` and `predict(model, x) -> array` in `model.py`. `task` is `"classification"` or `"regression"`; `predict` returns the probability of class 1 (classification) or the predicted value (regression), one per row.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean score** (primary) | Per held-out dataset, test accuracy (classification, class 1 when the predicted probability is ≥ 0.5) or R² (regression, counted as 0 when negative); averaged over the 20 datasets and the two seeds; higher is better | — |
| Fit time | Seconds of the whole `fit()` loop over the 20 datasets (parent clock) | Threshold: ≤ 1,200 s |
| Per-dataset scores, AUC, raw R² | Reported for each dataset | reported |
| Relative to the baseline | Mean-score difference and fraction of datasets won against HistGradientBoosting | reported |

**Timing and decision**

- Each run is a new process with empty cache directories and 8 CPU threads, timed by the evaluator's parent process (`interface/abench_stage.py`). The training splits are in place before the clock starts; importing `model.py` and every `fit()` call are timed. Test inputs are given only afterwards; all `predict()` calls must finish within 300 seconds.
- A submission is valid only if the fit-time threshold holds in both runs; valid submissions are ranked by mean score. Differences inside the 95% interval reported with the score are treated as ties.
- If any call crashes, a time limit is exceeded, or predictions have the wrong length, are non-finite or (classification) lie outside [0, 1], the submission is invalid.
- The final ABench score is driven mainly by mean score, and also considers how different the method is from existing tabular learners and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (datasets, seeds, CPU allotment), the evaluator also trains the HistGradientBoosting baseline in `interface/baselines/hgb/` and reports your mean score relative to it and the fraction of datasets on which you beat it.
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
├── model.py            # required: fit() and predict()
├── src/                # optional: other .py modules imported by model.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total; no trained parameters or stored predictions.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The submission is code only; every model is trained by the evaluator from the training split it receives.
- `fit()` may use only its arguments; `predict()` only the model and the rows it is given, each row independently of the others (no fitting on test inputs).
- No pretrained tabular models, embeddings or meta-learned weights from outside data; no code that recognises specific datasets (by shape, values or statistics) to switch to hand-chosen settings. General rules that depend on dataset size, feature count or task are fine.
- No subprocesses that outlive `fit()` or `predict()`, no reading or writing files outside the stage's working directory.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
