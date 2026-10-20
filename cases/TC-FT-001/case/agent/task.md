# TC-FT-001: Text classification on AG News within a CPU budget

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov, *Bag of Tricks for Efficient Text Classification* (arXiv:1607.01759). Baseline used for comparison: a hashed unigram+bigram linear classifier in the spirit of the anchor (`interface/baseline/`).

## Task

Design a text classifier for the four-topic AG News task that reaches the **highest possible test accuracy** and still predicts quickly. Develop it on 8 CPU cores in the sandbox, train it there, and submit the trained model.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint and classifies the held-out test documents; it never retrains.
- Only the provided training documents may be used: no pretrained embeddings, language models, vocabularies built from external text, or external data.
- You may build on any public work from before the cutoff (linear models over n-grams, hashing, subword features, small neural encoders, ensembles), but the design report must state what you borrowed and what you changed. This task tests algorithm design under a tight CPU budget: feature design, model choice, optimisation and inference efficiency are all open.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only), from AG News (Zhang, Zhao & LeCun 2015; Hugging Face `fancyzhx/ag_news`):

| Key | Shape | Type | Description |
|---|---|---|---|
| `text` | (120000,) | str | Title and description of a news article, as distributed |
| `y` | (120000,) | int64 | Class 0 World, 1 Sports, 2 Business, 3 Sci/Tech |
| `dev_idx` | (6000,) | int64 | Dev split for self-checking (5% per class), used only by `selfcheck.py` |

The held-out set is the official AG News test split (7,600 documents, 1,900 per class); it is not in the sandbox.

## Interface

Implement `load(checkpoint_dir)` and `predict(model, texts)` in `model.py` at the root of the submission directory; `predict` receives a list of strings and returns integer labels (0–3), one per text. Your `train.py` writes `checkpoint/`; the evaluator imports `model.py` and calls `load` inside the timed phase.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Accuracy** (primary) | Top-1 accuracy on the 7,600 held-out test documents; higher is better | — |
| Prediction throughput | Held-out documents divided by the seconds `predict()` takes | Threshold: ≥ 2,000 docs/s |
| Macro-F1 | Unweighted mean F1 over the 4 classes | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py`) | reported (killed at 120 s) |

**Timing rules**

- Timing is done by the evaluator's parent process with its own clock (handshake protocol in `interface/abench_stage.py`), in a new process with 8 CPU threads (`OMP_NUM_THREADS=8`) and empty cache directories.
- `predict()` runs after the timed phase and must finish within 60 seconds.

**Decision**

- A submission is valid only if the throughput threshold holds; valid submissions are ranked by accuracy.
- If loading or prediction crashes, exceeds a time limit, or returns the wrong number of labels or labels outside 0–3, the submission is invalid.
- The final ABench score is driven mainly by accuracy, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 8 CPU cores, no GPU; the evaluation uses the same CPU allotment. No network and no package installation; available software: Python 3, NumPy, SciPy, scikit-learn.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 CPU-core-hours (one hour of the 8-core allotment).
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 100 MB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights, embeddings or vocabularies built from external text, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights, embeddings or vocabularies built from external text, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- `predict()` must not train on or adapt to the held-out texts beyond computing features of each document independently.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
