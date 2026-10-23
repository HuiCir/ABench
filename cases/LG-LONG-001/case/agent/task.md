# LG-LONG-001: Classifying long court decisions: violated articles in European Court of Human Rights cases

Anchor: Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, Nikolaos Aletras, *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English* (arXiv:2110.00976). Baseline used for comparison: tf-idf over the whole case with one logistic regression per article (`interface/baseline/`).

## Task

Design a **long-document classification architecture** that, given the facts of a European Court of Human Rights case (often several thousand tokens), predicts which of ten articles of the Convention the court found violated, with the **highest mean of micro- and macro-F1** on held-out cases. Develop and train it on one 24GB GPU in the sandbox, then submit the trained model.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint and labels the 1,000 held-out cases; it never retrains.
- Most cases exceed the 512-token window of the provided encoder: how you cover the rest of the document (hierarchical encoding of paragraphs, sparse or sliding attention, retrieval of the relevant paragraphs, or no encoder at all) is the design question.
- Only the provided data and the provided legal-bert-base-uncased may be used: no other pretrained model, no external data, no case law outside the training split.
- You may build on any public work (hierarchical transformers, Longformer-style attention, label-wise attention, linear models over n-grams), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/ecthr_a.npz` (read-only), LexGLUE ECtHR task A (Hugging Face `coastalcph/lex_glue`, config `ecthr_a`, CC BY 4.0):

| Key | Shape | Type | Description |
|---|---|---|---|
| `train_text_utf8`, `train_text_offsets` | packed, 9,000 cases | uint8, int64 | Facts of each case, paragraphs separated by a blank line; `interface/textpack.py` has `unpack()` |
| `train_y` | (9000, 10) | int8 | 1 if the court found the article violated; a row may be all zero |
| `val_text_utf8`, `val_text_offsets`, `val_y` | 1,000 cases | as above | The public validation split |
| `labels` | (10,) | str | Articles 2, 3, 5, 6, 8, 9, 10, 11, 14 and Article 1 of Protocol 1 |

`$ABENCH_MODELS_DIR/legal-bert-base-uncased/` holds the encoder and tokenizer (Hugging Face `nlpaueb/legal-bert-base-uncased`, CC BY-SA 4.0; 512-token positions). The held-out set is the official LexGLUE test split (1,000 cases, judged later than the training cases); it is not in the sandbox. The self-check uses the validation split as its held-out set.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `predict(model, texts)` in `model.py`; `predict` receives a list of case texts and returns an (M, 10) array of 0/1 decisions. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean F1** (primary) | Mean of micro-F1 and macro-F1 over the 10 articles plus a "no violation" label that is 1 when a row has no positive label (computed separately for gold and prediction, as in LexGLUE); higher is better | — |
| Micro-F1, macro-F1 | The two components | reported |
| Load time, prediction time | Seconds to `load()` and to label the 1,000 cases | reported (load killed at 300 s) |


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).

**Timing and decision**

- Loading and prediction run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`); creating the CUDA context is not timed. `predict()` runs after the timed phase and must finish within 600 seconds.
- If loading or prediction crashes, exceeds a time limit, or returns an array of the wrong shape or values other than 0/1, the submission is invalid; valid submissions are ranked by mean F1. Differences inside the 95% interval reported with the score are treated as ties.
- The final ABench score is driven mainly by mean F1, and also considers how different the architecture is from existing long-document models and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, and Hugging Face transformers, tokenizers and safetensors (versions frozen at activation; offline); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 1,000 MiB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights other than the provided legal-bert-base-uncased, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- `predict()` labels each case on its own; it may not fit or adapt anything to the held-out cases.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
