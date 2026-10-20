# LG-HOLD-002: Legal holding identification on CaseHOLD

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Lucia Zheng, Neel Guha, Brandon R. Anderson, Peter Henderson, Daniel E. Ho, *When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset* (arXiv:2104.08671). Baseline used for comparison: lexical-overlap features with a learned conditional-logit scorer (`interface/baseline/`).

## Task

Design a method that reads the citing context of a U.S. court decision and picks, among five candidate holding statements, **the holding of the cited case**, with the **highest macro-F1** on the CaseHOLD test split. Develop and train it, including any fine-tuning of the one provided pretrained encoder (roberta-base), on one 24GB GPU in the sandbox, then submit the trained model.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint (with read access to the provided roberta-base files) and scores the five candidates of every test question; it never retrains.
- The only pretrained weights allowed are the roberta-base files in `$ABENCH_MODELS_DIR/roberta-base` (no legal-domain checkpoints, no other models, no external text). Your checkpoint may contain fine-tuned copies of them.
- You may build on any public work from before the cutoff (cross-encoders for multiple choice, domain-adaptive objectives, parameter-efficient tuning, lexical features), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only), the "all" configuration of CaseHOLD (Hugging Face `casehold/casehold`; about 42.5k train and 5.3k validation questions):

| Key | Shape | Type | Description |
|---|---|---|---|
| `prompt` | (N,) | str | Citing context, with the holding of the cited case replaced by a placeholder |
| `holdings` | (N, 5) | str | Five candidate holding statements |
| `y` | (N,) | int64 | Index 0–4 of the correct holding |
| `val_prompt`, `val_holdings`, `val_y` | — | — | The official validation split, same formats |

`$ABENCH_MODELS_DIR/roberta-base/` holds the frozen roberta-base weights and tokenizer (Hugging Face `FacebookAI/roberta-base`, MIT). The held-out set is the official test split (about 5.3k questions); it is not in the sandbox. The self-check uses the validation split as its held-out set.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `predict(model, prompts, holdings)` in `model.py`. `predict` returns float (M, 5) scores; the argmax is the chosen holding. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Macro-F1** (primary) | Unweighted mean F1 over the five answer positions of argmax(scores) on the test split (the CaseHOLD paper's metric); higher is better | — |
| Accuracy | Fraction of questions answered correctly | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py` and loading weights) | reported (killed at 300 s) |
| Prediction time | Seconds `predict()` takes for the test split | reported (limit 300 s) |


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).
**Timing and decision**

- Loading and prediction run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`); creating the CUDA context is not timed. `predict()` runs after the timed phase and must finish within 300 seconds.
- If loading or prediction crashes, exceeds a time limit, or returns scores of the wrong shape or non-finite values, the submission is invalid; valid submissions are ranked by macro-F1.
- The final ABench score is driven mainly by macro-F1, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy, and Hugging Face transformers, tokenizers and safetensors (versions frozen at activation; offline mode).
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 1000 MB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights other than the provided roberta-base, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- The only pretrained weights allowed are those in `$ABENCH_MODELS_DIR/roberta-base`; no external text.
- `predict()` must answer each question independently, without training on the test questions.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
