# X-ROBUST-009: An intent classifier that knows when not to answer: selective risk on CLINC150 with out-of-scope, shifted and noisy queries

Anchor: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, *On Calibration of Modern Neural Networks* (arXiv:1706.04599). Baseline used for comparison: TF-IDF word and character n-grams with a logistic regression over the 150 intents, temperature scaling and an abstain threshold fitted on the validation split (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design an **intent classifier with calibrated confidence and an abstain decision** for the 150 intents of CLINC150 that **keeps its selective risk low** when the queries it receives mix in-scope requests, **out-of-scope requests, requests from another banking assistant's domain, and requests with typos** — measured by the **area under the risk-coverage curve** over all of them.

- Submit a trained checkpoint, the code that loads it and the training program that produced it. The evaluator calls `predict()` on batches of held-out queries; for each it gets the best in-scope intent, a confidence that this answer is correct, and whether the model abstains.
- Out-of-scope queries have no correct intent: answering them is always an error, so the confidence must rank them below the in-scope queries the model gets right. The held-out mix contains query types that are not in the training data.
- The provided roberta-base may be fine-tuned, but nothing else pretrained may be used.
- You may build on any public work (temperature and vector scaling, label smoothing and focal losses, ensembles, Mahalanobis and energy scores, outlier exposure, selective classification), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/clinc_public.npz` (read-only): CLINC150 (Larson et al. 2019, `github.com/clinc/oos-eval` @828f8093, data_full.json, CC BY 3.0), packed texts (`interface/textpack.py`):

| Key | Rows | Description |
|---|---|---|
| `train_text_*`, `train_label` | 15,100 | 100 queries per intent and 100 out-of-scope queries (label −1) |
| `val_text_*`, `val_label` | 3,100 | 20 per intent and 100 out-of-scope |

`$ABENCH_DATA_DIR/intents.json` lists the 150 intent names in label order. `$ABENCH_MODELS_DIR/roberta-base/` holds the encoder and tokenizer (MIT).

The held-out mix (13,080 queries) is not in the sandbox: the CLINC150 test set (4,500 in-scope, 1,000 out-of-scope), the Banking77 test set (Casanueva et al. 2020, CC BY 4.0; 3,080 banking queries, all outside CLINC's intents), and the 4,500 in-scope test queries with character noise (deleted, doubled, swapped or neighbouring-key letters at about 6% of letters, drawn with a secret seed). The self-check uses the validation split, plus a noisy copy of its in-scope queries.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `predict(state, texts) -> (labels, confidence, abstain)` in `model.py`: for a list of strings, `labels` are intent indices 0–149 (your best in-scope answer, always given), `confidence` your probability in [0, 1] that this answer is correct, and `abstain` booleans; write a `train.py` that produces the checkpoint from the public data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **AURC** (primary) | Area under the risk-coverage curve over all 13,080 held-out queries: queries sorted by confidence, risk at coverage k/N = fraction of errors among the k most confident, averaged over k; an error is a wrong intent or any answer to an out-of-scope or Banking77 query; lower is better | — |
| ECE | Expected calibration error (15 equal-width bins) of the confidence on the clean in-scope queries | reported |
| Out-of-scope AUROC | Confidence of in-scope (clean and noisy) vs out-of-scope and Banking77 queries | reported |
| Accuracy | Intent accuracy and macro-F1 on clean in-scope queries; accuracy on noisy ones | reported |
| Decision accuracy | Correct if an in-scope query is answered with the right intent or an out-of-scope/Banking77 query is abstained on; overall and per group; over-abstention on in-scope queries | reported |
| Cost | Milliseconds per query in batches of 512; checkpoint size | reported |

No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and the code follows the rules (see below).

**Measurement and decision**

- `predict` is called in batches of 512 queries in a fixed random order, so every batch mixes the four groups; loading must finish within 600 s and all queries within 900 s, on one GPU.
- The primary metric is reported with a 95% bootstrap interval over queries; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by AURC, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: reproduces the checkpoint from the public data
├── src/                # optional: other .py modules
├── checkpoint/         # required: trained parameters, at most 600 MB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Train only on the public data and the provided roberta-base; no other datasets, pretrained models or lists of out-of-scope phrases.
- `predict` must treat each query on its own: no use of the batch's composition, other queries or their order.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
