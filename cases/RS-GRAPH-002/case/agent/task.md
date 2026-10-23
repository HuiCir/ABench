# RS-GRAPH-002: Collaborative filtering on Gowalla with Recall@20 over all items

Anchor: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang, *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (arXiv:2002.02126). Baseline used for comparison: implicit ALS with frequency-scaled regularisation, in NumPy (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **collaborative-filtering model** that, trained only on users' past check-ins, ranks all 40,981 Gowalla locations for each of 29,858 users so that the locations they visit in the held-out set appear in the **top 20** — measured by **Recall@20 over all items** (no sampled negatives).

- Submit a trained checkpoint and the code that loads it. At evaluation the evaluator calls `recommend()` once for all users; every item the user has not interacted with in the training data is a candidate (no sampled negatives).
- The data and split are those of NGCF and LightGCN (a random 80/20 split of each user's interactions); the 217,242 held-out interactions are not in the sandbox.
- You may build on any public work (matrix factorisation, graph convolution over the user-item graph, autoencoders, linear item-item models, sampling and loss design), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only): the training interactions of the LightGCN Gowalla split (`github.com/kuandeng/LightGCN`, Data/gowalla; check-ins of the SNAP loc-gowalla dataset, 10-core filtered).

| Key | Shape | Type | Description |
|---|---|---|---|
| `indptr` | (29,859,) | int64 | User u's training items are `items[indptr[u]:indptr[u + 1]]` |
| `items` | (810,128,) | int32 | Item ids in 0..40,980, sorted per user |
| `n_items` | () | int64 | 40,981 |

The held-out interactions (on average 7.3 per user, at least 1) are not in the sandbox. The self-check holds out 20% of each user's training items (`stages.make_dev`).

## Interface

Implement `load(checkpoint_dir, device) -> model` and `recommend(model, train, users, k=20) -> (len(users), 20) item ids` in `model.py`. `train` holds the arrays above; each row must contain 20 distinct item ids, best first; items in the user's training list count as misses.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Recall@20** (primary) | Per user, the number of held-out items among the 20 returned divided by the number of held-out items; mean over all 29,858 users; higher is better | — |
| NDCG@20 | Binary-relevance DCG of the 20 returned items divided by the ideal DCG over min(held-out items, 20) positions; mean over users | reported |
| Load time | Seconds from the start signal to the end of `load()` (parent clock) | Threshold: ≤ 300 s |
| Recommendation time | Seconds of the `recommend()` call | Threshold: ≤ 600 s |

**Timing and decision**

- Each run is a new process with empty cache directories, 8 CPU threads and the reference GPU, timed by the evaluator's parent process (`interface/abench_stage.py`).
- A submission is valid only if every threshold holds; valid submissions are ranked by Recall@20. Differences inside the 95% interval over users reported with the score are treated as ties.
- If a run crashes, exceeds a time limit, or returns lists of the wrong shape or with repeated items, the submission is invalid.
- The final ABench score is driven mainly by Recall@20, and also considers how different the model is from existing models and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and recommend()
├── train.py            # required: reproduces the checkpoint from the public data
├── src/                # optional: other .py modules
├── checkpoint/         # required: trained parameters, at most 2 GiB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; the checkpoint may hold any files.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by `train.py` from the public training interactions only; no external data, pretrained models or stored recommendation lists.
- `recommend()` must rank items with the checkpoint and the training interactions it receives; it must not read other files.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
