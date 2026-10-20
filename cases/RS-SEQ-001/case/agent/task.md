# RS-SEQ-001: Next-item recommendation on MovieLens-1M

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Wang-Cheng Kang, Julian McAuley, *Self-Attentive Sequential Recommendation* (arXiv:1808.09781). Baseline used for comparison: first-order item transitions blended with time-decayed popularity (`interface/baseline/`).

## Task

Design a sequential recommendation model that ranks every user's **next item** as high as possible among all items (full ranking, leave-one-out). Develop and train it on one 24GB GPU in the sandbox, then submit the trained model.

- Submit your model code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint and calls `recommend()` with every user's public history; it never retrains.
- Only the provided interaction histories may be used: no pretrained weights, movie metadata, ratings values from outside the file, or external data.
- You may build on any public work from before the cutoff (self-attentive, recurrent or convolutional sequence models, factorised Markov chains, graph models), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only), from MovieLens-1M (Harper & Konstan 2015): all 1,000,209 ratings of 6,040 users are treated as implicit interactions, ordered by timestamp (ties by item id), with **each user's final interaction removed** (that item is the held-out target).

| Key | Shape | Type | Description |
|---|---|---|---|
| `items` | (994,169,) | int32 | Item ids 1..`n_items` (0 is reserved for padding), concatenated user by user |
| `times` | same | int64 | Unix timestamps of the interactions |
| `offsets` | (6041,) | int64 | User u's sequence is `items[offsets[u]:offsets[u+1]]`, users 0..6039 |
| `n_items` | () | int64 | Number of items (every movie with at least one rating) |

No items are filtered. The held-out targets are not in the sandbox; the self-check holds out the last public item of each user instead.

## Interface

Implement `load(checkpoint_dir, device)` and `recommend(model, histories, users, k)` in `model.py`. `histories` holds the arrays above (the user histories to condition on); `recommend` returns an int array (len(users), k) of distinct item ids, best first, for the next interaction of each user after their history. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **NDCG@10** (primary) | 1/log2(rank + 1) if the held-out item is in the user's top-10 list, else 0; mean over all 6,040 users; higher is better | — |
| Recall@10 | Fraction of users whose held-out item is in their top-10 list | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `model.py`) | reported (killed at 120 s) |
| Recommendation time | Seconds `recommend()` takes for all users | reported (limit 120 s) |

Ranking is over **all** items (no sampled negatives). Items already in a user's history are never the target (MovieLens-1M has one rating per user and movie), so recommending them only wastes slots.


No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed outputs (see below).
**Timing and decision**

- Loading and recommendation run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`); loading `train.npz` and creating the CUDA context are not timed.
- If loading or recommendation crashes, exceeds a time limit, or returns lists of the wrong shape or with repeated items, the submission is invalid; valid submissions are ranked by NDCG@10.
- The final ABench score is driven mainly by NDCG@10, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 2 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and recommend()
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
- No pretrained weights, metadata or external data.
- The data comes from MovieLens-1M under the GroupLens usage license (research use only, no redistribution); do not copy it into the submission.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
