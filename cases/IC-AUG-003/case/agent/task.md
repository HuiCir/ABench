# IC-AUG-003: Searching a data-augmentation policy from image-transformation primitives for a fixed CIFAR-10 recipe

Anchor: Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, *AutoAugment: Learning Augmentation Policies from Data* (arXiv:1805.09501). Baseline used for comparison: random horizontal flip and translation by up to 4 pixels, trained by the evaluator with the same recipe and seeds (`interface/baselines/flipcrop/`).

## Task

**Search for a data-augmentation policy** — which image transformations, how strong, how often, in what combinations and on what schedule over training — built **only from the primitives in `interface/aug_ops.py`**, that gives the **highest CIFAR-10 test accuracy** when the evaluator trains a fixed network with a fixed 40-epoch recipe using it.

- Submit code only (`policy.py` with `augment()`). At evaluation the evaluator trains the fixed recipe in `interface/recipe.py` from scratch with your policy, once for each of 5 seeds unknown to you, and measures accuracy on the CIFAR-10 test set and on CIFAR-10.1. It trains the flip-and-translate baseline the same way.
- The search space is the primitives, their magnitudes and probabilities and how you combine them; the network, optimiser, schedule and number of epochs are fixed. Use your development budget to run the search in the sandbox (for example with shortened runs, as the anchor did with a proxy task) and submit the best policy found.
- `policy.py` may not import anything but `math` and `typing`, nor touch torch, NumPy, files or the environment directly; it receives the primitives module and may use tensor indexing and arithmetic on what they return. Augmentation must not make training more than 1.5 times slower than the baseline's.
- You may build on any public work (AutoAugment, RandAugment, TrivialAugment, population-based augmentation, Fast AutoAugment, cutout, curricula of augmentation strength), but the design report must state what you borrowed and what you changed, and describe your search procedure.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only): the CIFAR-10 training set (Krizhevsky 2009).

| Key | Shape | Type | Description |
|---|---|---|---|
| `x` | (50000, 32, 32, 3) | uint8 | Images |
| `y` | (50000,) | int64 | Classes 0–9 |
| `dev_idx` | (5000,) | int64 | 500 images per class used as the self-check's test set (the self-check trains on the other 45,000) |

The held-out sets are the official CIFAR-10 test set (10,000 images) and CIFAR-10.1 v6 (2,000 images, Recht et al. 2018, a re-collected test set that measures robustness to distribution shift); neither is in the sandbox.

## Interface

Implement `augment(x, ops, gen, epoch, n_epochs) -> x_aug` in `policy.py`. `x` is the whole training set as a float tensor (N, 3, 32, 32) in [0, 1] on the GPU, `ops` the module `aug_ops`, `gen` the torch.Generator for all random draws; return a tensor of the same shape. The recipe calls it once at the start of every epoch.


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **CIFAR-10 accuracy** (primary) | Test accuracy of the fixed recipe trained with your policy (mirror-averaged logits), mean over 5 seeds; higher is better | — |
| CIFAR-10.1 accuracy | The same models on CIFAR-10.1 v6 | reported |
| Relative to the baseline | Accuracy differences to the flip-and-translate baseline on both test sets | reported |
| Training time ratio | Seconds of the timed phase (import and training) divided by the baseline's, mean over seeds | Threshold: ≤ 1.5 |
| Policy check | `policy.py` imports only `math`/`typing` and names no torch, NumPy, file or reflection API | Threshold: must pass (a failure stops the run) |

**Timing and decision**

- Each seed runs in a new process with empty cache directories, timed by the evaluator's parent process (`interface/abench_stage.py`); loading the training images is not timed. Training must finish within 900 s; prediction of both test sets within 120 s.
- A submission is valid only if every threshold holds; valid submissions are ranked by CIFAR-10 accuracy. Differences inside the 95% interval reported with the score are treated as ties.
- If any seed crashes, exceeds a time limit, or `augment()` returns a tensor of the wrong shape or with non-finite values, the whole submission is invalid.
- The final ABench score is driven mainly by CIFAR-10 accuracy, and also considers how different the policy and the search procedure are from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 5 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Under exactly the same conditions (recipe, seeds, GPU), the evaluator also trains the network with the flip-and-translate baseline in `interface/baselines/flipcrop/` and reports your accuracy and training time relative to it.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com`, `www.cs.toronto.edu` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── policy.py           # required: augment()
├── src/                # optional: helper modules, under the same import restrictions
├── README.md           # optional
├── report/index.html   # required: design report (including the search procedure)
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total; no trained parameters.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The submission is code only; the policy may not carry learned network weights, images or statistics of the test sets.
- Every transformation must be a composition of `aug_ops` primitives with tensor indexing and arithmetic; no other imports, no direct use of torch or NumPy, no file or environment access.
- Randomness must come from `gen`; the policy must not inspect labels (it does not receive them) or adapt to the evaluation seed.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
