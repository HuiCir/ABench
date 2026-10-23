# IC-SPD-001: CIFAR-10 speedrun to 94%

Anchor: Keller Jordan, *94% on CIFAR-10 in 3.29 Seconds on a Single GPU* (arXiv:2404.00498).

## Task

Design an image classification network and its training recipe so that, on one 24GB reference GPU, **training from random initialization reaches 94% CIFAR-10 test accuracy in as little training time as possible**.

- Submit code only. At evaluation your code trains from scratch once for each of 5 seeds unknown to you, on the same GPU model, and is then evaluated on the test set.
- Only the 50,000 provided training images may be used: no pretrained weights and no external data.
- You may build on any public work, but the design report must state what you borrowed and what you changed. This task tests your understanding of training algorithms and hardware efficiency: architecture, optimizer and learning-rate schedule, data augmentation, precision and kernels, and compilation strategy are all open.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only):

| Key | Shape | Type | Description |
|---|---|---|---|
| `x` | (50000, 32, 32, 3) | uint8 | CIFAR-10 training set, RGB, channels last |
| `y` | (50000,) | int64 | Class 0–9 |
| `dev_idx` | (5000,) | int64 | Dev split for self-checking (500 per class), used only by `selfcheck.py` |

The test data is not in the sandbox.

## Interface

Implement `train(data, seed, device)` and `predict(model, x, device)` in `method.py` at the root of the submission directory; see `interface/interface.md`. `speedrun_harness.py` and `speedrun_runner.py` in `interface/` are the evaluator's actual timing code, and `selfcheck.py` uses them to self-check on the dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Training time T** (primary) | Mean training time in seconds over one run for each of 5 seeds; lower is better | — |
| CIFAR-10 test accuracy | Top-1 accuracy on the 10,000-image official CIFAR-10 test set, mean over 5 seeds | Threshold: ≥ 94.00% |
| Out-of-distribution accuracy | Top-1 accuracy on CIFAR-10.1 (v6, 2,000 images, same classes, independently collected), mean over 5 seeds | Threshold: ≥ 80.0% |

**Timing rules**

- The clock starts when the timing script sends the start signal. It includes importing `method.py` and everything inside `train()` (moving data to the GPU, building the model, `torch.compile` compilation, warm-up, training), until `train()` returns and CUDA is synchronized.
- Not counted: loading `train.npz` into memory and creating the CUDA context (done by the timing script, identical for all submissions).
- Timing is done by the parent process with its own clock. Every run is a new process with an empty cache directory; compiled artifacts cannot be reused across runs.
- `predict()` must finish inference on all test images (about 12,000) within 5 seconds, otherwise that run is invalid.

**Decision**

- A submission is valid only if both thresholds are met; valid submissions are ranked by T. Thresholds apply to the 5-seed mean, not to each seed.
- If any seed crashes, times out (600 seconds per run), returns an output whose shape is not (M, 10), exceeds the inference limit or violates the timing protocol, the whole submission is invalid.
- The final ABench score is driven mainly by T, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs `speedrun_runner.py` once per seed for 5 seeds on the same GPU model, training on all 50,000 training images.
3. Test images are passed to `predict()` without labels; labels stay in the evaluator process and are used only to compute accuracy.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/gpu_sandbox.md`. Key points:

- One 24GB reference GPU; Python 3, PyTorch (with CUDA and Triton) and NumPy are available, and your submission may depend only on them.
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. GPU usage during development is capped at 1 GPU-hour.
- Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface), `--seeds 2` (timing and dev accuracy).

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── method.py           # required: train() and predict()
├── src/                # optional: other .py modules imported by method.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No pretrained weights, no external data, and no trained parameters embedded in the code.
- `predict()` must not update model parameters and must not use test images for training or adaptation.
- Do not read or write files outside the submission directory and the paths given by the timing script, do not access the network (the evaluation runs without one), and do not touch the timing pipe (the file descriptor given by `--fd`).
- Do not use extra processes or background threads to move training work outside the timing window.
- Submission is one-shot. A submission that breaks the rules is invalid.
