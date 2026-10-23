# V-DST-004: Distilling a CIFAR-10 diffusion model into a one- to four-step generator

Anchor: Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever, *Consistency Models* (arXiv:2303.01469). Baseline used for comparison: consistency distillation with a squared-error distance, student initialised from the teacher (`interface/baseline/`).

## Task

Design a **few-step distillation scheme** that turns the frozen CIFAR-10 DDPM teacher into a generator producing the **best images (lowest FID) with only 1 or 2 network evaluations per image** (4 is reported too). Develop and train the student on one 24GB GPU in the sandbox, then submit it.

- Submit your model code, the trained student checkpoint and the training program that produced it. The evaluator loads the student — without the teacher — and generates 10,000 images with budgets of 1, 2 and 4 network calls per image; it never retrains.
- Only the teacher, the CIFAR-10 training images and networks you train yourself may be used; the student may have at most 72M parameters (twice the teacher) and may be initialised from the teacher's weights. No other pretrained networks (no discriminator backbones, no feature networks for perceptual losses), no external images.
- You may build on any public work (consistency distillation and training, progressive distillation, rectified flow and reflow, distribution matching and score distillation, adversarial distillation with discriminators you train yourself), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only):

| File | Key | Shape | Type | Description |
|---|---|---|---|---|
| `cifar10_train.npz` | `x` | (50000, 32, 32, 3) | uint8 | The CIFAR-10 training images |
| | `y` | (50000,) | int64 | Class labels 0–9 (you may ignore them; the evaluation is unconditional) |
| `cifar10_train_inception_stats.npz` | `mu`, `sigma` | (2048,), (2048, 2048) | float64 | Inception pool-3 statistics of the 50,000 training images: the FID reference at evaluation too |

`$ABENCH_MODELS_DIR/ddpm-cifar10-32/` holds the frozen teacher (Hugging Face `google/ddpm-cifar10-32`, Apache-2.0; a diffusers `UNet2DModel`, ε-prediction, linear β from 1e-4 to 0.02 over 1,000 steps), available during training only; `$ABENCH_MODELS_DIR/inception/` holds the pytorch-fid Inception weights. The held-out part of this case is only the evaluation seeds.

## Interface

Implement `load(checkpoint_dir, device) -> torch.nn.Module` and `generate(net, n, steps, generator, device) -> torch.Tensor` in `model.py`: `generate` draws all randomness from `generator`, calls `net` at most `steps` times and returns n images of shape (n, 3, 32, 32) in [-1, 1]. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean FID at 1 and 2 calls** (primary) | Fréchet Inception distance of 10,000 images generated with 1 call each and with 2 calls each, averaged; lower is better | — |
| FID per budget | Fréchet Inception distance between 10,000 samples and the CIFAR-10 training set (pytorch-fid Inception, pool-3 features) at each step budget (1, 2, 4) | reported |
| Diversity ratio | Trace of the generated images' feature covariance divided by that of the training set, per budget (well below 1 signals mode dropping) | reported |
| Network calls | Calls of the returned module during one `generate()`, the largest over batches | Threshold: ≤ the step budget |
| Model size | Parameters and buffers of the returned module | Threshold: ≤ 72,000,000 |

**Timing and decision**

- Loading and sampling run in a new process with empty cache directories, on the same GPU model, timed by the evaluator's parent process (`interface/abench_stage.py`). Images are generated in batches of 500; batch b of a budget uses a generator seeded from that budget's held-out seed and b (`stages.py`). All images must be generated within 600 s.
- A submission is valid only if every threshold holds; valid submissions are ranked by mean FID at 1 and 2 calls.
- If loading or sampling crashes, exceeds a time limit, or returns images of the wrong shape or with non-finite values, the submission is invalid.
- The final ABench score is driven mainly by mean FID at 1 and 2 calls, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, diffusers and pytorch-fid (versions frozen at activation; offline); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com`, `www.cs.toronto.edu` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and generate()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained weights, any file types, at most 400 MiB
├── src/                # optional: other .py modules imported by model.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights other than the provided teacher, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- Images must come from `generate()` with randomness drawn from `generator`; no stored images, no selection or filtering of samples, and no classifier or discriminator at sampling time.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
