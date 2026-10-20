# V-SMP-002: A training-free sampler for a frozen diffusion model at small step counts

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Jiaming Song, Chenlin Meng, Stefano Ermon, *Denoising Diffusion Implicit Models* (arXiv:2010.02502). Baseline used for comparison: deterministic DDIM with uniform timesteps, run by the evaluator under identical conditions (`interface/baselines/ddim/`).

## Task

Design a **training-free sampler** — integrator, timestep schedule, corrector or any other procedure that only queries the network — for a frozen ε-prediction DDPM on CIFAR-10, that produces the **best images (lowest FID) with only 5, 10 or 20 network evaluations per image**.

- Submit code only (`sampler.py`). At evaluation the evaluator runs your `sample()` with the frozen model at NFE 5, 10 and 20, 10,000 images each, once for each of 3 seed sets unknown to you, and computes FID against the CIFAR-10 training set. It runs the DDIM baseline the same way.
- No training or fine-tuning of any network, no other networks (distilled students, classifiers, discriminators), and no stored images; schedules or coefficients you tune in the sandbox may be written into the code.
- The NFE budget is enforced: every call of the denoiser counts once per batch, and calls beyond the budget fail the run.
- You may build on any public work from before the cutoff (DDIM, DPM-Solver and DPM-Solver++, UniPC, EDM samplers, optimised or learned timestep schedules, flow-matching views of the probability-flow ODE), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_MODELS_DIR/ddpm-cifar10-32/` holds the frozen model (Hugging Face `google/ddpm-cifar10-32`, Apache-2.0): a diffusers `UNet2DModel` (Ho et al. 2020) trained with ε-prediction and a linear β schedule from 1e-4 to 0.02 over 1,000 steps; `interface/diffusion.py` gives ᾱ_t and the counting `Denoiser`. `$ABENCH_MODELS_DIR/inception/` holds the pytorch-fid Inception weights.

`$ABENCH_DATA_DIR/cifar10_train_inception_stats.npz` (read-only) holds `mu` (2048,) and `sigma` (2048, 2048), the Inception pool-3 statistics of the 50,000 CIFAR-10 training images, which are the FID reference at evaluation too. The held-out part of this case is only the evaluation seeds.

## Interface

Implement `sample(denoiser, n, nfe, generator, device) -> torch.Tensor` in `sampler.py`: draw initial noise with `generator`, call `denoiser(x, t)` at most `nfe` times, and return n images of shape (n, 3, 32, 32) in [-1, 1].


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **FID at NFE 10** (primary) | Fréchet Inception distance between 10,000 samples and the CIFAR-10 training set (pytorch-fid Inception, pool-3 features), mean over 3 seed sets; lower is better | — |
| FID at NFE 5 and NFE 20 | The same at the other two budgets | reported |
| FID vs DDIM | Your FID minus the DDIM baseline's at each budget | reported |
| NFE violations | Budgets exceeded (a violation also fails the run) | Threshold: 0 |

**Timing and decision**

- Each seed set runs in a new process with empty caches (`interface/abench_stage.py`). Images are generated in batches of 500; batch b of a budget uses a generator seeded from that budget's held-out seed, the run's seed and b (`stages.py`). Generating all 30,000 images must finish within 900 s.
- A submission is valid only if no budget is exceeded; valid submissions are ranked by FID at NFE 10.
- If any run crashes, exceeds a time limit, or returns images of the wrong shape or with non-finite values, the whole submission is invalid.
- The final ABench score is driven mainly by FID at NFE 10 relative to the baseline (the other budgets are considered too), and also by how different the method is from existing samplers and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (model, seeds, number of samples), the evaluator also runs the DDIM baseline in `interface/baselines/ddim/` and reports your FID relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, diffusers and pytorch-fid (versions frozen at activation; offline).
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── sampler.py          # required: sample()
├── src/                # optional: helper modules imported by sampler.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Query the model only through the `denoiser` you are given; do not reach the underlying network by other means, load other networks or train anything.
- Initial noise and any other randomness must come from `generator`; samples must not be selected, filtered or re-ranked.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
