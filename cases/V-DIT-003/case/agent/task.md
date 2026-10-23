# V-DIT-003: A latent-diffusion backbone at the FLOPs of DiT-S/2: class-conditional Imagenette at 128 pixels under a fixed training recipe

Anchor: William Peebles, Saining Xie, *Scalable Diffusion Models with Transformers* (arXiv:2212.09748). Baseline used for comparison: DiT-S/2 (patch 2 on the 4 x 16 x 16 latent, width 384, 12 blocks, adaLN-Zero), trained by the evaluator under identical conditions (`interface/baselines/dit_s/`).

## Task

Design the **denoising backbone of a latent diffusion model** — tokenisation of the latent, block design, normalisation, conditioning on time and class, skip connections, attention variants, convolutional or hybrid stages — that, at **no more than 1.1 times the forward FLOPs of DiT-S/2** and trained by a **fixed recipe**, gives the **lowest FID** for class-conditional generation of Imagenette at 128 pixels.

- Submit code only: `backbone.py` with `make_backbone(latent_shape, num_classes)`. The evaluator trains it from scratch on the public VAE latents for 30,000 steps with the recipe of `interface/ldm.py` (noise prediction, linear schedule, AdamW, EMA), samples 5,000 images with 50 DDIM steps and classifier-free guidance 1.5 (and 5,000 without guidance), decodes them with the VAE and measures FID. It trains DiT-S/2 the same way.
- Only the backbone changes: objective, noise schedule, optimiser, number of steps, sampler and VAE are fixed. FLOPs are counted by the evaluator for one latent.
- You may build on any public work (DiT, U-ViT, U-Net backbones of ADM, MDT, DiffiT, SiT and REPA architectures, Hourglass and windowed attention, RMSNorm, SwiGLU, QK-norm), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): `latents.npz` with `latents` (9,469, 2, 4, 16, 16) float16, the Stable Diffusion VAE latents (posterior mean x 0.18215) of the Imagenette training images centre-cropped and resized to 128 pixels and of their mirror images, and `labels` (10 classes); `train_inception_stats.npz` (Inception statistics of the 128-pixel training images, for the self-check). Imagenette: fastai, Apache-2.0, images from ImageNet. `$ABENCH_MODELS_DIR/` holds the VAE (`sd-vae-ft-ema`, @f04b2c4b, MIT) and the pytorch-fid Inception weights.

The held-out set is the Inception statistics of the 3,925 Imagenette validation images at 128 pixels, and the sampling seeds (drawn with a secret seed).

## Interface

Implement `make_backbone(latent_shape, num_classes) -> torch.nn.Module` in `backbone.py`; `forward(x, t, y)` takes noisy latents x (B, 4, 16, 16), integer timesteps t (B,) in [0, 1000) and labels y (B,) in [0, num_classes], where num_classes is the null class of classifier-free guidance, and returns the predicted noise (B, 4, 16, 16).

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **FID with guidance** (primary) | FID of 5,000 class-balanced samples (guidance 1.5, 50 DDIM steps) against the Imagenette validation images; lower is better | — |
| FID without guidance | The same with guidance 1 | reported |
| Relative to DiT-S/2 | Both FIDs minus DiT-S/2's in the same run | reported |
| Cost | Forward FLOPs for one latent, divided by DiT-S/2's; parameters; training time | Threshold: ≤ 1.10 |
| Stability | Divergence of training, non-finite samples | Threshold: none |
| Diversity | Trace of the sample feature covariance over the reference one | reported |

**Measurement and decision**

- One run with one seed. FID from 5,000 samples against 3,925 reference images is biased upwards and varies by about 0.5 between seeds; differences smaller than that are within noise.
- A submission is valid if it trains without divergence within 3,600 s, its samples are finite and its forward FLOPs are at most 1.10 times DiT-S/2's; otherwise it is invalid.
- The final ABench score is driven mainly by FID with guidance, and also considers how different the backbone is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seed, latents, recipe, sampler), the evaluator also trains DiT-S/2 from `interface/baselines/dit_s/`; your FLOPs limit and FID differences are relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, SciPy, diffusers, pytorch-fid; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── backbone.py         # required: make_backbone()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The backbone may only compute from its inputs (x, t, y): no pretrained weights, no files, no state kept between calls beyond its parameters and buffers.
- The FLOPs counted for one latent must be representative: the computation may not depend on the batch size, on whether gradients are enabled or on the training mode in a way that lowers the count.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
