# V-3DG-005: Sparse-view novel-view synthesis in ten minutes per scene: 24 views of unbounded Mip-NeRF 360 scenes

Anchor: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (arXiv:2308.04079). Baseline used for comparison: plain 3D Gaussian splatting with the gsplat rasteriser from a random initialisation (7,000 steps, L1 + D-SSIM, gsplat's default densification), run by the evaluator under identical conditions (`interface/baselines/gs3d/`).

## Task

Design a **scene-reconstruction method for sparse views** — representation (3D Gaussians, radiance fields, hybrids), initialisation without structure-from-motion points, regularisation of unseen regions, densification and pruning, appearance and depth priors computed from the views themselves — that, **fitted in at most 600 seconds** to **24 posed photographs** of an unbounded scene, renders **held-out viewpoints** as accurately as possible.

- Submit code only: `method.py` with `fit(views, budget_seconds, device)` and `render(state, cameras)`. For each of six held-out Mip-NeRF 360 scenes the evaluator calls `fit` on 24 training views (images, intrinsics, camera poses) and then `render` at the test cameras; it runs plain 3D Gaussian splatting the same way.
- Nothing else is given: no point cloud, no depth, no test images. The gsplat rasteriser is installed; any other component must be written in the submission.
- No pretrained networks (no monocular-depth or diffusion priors) and no external data.
- You may build on any public work (3D Gaussian splatting, Mip-Splatting, FSGS, DNGaussian, SparseGS, CoR-GS, RegNeRF, FreeNeRF, Instant-NGP, Zip-NeRF), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): `dev_bonsai.npz`, `dev_flowers.npz`, `dev_treehill.npz` — three Mip-NeRF 360 scenes (Barron et al. 2022; `storage.googleapis.com/gresearch/refraw360`, no licence stated on the project page) with every image at the published 1/8 resolution (`images` uint8 (N, H, W, 3)) and its camera (`K` (N, 3, 3) in pixels, `w2c` (N, 4, 4) world-to-camera, OpenCV convention) from the dataset's COLMAP reconstruction.

The held-out set is the six other scenes of the main release (outdoor and indoor, in a secret order): for each, the 24 training views (spread evenly over the images that are not test views, `interface/nvs.py: split`) are given to `fit`; the test cameras (every 8th image) to `render`; the test images are never given.

## Interface

Implement in `method.py`: `fit(views, budget_seconds, device)` returning any state, where `views` has `images` uint8 (24, H, W, 3), `K` (24, 3, 3) and `w2c` (24, 4, 4); and `render(state, cameras)` returning uint8 (M, H, W, 3), where `cameras` has `K` (M, 3, 3), `w2c` (M, 4, 4), `height` and `width`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean PSNR** (primary) | PSNR of each rendered test view against the photograph, averaged per scene and then over the six scenes; higher is better | — |
| PSNR relative to 3DGS | Mean PSNR minus plain 3D Gaussian splatting's in the same run; per scene | reported |
| SSIM | Structural similarity (11 x 11 Gaussian window), averaged the same way | reported |
| Fitting time | Seconds spent in `fit` per scene | Threshold: ≤ 630 s on every scene |

**Measurement and decision**

- One run; each scene has about 20 to 40 test views at roughly 1/8 of the original resolution.
- A submission is valid if `fit` returns within 630 s (the 600 s budget plus 5%) on every scene and `render` returns images of the right size within 300 s; otherwise it is invalid.
- The final ABench score is driven mainly by mean PSNR, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, views, budget), the evaluator also runs plain 3D Gaussian splatting from `interface/baselines/gs3d/` on every scene and reports your PSNR relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), gsplat, NumPy, SciPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `jonbarron.info`.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── method.py           # required: fit(), render()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- `fit` may only use the views it is given; `render` may only use the state returned by `fit` and the cameras. No files from other scenes, no pretrained weights, no network.
- All computation must happen inside `fit` and `render`; nothing may be computed in advance for the held-out scenes.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
