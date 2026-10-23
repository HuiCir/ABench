# MD-SEG-001: A 3D segmentation network for a fixed nnU-Net-style recipe: four Medical Segmentation Decathlon tasks, one of them unseen

Anchor: Fabian Isensee, Jens Petersen, Andre Klein, David Zimmerer, Paul F. Jaeger, Simon Kohl, Jakob Wasserthal, Gregor Koehler, Tobias Norajitra, Sebastian Wirkert, Klaus H. Maier-Hein, *nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation* (arXiv:1809.10486). Baseline used for comparison: nnU-Net's plain convolutional U-Net, planned from the patch size and spacing (downsampling and kernel sizes as in nnU-Net's default planner, features 32 to 320, deep supervision), trained by the evaluator under identical conditions (`interface/baselines/unet/`).

## Task

Design a **3D segmentation network** — encoder and decoder blocks, normalisation, residual or dense connections, attention or large kernels, how resolution is reduced along anisotropic axes, deep supervision — that, trained by a **fixed nnU-Net-style recipe** (patch sampling, augmentation, Dice + cross-entropy, SGD with polynomial decay), gives the **highest Dice on four Medical Segmentation Decathlon tasks**: hippocampus and prostate MRI, left atrium MRI, and a **fourth task you never see**.

- Submit code only: `network.py` with `make_network(in_channels, num_classes, patch_size, spacing)`. The evaluator builds the network for each task from that task's configuration, trains it from scratch for 2,500 iterations with the recipe of `interface/seg3d.py` and segments the held-out volumes by sliding windows. It trains the nnU-Net U-Net baseline the same way.
- The network must adapt to the configuration it is given (1 or 2 channels, 2 or 3 classes, isotropic or strongly anisotropic spacing, patch sizes from 40 x 56 x 40 to 20 x 192 x 160); the unseen task's configuration is only given to `make_network` at training time.
- At most 64 million parameters for every task; the recipe, preprocessing and inference are fixed.
- You may build on any public work (nnU-Net and its residual-encoder variants, 3D U-Net, V-Net, attention U-Net, UNETR, Swin UNETR, MedNeXt, STU-Net, nnFormer, large-kernel convolutions), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): `hippocampus.npz`, `prostate.npz`, `heart.npz` with the preprocessed labelled training volumes of MSD Task04 (208 of 260 volumes), Task05 (26 of 32) and Task02 (16 of 20) (Simpson et al. 2019, Antonelli et al. 2022; CC-BY-SA 4.0): `img`, `img_shape` (C, D, H, W) float16 and `seg`, `seg_shape` (D, H, W) uint8 volumes packed one after the other (`interface/seg3d.py: unpack`), resampled to each task's target spacing and normalised as described in `seg3d.py`.

The held-out set is the remaining 20% of the labelled volumes of each of the three tasks (52, 6 and 4, chosen with a secret seed) and a fourth MSD task: its training volumes and configuration are given to the evaluator's training stage, and 10 of its labelled volumes are held out. Only the images are given to the network, after training.

## Interface

Implement `make_network(in_channels, num_classes, patch_size, spacing) -> torch.nn.Module` in `network.py`; `forward(x)` maps (B, in_channels, *patch_size) to logits (B, num_classes, *patch_size) or, in training mode, optionally to a list of logits at full and successively lower resolutions (deep supervision; full resolution first). `spacing` is in millimetres, slowest axis first.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean Dice** (primary) | Dice per foreground class and held-out volume, averaged over classes, then volumes, then the four tasks; higher is better | — |
| Dice relative to U-Net | Mean Dice minus the U-Net baseline's in the same run; per task | reported |
| Surface accuracy | Normalised surface Dice at 1 mm (hippocampus) or 2 mm (others), averaged the same way | reported |
| Per task and class | Dice of each task and each class; peak GPU memory; training time | reported |
| Size and stability | Parameters of the network on every task; divergence | Threshold: ≤ 64,000,000; no divergence |

**Measurement and decision**

- One run with one seed; Dice and surface Dice are measured at the preprocessed resolution. With 4 to 52 held-out volumes per task, differences of about 0.01 in mean Dice are within run-to-run noise.
- A submission is valid if it trains without divergence on all four tasks within the time limits (1,800 s of training and 900 s of inference per task) and has at most 64,000,000 parameters on each; otherwise it is invalid.
- The final ABench score is driven mainly by mean Dice, and also considers how different the network is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seed, data, recipe, iterations), the evaluator also trains the U-Net from `interface/baselines/unet/` on all four tasks and reports your Dice relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, SciPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── network.py          # required: make_network()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The network may only compute from its input patch; it must not read files, keep data between calls beyond its parameters and buffers, or change the recipe (optimiser, loss, sampling, inference).
- No pretrained weights: the network is initialised from scratch by `make_network`.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
