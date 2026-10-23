# IC-RES-002: An image backbone for small data at two compute budgets (Imagenette, 128 pixels, 20 epochs)

Anchor: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Deep Residual Learning for Image Recognition* (arXiv:1512.03385). Baseline used for comparison: ResNet-18 (small budget) and ResNet-50 (large budget), run by the evaluator under the same recipe (`interface/baselines/resnet/`).

## Task

Design an **image-classification backbone** that, trained from scratch by the evaluator for 20 epochs on the 9,469 training images of Imagenette at 128 pixels, reaches the **highest test accuracy at two compute budgets**: at most 1.25 GFLOPs and at most 2.8 GFLOPs per image (ResNet-18 and ResNet-50 sizes).

- Submit code only (`model.py` with `build(variant, num_classes)` for `"small"` and `"large"`). At evaluation the evaluator trains each variant with the same fixed recipe, once for each of 3 seeds unknown to you, then classifies the 3,925 held-out test images. It trains the ResNet baselines the same way.
- The evaluator measures each variant's forward FLOPs on one 128 × 128 image (FlopCounterMode); a variant above its budget makes the submission invalid. Parameters are reported, not limited.
- The primary metric is the mean of the two accuracies — two points of the accuracy–FLOPs curve.
- This is a calibration task with a mature anchor; improvements must hold at both budgets. You may build on any public work (residual and inverted-residual blocks, normalisation, attention or large-kernel convolutions, scaling rules), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only): the training split of Imagenette (fastai, the 160-pixel release of December 2019; 10 ImageNet classes), centre-cropped to 160 × 160.

| Key | Shape | Type | Description |
|---|---|---|---|
| `x` | (9469, 160, 160, 3) | uint8 | RGB images |
| `y` | (9469,) | int64 | Class 0–9 (WordNet ids in sorted order: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute) |

The held-out set is the official validation split (3,925 images); it is not in the sandbox. The self-check holds out 150 training images per class.

## Interface

Implement `build(variant, num_classes=10) -> nn.Module` in `model.py`; `variant` is `"small"` or `"large"`. The module maps normalised images (B, 3, 128, 128) to logits (B, 10). Training, augmentation and evaluation are the evaluator's (`interface/img_train.py`).

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean accuracy** (primary) | Test accuracy of the small and the large variant on the 3,925 held-out images, averaged over the two variants and the 3 seeds; higher is better | — |
| Accuracy per variant | The same per variant; differences to ResNet-18 and ResNet-50 | reported |
| Compute | Forward FLOPs per 128 × 128 image divided by the variant's budget (1.25 G, 2.8 G), the larger of the two | Threshold: ≤ 1.0 |
| Parameters | Per variant | reported |
| Training time | Seconds of one variant's training (parent clock), the larger | Threshold: ≤ 1,500 s |
| Divergence | 1 if a training loss became non-finite | Threshold: 0 |

**Fixed recipe** (`interface/img_train.py`, `interface/stages.py`): 20 epochs, batches of 128, SGD with Nesterov momentum 0.9, peak learning rate 0.1 with one warm-up epoch then cosine decay, weight decay 5e-4 on weights with more than one dimension, label smoothing 0.1, bf16 autocast, channels-last; augmentation is a random resized crop (35–100% of the area, aspect 3/4–4/3) to 128 pixels and a horizontal flip; test images are resized from 160 to 128 pixels.

**Decision**

- A submission is valid only if every threshold holds for the 3-seed means; valid submissions are ranked by mean accuracy. Differences inside the 95% interval over test images reported with the score are treated as ties.
- If any run crashes or exceeds a time limit (a stage is killed at 1,620 s), the whole submission is invalid.
- The final ABench score is driven mainly by mean accuracy relative to the baseline, and also considers how different the backbone is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Under exactly the same conditions (hardware, seeds, data, recipe), the evaluator also trains ResNet-18 and ResNet-50 from `interface/baselines/resnet/` and reports your accuracies relative to them.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 5 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: build(variant, num_classes)
├── src/                # optional: other .py modules imported by model.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total; no trained parameters.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The backbone must be built from `build()` alone and trained by the evaluator; no pretrained weights, files or network access.
- All computation must use PyTorch operators that the FLOP counter sees (no custom Triton or CUDA kernels).
- The model must not change the recipe, the augmentation or the data (for example by training itself inside `forward`).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
