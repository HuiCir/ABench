# T-MM-004: Aligning a frozen image encoder and a frozen text encoder with light trained adapters: COCO retrieval and compositional checks

Anchor: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, *Learning Transferable Visual Models From Natural Language Supervision* (arXiv:2103.00020). Baseline used for comparison: two-layer projections of the frozen encoders' pooled features into a shared 256-d space, trained with CLIP's symmetric contrastive loss (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design **light trainable components** — projections, adapters inside the frozen networks, cross-attention or pooling over patch and token features, a contrastive or other objective — that **align a frozen DINOv2-small image encoder with a frozen MiniLM text encoder** for **image-text retrieval** on COCO, while staying robust to **compositional changes** of captions (swapped, replaced or added attributes, objects and relations).

- Submit the trained tensors (`checkpoint/states.pt`, at most 10 million numbers), the code that uses them with the frozen encoders, and the training program. The evaluator calls your `encode_images()` and `encode_texts()` on held-out images and captions and ranks by cosine similarity; it never retrains.
- The encoders' weights may not be changed: anything trained must be in `states.pt`, which also counts as the trained-parameter cost.
- You may build on any public work (CLIP and its losses, locked-image tuning, adapters, hard-negative mining, compositional training), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/coco_train.npz` (read-only): COCO 2014 images of the Karpathy train and restval splits (Lin et al. 2014; Karpathy split from the parquet export `yerevann/coco-karpathy` @448fdb1b), minus every image used by the compositional test, each with its first five captions; `coco_val.npz`: the Karpathy val split (5,000 images). Keys: `jpeg`/`jpeg_offsets` (256×256 JPEG files, shorter side resized and centre-cropped, quality 90), `caption_utf8`/`caption_offsets` (packed captions, `interface/textpack.py`), `caption_image` (image index of each caption), `cocoid`. `$ABENCH_MODELS_DIR/dinov2-small/` (`facebook/dinov2-small` @ed25f3a3, Apache-2.0) and `all-MiniLM-L12-v2/` (`sentence-transformers/all-MiniLM-L12-v2` @a50ef001, Apache-2.0).

The held-out sets are not in the sandbox: the Karpathy test split (5,000 images, 25,000 captions) and the 7,511 items of SugarCrepe (Hsieh et al. 2023, MIT; COCO images with a true caption and a hard-negative caption, seven subsets). COCO annotations are CC BY 4.0; the images are used under the COCO terms of use.

## Interface

Implement `load(states, models_dir, device)`, `encode_images(state, jpegs) -> (n, d) array` and `encode_texts(state, texts) -> (n, d) array` in `model.py` (d ≤ 1,024; `jpegs` are JPEG file contents as bytes); write a `train.py` that produces `checkpoint/states.pt`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean recall** (primary) | Mean of R@1, R@5 and R@10 for image→text (an image is retrieved correctly if any of its five captions is in the top K) and text→image on the 5,000 Karpathy test images and 25,000 captions, by cosine similarity, ties counted against you; higher is better | — |
| Recall per direction and K | The six recalls | reported |
| Compositional accuracy | SugarCrepe: fraction of items where the true caption is more similar to the image than the hard negative, per subset and averaged over the seven subsets | reported |
| Trained parameters | Numbers in `states.pt` | Threshold: ≤ 10,000,000 |
| Latency | Milliseconds per image and per caption in batches of 256 on the reference GPU | reported |

**Measurement and decision**

- One stage in a new process on the reference GPU; loading is killed at 300 s and encoding of all held-out images and captions at 1,200 s.
- Valid submissions are ranked by mean recall; differences below 0.5 points are treated as ties.
- The final ABench score is driven mainly by mean recall, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), torchvision, transformers, Pillow, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load(), encode_images(), encode_texts()
├── train.py            # required: reproduces checkpoint/states.pt
├── src/                # optional: other .py modules
├── checkpoint/
│   └── states.pt       # required: {name: tensor}, at most 10 million numbers
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The encoders are loaded from `models_dir` and must not be modified; only tensors in `states.pt` may carry what was trained.
- Train only on the public COCO data; no other images, captions or pretrained models. Hard negatives may be generated from the public captions by rule.
- `encode_images` and `encode_texts` must embed each item on its own (no use of the other items in a batch).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
