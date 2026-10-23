# U-BYTE-007: A tokenizer-free byte-level language model with dynamic patching, scored in bits per byte at equal compute and under character noise

Anchor: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer, *Byte Latent Transformer: Patches Scale Better Than Tokens* (arXiv:2412.09871). Baseline used for comparison: a local-global byte model with static 4-byte patches (MegaByte layout), run by the evaluator under identical conditions (`interface/baselines/static4/`).

## Task

Design a **tokenizer-free language model over raw bytes with a dynamic patching scheme** — how bytes are grouped into patches, how patches are encoded and decoded — that reaches the **lowest bits per byte** on held-out text at no more compute per byte than a static-patch baseline, and stays robust to character-level noise.

- Submit the model only (`model.py` with `build()`). At evaluation the evaluator trains it from scratch with a fixed recipe — 3,000 steps of 32 windows of 4,096 bytes of FineWeb text — once for each of 2 seeds unknown to you, then measures bits per byte on held-out text, and on the same text after 5% of its letters are replaced, re-cased or deleted. It trains the static 4-byte-patch baseline the same way.
- The evaluator measures the compute itself: forward FLOPs per byte on held-out text may exceed the baseline's by at most 5%, parameters by at most 10%.
- Patching must be causal: whether a patch ends at byte j may depend only on bytes up to j.
- Scaling claims (how patching compares with tokenisation at larger scales) are not measured here; this case scores one scale.
- You may build on any public work (byte-level and character-level models, local-global architectures, entropy- and space-based patching, hash n-gram embeddings), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only) holds FineWeb text as UTF-8 bytes: the first two training shards of the GPT-2-tokenised FineWeb-10B of B-OPT-001 (Hugging Face `kjj0/fineweb10B-gpt2`), decoded back to bytes.

| File | Contents |
|---|---|
| `bytes_train_000001.bin`, `bytes_train_000002.bin` | raw uint8 bytes (about 450 MB each, no header); document boundaries are the byte 0 |

Read them with `interface/byte_data.py`. The held-out set is the first 8,388,608 bytes of the decoded official FineWeb validation shard; it is not in the sandbox. The self-check holds out the last 8,388,608 bytes of the second file.

## Interface

Implement `build() -> nn.Module` in `model.py`. The module's `forward(x)` maps int64 bytes (B, T) to logits (B, T, 256), where `logits[:, j]` predicts byte j from the bytes before it, and `loss(x)` returns the mean cross-entropy over all positions. `interface/byte_blocks.py` provides windowed-attention transformer blocks and a segment-mean helper; the baseline shows a complete model.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Bits per byte** (primary) | Mean cross-entropy (bits) per byte on the 8,388,608 held-out bytes in windows of 4,096, mean over 2 seeds; lower is better | — |
| Bits per byte under noise | The same on the held-out bytes after character-level noise (5% of ASCII letters replaced by a random letter, re-cased or deleted, one third each; fixed seed) | reported |
| Noise penalty | Bits per byte under noise minus bits per byte | reported |
| Compute | Forward FLOPs per byte on a held-out window of 4,096 bytes (FlopCounterMode) relative to the baseline | Threshold: ≤ 1.05 |
| Parameters | Relative to the baseline | Threshold: ≤ 1.10 |
| Causality | Relative change of logits at positions ≤ 2,048 when bytes from 2,048 on change | Threshold: ≤ 1e-3 |
| Training time | Seconds of the 3,000 steps (parent clock) | Threshold: ≤ 1,800 s |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (`interface/lm_train.py`, `interface/stages.py`): AdamW at 1e-3 (β = 0.9, 0.95; weight decay 0.1 on matrices), 150 warm-up steps then cosine decay to 0.1, clipping at 1.0, bf16 autocast with fp32 weights; the loss is your `model.loss(x)` on windows of 4,096 bytes.

**Decision**

- A submission is valid only if every threshold holds for the 2-seed means; valid submissions are ranked by bits per byte. Differences smaller than the spread over the two seeds are treated as ties.
- If any run crashes or exceeds a time limit (the stage is killed at 2,100 s), the whole submission is invalid.
- The final ABench score is driven mainly by bits per byte relative to the baseline, and also considers the noise penalty, how different the scheme is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe), the evaluator also trains the static-patch baseline in `interface/baselines/static4/` and reports your bits per byte and noise penalty relative to it; the compute and parameter gates are relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: build()
├── src/                # optional: helper modules imported by model.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters, vocabularies or tokenizers.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No tokenizer: the model must read raw bytes; no learned or stored vocabulary of byte sequences (a patching rule may use byte classes such as spaces, or quantities the model computes).
- All computation must use PyTorch operators that the FLOP counter sees (no custom Triton or CUDA kernels).
- The model must be causal and built from `build()` alone; no pretrained weights, files or network access.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
