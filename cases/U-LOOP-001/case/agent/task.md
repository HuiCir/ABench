# U-LOOP-001: A recurrent-depth language model that turns more loops into lower loss at the parameter budget of a 4-layer transformer

Anchor: Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser, *Universal Transformers* (arXiv:1807.03819). Baseline used for comparison: the B-OPT-001 decoder with 4 layers of width 384 and no weight sharing (the same parameter budget), run by the evaluator under identical conditions (`interface/baselines/fourlayer/`).

## Task

Design a **language model with recurrent (weight-shared) depth** that, at the parameter budget of a 4-layer transformer of width 384, reaches the **lowest held-out loss when run with 8 loops** — and whose loss keeps improving as the number of loops grows from 1 to 8.

- Submit the model only (`model.py` with `build(cfg)`). At evaluation the evaluator trains it from scratch with a fixed recipe (3,000 steps of 32 × 1,024 FineWeb tokens; how many loops to use in each training step is up to your model), once for each of 2 seeds unknown to you, then measures the held-out loss at 1, 2, 4 and 8 loops and, at 8 loops, on windows of 2,048 tokens (twice the training length). It trains the 4-layer baseline the same way.
- Parameters may exceed the baseline's by at most 2%; forward compute at 8 loops may be at most 3 times the baseline's. Equal parameters, not equal compute, is the comparison this case makes; the loop-count curve shows what extra compute buys.
- You may build on any public work (universal and looped transformers, adaptive computation, recurrent-depth models, input injection, deep-equilibrium models), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only) holds the first two training shards of FineWeb-10B pre-tokenized with the GPT-2 BPE (Hugging Face `kjj0/fineweb10B-gpt2`, the data of the modded-nanogpt speedrun):

| File | Contents |
|---|---|
| `fineweb_train_000001.bin`, `fineweb_train_000002.bin` | 256 int32 header (magic 20240520, version 1, token count), then about 100M uint16 tokens each |

Read them with `interface/token_data.py`. The held-out set is the first 2,097,152 tokens of the official FineWeb validation shard (`fineweb_val_000000.bin`); it is not in the sandbox. The self-check holds out the last 2,097,152 tokens of the second training shard.

## Interface

Implement `build(cfg) -> nn.Module` in `model.py`. `cfg` is `lm_model.Config(n_layers=4, max_seq=2048)` (width 384); the module's `forward(idx, loops)` returns logits (B, T, vocab) causally and `loss(idx, targets, loops=None)` the mean cross-entropy, where `loops=None` (training) lets the model choose.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss at 8 loops** (primary) | Mean next-token cross-entropy (nats) on the 2,097,152 held-out tokens in windows of 1,024 with `loops=8`, mean over 2 seeds; lower is better | — |
| Loop-count curve | The same at 1, 2, 4 and 8 loops; gain from 1 to 8 loops; whether it is monotone | reported |
| Length extrapolation | Held-out loss at 8 loops on windows of 2,048 tokens, and its difference to 1,024 | reported |
| Loss vs 4-layer | Primary minus the baseline's held-out loss on the same seeds | reported |
| Parameters | Relative to the 4-layer baseline | Threshold: ≤ 1.02 |
| Compute at 8 loops | Forward FLOPs per token at T = 1,024 with `loops=8` (FlopCounterMode) relative to the baseline | Threshold: ≤ 3.0 |
| Causality | Relative change of outputs at positions < 512 when later tokens change (8 loops) | Threshold: ≤ 1e-3 |
| Training time | Seconds of the 3,000 steps (parent clock) | Threshold: ≤ 1,800 s |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (`interface/lm_train.py`, `interface/stages.py`): AdamW at 1e-3 (β = 0.9, 0.95; weight decay 0.1 on matrices), 150 warm-up steps then cosine decay to 0.1, clipping at 1.0, bf16 autocast with fp32 weights; the loss is your `model.loss(x, y)`.

**Decision**

- A submission is valid only if every threshold holds for the 2-seed means; valid submissions are ranked by held-out loss at 8 loops. Differences smaller than the spread over the two seeds are treated as ties.
- If any run crashes or exceeds a time limit (the stage is killed at 2,100 s), the whole submission is invalid.
- The final ABench score is driven mainly by the primary metric relative to the baseline, and also considers how different the model is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe), the evaluator also trains the 4-layer baseline in `interface/baselines/fourlayer/` and reports your loss relative to it.
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
├── model.py            # required: build(cfg)
├── src/                # optional: helper modules imported by model.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The model must be built from `cfg` alone and trained by the evaluator; no pretrained weights, files, environment variables or network access.
- `loops` must control the amount of computation: the evaluator compares losses across loop counts, and the compute gate is measured at 8 loops.
- The model must be causal; the training loss may add terms (for example over several loop counts) but must stay a next-token objective on the given batch.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
