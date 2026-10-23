# A-NORM-005: A normalisation and residual scheme for small language models that stays stable at a high learning rate (two scales, two learning rates)

Anchor: Biao Zhang, Rico Sennrich, *Root Mean Square Layer Normalization* (arXiv:1910.07467). Baseline used for comparison: pre-normalisation with RMSNorm (the anchor, as in Llama), run by the evaluator under identical conditions (`interface/baselines/prenorm/`).

## Task

Design a **normalisation and residual scheme** — where activations are normalised, with what, how the residual stream is formed and scaled — that lets a fixed decoder-only language model train to the **lowest held-out loss at two scales and two learning rates**, including a learning rate four times higher than usual, where loss spikes and divergence are common.

- Submit the scheme only (`norm.py`). At evaluation the evaluator trains the fixed model (attention and MLP sublayers, tied embeddings, data, optimiser and schedule are all fixed) from scratch with your scheme in four runs — scale μ1 (4 layers, width 256, 1,500 steps) and μ2 (8 layers, width 384, 3,000 steps), each at learning rates 1e-3 and 4e-3 — once for each of 2 seeds unknown to you. It trains the pre-RMSNorm baseline the same way.
- The scheme may add normalisation parameters and cheap element-wise operations, but no matrix multiplications: forward FLOPs per token may exceed the baseline's by at most 2%.
- This is a calibration task with a mature anchor; improvements must hold at both scales. You may build on any public work (pre-, post- and sandwich normalisation, residual scaling and gating, QK normalisation, initialisation schemes), but the design report must state what you borrowed and what you changed.

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

Read them with `interface/token_data.py`. The evaluator samples training windows from both shards with the seed. The held-out set is the first 4,194,304 tokens of the official FineWeb validation shard (`fineweb_val_000000.bin`); it is not in the sandbox. The self-check holds out the last 2,097,152 tokens of the second shard.

## Interface

Implement a scheme class in `norm.py` and export it as `SCHEME`: `SCHEME(d_model, n_layers, n_heads, head_dim)` with `make_layer(layer) -> nn.Module`, `layer_forward(mod, x, attn, mlp, layer) -> x` and `make_final() -> nn.Module`, optionally `qk(mod, q, k, layer)`, `embed(final, x)` and `init(model)`. `interface/norm_lm.py` shows exactly where each is called.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean held-out loss** (primary) | Mean next-token cross-entropy (nats) on the 4,194,304 held-out tokens in windows of 1,024, averaged over the four runs and the 2 seeds; a diverged run counts as ln(50,304) = 10.83; lower is better | — |
| Loss per run | The same for each scale and learning rate | reported |
| High-learning-rate penalty | Loss at 4e-3 minus loss at 1e-3, mean over scales | reported |
| Loss spikes | Training steps after warm-up whose loss exceeds 1.15 × the median of the previous 100 steps (a multi-step spike counts once), summed over runs | reported |
| Diverged runs | Runs whose training loss became non-finite or exceeded 20 | reported |
| Compute | Forward FLOPs per token (FlopCounterMode, sequence 1,024) relative to the baseline scheme at the same scale, the largest over runs | Threshold: ≤ 1.02 |
| Parameters | Model parameters relative to the baseline scheme | Threshold: ≤ 1.001 |
| Training time | Seconds of a μ2 run (parent clock), the largest | Threshold: ≤ 1,500 s |

**Fixed training setting** (`interface/lm_train.py`, `interface/stages.py`): AdamW (β = 0.9, 0.95; weight decay 0.1 on matrices only), 150 warm-up steps then cosine decay to 0.1 of the peak, gradient clipping at 1.0, batches of 32 × 1,024 tokens, bf16 autocast with fp32 weights; initialisation N(0, 0.02) with residual output projections scaled by 1/√(2 · layers), after which your `init` may rescale weights.

**Decision**

- A submission is valid only if every threshold holds for the 2-seed means; valid submissions are ranked by mean held-out loss. Differences between submissions smaller than the spread over the two seeds are treated as ties.
- If any run crashes or exceeds a time limit (a μ2 stage is killed at 1,800 s), the whole submission is invalid.
- The final ABench score is driven mainly by the mean held-out loss relative to the baseline, and also considers how different the scheme is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe, scales and learning rates), the evaluator also trains the model with the pre-RMSNorm baseline in `interface/baselines/prenorm/` and reports your losses and spike counts relative to it.
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
├── norm.py             # required: SCHEME
├── src/                # optional: helper modules imported by norm.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The scheme may only normalise, scale, gate or combine the residual stream and the sublayer outputs it is given; it must not replace the attention or MLP sublayers, change the optimiser, the learning rate or the loss, or read files, environment variables or the network.
- `init` may only rescale or re-draw the model's existing weights at initialisation.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
