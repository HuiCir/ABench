# A-MOE-003: Expert capacity and routing for a small mixture-of-experts language model at fixed active and total parameters

Anchor: William Fedus, Barret Zoph, Noam Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (arXiv:2101.03961). Baseline used for comparison: top-2 of 8 SwiGLU experts with the Switch load-balancing loss, run by the evaluator under identical conditions (`interface/baselines/top2/`).

## Task

Design the **feed-forward layer of a mixture-of-experts language model** — how many experts and of what size, how tokens are routed, how load is balanced — that trains a fixed 8-layer decoder to the **lowest held-out loss** without exceeding the **total parameters of eight dense MLPs** or the **active compute of two** per token.

- Submit the layer only (`moe.py`). At evaluation the evaluator puts your layer in every block of the fixed μ2 model (attention, normalisation, embeddings, data, optimiser and schedule are fixed) and trains it from scratch for 3,000 steps of 32 × 1,024 tokens, once for each of 3 seeds unknown to you. It trains the top-2 baseline the same way.
- The evaluator measures the compute itself: forward FLOPs per token on held-out data must stay within 5% of a dense model with two SwiGLU MLPs of width 1,024 per layer, so computing all experts and masking does not fit.
- Routing must be per token and causal in evaluation mode: a token's output may not depend on later tokens or on other sequences in the batch (no batch-level capacity limits at evaluation).
- Load balance and expert utilisation are reported, not scored: they matter only through the loss and the compute.
- You may build on any public work (Switch and top-k routing, expert choice, fine-grained and shared experts, balancing losses and loss-free balancing, router regularisation), but the design report must state what you borrowed and what you changed.

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

Implement a layer class in `moe.py` and export it as `FFN`: `FFN(d_model, layer, n_layers)` is an `nn.Module` whose `forward(x)` maps (B, T, d_model) to `(y, aux)`; `aux` may hold `loss` (a scalar added to the training loss) and `counts` (tokens per expert, for the load metrics). `interface/moe_lm.py` shows where it is called and the budgets.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy (nats) on the 4,194,304 held-out tokens in windows of 1,024, mean over 3 seeds; lower is better | — |
| Loss vs top-2 | Held-out loss minus the baseline's on the same seeds | reported |
| Active compute | Forward FLOPs per token of the model on 8 held-out windows (FlopCounterMode) divided by those of the dense model with two MLPs of width 1,024 per layer | Threshold: ≤ 1.05 |
| Total parameters | Largest per-layer parameter count of your layer divided by 8 · 3 · d · 1,024 + 64 · d (eight dense MLPs and a router) | Threshold: ≤ 1.0 |
| Causality and batch independence | In evaluation mode, outputs at positions ≤ t unchanged when later tokens change, and a sequence's outputs unchanged when the other sequences change (relative tolerance 1e-3) | Threshold: both hold |
| Load imbalance | Per layer, the largest expert share of held-out tokens times the number of experts (1 = balanced), mean over layers | reported |
| Expert utilisation | Fraction of experts receiving at least 1% of a layer's held-out tokens | reported |
| Throughput | Training tokens per second, and its ratio to the baseline's | reported |
| Training time | Seconds of the 3,000 steps (parent clock) | Threshold: ≤ 1,800 s |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (`interface/lm_train.py`, `interface/stages.py`): AdamW at 1e-3 (β = 0.9, 0.95; weight decay 0.1 on matrices only, including expert weights), 150 warm-up steps then cosine decay to 0.1, gradient clipping at 1.0, bf16 autocast with fp32 weights; the attention and embedding initialisation is fixed, your layer initialises itself.

**Decision**

- A submission is valid only if every threshold holds in every seed; valid submissions are ranked by held-out loss. Differences smaller than the spread over the three seeds are treated as ties.
- If any run crashes or exceeds a time limit (the stage is killed at 2,100 s), the whole submission is invalid.
- The final ABench score is driven mainly by the held-out loss relative to the baseline, and also considers how different the routing is from existing designs and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe), the evaluator also trains the model with the top-2 baseline layer in `interface/baselines/top2/` and reports your loss and throughput relative to it.
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
├── moe.py              # required: FFN
├── src/                # optional: helper modules imported by moe.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The layer may only transform its input; it must not read files, environment variables or the network, or change the rest of the model, the optimiser or the data.
- Computation must use PyTorch operators that the FLOP counter sees (no custom CUDA or Triton kernels that hide matrix multiplications).
- Non-gradient state (for example balancing biases) may be updated during training forward passes only.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
