# U-DLM-004: A noise process and objective that narrow the likelihood gap between masked diffusion and autoregressive language models (two scales)

Anchor: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov, *Simple and Effective Masked Diffusion Language Models* (arXiv:2406.07524). Baseline used for comparison: MDLM (linear schedule, the bound as the training loss), run by the evaluator under identical conditions (`interface/baselines/mdlm/`).

## Task

Design the **noise process and training objective of a masked (absorbing-state) diffusion language model** — per-position masking schedules, time sampling, weighting, auxiliary terms — so that a fixed denoiser reaches the **lowest likelihood bound** on held-out text at two scales, closing as much as possible of the gap to an autoregressive model of the same size.

- Submit the recipe only (`recipe.py` with `loss(model, x)` and optionally `alpha(t, seq)`). At evaluation the evaluator trains the fixed bidirectional denoiser from scratch with your loss at scale μ2 (8 layers, width 384) and μ3 (10 layers, width 768), 3,000 steps of 32 × 1,024 FineWeb tokens each, once for each of 2 seeds unknown to you, then computes the masked-diffusion bound itself under your schedule. It trains MDLM the same way.
- Your schedule `alpha_i(t)` may differ by position; the bound stays valid for any schedule that decreases from 1 to 0. To keep the model a parallel (non-autoregressive) generator, every position must take at least 0.2 of diffusion time to go from 90% to 10% unmasked.
- The gap to an autoregressive model with the same sizes, data and recipe is reported at both scales, and how the bound improves from μ2 to μ3.
- You may build on any public work (masked and uniform discrete diffusion, learned and state-dependent schedules, block and semi-autoregressive diffusion, variance reduction), but the design report must state what you borrowed and what you changed.

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

Read them with `interface/token_data.py`. The held-out set is the first 2,097,152 tokens of the official FineWeb validation shard (`fineweb_val_000000.bin`); it is not in the sandbox. The self-check holds out the last 2,097,152 tokens of the second training shard. The mask token is id 50,257 (`dlm.MASK`).

## Interface

Implement `loss(model, x) -> scalar tensor` and optionally `alpha(t, seq) -> (B, seq)` in `recipe.py` (default: the linear schedule 1 − t). `x` is an int64 (32, 1,024) batch of clean tokens (μ3 receives two micro-batches of 16 per step); `model(z, t)` maps corrupted tokens and times (B,) to logits (B, T, 50,304); `alpha` maps times t (B,) to the probability that each position is still unmasked. `interface/dlm.py` provides corruption, the bound and helpers.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean NELBO** (primary) | The masked-diffusion bound in nats per token under your schedule on the 2,097,152 held-out tokens (windows of 1,024, 8 stratified times per window), averaged over μ2 and μ3 and the 2 seeds; lower is better | — |
| NELBO and perplexity bound per scale | The same per scale, and exp(NELBO) | reported |
| Gap to autoregression | NELBO minus the held-out loss of the autoregressive reference of the same scale | reported |
| Scale gain | NELBO at μ2 minus NELBO at μ3 | reported |
| NELBO vs MDLM | Differences to the baseline, per scale and on the mean | reported |
| Schedule | alpha in [0, 1], decreasing from 1 to 0; every position needs ≥ 0.2 of time from alpha 0.9 to 0.1 | Threshold: holds (checked before training) |
| Training time | Seconds of the μ2 and μ3 training (parent clock) | Threshold: ≤ 1,800 s (μ2), ≤ 3,600 s (μ3) |

**The bound** (`dlm.nelbo_per_token`): NELBO = Σ_i ∫ −alpha_i'(t) / (1 − alpha_i(t)) · 1[position i masked] · (−log p(x_i | z_t, t)) dt per window, estimated with 8 stratified times and divided by the window length (Sahoo et al. 2024; Shi et al. 2024 for per-position schedules); derivatives by central differences. A diverged run counts as 10.83.

**Fixed training setting** (`interface/lm_train.py`, `interface/dlm.py`): the denoiser is B-OPT-001's decoder without the causal mask, with the time added to the token embeddings through a sinusoidal embedding and a linear map; AdamW at 1e-3 (β = 0.9, 0.95; weight decay 0.1 on matrices), 150 warm-up steps then cosine decay to 0.1, clipping at 1.0, bf16 autocast with fp32 weights.

**Decision**

- A submission is valid only if every threshold holds for the 2-seed means; valid submissions are ranked by mean NELBO. Differences smaller than the spread over the two seeds are treated as ties.
- If any run crashes, fails the schedule check or exceeds a time limit (the stages are killed at 2,100 s and 3,900 s), the whole submission is invalid.
- The final ABench score is driven mainly by the mean NELBO relative to the baseline, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.
- Generative perplexity of samples is not measured in this case (it needs an external judge model); the likelihood bound is the only quality metric.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe, denoiser), the evaluator also trains MDLM from `interface/baselines/mdlm/` and reports your bounds relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── recipe.py           # required: loss(model, x); optional alpha(t, seq)
├── src/                # optional: helper modules imported by recipe.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- `alpha` must be a deterministic function of t, the position and the sequence length only (no dependence on the tokens or the model).
- The loss may only call the model on corruptions of the batch it receives; it must not change the model's parameters other than through the returned loss, read files or the network, or use a second model.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
