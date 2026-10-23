# A-LIN-002: A better state-update rule for linear attention, measured by associative recall

Anchor: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim, *Gated Linear Attention Transformers with Hardware-Efficient Training* (arXiv:2312.06635). Baseline used for comparison: a gated linear attention layer in plain PyTorch (`interface/baselines/gla/`), trained by the evaluator with the same recipe and seeds.

## Task

Design a **linear-time token mixer** — a recurrent state-update rule in the family of linear attention (decay, gating, delta rule and beyond) — that, placed in a fixed two-layer model and trained with a fixed recipe on multi-query associative recall (MQAR), **recalls the most key-value associations on longer sequences with more pairs than seen in training**, while training as fast as a linear-attention layer should.

- Submit your mixer (`mixer.py`) as code. The evaluator trains the fixed model built with it from scratch, with the fixed recipe in `interface/stages.py` (`PARAMS`), on two secret seeds, then tests recall on a held-out grid of sequence lengths and pair counts and measures causality and forward+backward throughput. The recipe is fixed so that differences in recall come from the update rule, not from training compute.
- The GLA baseline is trained by the evaluator alongside your mixer under identical conditions.
- The mixer must be causal and sub-quadratic in sequence length: no softmax attention over the full sequence, no sliding-window attention used as a substitute for the state. Both properties are checked.
- You may build on any public work (GLA, DeltaNet and gated variants, Mamba-style selective state spaces, RetNet, chunkwise-parallel training forms), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data file. Training batches come from `interface/mqar.py` (after Zoology, Arora et al. 2023): a sequence of length T starts with K key-value pairs, then each key reappears once at a random later position, and the model must output that key's value there. Vocabulary 8,192; keys from the lower half, values from the upper half; filler token 0.

Training uses the (T, K) grid `mqar.TRAIN_GRID` = (64, 4), (64, 8), (128, 8), (128, 16), (256, 16), (256, 32), (256, 64). The held-out set is 256 sequences per cell of a grid with T up to 2,048 and K up to 256, generated with a secret seed; it is not in the sandbox. The self-check uses `DEV_GRID` in `interface/stages.py` with a public seed.

## Interface

Implement `class Mixer(torch.nn.Module)` in `mixer.py` with `__init__(self, d_model, n_heads, layer_idx)` and `forward(x) -> y`, mapping (B, T, d_model) to (B, T, d_model) causally. The evaluator builds `lin_model.Model(Mixer)` (d_model = 128, n_heads = 4, two layers; `interface/lin_model.py`) and trains it with the recipe in `interface/stages.py`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **MQAR accuracy** (primary) | Fraction of query positions where argmax of the model's output equals the key's value, per held-out (T, K) cell, averaged over cells and over the two seeds; higher is better | — |
| Causality | Relative change of outputs at positions < 256 when tokens from position 256 on are replaced (sequence length 512) | Threshold: ≤ 1e-3 |
| Scaling exponent | 1 + log(t₄₀₉₆ / t₁₀₂₄) / log 4, where t_T is the forward+backward time of the model on 32,768 tokens arranged as sequences of length T (1 = linear, 2 = quadratic) | Threshold: ≤ 1.3 |
| Throughput ratio | Forward+backward tokens per second at T = 4,096, divided by that of the GLA-baseline model measured by the evaluator in the same way | Threshold: ≥ 0.5 |
| Training time | Wall-clock seconds of the fixed 8,000-step recipe | Threshold: ≤ 1,500 s |
| Per-cell accuracy, accuracy relative to GLA | Reported for each held-out cell; your accuracy minus the GLA baseline's | reported |

**Decision**

- A submission is valid only if every threshold holds; valid submissions are ranked by MQAR accuracy. Differences inside the 95% interval reported with the score are treated as ties.
- If training or the measurements crash or exceed a time limit (training is killed at 1,620 s; held-out predictions and measurements must finish within 300 s), the submission is invalid.
- The final ABench score is driven mainly by MQAR accuracy, and also considers how different the update rule is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Under exactly the same conditions (fixed recipe, seeds, held-out grid, GPU), the evaluator also trains the model with the gated-linear-attention baseline in `interface/baselines/gla/` and reports your recall and throughput relative to it.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── mixer.py            # required: class Mixer
├── src/                # optional: other .py modules imported by mixer.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The submission is code only; no weights, checkpoints or stored tensors. Parameters start from your mixer's initialisation and are trained only by the evaluator's recipe.
- The mixer must be causal and its cost must grow linearly (up to logarithmic factors) with sequence length; no full or sliding-window softmax attention.
- The mixer may not inspect token ids or the MQAR structure; it sees only its input activations. The rest of the model (`lin_model.py`) and the training recipe are fixed.
- The mixer may not read files, the environment, the clock or the random state to change its behaviour between training and evaluation, and may not change the optimiser, schedule or other model parameters.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
