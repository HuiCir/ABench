# A-LIN-002: A better state-update rule for linear attention, measured by associative recall

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim, *Gated Linear Attention Transformers with Hardware-Efficient Training* (arXiv:2312.06635). Baseline used for comparison: a gated linear attention layer in plain PyTorch (`interface/baselines/gla/`; its trained reference is `interface/baseline/`).

## Task

Design a **linear-time token mixer** — a recurrent state-update rule in the family of linear attention (decay, gating, delta rule and beyond) — that, placed in a fixed two-layer model and trained on multi-query associative recall (MQAR), **recalls the most key-value associations on longer sequences with more pairs than seen in training**, while training as fast as a linear-attention layer should. Develop and train it on one 24GB GPU in the sandbox, then submit the trained model.

- Submit your mixer (`mixer.py`), the trained checkpoint of the fixed model built with it (`checkpoint/model.pt`) and the training program that produced it. The evaluator loads the checkpoint, tests recall on a held-out grid of sequence lengths and pair counts, and measures causality and forward+backward throughput; it never retrains.
- Training data may come only from `interface/mqar.py` on the training grid (`mqar.TRAIN_GRID`); the training recipe (steps, optimiser, schedule) is yours to choose and must be in `train.py`.
- The mixer must be causal and sub-quadratic in sequence length: no softmax attention over the full sequence, no sliding-window attention used as a substitute for the state. Both properties are checked.
- You may build on any public work from before the cutoff (GLA, DeltaNet and gated variants, Mamba-style selective state spaces, RetNet, chunkwise-parallel training forms), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data file. Training batches come from `interface/mqar.py` (after Zoology, Arora et al. 2023): a sequence of length T starts with K key-value pairs, then each key reappears once at a random later position, and the model must output that key's value there. Vocabulary 8,192; keys from the lower half, values from the upper half; filler token 0.

Training uses the (T, K) grid `mqar.TRAIN_GRID` = (64, 4), (64, 8), (128, 8), (128, 16), (256, 16), (256, 32), (256, 64). The held-out set is 256 sequences per cell of a grid with T up to 2,048 and K up to 256, generated with a secret seed; it is not in the sandbox. The self-check uses `DEV_GRID` in `interface/stages.py` with a public seed.

## Interface

Implement `class Mixer(torch.nn.Module)` in `mixer.py` with `__init__(self, d_model, n_heads, layer_idx)` and `forward(x) -> y`, mapping (B, T, d_model) to (B, T, d_model) causally. The evaluator builds `lin_model.Model(Mixer)` (d_model = 128, n_heads = 4, two layers; `interface/lin_model.py`) and loads `checkpoint/model.pt`, its `state_dict()`, written by your `train.py`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **MQAR accuracy** (primary) | Fraction of query positions where argmax of the model's output equals the key's value, per held-out (T, K) cell, averaged over cells; higher is better | — |
| Causality | Relative change of outputs at positions < 256 when tokens from position 256 on are replaced (sequence length 512) | Threshold: ≤ 1e-3 |
| Scaling exponent | 1 + log(t₄₀₉₆ / t₁₀₂₄) / log 4, where t_T is the forward+backward time of the model on 32,768 tokens arranged as sequences of length T (1 = linear, 2 = quadratic) | Threshold: ≤ 1.3 |
| Throughput ratio | Forward+backward tokens per second at T = 4,096, divided by that of a freshly initialised GLA-baseline model timed in the same process | Threshold: ≥ 0.5 |
| Per-cell accuracy, load time | Reported for each held-out cell; seconds to build the model and load the checkpoint | reported |

**Decision**

- A submission is valid only if every threshold holds; valid submissions are ranked by MQAR accuracy.
- If loading or the measurements crash or exceed a time limit (loading is killed at 120 s; held-out predictions and measurements must finish within 600 s), or `model.pt` does not load into `Model(Mixer)` with `strict=True`, the submission is invalid.
- The final ABench score is driven mainly by MQAR accuracy, and also considers how different the update rule is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. In the same evaluation process, the evaluator also times a freshly initialised model with the gated-linear-attention baseline in `interface/baselines/gla/` and reports your throughput relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy.
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── mixer.py            # required: class Mixer
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: model.pt, the state_dict of Model(Mixer), at most 100 MB
├── src/                # optional: other .py modules imported by mixer.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights, no data other than `mqar.batch` on `mqar.TRAIN_GRID` and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- The mixer must be causal and its cost must grow linearly (up to logarithmic factors) with sequence length; no full or sliding-window softmax attention.
- The mixer may not inspect token ids or the MQAR structure; it sees only its input activations. The rest of the model (`lin_model.py`) is fixed.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
