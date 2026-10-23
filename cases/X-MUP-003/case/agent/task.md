# X-MUP-003: A parametrisation whose learning rate, tuned on a narrow model, stays optimal on models four and eight times wider

Anchor: Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (arXiv:2203.03466). Baseline used for comparison: the standard parametrisation (width-independent initialisation and one learning rate), run by the evaluator under the same protocol (`interface/baselines/sp/`).

## Task

Design a **parametrisation** — per-tensor initialisation, per-tensor learning-rate and weight-decay multipliers, output and attention scales — so that **a learning rate tuned on a width-128 model transfers** to models of width 512 and 1,024, and the width-1,024 model trained with it reaches the **lowest held-out loss**.

- Submit the parametrisation only (`param.py`). At evaluation the evaluator runs a fixed protocol once for each of 2 seeds unknown to you: it trains the 8-layer decoder at width 128 with 7 global learning rates (2⁻¹² to 2⁻⁶), picks the best on a tuning set, trains width 512 with that rate times ¼, ½, 1, 2 and 4 (to measure the transfer gap) and width 1,024 with that rate. Every run uses the same data, steps, optimiser and schedule; only your parametrisation and the global rate differ.
- The primary metric rewards both transfer and the quality of the parametrisation itself: the loss of the wide model trained with the transferred rate. The transfer gap at width 512 is reported.
- You may build on any public work (μP and its variants, unit scaling, depth and width scaling rules, per-layer learning rates), but the design report must state what you borrowed and what you changed.

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

Read them with `interface/token_data.py`. The evaluator's tuning set is the last 1,048,576 tokens of the second shard (not used for training). The held-out set is the first 4,194,304 tokens of the official FineWeb validation shard; it is not in the sandbox. The self-check holds out the last 2,097,152 tokens of the second shard and tunes on the 1,048,576 before them.

## Interface

Implement a class in `param.py` and export it as `PARAM` with `init_std(name, shape, width, base_width)`, `lr_scale(name, shape, width, base_width)` and `output_multiplier(width, base_width)`, optionally `wd_scale(...)` and `attention_scale(head_dim, width, base_width)`; base_width is 128. `interface/mup_lm.py` shows exactly how each is applied.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss at width 1,024** (primary) | Mean next-token cross-entropy (nats) on the 4,194,304 held-out tokens in windows of 1,024 of the width-1,024 model trained with the learning rate selected at width 128, mean over 2 seeds; lower is better | — |
| Transfer gap at width 512 | Tuning-set loss at width 512 with the selected rate minus the best over the five multiples | reported |
| Optimum shift at width 512 | log₂ of the multiple with the lowest width-512 loss (0 = perfect transfer) | reported |
| Held-out loss at width 512 | With the selected rate | reported |
| Width-128 sweep | Tuning-set loss of the 7 rates, the selected rate, and how many rates lie within 0.02 of the best | reported |
| Protocol time | Seconds of the whole timed protocol (13 training runs, parent clock) | Threshold: ≤ 3,600 s |

**Fixed protocol** (`interface/stages.py`, `interface/lm_train.py`, `interface/mup_lm.py`): 8 layers, head dimension 64, MLP width 2.75 × width, tied embeddings; each run 1,500 steps of 32 × 1,024 tokens, AdamW (β = 0.9, 0.95; weight decay 0.1 × your `wd_scale` on matrices), per-tensor rate η × `lr_scale`, 150 warm-up steps then cosine decay to 0.1, clipping at 1.0, bf16 autocast with fp32 weights. A diverged run counts as ln(50,304) = 10.83.

**Decision**

- A submission is valid only if the time threshold holds in every seed; valid submissions are ranked by held-out loss at width 1,024. Differences smaller than the spread over the two seeds are treated as ties.
- If a seed's protocol crashes or exceeds the time limit (killed at 4,200 s), the submission is invalid.
- The final ABench score is driven mainly by the primary metric relative to the baseline, and also considers how different the parametrisation is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same protocol (hardware, seeds, data, sweep, transfer), the evaluator also runs the standard parametrisation in `interface/baselines/sp/` and reports your losses and transfer gap relative to it.
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
├── param.py            # required: PARAM
├── src/                # optional: helper modules imported by param.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The parametrisation may depend only on the parameter's name and shape, the width and the base width; it must not read files, environment variables or the network, or depend on the seed or the data.
- It may not change the architecture, the optimiser, the schedule or the number of steps.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
