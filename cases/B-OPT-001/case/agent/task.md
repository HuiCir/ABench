# B-OPT-001: An optimizer for small language-model pretraining

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Diederik P. Kingma, Jimmy Ba, *Adam: A Method for Stochastic Optimization* (arXiv:1412.6980). Baseline used for comparison: AdamW (Loshchilov & Hutter 2019), run by the evaluator under identical conditions (`interface/baselines/adamw/`).

## Task

Design a new **update rule** (optimizer) that, used to pretrain a fixed 32M-parameter decoder-only language model for a fixed budget of 3,000 steps on one 24GB GPU, reaches the **lowest held-out loss**, without being slower per step than allowed.

- Submit the update rule only. At evaluation the evaluator trains the fixed model from scratch with your rule once for each of 3 seeds unknown to you, then measures the loss on held-out tokens. It trains the AdamW baseline the same way.
- The rule must be new: not a renamed or re-tuned existing optimizer or a plain combination of existing ones. Its behaviour should be distinguishable from existing methods, and the design report must state precisely what you borrowed (including ideas from other fields such as numerical linear algebra) and where it is in the code.
- The evaluator does not tune hyperparameters: it uses the defaults you declare in `hparam_space.yaml`.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
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

Implement a functional update rule in `optimizer.py` (subclass `optim_api.FunctionalRule`, export the class as `RULE`) and declare its hyperparameters in `hparam_space.yaml`. The training framework wraps the rule as a torch optimizer, applies it to every parameter (2D matrices and 1D gains alike) and runs the fixed loop in `interface/stages.py` on the model in `interface/lm_model.py`.


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy (nats) of the trained model on the 4,194,304 held-out tokens in windows of 1,024, mean over 3 seeds; lower is better | — |
| Loss vs AdamW | Held-out loss minus the AdamW baseline's held-out loss on the same seeds | reported |
| Step-time ratio | Training time of your run divided by that of the AdamW baseline, mean over seeds | Threshold: ≤ 1.25 |
| Training time | Seconds from the start signal to the end of the 3,000 steps (includes importing `optimizer.py`), mean over seeds | Threshold: ≤ 1,500 s |
| Optimizer state | Elements of all state tensors divided by the number of model parameters | Threshold: ≤ 3.0 |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (all in `interface/stages.py`): model 8 layers, width 384, 6 query / 2 key-value heads, SwiGLU 1,024, RoPE, RMSNorm, tied embeddings (about 32M parameters); 3,000 steps of 32 × 1,024 tokens (about 98M tokens); bf16 autocast with fp32 weights; the framework's learning-rate factor is linear warm-up over 150 steps then cosine decay to 0.1 (unless your rule sets `owns_schedule: true`); 1D parameters use `lr × 0.15`.

**Decision**

- A submission is valid only if every threshold holds for the 3-seed means; valid submissions are ranked by held-out loss.
- If any seed crashes or exceeds a time limit (the training stage is killed at 1,620 s), the whole submission is invalid.
- The final ABench score is driven mainly by the held-out loss relative to the baseline, and also considers how different the rule is from existing optimizers and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, schedule), the evaluator also trains the model with the AdamW baseline in `interface/baselines/adamw/` and reports your loss and step time relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── optimizer.py        # required: exports RULE (a FunctionalRule subclass)
├── hparam_space.yaml   # required: rule name, ≤ 4 tunable hyperparameters with defaults, fixed hyperparameters
├── src/                # optional: helper modules imported by optimizer.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The rule may use only `(p, g, state, step, lr)` and its own hyperparameters; it must not read files, environment variables or the network, inspect the model or data, or modify `p` or `g` in place.
- All tensor arithmetic goes through `ops` (`optim_api.Ops`); state lives only in the `state` dict.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
