# A-LCTX-006: Position encoding for context extension: a 32M language model trained at 1,024 tokens, used at up to 16,384

Anchor: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole, *YaRN: Efficient Context Window Extension of Large Language Models* (arXiv:2309.00071). Baseline used for comparison: rotary position embedding (base 10,000) with no change beyond the training length, trained by the evaluator under identical conditions (`interface/baselines/rope/`).

## Task

Design the **position mechanism** of a fixed decoder — how queries and keys carry position, and how that changes beyond the training length (rotary variants, frequency interpolation and extrapolation rules, attention temperature, learnable or data-dependent position signals) — so that a **32M-parameter language model trained at 1,024 tokens** and briefly extended at 4,096 **uses contexts of 4,096 to 16,384 tokens well**, without losing quality at 1,024.

- Submit code only: `positions.py` with `make_positions(head_dim, n_heads, n_kv_heads, train_len)` returning a module that maps (q, k) to (q, k) inside every attention layer; it sees the sequence length and may behave differently beyond `train_len`. At most 65,536 parameters.
- The evaluator trains the fixed model (B-OPT-001's μ2 decoder) from scratch with your module: 3,000 steps at 1,024 tokens, then 200 steps at 4,096 tokens (the extension phase), and measures loss on PG-19 books with 4,096, 8,192 and 16,384 tokens of context, retrieval of a repeated segment across 8,192 and 16,384 tokens, and loss at 1,024. It trains the RoPE baseline the same way.
- Attention stays exact and causal (scaled dot-product attention after your module); attention biases and changes to the rest of the model are out of scope.
- You may build on any public work (RoPE, position interpolation, NTK-aware and YaRN scaling, LongRoPE, ALiBi-like decay expressed through queries and keys, NoPE, xPos, attention temperature), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): the B-OPT-001 training shards `fineweb_train_000001.bin` and `fineweb_train_000002.bin` (FineWeb-10B GPT-2 tokens, `kjj0/fineweb10B-gpt2` @889765ea, MIT; text ODC-BY), and `pg19_dev.npz` with `windows`: one 16,385-token window from each PG-19 validation book long enough (Rae et al. 2020, Apache-2.0; GPT-2 tokenizer).

The held-out set is 4,194,305 FineWeb validation tokens (short-context loss), one 16,385-token window from each of the 100 PG-19 test books (offsets from a secret seed), and copy-test sequences (64 of 8,192 and 32 of 16,384 tokens, `interface/longctx.py`) built from the FineWeb validation tokens with the same seed.

## Interface

Implement `make_positions(head_dim, n_heads, n_kv_heads, train_len) -> torch.nn.Module` in `positions.py`; the module's `forward(q, k)` takes q (B, n_heads, T, head_dim) and k (B, n_kv_heads, T, head_dim) and returns transformed q and k of the same shapes. One module is shared by all layers; its parameters (at most 65,536) are trained with the model.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Long-context loss** (primary) | Mean next-token loss on the last 1,024 tokens of the PG-19 windows when the model reads 4,096, 8,192 and 16,384 tokens, averaged over the three lengths and over the two training seeds; lower is better | — |
| Short-context loss | Held-out FineWeb loss at 1,024 tokens, minus the RoPE baseline's in the same run | Threshold: ≤ +0.02 |
| Copy test | Loss on a 64-token segment repeated about 3/4 of the context after its first occurrence, at 8,192 and 16,384 tokens | reported |
| Per length | Loss at each length; differences to RoPE | reported |
| Size and stability | Parameters of the positions module; divergence in either training phase | Threshold: ≤ 65,536; no divergence |

**Measurement and decision**

- A submission is valid if it trains without divergence within the time limit (2,100 s for both phases), keeps the short-context loss within 0.02 of RoPE's, and its module has at most 65,536 parameters; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over windows; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by the long-context loss, and also considers how different the mechanism is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe, extension phase), the evaluator also trains the model with RoPE from `interface/baselines/rope/` and reports your losses relative to it; the short-context requirement is relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── positions.py        # required: make_positions()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The module may only transform queries and keys from their values, positions and the sequence length; it must not look at token ids, change other parts of the model, or depend on anything outside its inputs.
- Causality must hold: no position may receive information from later positions.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
