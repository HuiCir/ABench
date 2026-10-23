# C-SPEC-001: Lossless speculative decoding for a 1.5B target with a 0.5B draft: end-to-end speedup on five workloads

Anchor: Yaniv Leviathan, Matan Kalman, Yossi Matias, *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192). Baseline used for comparison: speculative decoding with the provided draft model, greedy verification and an adaptive draft length (the reference submission; its code is shown under `interface/baseline/`); speedups are measured against the evaluator's plain greedy decoding.

## Task

Design a **drafting and verification strategy** that makes greedy generation with **Qwen2.5-1.5B-Instruct** as fast as possible end to end while producing **exactly the target model's greedy output**, on five workloads — including one where a small draft model is expected to agree rarely.

- Submit the code and an optional checkpoint (for example trained draft heads, at most 2 GB). The evaluator calls `generate()` once per prompt (batch size 1, at most 256 new tokens) and compares the tokens with its own greedy decoding of the target in fp32; it times both on the same GPU.
- The provided draft model Qwen2.5-0.5B-Instruct may be used, modified or replaced by something you train in the sandbox; verification must use the provided target weights.
- Workloads: math word problems (GSM8K), code completion (HumanEval), news summarisation and German translation (CNN/DailyMail), open instructions (Dolly) — 32 held-out prompts each.
- You may build on any public work (draft models, self-drafting heads, tree verification, n-gram lookup, adaptive speculation), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/dev_prompts.npz` (read-only): 16 development prompts per workload, already chat-formatted and tokenised for the target (`<workload>_tokens`, `<workload>_offsets`: prompt i is `tokens[offsets[i]:offsets[i+1]]`), drawn from other splits than the held-out prompts. `$ABENCH_MODELS_DIR/Qwen2.5-1.5B-Instruct` and `Qwen2.5-0.5B-Instruct`: the target and draft models (Apache-2.0).

| Workload | Source (licence) | Held-out prompts from |
|---|---|---|
| `math` | GSM8K (MIT) | test split |
| `code` | HumanEval (MIT) | the last 64 problems |
| `summarize` | CNN/DailyMail 3.0.0 articles, at most 1,500 tokens (Apache-2.0) | test split |
| `chat` | Dolly 15k instructions with context (CC BY-SA 3.0) | a random half |
| `translate` | CNN/DailyMail highlights into German (Apache-2.0) | test split |

The 160 held-out prompts are chosen with a secret seed and are not in the sandbox; you may build your own training prompts from anything you generate with the provided models.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `generate(model, prompt_ids, max_new_tokens) -> list[int]` in `model.py`. `prompt_ids` is a list of token ids (chat template applied); `generate` returns the new tokens, stopping after the target's end-of-sequence token or at `max_new_tokens`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Geometric-mean speedup** (primary) | Per workload, the evaluator's total time of plain greedy decoding (target in fp32 with its key-value cache) divided by your total `generate()` time over the same prompts; geometric mean over the five workloads; higher is better | — |
| Speedup per workload | The same per workload, and the smallest | reported |
| Exactness | Fraction of prompts whose output equals the target's greedy output, a first difference being accepted only where the target's two best logits differ by less than 1e-3 | Threshold: = 1.0 |
| Load time | Seconds from the start signal to the return of `load()` | reported (killed at 600 s) |

**Timing and decision**

- Loading, the evaluator's greedy decoding and your `generate()` run in one new process on the reference GPU, timed with device synchronisation; prompt by prompt the evaluator's decoding runs first, then yours. All 160 prompts must finish within 3,000 seconds.
- A submission is valid only if every output is exact; valid submissions are ranked by geometric-mean speedup. Differences below 3% are treated as ties (timing noise).
- If loading or decoding crashes or exceeds a time limit, the submission is invalid.
- The final ABench score is driven mainly by the speedup, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and generate()
├── train.py            # required: reproduces the checkpoint (may only write settings)
├── src/                # optional: other .py modules
├── checkpoint/         # required: trained draft parameters or settings, at most 2 GB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Verification must use the provided target weights; the output must be the target's greedy output, not an approximation.
- The checkpoint may only be trained from the provided models and data generated with them; no other pretrained models or external data.
- `generate` must not keep results between calls (no caching of earlier prompts' outputs) and must not depend on the order or identity of the prompts.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
