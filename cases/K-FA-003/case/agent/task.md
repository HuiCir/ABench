# K-FA-003: An attention forward kernel faster than PyTorch SDPA

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (arXiv:2205.14135). Baseline used for comparison: PyTorch `scaled_dot_product_attention` with the flash backend, timed by the evaluator on every shape.

## Task

Write an **exact attention forward kernel** for bf16 inputs on one 24GB reference GPU that is **as fast as possible relative to PyTorch's `scaled_dot_product_attention` (flash backend)** across a set of shapes you do not know in advance, while staying numerically as accurate as SDPA.

- Submit code only: `kernel.py` with `attention(q, k, v, causal)`. At evaluation it is called on the held-out shapes with random inputs, checked against an fp32 reference, and timed against SDPA on the same GPU, once for each of 3 seeds (separate processes).
- Triton and hand-written CUDA via PyTorch's inline extension loading are allowed; calling SDPA, FlashAttention, xFormers or any other attention library from your code is not.
- You may build on any public work from before the cutoff (FlashAttention 1–3, tiling and warp specialisation strategies, Triton tutorials), but the design report must state what you borrowed and what you changed. The task tests your understanding of the hardware: memory hierarchy, tiling, occupancy and numerics.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data file. Inputs are random bf16 tensors of layout (B, H, T, D) drawn on the GPU from a seed.

The held-out set is a list of about a dozen shapes (B, H, T, D, causal) with head dimension 64 or 128, causal and non-causal masking, sequence lengths from a few hundred to 8,192 including values that are not multiples of 64, and B·H·T² small enough for an fp32 reference. It is not in the sandbox; the self-check uses six public shapes listed in `interface/stages.py`.

## Interface

Implement `attention(q, k, v, causal: bool) -> torch.Tensor` in `kernel.py`. Inputs are contiguous bf16 CUDA tensors of shape (B, H, T, D) (same T for queries and keys); the output has the same shape and dtype; the softmax scale is 1/sqrt(D); with `causal=True`, query i attends to keys 0..i.


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Relative speed** (primary) | Geometric mean over the held-out shapes of t_SDPA / t_yours, where each t is the median of 20 CUDA-event-timed calls after 5 warm-up calls, then mean over 3 seeds; higher is better | — |
| Correctness | On every shape, max \|out − ref\| ≤ max(0.02, 4 × SDPA's own max error) and mean \|out − ref\| ≤ max(0.002, 4 × SDPA's mean error), against an fp32 reference | Threshold: all shapes |
| Forbidden APIs | Occurrences of SDPA, FlashAttention, xFormers or cuDNN-attention entry points in your source files | Threshold: 0 |
| Slowest shape | The smallest per-shape t_SDPA / t_yours | reported |

**Timing and decision**

- Each seed runs in a new process with empty compiler caches (`interface/abench_stage.py`). Importing `kernel.py` (including any compilation or autotuning at import) must finish within 600 s; all held-out shapes, including the reference computations and timing, must finish within 900 s.
- A submission is valid only if every shape is correct and no forbidden API is used; valid submissions are ranked by relative speed.
- If any seed crashes, exceeds a time limit or returns a wrong shape, the whole submission is invalid.
- The final ABench score is driven mainly by relative speed, and also considers how different the method is from existing kernels and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton; `torch.utils.cpp_extension.load_inline` with the CUDA toolkit), NumPy.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 1 GPU-hour.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── kernel.py           # required: attention()
├── src/                # optional: helper modules, .py only (CUDA sources go in Python strings)
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No attention library calls (SDPA, FlashAttention, xFormers, cuDNN attention) and no stored results; the kernel must compute exact attention on the inputs it is given.
- Do not cache outputs across calls or detect the timing loop; every call must do the full computation.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
