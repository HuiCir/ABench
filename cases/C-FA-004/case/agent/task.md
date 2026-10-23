# C-FA-004: A trainable attention: forward and backward faster than PyTorch SDPA in bounded memory

Anchor: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (arXiv:2205.14135). Baseline used for comparison: PyTorch `scaled_dot_product_attention` (flash backend) forward + backward, timed by the evaluator on every shape.

## Task

Write an **exact, trainable attention** — forward pass and the gradients with respect to q, k and v — for bf16 inputs on one 24GB reference GPU that runs **forward + backward as fast as possible relative to PyTorch's `scaled_dot_product_attention` (flash backend)** on shapes you do not know in advance, while staying as accurate as SDPA and never using more than twice its peak memory.

- Submit code only: `kernel.py` with `attention(q, k, v, causal)` returning a tensor that supports `.backward()`. At evaluation it is called on the held-out shapes with random inputs and a random output gradient; the output and the three gradients are checked against an fp32 reference, and forward + backward is timed against SDPA on the same GPU, once for each of 3 seeds (separate processes).
- This is a calibration task with a mature anchor: the backward pass is where a faithful IO-aware implementation earns its speed and memory, by recomputing attention probabilities from saved row statistics instead of storing them.
- Triton and hand-written CUDA via PyTorch's inline extension loading are allowed; calling SDPA, FlashAttention, xFormers or any other attention library from your code is not.
- You may build on any public work (FlashAttention 1–3, the Triton fused-attention tutorial, sequence-parallel backward schemes), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data file. Inputs are random bf16 tensors of layout (B, H, T, D) and a random output gradient of the same shape, drawn on the GPU from a seed.

The held-out set is a list of about a dozen shapes (B, H, T, D, causal) with head dimension 64 or 128, causal and non-causal masking, sequence lengths from a few hundred to 8,192 including values that are not multiples of 64, and B·H·T² small enough for an fp32 reference. It is not in the sandbox; the self-check uses six public shapes listed in `interface/stages.py`.

## Interface

Implement `attention(q, k, v, causal: bool) -> torch.Tensor` in `kernel.py`. Inputs are contiguous bf16 CUDA tensors of shape (B, H, T, D) with `requires_grad=True` (same T for queries and keys); the output has the same shape and dtype and must be differentiable with respect to q, k and v (for example through a `torch.autograd.Function`); the softmax scale is 1/sqrt(D); with `causal=True`, query i attends to keys 0..i.


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Relative speed** (primary) | Geometric mean over the held-out shapes of t_SDPA / t_yours, where t is the median of 10 CUDA-event-timed forward + backward calls after 3 warm-up calls, then mean over 3 seeds; higher is better | — |
| Correctness | On every shape and for each of out, dq, dk, dv: max \|x − ref\| ≤ max(floor, 4 × SDPA's own max error) and mean \|x − ref\| ≤ max(floor / 10, 4 × SDPA's mean error) against an fp32 reference; floor 0.02 for out, 0.05 for the gradients | Threshold: all shapes |
| Peak memory | Peak memory allocated during one forward + backward beyond the inputs, divided by SDPA's, largest over the shapes | Threshold: ≤ 2.0 |
| Forbidden APIs | Occurrences of SDPA, FlashAttention, xFormers or cuDNN-attention entry points in your source files | Threshold: 0 |
| Slowest shape | The smallest per-shape t_SDPA / t_yours | reported |

**Timing and decision**

- Each seed runs in a new process with empty compiler caches (`interface/abench_stage.py`). Importing `kernel.py` (including any compilation or autotuning at import) must finish within 600 s; all held-out shapes, including the reference computations, timing and memory measurement, must finish within 1,200 s.
- A submission is valid only if every shape is correct, the memory threshold holds and no forbidden API is used; valid submissions are ranked by relative speed.
- If any seed crashes, exceeds a time limit or returns a wrong shape, the whole submission is invalid.
- The final ABench score is driven mainly by relative speed, and also considers how different the method is from existing kernels and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton; `torch.utils.cpp_extension.load_inline` with the CUDA toolkit), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 3 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 1 GPU-hour; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── kernel.py           # required: attention(), differentiable
├── src/                # optional: helper modules, .py only (CUDA sources go in Python strings)
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No attention library calls (SDPA, FlashAttention, xFormers, cuDNN attention) and no stored results; the kernel must compute exact attention and its gradients on the inputs it is given.
- Do not cache outputs or gradients across calls or detect the timing loop; every call must do the full computation.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
