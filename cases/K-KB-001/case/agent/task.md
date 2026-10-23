# K-KB-001: Faster kernels for 30 KernelBench Level-1 operators: correct on five input distributions, timed against eager PyTorch and torch.compile

Anchor: Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini, *KernelBench: Can LLMs Write Efficient GPU Kernels?* (arXiv:2502.10517). Baseline used for comparison: for every problem the faster of PyTorch eager and `torch.compile` (default mode) on the problem's own code, timed by the evaluator in a separate process on the same GPU; the reference submission (not shown) replaces the five activation problems (ReLU, LeakyReLU, sigmoid, hardsigmoid, hardtanh) with a one-pass Triton elementwise kernel and keeps the problem's own PyTorch code for the other 25.

## Task

Write **GPU kernels** for 30 problems of **KernelBench Level 1** — single operators — matrix products, activations, normalisations, pooling, reductions, convolutions, scans and attention — that are **correct** on inputs you have not seen and **as fast as possible relative to the faster of PyTorch eager and `torch.compile`** on one 24GB reference GPU.

- Submit code only: one file `kernels/<problem>.py` per problem with a `ModelNew` class that has the same constructor and `forward` signature as the problem's `Model`. Problems without a file count as failed; you may submit the problem's own PyTorch code as a fallback.
- At evaluation every ModelNew gets Model's weights (same seed, matching parameter names copied), is checked on five input distributions with seeds you do not know (it fails if it modifies any input), timed on the problem's own inputs, and checked once more on fresh inputs after timing and again after those input buffers are overwritten in place (a cache keyed on buffer addresses fails here).
- The problem files are public (KernelBench, MIT licence) and given in the sandbox; the held-out part is the seeds of the weights and inputs, and the timing on the reference GPU.
- Triton, hand-written CUDA via PyTorch's inline extension loading and PyTorch operators are allowed; `torch.compile`, TorchInductor, TorchDynamo and TorchScript are not, and all work must run on the current CUDA stream.
- You may build on any public work (KernelBench solutions, Triton tutorials, fusion and tiling techniques, library kernels), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): the 30 problem files, renamed `L1_<KernelBench file name>` (KernelBench @423217d9, `KernelBench/level1/`, MIT licence), and `problems.json` (name, file and shape overrides of each problem, in evaluation order).

Overrides set module-level size variables of a problem file before `get_inputs()` and `get_init_inputs()` are called, so that every problem's inputs fit within 1 GiB of fp32; problems not listed keep their own sizes. The same overrides apply in the self-check and at evaluation.

| Problem | Overrides |
|---|---|
| `L1_2_Standard_matrix_multiplication_` | — |
| `L1_3_Batched_matrix_multiplication` | `n` = 1024 |
| `L1_4_Matrix_vector_multiplication_` | `K` = 65536 |
| `L1_5_Matrix_scalar_multiplication` | `M` = 16384 |
| `L1_9_Tall_skinny_matrix_multiplication_` | — |
| `L1_18_Matmul_with_transposed_both` | — |
| `L1_19_ReLU` | `dim` = 49152 |
| `L1_20_LeakyReLU` | `dim` = 49152 |
| `L1_21_Sigmoid` | `dim` = 49152 |
| `L1_28_HardSigmoid` | `dim` = 49152 |
| `L1_32_HardTanh` | `dim` = 49152 |
| `L1_35_GroupNorm_` | `dim1` = 128, `dim2` = 256 |
| `L1_36_RMSNorm_` | `dim1` = 128, `dim2` = 256 |
| `L1_37_FrobeniusNorm_` | `dim1` = 128, `dim2` = 256 |
| `L1_38_L1Norm_` | `dim` = 16383, `batch_size` = 16384 |
| `L1_41_Max_Pooling_1D` | `sequence_length` = 16384 |
| `L1_43_Max_Pooling_3D` | `dim1` = 64, `dim2` = 64 |
| `L1_44_Average_Pooling_1D` | `input_length` = 32768 |
| `L1_47_Sum_reduction_over_a_dimension` | `dim1` = 1024, `dim2` = 2047 |
| `L1_51_Argmax_over_a_dimension` | `dim1` = 1024, `dim2` = 2047 |
| `L1_52_Argmin_over_a_dimension` | `dim1` = 1024, `dim2` = 2047 |
| `L1_58_conv_transposed_3D__asymmetric_input__asymmetric_kernel` | — |
| `L1_60_conv_standard_3D__square_input__asymmetric_kernel` | — |
| `L1_64_conv_transposed_1D` | `length` = 32768 |
| `L1_71_conv_transposed_2D__asymmetric_input__square_kernel` | — |
| `L1_80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__` | — |
| `L1_84_conv_depthwise_2D_asymmetric_input_square_kernel` | `width_in` = 128 |
| `L1_90_cumprod` | `batch_size` = 8192 |
| `L1_92_cumsum_exclusive` | `batch_size` = 8192 |
| `L1_97_ScaledDotProductAttention` | `embedding_dimension` = 256, `sequence_length` = 256 |

The held-out set is a table of seeds per problem (weights, five input distributions, the check after timing), drawn from a secret seed; it is not in the sandbox. The self-check uses the same problems with public seeds.

## Interface

One file `kernels/<problem>.py` per problem defining `ModelNew(nn.Module)` with the same constructor arguments as `Model` (from `get_init_inputs()`) and the same `forward` inputs and outputs (from `get_inputs()`); outputs are compared in fp32. Helper modules go in `src/` (the submission directory is on `sys.path`).

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Geometric-mean speedup** (primary) | Per problem, the time of the faster of PyTorch eager and `torch.compile` divided by your time, clipped to [0.1, 10], a failed or missing problem counting as 0.1; geometric mean over the 30 problems; higher is better | — |
| Fraction correct | Problems whose outputs are within 1e-2 + 1e-2·\|ref\| of Model's fp32 output on all five input distributions and on the checks after timing | reported |
| fast_1 | Fraction of problems that are correct and faster than the faster baseline; also against eager alone | reported |
| Geometric-mean speedup against eager | As the primary metric with PyTorch eager as the baseline | reported |
| Memory | Median over correct problems of your peak memory during one call divided by eager's | reported |
| Forbidden APIs | Occurrences of `torch.compile`, TorchInductor, TorchDynamo, TorchScript or stream creation in your source files | Threshold: 0 |
| Changed torch entry points | torch, `torch.nn.functional`, `torch.Tensor`, `torch.cuda` callables or `nn` module `forward` methods replaced by your code | Threshold: 0 |

**Measurement and decision**

- Input distributions: the problem's own `get_inputs()` (native), and the same shapes filled with N(0,1), 8·N(0,1), N(0,1) with 75% zeros, and 100·N(0,1) (numerical range); integer and non-tensor inputs are kept. Positions where Model's output is not finite are skipped.
- One call is timed as the wall-clock time between two device-wide synchronisations, median of 10 calls after 3 warm-up calls, on fixed native inputs. The baselines run first in their own process (`reference` stage), your kernels in a second process (`bench` stage) on the same GPU.
- The reference runs matmuls and convolutions in full fp32 (TF32 disabled), and the evaluator restores these flags before every call; lower precision is allowed inside your own kernels as long as the tolerance holds.
- Importing all `kernels/*.py` (including compilation and autotuning at import) must finish within 900 s, checking and timing all problems within 2,400 s. A problem that raises an error fails; after a CUDA error that may corrupt the context, all later problems fail.
- A submission is valid if both thresholds hold; valid submissions are ranked by the geometric-mean speedup. Differences below 3% are treated as ties (timing noise).
- The final ABench score is driven mainly by the geometric-mean speedup, and also considers how different the kernels are from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton; `torch.utils.cpp_extension.load_inline` with the CUDA toolkit), NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── kernels/            # required: one <problem>.py per problem, each defining ModelNew
├── src/                # optional: helper modules, .py only (CUDA sources go in Python strings)
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 400,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Each ModelNew must compute its problem's output from the inputs it is given on every call: no cached outputs, no detection of the timing loop or of the input distribution, no work on other CUDA streams.
- Do not change torch's functions, modules or global settings for other code (monkey-patching); the evaluator checks torch's entry points before and after every problem.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
