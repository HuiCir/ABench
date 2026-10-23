# U-KAN-009: A network layer that beats the MLP layer at equal parameters: held-out function fitting and Fashion-MNIST

Anchor: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark, *KAN: Kolmogorov-Arnold Networks* (arXiv:2404.19756). Baseline used for comparison: the MLP layer (Linear followed by SiLU), trained by the evaluator next to your layer in every task at the same parameter budget, seed, data and recipe (`interface/layerbench.py`); the reference submission is that same layer (`interface/baseline/layer.py`).

## Task

Design a **network layer** — a replacement for the MLP's linear map followed by a fixed activation, for example learnable activations on edges (Kolmogorov-Arnold networks), basis expansions, gated or rational units — that **gives lower test error than the MLP layer at the same number of parameters**, measured on **twelve held-out function-fitting tasks** and on **Fashion-MNIST** by the **error ratio** to an MLP the evaluator trains next to it.

- Submit code only: `layer.py` with `make_layer(d_in, d_out, final)`. The evaluator builds three-layer networks d_in → h → h → d_out from your layer, chooses the largest width h within each task's parameter budget, picks a learning rate for each task on validation points and trains with a fixed recipe; it does the same for the MLP layer, on the same data and seeds.
- The comparison is the point of the case: claims that new layers beat MLPs often rest on unequal budgets or untuned baselines. Here parameters are matched exactly, the baseline gets the same learning-rate search, and your network's training step may take at most 3 times as long as the MLP's.
- The held-out functions (1–8 inputs; smooth, multiplicative, oscillating, sharp and piecewise-linear parts) are not shown; five dev functions of the same kinds are public.
- You may build on any public work (KANs and their spline, Chebyshev, Fourier, wavelet and radial-basis variants, rational and gated activations, SIREN-style periodic units, mixtures of activations), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only):

| File | Contents |
|---|---|
| `dev_functions.npz` | Five dev functions of 2 inputs (`d1`–`d5`), inputs uniform on [-1, 1]: `<name>_x_train` (2,000), `_x_val` (500), `_x_test` (5,000) and the matching `_y_*`, targets standardised by the training mean and std |
| `fashion_train.npz` | The 60,000 Fashion-MNIST training images (Xiao et al. 2017, MIT licence, `github.com/zalandoresearch/fashion-mnist` @b2617bb6): `x` uint8 (60000, 784), `y` int64 labels 0–9 |

The held-out part is not in the sandbox: twelve functions with the same layout (points drawn with a secret seed; their training and validation points go to the evaluator's training run, their test points to the final test) and the 10,000 Fashion-MNIST test images.

## Interface

Implement `make_layer(d_in, d_out, final) -> torch.nn.Module` in `layer.py`: a module mapping float32 (B, d_in) to (B, d_out); `final` is True for the output layer (it should not squash its output). The evaluator stacks three of them; it counts every parameter of the network against the budget.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Error ratio** (primary) | exp(½·mean over the twelve functions of ln(your error / MLP error) + ½·ln(your Fashion-MNIST error rate / MLP error rate)); function error = test RMSE / std of the test targets (floored at 1e-4), image error floored at 1e-3; lower is better, 1 = the MLP | — |
| Step time | Median training-step time of your network divided by the MLP's, the largest over tasks | Threshold: ≤ 3.0 |
| Budget | Tasks where your network exceeds the budget even at width 1 | Threshold: = 0 |
| Per task | Both errors, the ratio, widths, parameter counts, chosen learning rates, diverged runs; function ratio and image ratio separately; share of functions where you beat the MLP | reported |

**Measurement and decision**

- Budgets: 3,000 parameters per function network, 100,000 for the image network. Widths are chosen by bisection up to 4,096 assuming the parameter count grows with the width.
- Recipe (`interface/layerbench.py`): Adam, learning rate from {1e-3, 3e-3, 1e-2} by validation error, cosine decay, gradient clipping at 1.0, float32; functions: 3,000 full-batch steps on 2,000 points; images: 10 epochs, batches of 256, 55,000 training and 5,000 validation images. Step times are medians over the run on the same GPU.
- The functions stage is killed at 2,400 s and the image stage at 1,200 s (training of both networks included).
- Two runs with seeds you do not know; the error ratio is averaged over runs and reported with a 95% bootstrap interval over test points; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by the error ratio, and also considers how different the layer is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── layer.py            # required: make_layer()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The layer must be a function of its input only: no state carried between calls, no access to the data, the task or the step count, and no change to the optimiser, the recipe or global settings.
- No pretrained weights and no data inside the code.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
